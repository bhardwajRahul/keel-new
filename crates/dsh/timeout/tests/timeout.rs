//! Port of `packages/util/timeout/tests/timeout.spec.ts`. Fake timers map to
//! `tokio::test(start_paused = true)` + `tokio::time::advance`; the
//! `using`-disposer assertions map to `drop`/`dispose`; iterator fixtures map
//! to `futures` unbounded channels polled manually so arming, rearming, and
//! expiry are observed at exact instants.

use std::error::Error;
use std::pin::pin;
use std::task::Poll;
use std::time::Duration;

use dsh_timeout::{
    AbortController, AbortReason, IdleWatchdogError, MAX_TIMER_DELAY_MS, TimeoutReason,
    clamp_timeout, deadline, idle_watchdog, timeout_of,
};
use futures::channel::mpsc;
use futures::poll;
use tokio::time::advance;

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

// ---- TimeoutReason ----

#[test]
fn timeout_reason_is_an_error_carrying_the_code_and_elapsed_ms() {
    let reason = TimeoutReason::new("BASH_TIMEOUT", 100.0);
    assert_eq!(reason.code, "BASH_TIMEOUT");
    assert_eq!(reason.timeout_ms, 100.0);
    assert_eq!(reason.to_string(), "BASH_TIMEOUT after 100ms");
    let _: &dyn Error = &reason; // participates in the error trait like the JS Error subclass
}

// ---- clamp_timeout ----

#[test]
fn fills_the_default_when_the_hint_is_absent() {
    assert_eq!(
        clamp_timeout(None, 120_000.0, 600_000.0, "timeoutMs"),
        Ok(120_000.0)
    );
}

#[test]
fn caps_the_hint_at_max() {
    assert_eq!(
        clamp_timeout(Some(999_999.0), 120_000.0, 600_000.0, "timeoutMs"),
        Ok(600_000.0)
    );
}

#[test]
fn keeps_a_valid_hint_under_the_cap() {
    assert_eq!(
        clamp_timeout(Some(5_000.0), 120_000.0, 600_000.0, "timeoutMs"),
        Ok(5_000.0)
    );
}

#[test]
fn caps_the_default_itself_when_the_default_exceeds_max() {
    // min(def, max) applies even with no hint — a misconfigured backend never
    // exceeds its own cap.
    assert_eq!(
        clamp_timeout(None, 900_000.0, 600_000.0, "timeoutMs"),
        Ok(600_000.0)
    );
}

#[test]
fn rejects_a_non_finite_hint_with_the_caller_provided_name() {
    let error = clamp_timeout(
        Some(f64::NAN),
        100.0,
        200.0,
        "bash-local: request.timeoutMs",
    )
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "bash-local: request.timeoutMs must be a positive finite number"
    );
    let error = clamp_timeout(Some(f64::INFINITY), 100.0, 200.0, "timeoutMs").unwrap_err();
    assert_eq!(
        error.to_string(),
        "timeoutMs must be a positive finite number"
    );
}

#[test]
fn rejects_a_non_positive_hint() {
    assert!(clamp_timeout(Some(0.0), 100.0, 200.0, "timeoutMs").is_err());
    assert!(clamp_timeout(Some(-1.0), 100.0, 200.0, "timeoutMs").is_err());
}

// ---- deadline: timeout arm ----

#[tokio::test(start_paused = true)]
async fn aborts_on_timeout_with_a_timeout_reason_after_the_elapsed_ms() {
    let d = deadline(None, 100.0, "BASH_TIMEOUT");
    assert!(!d.signal().aborted());
    advance(ms(100)).await;
    assert!(d.signal().aborted());
    let reason = timeout_of(&d.signal(), None).expect("timeout classification");
    assert_eq!(reason.code, "BASH_TIMEOUT");
    assert_eq!(reason.timeout_ms, 100.0);
}

#[tokio::test(start_paused = true)]
async fn drop_clears_the_timer_so_no_abort_fires_afterward() {
    let d = deadline(None, 100.0, "BASH_TIMEOUT");
    let signal = d.signal();
    drop(d);
    advance(ms(1_000)).await;
    assert!(!signal.aborted());
    assert!(timeout_of(&signal, None).is_none());
}

#[test]
#[should_panic(expected = "no greater than 2147483647")]
fn rejects_delays_above_the_shared_timer_bound() {
    deadline(None, MAX_TIMER_DELAY_MS + 1.0, "BASH_TIMEOUT");
}

#[test]
#[should_panic(expected = "no greater than 2147483647")]
fn rejects_an_infinite_delay() {
    deadline(None, f64::INFINITY, "BASH_TIMEOUT");
}

// ---- deadline: fuse with upstream ----

#[tokio::test]
async fn aborts_on_upstream_cancellation_classified_as_not_a_timeout() {
    let upstream = AbortController::new();
    let d = deadline(Some(&upstream.signal()), 60_000.0, "BASH_TIMEOUT");
    upstream.abort("user cancelled");
    assert!(d.signal().aborted());
    assert!(timeout_of(&d.signal(), None).is_none());
}

#[tokio::test(start_paused = true)]
async fn cancel_wins_when_it_fires_before_the_timeout() {
    let upstream = AbortController::new();
    let d = deadline(Some(&upstream.signal()), 100.0, "BASH_TIMEOUT");
    upstream.abort("user cancelled"); // fires first, before the 100ms timer
    advance(ms(200)).await;
    assert!(d.signal().aborted());
    // Fusion adopts the FIRST source's reason: cancel won, so no
    // TimeoutReason even though the timer later elapsed.
    assert!(timeout_of(&d.signal(), None).is_none());
}

#[tokio::test(start_paused = true)]
async fn timeout_wins_when_it_fires_before_upstream_cancellation() {
    let upstream = AbortController::new();
    let d = deadline(Some(&upstream.signal()), 100.0, "WEB_FETCH_TIMEOUT");
    advance(ms(150)).await; // past the 100ms deadline: the timer fires first
    assert!(d.signal().aborted());
    assert_eq!(
        timeout_of(&d.signal(), None).unwrap().code,
        "WEB_FETCH_TIMEOUT"
    );
    // A later upstream abort is a no-op on the already-aborted fused signal:
    // the first cause stands, so the timeout classification survives.
    upstream.abort("too late");
    assert_eq!(
        timeout_of(&d.signal(), None).unwrap().code,
        "WEB_FETCH_TIMEOUT"
    );
}

#[tokio::test]
async fn forwards_a_pre_aborted_upstream_signal_immediately() {
    let upstream = AbortController::new();
    upstream.abort("already gone");
    let d = deadline(Some(&upstream.signal()), 60_000.0, "BASH_TIMEOUT");
    assert!(d.signal().aborted());
    assert!(timeout_of(&d.signal(), None).is_none());
}

// ---- deadline: timeout_ms <= 0 (no-timeout sentinel) ----

#[tokio::test(start_paused = true)]
async fn arms_no_timer_and_forwards_only_the_upstream_signal() {
    let upstream = AbortController::new();
    let d = deadline(Some(&upstream.signal()), 0.0, "BASH_TIMEOUT");
    advance(ms(1_000_000)).await;
    assert!(!d.signal().aborted()); // no timer ever armed
    upstream.abort("kill");
    assert!(d.signal().aborted());
    assert!(timeout_of(&d.signal(), None).is_none()); // never a timeout
}

#[tokio::test(start_paused = true)]
async fn returns_a_never_aborting_signal_with_a_noop_disposer_when_there_is_no_upstream() {
    let d = deadline(None, 0.0, "BASH_TIMEOUT");
    let signal = d.signal();
    drop(d); // the no-timer disposer is a no-op
    advance(ms(1_000_000)).await;
    assert!(!signal.aborted());
    assert!(timeout_of(&signal, None).is_none());
}

#[test]
fn treats_a_negative_timeout_the_same_as_zero() {
    let d = deadline(None, -5.0, "BASH_TIMEOUT");
    assert!(!d.signal().aborted());
    drop(d);
}

// ---- timeout_of / AbortReason::as_timeout ----

#[test]
fn classifies_a_bare_reason_carrier_that_holds_a_timeout_reason() {
    let inner = TimeoutReason::new("WEB_FETCH_TIMEOUT", 50.0);
    let reason = AbortReason::Timeout(inner.clone());
    assert_eq!(reason.as_timeout(None), Some(&inner));
}

#[test]
fn returns_none_for_a_non_timeout_reason() {
    assert_eq!(
        AbortReason::Cancelled("other".into()).as_timeout(None),
        None
    );
    assert_eq!(
        AbortReason::Cancelled("user cancelled".into()).as_timeout(None),
        None
    );
    // The empty reason carrier: a signal that never aborted classifies as no timeout.
    assert!(timeout_of(&dsh_timeout::AbortSignal::never(), None).is_none());
}

#[test]
fn matches_only_the_requested_code_when_one_is_given() {
    let inner = TimeoutReason::new("BASH_TIMEOUT", 100.0);
    let reason = AbortReason::Timeout(inner.clone());
    assert_eq!(reason.as_timeout(Some("BASH_TIMEOUT")), Some(&inner));
    assert_eq!(reason.as_timeout(Some("WEB_FETCH_TIMEOUT")), None);
}

// ---- deadline: nested deadlines ----

#[tokio::test]
async fn does_not_misclassify_an_outer_deadlines_timeout_as_the_inner_code() {
    // The upstream handed to the inner deadline has ITSELF timed out (outer).
    // Fusion preserves that reason, but scoping timeout_of to the inner code
    // must classify it as upstream cancellation rather than the inner
    // capability's timeout.
    let outer = AbortController::new();
    outer.abort(TimeoutReason::new("OUTER_TIMEOUT", 30.0));
    let inner = deadline(Some(&outer.signal()), 60_000.0, "BASH_TIMEOUT");
    assert!(inner.signal().aborted());
    assert!(timeout_of(&inner.signal(), Some("BASH_TIMEOUT")).is_none()); // not ours → upstream-cancel path
    assert_eq!(
        timeout_of(&inner.signal(), None).unwrap().code,
        "OUTER_TIMEOUT"
    ); // but IS a timeout, unscoped
}

// ---- AbortSignal::wait ----

#[tokio::test(start_paused = true)]
async fn wait_resolves_with_the_first_abort_reason() {
    let d = deadline(None, 100.0, "WAIT_TIMEOUT");
    let signal = d.signal();
    // The paused clock auto-advances to the armed deadline while the runtime idles.
    let reason = tokio::spawn(async move { signal.wait().await })
        .await
        .unwrap();
    assert_eq!(reason.as_timeout(None).unwrap().code, "WAIT_TIMEOUT");

    let ctl = AbortController::new();
    ctl.abort("done");
    assert_eq!(
        ctl.signal().wait().await,
        AbortReason::Cancelled("done".into())
    );
}

// ---- idle_watchdog ----

#[tokio::test(start_paused = true)]
async fn arms_only_while_next_is_outstanding_and_rearms_the_same_signal_for_later_demand() {
    let (tx, mut rx) = mpsc::unbounded::<i32>();
    let watchdog = idle_watchdog(None, 100.0, "LLM_STREAM_IDLE_TIMEOUT");
    let stable = watchdog.signal();

    {
        let mut next = pin!(watchdog.next(&mut rx));
        assert!(poll!(next.as_mut()).is_pending());
        advance(ms(99)).await;
        assert!(poll!(next.as_mut()).is_pending());
        assert!(!stable.aborted());
        tx.unbounded_send(1).unwrap();
        assert_eq!(poll!(next.as_mut()), Poll::Ready(Ok(Some(1))));
    }

    // Demand settled: consumer think time does not count as provider idle time.
    advance(ms(10_000)).await;
    assert!(!stable.aborted());

    let mut next = pin!(watchdog.next(&mut rx));
    assert!(poll!(next.as_mut()).is_pending());
    advance(ms(100)).await;
    assert!(poll!(next.as_mut()).is_pending()); // signal aborted; the stream is still awaited
    let reason = timeout_of(&stable, Some("LLM_STREAM_IDLE_TIMEOUT")).unwrap();
    assert_eq!(reason.timeout_ms, 100.0);
    drop(tx); // the provider observes the signal and ends the stream
    assert_eq!(poll!(next.as_mut()), Poll::Ready(Ok(None)));
}

#[tokio::test(start_paused = true)]
async fn rearms_outstanding_demand_on_an_out_of_band_activity_pulse() {
    let (tx, mut rx) = mpsc::unbounded::<i32>();
    let watchdog = idle_watchdog(None, 100.0, "LLM_STREAM_IDLE_TIMEOUT");
    watchdog.pulse(); // no demand outstanding: a no-op
    advance(ms(1_000)).await;
    assert!(!watchdog.signal().aborted());

    {
        let mut next = pin!(watchdog.next(&mut rx));
        assert!(poll!(next.as_mut()).is_pending());
        advance(ms(99)).await;
        watchdog.pulse();
        assert!(poll!(next.as_mut()).is_pending());
        advance(ms(99)).await;
        assert!(poll!(next.as_mut()).is_pending());
        assert!(!watchdog.signal().aborted());
        advance(ms(1)).await;
        assert!(poll!(next.as_mut()).is_pending());
        let reason = timeout_of(&watchdog.signal(), Some("LLM_STREAM_IDLE_TIMEOUT")).unwrap();
        assert_eq!(reason.timeout_ms, 100.0);
        drop(tx);
        assert_eq!(poll!(next.as_mut()), Poll::Ready(Ok(None)));
    }

    watchdog.dispose();
    watchdog.pulse(); // pulse after dispose stays a no-op
}

#[tokio::test(start_paused = true)]
async fn keeps_an_earlier_upstream_abort_distinct_from_its_own_timeout() {
    let upstream = AbortController::new();
    let watchdog = idle_watchdog(Some(&upstream.signal()), 100.0, "LLM_STREAM_IDLE_TIMEOUT");
    upstream.abort("caller cancelled");
    assert!(watchdog.signal().aborted());
    assert!(timeout_of(&watchdog.signal(), Some("LLM_STREAM_IDLE_TIMEOUT")).is_none());
    advance(ms(1_000)).await;
    assert_eq!(
        watchdog.signal().reason(),
        Some(AbortReason::Cancelled("caller cancelled".into()))
    );
}

#[tokio::test(start_paused = true)]
async fn clears_an_outstanding_arm_on_disposal() {
    let (_tx, mut rx) = mpsc::unbounded::<i32>();
    let watchdog = idle_watchdog(None, 100.0, "LLM_STREAM_IDLE_TIMEOUT");
    {
        let mut next = pin!(watchdog.next(&mut rx));
        assert!(poll!(next.as_mut()).is_pending());
        watchdog.dispose();
        advance(ms(1_000)).await;
        assert!(poll!(next.as_mut()).is_pending()); // armed timer cleared: no abort fired
        assert!(!watchdog.signal().aborted());
    }
    let (_tx2, mut rx2) = mpsc::unbounded::<i32>();
    assert_eq!(
        watchdog.next(&mut rx2).await,
        Err(IdleWatchdogError::Disposed)
    );
    watchdog.dispose(); // idempotent
}

#[test]
#[should_panic(expected = "positive finite")]
fn idle_watchdog_rejects_a_zero_interval() {
    idle_watchdog(None, 0.0, "IDLE");
}

#[test]
#[should_panic(expected = "positive finite")]
fn idle_watchdog_rejects_a_nan_interval() {
    idle_watchdog(None, f64::NAN, "IDLE");
}

#[test]
#[should_panic(expected = "no greater than 2147483647")]
fn idle_watchdog_rejects_an_interval_above_the_shared_timer_bound() {
    idle_watchdog(None, MAX_TIMER_DELAY_MS + 1.0, "IDLE");
}

#[tokio::test(start_paused = true)]
async fn rejects_concurrent_stream_demand() {
    let (tx, mut rx) = mpsc::unbounded::<i32>();
    let (_tx2, mut rx2) = mpsc::unbounded::<i32>();
    let watchdog = idle_watchdog(None, 100.0, "IDLE");
    let mut next = pin!(watchdog.next(&mut rx));
    assert!(poll!(next.as_mut()).is_pending());
    assert_eq!(
        watchdog.next(&mut rx2).await,
        Err(IdleWatchdogError::AlreadyOutstanding)
    );
    tx.unbounded_send(0).unwrap();
    assert_eq!(poll!(next.as_mut()), Poll::Ready(Ok(Some(0))));
}
