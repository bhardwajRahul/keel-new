//! Port of `packages/util/timeout` (`@deepseek-ai/dsh-timeout`): shared
//! timeout arithmetic, signal fusion, and classification. The library only
//! notifies through abort signals; each capability still owns the mechanism
//! that stops its work and translates timeout reasons into public outcomes.
//!
//! Divergences from the TS original:
//! - The Web `AbortSignal`/`AbortController` pair is reimplemented here on
//!   tokio time ([`AbortSignal`], [`AbortController`]): the JS classes do not
//!   exist in Rust and no workspace dependency supplies an equivalent that
//!   carries an abort *reason*. Arbitrary JS abort reasons narrow to
//!   [`AbortReason`]: a [`TimeoutReason`] or a cancellation message.
//! - Fusion (`AbortSignal.any`) adopts the reason of whichever source aborted
//!   first, like the platform primitive; deadline timers are evaluated
//!   lazily against the tokio clock, so an armed timer's expiry is observable
//!   synchronously without a background task.
//! - `[Symbol.dispose]` becomes `Drop` ([`Deadline`]) plus an idempotent
//!   explicit [`IdleWatchdog::dispose`].
//! - `clampTimeout`'s invalid-hint throw becomes a `Result`
//!   ([`InvalidTimeoutError`]); the internal timer-delay assertion in
//!   [`deadline`] / [`idle_watchdog`] panics, as those arguments are
//!   programmer-owned, not caller hints.
//! - [`MAX_TIMER_DELAY_MS`] is Node's timer clamp; tokio has no such clamp,
//!   but the bound is kept so both implementations reject the same
//!   configurations.
//! - The async-iterator guarded by [`IdleWatchdog::next`] is a
//!   [`futures::Stream`]; `IteratorResult` maps to the stream's
//!   `Option<Item>`.
//! - The Cordis `./invariant` companion is not ported: the package declares
//!   no runtime invariant.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::{Stream, StreamExt};
use tokio::sync::Notify;
use tokio::time::Instant;

/// Internal abort reason carrying a capability-owned code and the deadline
/// that elapsed. Providers translate it through [`timeout_of`] before
/// returning to callers.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{code} after {timeout_ms}ms")]
pub struct TimeoutReason {
    /// Capability-owned timeout code (e.g. `BASH_TIMEOUT`).
    pub code: String,
    /// The deadline that elapsed, in milliseconds.
    pub timeout_ms: f64,
}

impl TimeoutReason {
    /// Build a reason from a capability code and the elapsed deadline.
    pub fn new(code: impl Into<String>, timeout_ms: f64) -> Self {
        Self {
            code: code.into(),
            timeout_ms,
        }
    }
}

/// Largest delay Node schedules without clamping it to one millisecond; kept
/// as the shared upper bound so both implementations reject the same values.
pub const MAX_TIMER_DELAY_MS: f64 = 2_147_483_647.0;

fn assert_timer_delay(timeout_ms: f64, name: &str) {
    if !timeout_ms.is_finite() || timeout_ms <= 0.0 || timeout_ms > MAX_TIMER_DELAY_MS {
        panic!("{name} must be a positive finite number no greater than {MAX_TIMER_DELAY_MS}");
    }
}

/// A caller-supplied timeout hint was rejected; the message names the field
/// so the caller sees which input was bad.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{name} must be a positive finite number")]
pub struct InvalidTimeoutError {
    /// The field name given to [`clamp_timeout`].
    pub name: String,
}

/// Validate a caller's optional timeout hint, use the backend default, then
/// cap it: `min(requested.unwrap_or(def), max)` in milliseconds.
///
/// A supplied hint must be positive and finite; zero is not a public
/// disable-timeout sentinel. The cap also applies to the default itself, so a
/// misconfigured backend never exceeds its own bound.
pub fn clamp_timeout(
    requested: Option<f64>,
    def: f64,
    max: f64,
    name: &str,
) -> Result<f64, InvalidTimeoutError> {
    if let Some(hint) = requested {
        if !hint.is_finite() || hint <= 0.0 {
            return Err(InvalidTimeoutError {
                name: name.to_owned(),
            });
        }
    }
    Ok(requested.unwrap_or(def).min(max))
}

/// Why a signal aborted: this deadline's (or a nested deadline's) timeout, or
/// an ordinary cancellation.
#[derive(Debug, Clone, PartialEq)]
pub enum AbortReason {
    /// A deadline elapsed; classify through [`timeout_of`].
    Timeout(TimeoutReason),
    /// Ordinary cancellation with the canceller's message.
    Cancelled(String),
}

impl AbortReason {
    /// The contained [`TimeoutReason`], scoped to `code` when one is given: a
    /// foreign code follows the ordinary cancellation path (`None`).
    pub fn as_timeout(&self, code: Option<&str>) -> Option<&TimeoutReason> {
        match self {
            Self::Timeout(reason) => match code {
                None => Some(reason),
                Some(code) if reason.code == code => Some(reason),
                Some(_) => None,
            },
            Self::Cancelled(_) => None,
        }
    }
}

impl From<TimeoutReason> for AbortReason {
    fn from(reason: TimeoutReason) -> Self {
        Self::Timeout(reason)
    }
}

impl From<&str> for AbortReason {
    fn from(message: &str) -> Self {
        Self::Cancelled(message.to_owned())
    }
}

impl From<String> for AbortReason {
    fn from(message: String) -> Self {
        Self::Cancelled(message)
    }
}

struct ControllerState {
    // The first abort wins and records when it happened, so fusion can adopt
    // the earliest source's reason.
    aborted: Mutex<Option<(AbortReason, Instant)>>,
    notify: Notify,
}

/// Owner side of an [`AbortSignal`]; the first [`abort`](Self::abort) wins.
pub struct AbortController {
    state: Arc<ControllerState>,
}

impl AbortController {
    /// A controller whose signal has not aborted.
    pub fn new() -> Self {
        Self {
            state: Arc::new(ControllerState {
                aborted: Mutex::new(None),
                notify: Notify::new(),
            }),
        }
    }

    /// The controller's signal; clones observe the same state.
    pub fn signal(&self) -> AbortSignal {
        AbortSignal {
            inner: SignalInner::Controller(self.state.clone()),
        }
    }

    /// Abort with a reason. Only the first abort takes effect; later calls
    /// are no-ops, so an already-settled classification never changes.
    pub fn abort(&self, reason: impl Into<AbortReason>) {
        {
            let mut aborted = self.state.aborted.lock().unwrap();
            if aborted.is_some() {
                return;
            }
            *aborted = Some((reason.into(), Instant::now()));
        }
        self.state.notify.notify_waiters();
    }
}

impl Default for AbortController {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum TimerPhase {
    Armed,
    Cleared,
    Fired,
}

struct TimerState {
    deadline_at: Instant,
    reason: TimeoutReason,
    phase: Mutex<TimerPhase>,
}

impl TimerState {
    fn abort_info(&self) -> Option<(AbortReason, Instant)> {
        let mut phase = self.phase.lock().unwrap();
        match *phase {
            TimerPhase::Cleared => None,
            TimerPhase::Fired => {
                Some((AbortReason::Timeout(self.reason.clone()), self.deadline_at))
            }
            TimerPhase::Armed => {
                if Instant::now() >= self.deadline_at {
                    *phase = TimerPhase::Fired;
                    Some((AbortReason::Timeout(self.reason.clone()), self.deadline_at))
                } else {
                    None
                }
            }
        }
    }

    fn clear(&self) {
        let mut phase = self.phase.lock().unwrap();
        if *phase == TimerPhase::Armed {
            // A timer whose deadline already elapsed stays aborted; clearing
            // only prevents a future expiry.
            *phase = if Instant::now() >= self.deadline_at {
                TimerPhase::Fired
            } else {
                TimerPhase::Cleared
            };
        }
    }
}

#[derive(Clone)]
enum SignalInner {
    Never,
    Controller(Arc<ControllerState>),
    Timer(Arc<TimerState>),
    Any(Arc<[AbortSignal]>),
}

/// A cancellation signal carrying an [`AbortReason`]. The signal only
/// notifies: observers must stop their own work when it aborts. Clones share
/// state.
#[derive(Clone)]
pub struct AbortSignal {
    inner: SignalInner,
}

impl AbortSignal {
    /// A signal that never aborts.
    pub fn never() -> Self {
        Self {
            inner: SignalInner::Never,
        }
    }

    /// Fuse sources: the result is aborted once any source is, and its
    /// reason is the reason of whichever source aborted first (list order
    /// breaks exact ties).
    pub fn any(signals: impl IntoIterator<Item = AbortSignal>) -> Self {
        let mut sources: Vec<AbortSignal> = signals.into_iter().collect();
        match sources.len() {
            0 => Self::never(),
            1 => sources.pop().expect("length checked"),
            _ => Self {
                inner: SignalInner::Any(sources.into()),
            },
        }
    }

    fn abort_info(&self) -> Option<(AbortReason, Instant)> {
        match &self.inner {
            SignalInner::Never => None,
            SignalInner::Controller(state) => state.aborted.lock().unwrap().clone(),
            SignalInner::Timer(timer) => timer.abort_info(),
            SignalInner::Any(children) => children
                .iter()
                .filter_map(AbortSignal::abort_info)
                // min_by keeps the FIRST of equally-early sources, matching
                // the fusion list order on exact ties.
                .min_by(|a, b| a.1.cmp(&b.1)),
        }
    }

    /// Whether the signal has aborted.
    pub fn aborted(&self) -> bool {
        self.abort_info().is_some()
    }

    /// The abort reason, once aborted.
    pub fn reason(&self) -> Option<AbortReason> {
        self.abort_info().map(|(reason, _)| reason)
    }

    /// Resolve when the signal aborts; pends forever on a signal that never
    /// does. Must run inside a tokio runtime when a deadline timer is fused
    /// in.
    pub async fn wait(&self) -> AbortReason {
        self.wait_boxed().await
    }

    // Explicitly boxed so the fused arm can recurse without a Send-inference cycle.
    fn wait_boxed(&self) -> Pin<Box<dyn Future<Output = AbortReason> + Send + '_>> {
        Box::pin(self.wait_inner())
    }

    async fn wait_inner(&self) -> AbortReason {
        match &self.inner {
            SignalInner::Never => std::future::pending().await,
            SignalInner::Controller(state) => loop {
                let notified = state.notify.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                if let Some((reason, _)) = state.aborted.lock().unwrap().clone() {
                    return reason;
                }
                notified.await;
            },
            SignalInner::Timer(timer) => {
                tokio::time::sleep_until(timer.deadline_at).await;
                match timer.abort_info() {
                    Some((reason, _)) => reason,
                    // Cleared before expiry: this timer never aborts.
                    None => std::future::pending().await,
                }
            }
            SignalInner::Any(children) => {
                let waits: Vec<Pin<Box<dyn Future<Output = AbortReason> + Send + '_>>> =
                    children.iter().map(AbortSignal::wait_boxed).collect();
                let (reason, _, _) = futures::future::select_all(waits).await;
                // Re-read the fused classification: a slower waker may lose
                // the first-abort race to an earlier source.
                self.reason().unwrap_or(reason)
            }
        }
    }
}

/// Convert a millisecond count to a [`Duration`] without rounding integral
/// values through the float path.
fn ms_duration(timeout_ms: f64) -> Duration {
    if timeout_ms.fract() == 0.0 {
        Duration::from_millis(timeout_ms as u64)
    } else {
        Duration::from_secs_f64(timeout_ms / 1000.0)
    }
}

/// A deadline signal plus the timer it may have armed; dropping the value
/// clears the timer (an already-elapsed expiry keeps its classification).
pub struct Deadline {
    signal: AbortSignal,
    timer: Option<Arc<TimerState>>,
}

impl Deadline {
    /// The fused signal: aborts on upstream cancellation OR on timeout (the
    /// timeout carries a [`TimeoutReason`]). Clones stay valid after the
    /// `Deadline` is dropped.
    pub fn signal(&self) -> AbortSignal {
        self.signal.clone()
    }
}

impl Drop for Deadline {
    fn drop(&mut self) {
        if let Some(timer) = &self.timer {
            timer.clear();
        }
    }
}

/// Fuse upstream cancellation with an identifiable timeout.
///
/// `timeout_ms <= 0` is the internal no-timer sentinel: no timer is armed and
/// only the upstream signal (or a never-aborting one) is forwarded. The
/// signal only notifies, so callers must stop their own work; classification
/// goes through [`timeout_of`].
///
/// # Panics
/// When a positive `timeout_ms` is not finite or exceeds
/// [`MAX_TIMER_DELAY_MS`].
pub fn deadline(upstream: Option<&AbortSignal>, timeout_ms: f64, code: &str) -> Deadline {
    if timeout_ms <= 0.0 {
        return Deadline {
            signal: upstream.cloned().unwrap_or_else(AbortSignal::never),
            timer: None,
        };
    }
    assert_timer_delay(timeout_ms, "deadline timeout_ms");
    let timer = Arc::new(TimerState {
        deadline_at: Instant::now() + ms_duration(timeout_ms),
        reason: TimeoutReason::new(code, timeout_ms),
        phase: Mutex::new(TimerPhase::Armed),
    });
    let timer_signal = AbortSignal {
        inner: SignalInner::Timer(timer.clone()),
    };
    let signal = match upstream {
        Some(upstream) => AbortSignal::any([upstream.clone(), timer_signal]),
        None => timer_signal,
    };
    Deadline {
        signal,
        timer: Some(timer),
    }
}

/// Recover a timeout reason from a signal. Supplying `code` distinguishes
/// this deadline from a nested upstream deadline: a foreign code follows the
/// ordinary cancellation path (`None`).
pub fn timeout_of(signal: &AbortSignal, code: Option<&str>) -> Option<TimeoutReason> {
    signal.reason()?.as_timeout(code).cloned()
}

/// Misuse of [`IdleWatchdog::next`], mirroring the upstream throws.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IdleWatchdogError {
    /// [`IdleWatchdog::dispose`] already ran.
    #[error("idle_watchdog is disposed")]
    Disposed,
    /// Another [`IdleWatchdog::next`] call is still outstanding.
    #[error("idle_watchdog next is already outstanding")]
    AlreadyOutstanding,
}

struct WatchdogInner {
    outstanding: bool,
    disposed: bool,
    /// The armed expiry; `None` while no demand is outstanding (or after the
    /// timer fired or the watchdog was disposed).
    deadline: Option<Instant>,
}

/// Rearmable timeout around one outstanding stream demand. The timer exists
/// only while [`next`](Self::next) is outstanding, so consumer think time
/// does not count as provider idle time. Construct with [`idle_watchdog`].
pub struct IdleWatchdog {
    signal: AbortSignal,
    ctl: AbortController,
    timeout: Duration,
    timeout_ms: f64,
    code: String,
    inner: Mutex<WatchdogInner>,
}

/// Create a rearmable idle watchdog for an async stream. The returned
/// watchdog's signal is stable for its whole life and only notifies; the
/// stream's producer must observe it to terminate its work.
///
/// # Panics
/// When `timeout_ms` is not a positive finite number no greater than
/// [`MAX_TIMER_DELAY_MS`].
pub fn idle_watchdog(upstream: Option<&AbortSignal>, timeout_ms: f64, code: &str) -> IdleWatchdog {
    assert_timer_delay(timeout_ms, "idle_watchdog timeout_ms");
    let ctl = AbortController::new();
    let signal = match upstream {
        Some(upstream) => AbortSignal::any([upstream.clone(), ctl.signal()]),
        None => ctl.signal(),
    };
    IdleWatchdog {
        signal,
        ctl,
        timeout: ms_duration(timeout_ms),
        timeout_ms,
        code: code.to_owned(),
        inner: Mutex::new(WatchdogInner {
            outstanding: false,
            disposed: false,
            deadline: None,
        }),
    }
}

impl IdleWatchdog {
    /// Stable signal aborted by upstream cancellation or this watchdog's
    /// timeout; clones share state.
    pub fn signal(&self) -> AbortSignal {
        self.signal.clone()
    }

    /// Await one stream demand while the idle timer is armed. On expiry the
    /// watchdog's signal aborts with a [`TimeoutReason`], and the call keeps
    /// awaiting the stream — the signal only notifies, so the producer must
    /// observe it and end the stream.
    pub async fn next<S: Stream + Unpin>(
        &self,
        stream: &mut S,
    ) -> Result<Option<S::Item>, IdleWatchdogError> {
        {
            let mut inner = self.inner.lock().unwrap();
            if inner.disposed {
                return Err(IdleWatchdogError::Disposed);
            }
            if inner.outstanding {
                return Err(IdleWatchdogError::AlreadyOutstanding);
            }
            inner.outstanding = true;
            inner.deadline = Some(Instant::now() + self.timeout);
        }
        let item = self.guarded_next(stream).await;
        let mut inner = self.inner.lock().unwrap();
        inner.outstanding = false;
        inner.deadline = None;
        Ok(item)
    }

    async fn guarded_next<S: Stream + Unpin>(&self, stream: &mut S) -> Option<S::Item> {
        loop {
            let deadline = self.inner.lock().unwrap().deadline;
            let idle_timer = async {
                match deadline {
                    Some(at) => tokio::time::sleep_until(at).await,
                    None => std::future::pending().await,
                }
            };
            tokio::select! {
                item = stream.next() => return item,
                () = idle_timer => {
                    let mut inner = self.inner.lock().unwrap();
                    match inner.deadline {
                        // A pulse moved the expiry while we slept: rearm.
                        Some(actual) if actual > Instant::now() => {}
                        Some(_) => {
                            inner.deadline = None;
                            drop(inner);
                            self.ctl.abort(TimeoutReason::new(&self.code, self.timeout_ms));
                        }
                        // Disposed while sleeping: only stream demand remains.
                        None => {}
                    }
                }
            }
        }
    }

    /// Rearm an outstanding demand after transport activity that yields no
    /// stream item; otherwise a no-op.
    pub fn pulse(&self) {
        let mut inner = self.inner.lock().unwrap();
        if inner.disposed || !inner.outstanding {
            return;
        }
        inner.deadline = Some(Instant::now() + self.timeout);
    }

    /// Clear an armed timer and refuse further demand; idempotent, and also
    /// run by `Drop`.
    pub fn dispose(&self) {
        let mut inner = self.inner.lock().unwrap();
        if inner.disposed {
            return;
        }
        inner.disposed = true;
        inner.deadline = None;
    }
}

impl Drop for IdleWatchdog {
    fn drop(&mut self) {
        self.dispose();
    }
}
