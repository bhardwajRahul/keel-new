//! Behavior tests for the managed subprocess layer against real child
//! processes, mirroring the portable parts of upstream
//! `subprocess-local/tests/spawn.spec.ts` + `local.spec.ts`. Not ported:
//! Windows taskkill/PATHEXT suites, spill-file coverage (spill is out of
//! scope), the terminal tier, and the /proc zombie-group inspector.

use dsh_cordis::App;
use dsh_shell::{
    LocalSubprocessRuntime, SubprocessCollect, SubprocessSpawnSpec, SubprocessStdinMode,
    SubprocessStdio, child_env, is_sensitive_env_key, scrubbed_parent_env,
};
use dsh_timeout::AbortController;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::time::Duration;

fn runtime() -> (App, Rc<LocalSubprocessRuntime>) {
    let app = App::new();
    let ctx = app.root();
    let runtime = LocalSubprocessRuntime::provide(&ctx).unwrap();
    (app, runtime)
}

fn bash_spec(command: &str) -> SubprocessSpawnSpec {
    SubprocessSpawnSpec {
        argv: vec!["bash".into(), "-c".into(), command.into()],
        cwd: std::env::temp_dir().to_string_lossy().into_owned(),
        stdio: SubprocessStdio {
            stdin: SubprocessStdinMode::Ignore,
            stdout: SubprocessCollect { max_bytes: 64_000 },
            stderr: SubprocessCollect { max_bytes: 64_000 },
        },
        grace_ms: 3_000.0,
        signal: None,
        env: BTreeMap::new(),
    }
}

async fn done_within(
    handle: &dsh_shell::SubprocessHandle,
    seconds: u64,
) -> dsh_shell::SubprocessOutcome {
    tokio::time::timeout(Duration::from_secs(seconds), handle.done())
        .await
        .expect("process must settle in time")
}

#[test]
fn spawn_captures_stdout_and_exit_facts() {
    dsh_cordis::run(async {
        let (_app, runtime) = runtime();
        let handle = runtime.spawn(bash_spec("echo hi; echo err 1>&2")).unwrap();
        let outcome = done_within(&handle, 10).await;
        assert_eq!(outcome.exit_code, Some(0));
        assert_eq!(outcome.signal, None);
        assert_eq!(handle.stdout().read_from(0).text, "hi\n");
        assert_eq!(handle.stderr().read_from(0).text, "err\n");
    });
}

#[test]
fn nonzero_exits_resolve_with_the_code() {
    dsh_cordis::run(async {
        let (_app, runtime) = runtime();
        let handle = runtime.spawn(bash_spec("exit 3")).unwrap();
        assert_eq!(done_within(&handle, 10).await.exit_code, Some(3));
    });
}

#[test]
fn the_environment_scrub_drops_secrets_and_dsh_names() {
    // set_var mutates process state; this is the only test that does so.
    unsafe {
        std::env::set_var("MY_PROBE_TOKEN", "leak");
        std::env::set_var("DSH_PROBE_FACT", "leak");
        std::env::set_var("MY_PROBE_PLAIN", "keep");
    }
    let env = scrubbed_parent_env();
    assert!(!env.contains_key("MY_PROBE_TOKEN"));
    assert!(!env.contains_key("DSH_PROBE_FACT"));
    assert_eq!(env.get("MY_PROBE_PLAIN").map(String::as_str), Some("keep"));
    assert!(
        env.contains_key("PATH"),
        "PATH survives so child CLIs run normally"
    );

    // Explicit entries merge AFTER the scrub: a deliberate forward survives,
    // a tombstone removes an ambient entry.
    let mut extra: BTreeMap<String, Option<String>> = BTreeMap::new();
    extra.insert("MY_PROBE_TOKEN".into(), Some("forwarded".into()));
    extra.insert("MY_PROBE_PLAIN".into(), None);
    let merged = child_env(&extra);
    assert_eq!(
        merged.get("MY_PROBE_TOKEN").map(String::as_str),
        Some("forwarded")
    );
    assert!(!merged.contains_key("MY_PROBE_PLAIN"));
    unsafe {
        std::env::remove_var("MY_PROBE_TOKEN");
        std::env::remove_var("DSH_PROBE_FACT");
        std::env::remove_var("MY_PROBE_PLAIN");
    }

    assert!(is_sensitive_env_key("aws_secret_access_key"));
    assert!(is_sensitive_env_key("GITHUB_TOKEN"));
    assert!(!is_sensitive_env_key("HOME"));
}

#[test]
fn the_scrubbed_environment_reaches_the_child() {
    dsh_cordis::run(async {
        let (_app, runtime) = runtime();
        let mut spec = bash_spec(
            "printf 'token=%s dsh=%s mark=%s' \"${FORWARDED_TOKEN:-none}\" \"${DSH_CHILD_FACT:-none}\" \"${PLAIN_MARK:-none}\"",
        );
        spec.env
            .insert("FORWARDED_TOKEN".into(), Some("yes".into()));
        spec.env
            .insert("DSH_CHILD_FACT".into(), Some("fact".into()));
        spec.env.insert("PLAIN_MARK".into(), Some("m".into()));
        let handle = runtime.spawn(spec).unwrap();
        done_within(&handle, 10).await;
        assert_eq!(
            handle.stdout().read_from(0).text,
            "token=yes dsh=fact mark=m"
        );
    });
}

#[test]
fn batch_stdin_is_written_and_closed() {
    dsh_cordis::run(async {
        let (_app, runtime) = runtime();
        let mut spec = bash_spec("cat");
        spec.stdio.stdin = SubprocessStdinMode::Data("ping\n".into());
        let handle = runtime.spawn(spec).unwrap();
        let outcome = done_within(&handle, 10).await;
        assert_eq!(outcome.exit_code, Some(0));
        assert_eq!(handle.stdout().read_from(0).text, "ping\n");
    });
}

#[test]
fn collection_keeps_a_byte_exact_tail_and_reports_the_loss() {
    dsh_cordis::run(async {
        let (_app, runtime) = runtime();
        let mut spec = bash_spec("printf 1234567890ABCDEF");
        spec.stdio.stdout = SubprocessCollect { max_bytes: 10 };
        let handle = runtime.spawn(spec).unwrap();
        done_within(&handle, 10).await;
        let read = handle.stdout().read_from(0);
        assert_eq!(read.text, "7890ABCDEF");
        assert!(read.lossy, "the head slid out of the tail window");
        assert_eq!(read.next_offset, 16);
    });
}

#[test]
fn reads_are_offset_based_and_non_consuming() {
    dsh_cordis::run(async {
        let (_app, runtime) = runtime();
        let handle = runtime.spawn(bash_spec("printf hello")).unwrap();
        done_within(&handle, 10).await;
        let first = handle.stdout().read_from(0);
        assert_eq!(first.text, "hello");
        assert!(!first.lossy);
        // Resuming from the cursor yields the empty delta; an independent
        // reader from zero still sees everything.
        let resumed = handle.stdout().read_from(first.next_offset);
        assert_eq!(resumed.text, "");
        assert!(!resumed.lossy);
        assert_eq!(handle.stdout().read_from(0).text, "hello");
    });
}

#[test]
fn terminate_escalates_from_sigterm() {
    dsh_cordis::run(async {
        let (_app, runtime) = runtime();
        let handle = runtime.spawn(bash_spec("sleep 30")).unwrap();
        handle.terminate();
        let outcome = done_within(&handle, 10).await;
        assert_eq!(outcome.exit_code, None);
        assert_eq!(outcome.signal.as_deref(), Some("SIGTERM"));
        // Idempotent after settlement.
        handle.terminate();
    });
}

#[test]
fn the_grace_period_escalates_a_term_trapping_group_to_sigkill() {
    dsh_cordis::run(async {
        let (_app, runtime) = runtime();
        let mut spec = bash_spec("trap '' TERM; while true; do sleep 0.05; done");
        spec.grace_ms = 300.0;
        let handle = runtime.spawn(spec).unwrap();
        // Give bash a beat to install the trap before signalling.
        tokio::time::sleep(Duration::from_millis(150)).await;
        handle.terminate();
        let outcome = done_within(&handle, 10).await;
        assert_eq!(outcome.signal.as_deref(), Some("SIGKILL"));
    });
}

#[test]
fn the_abort_signal_starts_the_escalation() {
    dsh_cordis::run(async {
        let (_app, runtime) = runtime();
        let controller = AbortController::new();
        let mut spec = bash_spec("sleep 30");
        spec.signal = Some(controller.signal());
        let handle = runtime.spawn(spec).unwrap();
        tokio::task::spawn_local(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            controller.abort("caller cancelled");
        });
        let outcome = done_within(&handle, 10).await;
        assert_eq!(outcome.signal.as_deref(), Some("SIGTERM"));
    });
}

#[test]
fn spawn_validates_its_inputs() {
    dsh_cordis::run(async {
        let (_app, runtime) = runtime();
        // Pre-aborted signal.
        let controller = AbortController::new();
        controller.abort("early");
        let mut spec = bash_spec("echo never");
        spec.signal = Some(controller.signal());
        assert!(
            runtime
                .spawn(spec)
                .unwrap_err()
                .to_string()
                .contains("aborted before spawn")
        );
        // Grace bounds.
        let mut spec = bash_spec("echo never");
        spec.grace_ms = 0.0;
        assert!(
            runtime
                .spawn(spec)
                .unwrap_err()
                .to_string()
                .contains("grace_ms")
        );
        // Empty argv.
        let mut spec = bash_spec("echo never");
        spec.argv = vec![];
        assert!(
            runtime
                .spawn(spec)
                .unwrap_err()
                .to_string()
                .contains("argv")
        );
        // A bad cwd fails the spawn itself.
        let mut spec = bash_spec("echo never");
        spec.cwd = "/definitely/not/a/dir".into();
        assert!(runtime.spawn(spec).is_err());
    });
}

#[test]
fn group_kill_reaches_descendants() {
    dsh_cordis::run(async {
        let (_app, runtime) = runtime();
        // The child forks a grandchild that prints after a delay; killing the
        // GROUP prevents the grandchild's output from ever arriving.
        let handle = runtime
            .spawn(bash_spec("(sleep 5; echo grandchild) & wait"))
            .unwrap();
        tokio::time::sleep(Duration::from_millis(150)).await;
        handle.terminate();
        let outcome = done_within(&handle, 10).await;
        assert_eq!(outcome.exit_code, None);
        assert!(!handle.stdout().read_from(0).text.contains("grandchild"));
    });
}
