//! Behavior tests for the local bash executor, mirroring the portable parts
//! of upstream `bash-local/tests/executor.spec.ts` (foreground runs and
//! background handles) against real processes. Not ported: the settings
//! document suites (dsh-settings live-reconfiguration is out of scope) and
//! spill-path assertions.

use dsh_cordis::App;
use dsh_shell::{
    BashConfig, ExitStatusMarker, LocalBashExecutor, LocalSubprocessRuntime, ShellExecRequest,
    ShellProcessStatus, parse_exit_status,
};
use dsh_timeout::AbortController;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::time::Duration;

fn executor_with(config: BashConfig) -> (App, Rc<LocalBashExecutor>) {
    let app = App::new();
    let ctx = app.root();
    let subprocess = LocalSubprocessRuntime::provide(&ctx).unwrap();
    let shell = LocalBashExecutor::provide(&ctx, subprocess, config).unwrap();
    (app, shell)
}

fn executor() -> (App, Rc<LocalBashExecutor>) {
    executor_with(BashConfig::default())
}

#[test]
fn run_resolves_output_and_the_effective_timeout() {
    dsh_cordis::run(async {
        let (_app, shell) = executor();
        let result = shell
            .run(shell.resolve(ShellExecRequest::new("echo hello")).unwrap())
            .await
            .unwrap();
        assert_eq!(result.exit_code, Some(0));
        assert_eq!(result.stdout.text, "hello\n");
        assert!(!result.stdout.truncated);
        assert_eq!(result.timeout_ms, 120_000.0);
        assert!(!result.timed_out);
        assert!(!result.aborted);
    });
}

#[test]
fn resolve_fills_the_cwd_and_honors_the_override() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        let (_app, shell) = executor_with(BashConfig {
            cwd: Some(root.to_string_lossy().into_owned()),
            ..BashConfig::default()
        });
        let result = shell
            .run(shell.resolve(ShellExecRequest::new("pwd")).unwrap())
            .await
            .unwrap();
        assert_eq!(result.stdout.text.trim(), root.to_string_lossy());

        let other = tempfile::tempdir().unwrap();
        let other_root = std::fs::canonicalize(other.path()).unwrap();
        let request = ShellExecRequest {
            workdir: Some(other_root.to_string_lossy().into_owned()),
            ..ShellExecRequest::new("pwd")
        };
        let result = shell.run(shell.resolve(request).unwrap()).await.unwrap();
        assert_eq!(result.stdout.text.trim(), other_root.to_string_lossy());
    });
}

#[test]
fn resolve_caps_and_validates_timeout_overrides() {
    let app = App::new();
    let ctx = app.root();
    let subprocess = LocalSubprocessRuntime::provide(&ctx).unwrap();
    let shell = LocalBashExecutor::provide(
        &ctx,
        subprocess.clone(),
        BashConfig {
            timeout_ms: 1_000.0,
            max_timeout_ms: 2_000.0,
            ..BashConfig::default()
        },
    )
    .unwrap();
    let capped = shell
        .resolve(ShellExecRequest {
            timeout_ms: Some(50_000.0),
            ..ShellExecRequest::new("true")
        })
        .unwrap();
    assert_eq!(capped.timeout_ms, 2_000.0);
    let default = shell.resolve(ShellExecRequest::new("true")).unwrap();
    assert_eq!(default.timeout_ms, 1_000.0);
    for bad in [0.0, -5.0, f64::NAN, f64::INFINITY] {
        assert!(
            shell
                .resolve(ShellExecRequest {
                    timeout_ms: Some(bad),
                    ..ShellExecRequest::new("true")
                })
                .is_err()
        );
    }
    // Invalid executor config fails at provide, not at the next command.
    let second = App::new();
    let second_ctx = second.root();
    let second_subprocess = LocalSubprocessRuntime::provide(&second_ctx).unwrap();
    assert!(
        LocalBashExecutor::provide(
            &second_ctx,
            second_subprocess,
            BashConfig {
                timeout_ms: 0.0,
                ..BashConfig::default()
            }
        )
        .is_err()
    );
}

#[test]
fn the_per_call_timeout_kills_and_classifies() {
    dsh_cordis::run(async {
        let (_app, shell) = executor_with(BashConfig {
            grace_ms: 500.0,
            ..BashConfig::default()
        });
        let request = ShellExecRequest {
            timeout_ms: Some(200.0),
            ..ShellExecRequest::new("sleep 10")
        };
        let result = shell.run(shell.resolve(request).unwrap()).await.unwrap();
        assert!(result.timed_out);
        assert!(!result.aborted, "one fused deadline: the first cause wins");
        assert_eq!(result.signal.as_deref(), Some("SIGTERM"));
        assert_eq!(result.timeout_ms, 200.0);
    });
}

#[test]
fn a_caller_abort_classifies_as_aborted_not_timed_out() {
    dsh_cordis::run(async {
        let (_app, shell) = executor();
        let controller = AbortController::new();
        let request = ShellExecRequest {
            signal: Some(controller.signal()),
            ..ShellExecRequest::new("sleep 10")
        };
        tokio::task::spawn_local(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            controller.abort("caller cancelled");
        });
        let result = shell.run(shell.resolve(request).unwrap()).await.unwrap();
        assert!(result.aborted);
        assert!(!result.timed_out);
    });
}

#[test]
fn a_self_killed_command_is_neither_timed_out_nor_aborted() {
    dsh_cordis::run(async {
        let (_app, shell) = executor();
        let result = shell
            .run(
                shell
                    .resolve(ShellExecRequest::new("kill -TERM $$"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(result.signal.as_deref(), Some("SIGTERM"));
        assert!(!result.timed_out);
        assert!(!result.aborted);
    });
}

#[test]
fn a_bad_workdir_rejects_the_foreground_run() {
    dsh_cordis::run(async {
        let (_app, shell) = executor();
        let request = ShellExecRequest {
            workdir: Some("/definitely/not/a/dir".into()),
            ..ShellExecRequest::new("echo never")
        };
        assert!(shell.run(shell.resolve(request).unwrap()).await.is_err());
    });
}

#[test]
fn stdin_env_and_the_managed_snapshot_thread_through() {
    dsh_cordis::run(async {
        let (_app, shell) = executor();
        let mut env = BTreeMap::new();
        env.insert("PLAIN_MARK".to_string(), "m".to_string());
        // The caller cannot displace a managed fact: the snapshot merges last.
        env.insert("DSH_FACT".to_string(), "caller".to_string());
        let mut dsh_env = BTreeMap::new();
        dsh_env.insert("DSH_FACT".to_string(), "managed".to_string());
        let request = ShellExecRequest {
            stdin: Some("ping\n".into()),
            env: Some(env),
            dsh_env: Some(dsh_env),
            ..ShellExecRequest::new("cat; printf 'mark=%s fact=%s' \"$PLAIN_MARK\" \"$DSH_FACT\"")
        };
        let result = shell.run(shell.resolve(request).unwrap()).await.unwrap();
        assert_eq!(result.stdout.text, "ping\nmark=m fact=managed");
    });
}

#[test]
fn dsh_env_keys_outside_the_managed_namespace_fail_at_resolve() {
    let (_app, shell) = executor();
    let mut dsh_env = BTreeMap::new();
    dsh_env.insert("NOT_MANAGED".to_string(), "x".to_string());
    let error = shell
        .resolve(ShellExecRequest {
            dsh_env: Some(dsh_env),
            ..ShellExecRequest::new("true")
        })
        .unwrap_err();
    assert!(error.to_string().contains("DSH_"), "{error}");
}

#[test]
fn the_terminal_overrides_apply_but_a_caller_entry_wins() {
    dsh_cordis::run(async {
        let (_app, shell) = executor();
        let result = shell
            .run(
                shell
                    .resolve(ShellExecRequest::new("printf %s \"$TERM\""))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(result.stdout.text, "dumb");
        let mut env = BTreeMap::new();
        env.insert("TERM".to_string(), "xterm".to_string());
        let request = ShellExecRequest {
            env: Some(env),
            ..ShellExecRequest::new("printf %s \"$TERM\"")
        };
        let result = shell.run(shell.resolve(request).unwrap()).await.unwrap();
        assert_eq!(result.stdout.text, "xterm");
    });
}

#[test]
fn stdout_budget_overrides_raise_stdout_only() {
    dsh_cordis::run(async {
        let (_app, shell) = executor_with(BashConfig {
            max_output_bytes: 8,
            ..BashConfig::default()
        });
        // Default budget: both streams keep an 8-byte tail.
        let result = shell
            .run(
                shell
                    .resolve(ShellExecRequest::new(
                        "printf 1234567890; printf abcdefghij 1>&2",
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(result.stdout.text, "34567890");
        assert!(result.stdout.truncated);
        assert_eq!(result.stderr.text, "cdefghij");
        assert!(result.stderr.truncated);
        // A trusted caller may raise stdout; stderr keeps the executor cap.
        let request = ShellExecRequest {
            stdout_max_bytes: Some(64_000),
            ..ShellExecRequest::new("printf 1234567890; printf abcdefghij 1>&2")
        };
        let result = shell.run(shell.resolve(request).unwrap()).await.unwrap();
        assert_eq!(result.stdout.text, "1234567890");
        assert!(!result.stdout.truncated);
        assert!(result.stderr.truncated);
    });
}

#[test]
fn background_processes_settle_with_readable_output() {
    dsh_cordis::run(async {
        let (_app, shell) = executor();
        let process = shell.start(
            shell
                .resolve(ShellExecRequest::new("echo out; echo err 1>&2"))
                .unwrap(),
        );
        tokio::time::timeout(Duration::from_secs(10), process.done())
            .await
            .unwrap();
        assert_eq!(process.status(), ShellProcessStatus::Completed);
        assert_eq!(process.exit_code(), Some(0));
        let read = process.read_output();
        assert_eq!(read.delta, "out\n[stderr]\nerr\n");
        assert!(!read.lossy);
        // Reads are consuming: nothing is re-delivered.
        assert_eq!(process.read_output().delta, "");
    });
}

#[test]
fn stderr_only_deltas_carry_no_leading_newline() {
    dsh_cordis::run(async {
        let (_app, shell) = executor();
        let process = shell.start(
            shell
                .resolve(ShellExecRequest::new("echo err 1>&2"))
                .unwrap(),
        );
        tokio::time::timeout(Duration::from_secs(10), process.done())
            .await
            .unwrap();
        assert_eq!(process.read_output().delta, "[stderr]\nerr\n");
    });
}

#[test]
fn kill_terminates_once_and_settles_as_killed() {
    dsh_cordis::run(async {
        let (_app, shell) = executor();
        let process = shell.start(shell.resolve(ShellExecRequest::new("sleep 30")).unwrap());
        assert_eq!(process.status(), ShellProcessStatus::Running);
        assert!(process.kill());
        assert!(!process.kill(), "kill is idempotent");
        tokio::time::timeout(Duration::from_secs(10), process.done())
            .await
            .unwrap();
        assert_eq!(process.status(), ShellProcessStatus::Killed);
        assert_eq!(process.signal().as_deref(), Some("SIGTERM"));
    });
}

#[test]
fn kill_after_natural_completion_is_a_no_op() {
    dsh_cordis::run(async {
        let (_app, shell) = executor();
        let process = shell.start(shell.resolve(ShellExecRequest::new("true")).unwrap());
        tokio::time::timeout(Duration::from_secs(10), process.done())
            .await
            .unwrap();
        assert_eq!(process.status(), ShellProcessStatus::Completed);
        assert!(!process.kill());
        assert_eq!(process.status(), ShellProcessStatus::Completed);
    });
}

#[test]
fn a_background_abort_settles_the_handle_as_killed() {
    dsh_cordis::run(async {
        let (_app, shell) = executor();
        let controller = AbortController::new();
        let request = ShellExecRequest {
            signal: Some(controller.signal()),
            ..ShellExecRequest::new("sleep 30")
        };
        let process = shell.start(shell.resolve(request).unwrap());
        controller.abort("caller cancelled");
        tokio::time::timeout(Duration::from_secs(10), process.done())
            .await
            .unwrap();
        assert_eq!(process.status(), ShellProcessStatus::Killed);
    });
}

#[test]
fn a_self_signal_exit_settles_the_handle_as_killed() {
    dsh_cordis::run(async {
        let (_app, shell) = executor();
        let process = shell.start(
            shell
                .resolve(ShellExecRequest::new("kill -TERM $$"))
                .unwrap(),
        );
        tokio::time::timeout(Duration::from_secs(10), process.done())
            .await
            .unwrap();
        assert_eq!(process.status(), ShellProcessStatus::Killed);
        assert_eq!(process.signal().as_deref(), Some("SIGTERM"));
    });
}

#[test]
fn a_background_spawn_failure_settles_as_killed_with_the_note_on_stderr() {
    dsh_cordis::run(async {
        let (_app, shell) = executor();
        let request = ShellExecRequest {
            workdir: Some("/definitely/not/a/dir".into()),
            ..ShellExecRequest::new("echo never")
        };
        let process = shell.start(shell.resolve(request).unwrap());
        tokio::time::timeout(Duration::from_secs(10), process.done())
            .await
            .unwrap();
        assert_eq!(process.status(), ShellProcessStatus::Killed);
        let read = process.read_output();
        assert!(
            read.delta.starts_with("[stderr]\nspawn failed:"),
            "{}",
            read.delta
        );
        // The note is delivered exactly once.
        assert_eq!(process.read_output().delta, "");
    });
}

#[test]
fn background_output_truncation_flags_lossy_reads() {
    dsh_cordis::run(async {
        let (_app, shell) = executor_with(BashConfig {
            max_output_bytes: 8,
            ..BashConfig::default()
        });
        let process = shell.start(
            shell
                .resolve(ShellExecRequest::new("printf 1234567890ABCDEF"))
                .unwrap(),
        );
        tokio::time::timeout(Duration::from_secs(10), process.done())
            .await
            .unwrap();
        let read = process.read_output();
        assert!(read.lossy);
        assert_eq!(read.delta, "90ABCDEF");
    });
}

#[test]
fn parse_exit_status_recovers_the_marker_contract() {
    let clean = parse_exit_status("all good\n");
    assert_eq!(clean.status, ExitStatusMarker::Code(0));
    assert_eq!(clean.body, "all good\n");

    let coded = parse_exit_status("boom\n[exit code: 3]");
    assert_eq!(coded.status, ExitStatusMarker::Code(3));
    assert_eq!(coded.body, "boom");

    let signalled = parse_exit_status("out\n[killed by signal: SIGKILL]");
    assert_eq!(signalled.status, ExitStatusMarker::Signal("SIGKILL".into()));
    assert_eq!(signalled.body, "out");

    // Marker-like text that is not the final line stays in the body.
    let embedded = parse_exit_status("saw [exit code: 9] earlier\ndone");
    assert_eq!(embedded.status, ExitStatusMarker::Code(0));
    assert_eq!(embedded.body, "saw [exit code: 9] earlier\ndone");

    // Only the exit marker is consumed; other markers stay in the body.
    let with_timeout = parse_exit_status("out\n[timed out after 100ms]\n[exit code: 143]");
    assert_eq!(with_timeout.status, ExitStatusMarker::Code(143));
    assert_eq!(with_timeout.body, "out\n[timed out after 100ms]");
}
