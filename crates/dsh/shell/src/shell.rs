//! The `ctx.shell` capability (port of upstream `packages/shell/shell` +
//! `packages/shell/bash-local`, collapsed: the local bash executor is the
//! only backend ported — pwsh and the sandboxing executors are out of scope
//! — so it registers directly as the `shell` service and its methods are the
//! seam contract).
//!
//! The request/spec split is the load-bearing contract: defaulting is the
//! explicit [`LocalBashExecutor::resolve`] step that fills and caps every
//! field from the executor's config; [`LocalBashExecutor::run`] and
//! [`LocalBashExecutor::start`] take only resolved specs and never
//! re-default. Managed `DSH_*` variables merge last onto the child
//! environment, after the subprocess scrub dropped every ambient `DSH_*`
//! entry, so a stale harness fact can never leak through and an ordinary
//! `env` entry can never displace a managed one.

use crate::subprocess::{
    DSH_ENV_PREFIX, LocalSubprocessRuntime, SubprocessCollect, SubprocessOutcome,
    SubprocessSpawnSpec, SubprocessStdinMode, SubprocessStdio,
};
use dsh_cordis::{Context, Service};
use dsh_timeout::{AbortSignal, MAX_TIMER_DELAY_MS, clamp_timeout, deadline, timeout_of};
use futures::FutureExt;
use futures::future::{LocalBoxFuture, Shared};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::rc::Rc;

/// Model-friendly environment overrides: disable colors, pagers, and
/// interactive terminal features that would garble tool output. Merged first
/// into the spawn's explicit env, so a trusted caller's own entry still
/// wins.
pub const ENV_OVERRIDES: &[(&str, &str)] = &[
    ("NO_COLOR", "1"),
    ("TERM", "dumb"),
    ("PAGER", "cat"),
    ("GIT_PAGER", "cat"),
];

/// The timeout code this executor's own deadline carries; an outer deadline
/// with another code classifies as an abort, not a timeout.
pub const BASH_TIMEOUT_CODE: &str = "BASH_TIMEOUT";

/// One captured stream: the (possibly truncated) text plus the loss flag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectedOutput {
    /// Collected text — the TAIL of the stream when truncated.
    pub text: String,
    /// True when bytes were dropped from `text`.
    pub truncated: bool,
}

/// A caller's execution REQUEST: optional fields are filled by
/// [`LocalBashExecutor::resolve`] from the executor's config, never by a
/// hidden default inside `run`.
#[derive(Clone, Default)]
pub struct ShellExecRequest {
    pub command: String,
    /// Working directory override (default: the executor's configured cwd).
    pub workdir: Option<String>,
    /// Timeout override in milliseconds; the executor caps it.
    pub timeout_ms: Option<f64>,
    /// Foreground stdout capture budget override for trusted in-process
    /// consumers; the model-facing tool never exposes it.
    pub stdout_max_bytes: Option<usize>,
    /// The executor kills the command when this fires.
    pub signal: Option<AbortSignal>,
    /// Bytes written to stdin, then closed; absent leaves stdin closed.
    pub stdin: Option<String>,
    /// Ordinary environment entries, merged after the credential scrub and
    /// BEFORE the managed snapshot so they can never displace a `DSH_*`
    /// fact.
    pub env: Option<BTreeMap<String, String>>,
    /// Harness-owned `DSH_*` snapshot for this execution; merges last.
    pub dsh_env: Option<BTreeMap<String, String>>,
}

impl ShellExecRequest {
    pub fn new(command: impl Into<String>) -> Self {
        ShellExecRequest {
            command: command.into(),
            ..Default::default()
        }
    }
}

/// A fully-resolved execution spec from [`LocalBashExecutor::resolve`].
/// `timeout_ms` applies to foreground runs only; background starts have no
/// executor timeout.
#[derive(Clone)]
pub struct ShellExecSpec {
    // Debug is manual below: AbortSignal carries no useful Debug form and
    // env values must never be echoed into diagnostics.
    pub command: String,
    pub workdir: String,
    pub timeout_ms: f64,
    pub stdout_max_bytes: usize,
    pub signal: Option<AbortSignal>,
    pub stdin: Option<String>,
    pub env: Option<BTreeMap<String, String>>,
    pub dsh_env: Option<BTreeMap<String, String>>,
}

/// Environment VALUES never enter a diagnostic — they carry credentials —
/// so only their key counts are reported; `AbortSignal` is not `Debug`, so
/// only its presence is.
impl std::fmt::Debug for ShellExecSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShellExecSpec")
            .field("command", &self.command)
            .field("workdir", &self.workdir)
            .field("timeout_ms", &self.timeout_ms)
            .field("stdout_max_bytes", &self.stdout_max_bytes)
            .field("signal", &self.signal.is_some())
            .field("stdin", &self.stdin.as_ref().map(|value| value.len()))
            .field("env_keys", &self.env.as_ref().map(|env| env.len()))
            .field("dsh_env_keys", &self.dsh_env.as_ref().map(|env| env.len()))
            .finish()
    }
}

/// The outcome of one completed (or killed) foreground run. `timed_out` and
/// `aborted` are mutually exclusive: one fused deadline drives both causes
/// and the first abort wins.
#[derive(Debug, Clone, PartialEq)]
pub struct ShellRunResult {
    /// Exit code; `None` when the process died from a signal.
    pub exit_code: Option<i32>,
    /// Terminating signal name; `None` on normal exit.
    pub signal: Option<String>,
    /// True when the executor's own timeout was the first cause to cut the
    /// command short.
    pub timed_out: bool,
    /// True when the caller's signal was the first cause (and it was not
    /// this executor's timeout).
    pub aborted: bool,
    /// The effective timeout applied to this run, after default and cap.
    pub timeout_ms: f64,
    pub stdout: CollectedOutput,
    pub stderr: CollectedOutput,
}

/// Lifecycle of a background process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellProcessStatus {
    Running,
    Completed,
    Killed,
}

/// One incremental background read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellProcessRead {
    /// Output since the previous read; stderr rides in a `[stderr]` section.
    pub delta: String,
    /// True when truncation dropped unread bytes the delta cannot include.
    pub lossy: bool,
}

struct ShellProcessState {
    status: Cell<ShellProcessStatus>,
    exit_code: Cell<Option<i32>>,
    signal: RefCell<Option<String>>,
    handle: Option<crate::subprocess::SubprocessHandle>,
    /// Delivered exactly once through the read path after a spawn failure.
    spawn_failure: RefCell<Option<String>>,
    stdout_offset: Cell<u64>,
    stderr_offset: Cell<u64>,
}

/// A background process handle — the only access path to a started command.
/// Buffered output remains readable after exit; `done` never rejects (a
/// spawn failure settles as `Killed` with the error readable on stderr).
#[derive(Clone)]
pub struct ShellProcess {
    state: Rc<ShellProcessState>,
    done: Shared<LocalBoxFuture<'static, ()>>,
}

impl ShellProcess {
    pub fn status(&self) -> ShellProcessStatus {
        self.state.status.get()
    }

    /// Exit code once finished; `None` while running or when signal-killed.
    pub fn exit_code(&self) -> Option<i32> {
        self.state.exit_code.get()
    }

    /// Terminating signal name, when signal-killed.
    pub fn signal(&self) -> Option<String> {
        self.state.signal.borrow().clone()
    }

    /// Resolves when the underlying process closes; never rejects.
    pub fn done(&self) -> impl std::future::Future<Output = ()> + 'static {
        self.done.clone()
    }

    /// Read output produced since the previous read (consuming — consecutive
    /// reads never re-deliver). Reads that lost data flag `lossy`.
    pub fn read_output(&self) -> ShellProcessRead {
        let state = &self.state;
        let (out, err) = match &state.handle {
            Some(handle) => {
                let out = handle.stdout().read_from(state.stdout_offset.get());
                let err = handle.stderr().read_from(state.stderr_offset.get());
                state.stdout_offset.set(out.next_offset);
                state.stderr_offset.set(err.next_offset);
                (out.text, (err.text, out.lossy || err.lossy))
            }
            None => (String::new(), (String::new(), false)),
        };
        let (mut err_text, lossy) = err;
        if err_text.is_empty() {
            // Spawn-failure note and real stderr are mutually exclusive: a
            // failed spawn produced no process output.
            if let Some(note) = state.spawn_failure.borrow_mut().take() {
                err_text = note;
            }
        }
        let separator = if !out.is_empty() && !out.ends_with('\n') {
            "\n"
        } else {
            ""
        };
        let delta = if err_text.is_empty() {
            out
        } else {
            format!("{out}{separator}[stderr]\n{err_text}")
        };
        ShellProcessRead { delta, lossy }
    }

    /// Kill the process group; `false` when it had already finished.
    /// Idempotent.
    pub fn kill(&self) -> bool {
        if self.state.status.get() != ShellProcessStatus::Running {
            return false;
        }
        self.state.status.set(ShellProcessStatus::Killed);
        if let Some(handle) = &self.state.handle {
            handle.terminate();
        }
        true
    }
}

/// Configuration for the local bash executor.
#[derive(Debug, Clone)]
pub struct BashConfig {
    /// Default working directory for commands (default: the process cwd).
    pub cwd: Option<String>,
    /// Default foreground timeout in milliseconds.
    pub timeout_ms: f64,
    /// Upper bound for per-call timeout overrides.
    pub max_timeout_ms: f64,
    /// Per-stream in-memory output cap in bytes.
    pub max_output_bytes: usize,
    /// Grace period for the kill escalation and post-exit pipe drain.
    pub grace_ms: f64,
}

impl Default for BashConfig {
    fn default() -> Self {
        BashConfig {
            cwd: None,
            timeout_ms: 120_000.0,
            max_timeout_ms: 600_000.0,
            max_output_bytes: 64_000,
            grace_ms: 3_000.0,
        }
    }
}

fn assert_positive_finite(name: &str, value: f64) -> anyhow::Result<()> {
    if !value.is_finite() || value <= 0.0 {
        anyhow::bail!("bash-local: {name} must be a positive finite number");
    }
    Ok(())
}

/// Local bash executor over the subprocess service, registered as the
/// `shell` service. Public commands run as `bash -c` in a managed process
/// group; bounded output and group escalation are the subprocess service's
/// mechanics with this executor's configured budgets.
pub struct LocalBashExecutor {
    config: BashConfig,
    subprocess: Rc<LocalSubprocessRuntime>,
}

impl Service for LocalBashExecutor {
    const NAME: &'static str = "shell";
}

impl LocalBashExecutor {
    /// Validate the config, build, and register the executor.
    pub fn provide(
        ctx: &Context,
        subprocess: Rc<LocalSubprocessRuntime>,
        config: BashConfig,
    ) -> anyhow::Result<Rc<LocalBashExecutor>> {
        assert_positive_finite("timeoutMs", config.timeout_ms)?;
        assert_positive_finite("maxTimeoutMs", config.max_timeout_ms)?;
        assert_positive_finite("graceMs", config.grace_ms)?;
        if config.max_output_bytes == 0 {
            anyhow::bail!("bash-local: maxOutputBytes must be a positive integer");
        }
        if config.grace_ms > MAX_TIMER_DELAY_MS {
            anyhow::bail!("bash-local: graceMs must be no greater than {MAX_TIMER_DELAY_MS}");
        }
        let executor = Rc::new(LocalBashExecutor { config, subprocess });
        ctx.provide_service(executor.clone())?;
        Ok(executor)
    }

    /// Resolve a request into a fully-specified spec: `workdir` from the
    /// configured cwd (else the process cwd), `timeout_ms` from the default
    /// capped at the maximum, `stdout_max_bytes` from the output cap.
    /// Callers hand the result to [`Self::run`]/[`Self::start`], which never
    /// re-default. A `dsh_env` entry outside the managed `DSH_*` namespace
    /// fails loud here.
    pub fn resolve(&self, request: ShellExecRequest) -> anyhow::Result<ShellExecSpec> {
        let timeout_ms = clamp_timeout(
            request.timeout_ms,
            self.config.timeout_ms,
            self.config.max_timeout_ms,
            "bash-local: request.timeoutMs",
        )?;
        let stdout_max_bytes = request
            .stdout_max_bytes
            .unwrap_or(self.config.max_output_bytes);
        if stdout_max_bytes == 0 {
            anyhow::bail!("bash-local: request.stdoutMaxBytes must be a positive integer");
        }
        if let Some(dsh_env) = &request.dsh_env {
            for key in dsh_env.keys() {
                if !key.starts_with(DSH_ENV_PREFIX) {
                    anyhow::bail!(
                        "bash-local: dshEnv key {key:?} is outside the managed {DSH_ENV_PREFIX}* namespace"
                    );
                }
            }
        }
        let workdir = match request.workdir {
            Some(workdir) => workdir,
            None => match &self.config.cwd {
                Some(cwd) => cwd.clone(),
                None => std::env::current_dir()?.to_string_lossy().into_owned(),
            },
        };
        Ok(ShellExecSpec {
            command: request.command,
            workdir,
            timeout_ms,
            stdout_max_bytes,
            signal: request.signal,
            stdin: request.stdin,
            env: request.env,
            dsh_env: request.dsh_env,
        })
    }

    /// One explicit env map for the subprocess seam, layered so the managed
    /// `DSH_*` snapshot beats both the caller's env and the terminal
    /// overrides; the subprocess service applies its scrub independently.
    fn spawn_spec(
        &self,
        spec: &ShellExecSpec,
        stdout_max_bytes: usize,
        signal: Option<AbortSignal>,
    ) -> SubprocessSpawnSpec {
        let mut env: BTreeMap<String, Option<String>> = ENV_OVERRIDES
            .iter()
            .map(|(key, value)| ((*key).to_string(), Some((*value).to_string())))
            .collect();
        if let Some(extra) = &spec.env {
            for (key, value) in extra {
                env.insert(key.clone(), Some(value.clone()));
            }
        }
        if let Some(dsh_env) = &spec.dsh_env {
            for (key, value) in dsh_env {
                env.insert(key.clone(), Some(value.clone()));
            }
        }
        SubprocessSpawnSpec {
            argv: vec!["bash".to_string(), "-c".to_string(), spec.command.clone()],
            cwd: spec.workdir.clone(),
            stdio: SubprocessStdio {
                stdin: match &spec.stdin {
                    Some(data) => SubprocessStdinMode::Data(data.clone()),
                    None => SubprocessStdinMode::Ignore,
                },
                stdout: SubprocessCollect {
                    max_bytes: stdout_max_bytes,
                },
                stderr: SubprocessCollect {
                    max_bytes: self.config.max_output_bytes,
                },
            },
            grace_ms: self.config.grace_ms,
            signal,
            env,
        }
    }

    /// Run a command in the foreground. Nonzero exits, timeout kills, and
    /// abort kills resolve with a descriptive result; only infrastructure
    /// failures (a failed spawn) reject.
    pub async fn run(&self, spec: ShellExecSpec) -> anyhow::Result<ShellRunResult> {
        // One deadline fuses the executor timeout with upstream cancellation;
        // dropping it clears the timer.
        let d = deadline(spec.signal.as_ref(), spec.timeout_ms, BASH_TIMEOUT_CODE);
        let handle = self.subprocess.spawn(self.spawn_spec(
            &spec,
            spec.stdout_max_bytes,
            Some(d.signal()),
        ))?;
        let outcome = handle.done().await;
        // Only this executor's own timeout reason counts as timed out; outer
        // deadlines classify as aborts.
        let timed_out = timeout_of(&d.signal(), Some(BASH_TIMEOUT_CODE)).is_some();
        let aborted = d.signal().aborted() && !timed_out;
        let output = |read: crate::subprocess::SubprocessOutputRead| CollectedOutput {
            text: read.text,
            truncated: read.lossy,
        };
        Ok(ShellRunResult {
            exit_code: outcome.exit_code,
            signal: outcome.signal,
            timed_out,
            aborted,
            timeout_ms: spec.timeout_ms,
            stdout: output(handle.stdout().read_from(0)),
            stderr: output(handle.stderr().read_from(0)),
        })
    }

    /// Start a background process and return its handle immediately; no
    /// timeout applies. A spawn failure settles the handle as `Killed` with
    /// the error readable through `read_output` instead of failing.
    pub fn start(&self, spec: ShellExecSpec) -> ShellProcess {
        let spawned = self.subprocess.spawn(self.spawn_spec(
            &spec,
            self.config.max_output_bytes,
            spec.signal.clone(),
        ));
        match spawned {
            Ok(handle) => {
                let state = Rc::new(ShellProcessState {
                    status: Cell::new(ShellProcessStatus::Running),
                    exit_code: Cell::new(None),
                    signal: RefCell::new(None),
                    handle: Some(handle.clone()),
                    spawn_failure: RefCell::new(None),
                    stdout_offset: Cell::new(0),
                    stderr_offset: Cell::new(0),
                });
                let done_state = state.clone();
                let caller_signal = spec.signal;
                let done = async move {
                    let outcome: SubprocessOutcome = handle.done().await;
                    if done_state.status.get() == ShellProcessStatus::Running {
                        // Any signal termination is Killed, including a
                        // command signaling itself.
                        let killed = caller_signal.as_ref().is_some_and(AbortSignal::aborted)
                            || outcome.signal.is_some();
                        done_state.status.set(if killed {
                            ShellProcessStatus::Killed
                        } else {
                            ShellProcessStatus::Completed
                        });
                    }
                    done_state.exit_code.set(outcome.exit_code);
                    *done_state.signal.borrow_mut() = outcome.signal;
                }
                .boxed_local()
                .shared();
                ShellProcess { state, done }
            }
            Err(error) => {
                let state = Rc::new(ShellProcessState {
                    status: Cell::new(ShellProcessStatus::Killed),
                    exit_code: Cell::new(None),
                    signal: RefCell::new(None),
                    handle: None,
                    spawn_failure: RefCell::new(Some(format!("spawn failed: {error}"))),
                    stdout_offset: Cell::new(0),
                    stderr_offset: Cell::new(0),
                });
                ShellProcess {
                    state,
                    done: async {}.boxed_local().shared(),
                }
            }
        }
    }
}

/// The exit status recovered from one rendered shell-tool result.
#[derive(Debug, Clone, PartialEq)]
pub enum ExitStatusMarker {
    Code(i32),
    Signal(String),
}

/// A rendered result split into its output body and structured exit status.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedExitStatus {
    pub body: String,
    pub status: ExitStatusMarker,
}

/// Split a rendered shell-tool result into its output body and the exit
/// status — the inverse of the `[exit code: N]` / `[killed by signal: X]`
/// markers the renderer appends. The consumed marker leaves the body (a
/// terminal card shows the exit as its own pill); other markers stay. The
/// marker must be the final line, so ordinary output that merely resembles
/// one cannot match. Absent both markers means a clean exit 0.
pub fn parse_exit_status(text: &str) -> ParsedExitStatus {
    const SIGNAL_PREFIX: &str = "\n[killed by signal: ";
    const EXIT_PREFIX: &str = "\n[exit code: ";
    if let Some(index) = text.rfind(SIGNAL_PREFIX) {
        let rest = &text[index + SIGNAL_PREFIX.len()..];
        if let Some(name) = rest.strip_suffix(']') {
            if !name.is_empty() && !name.contains(']') && !name.contains('\n') {
                return ParsedExitStatus {
                    body: text[..index].to_string(),
                    status: ExitStatusMarker::Signal(name.to_string()),
                };
            }
        }
    }
    if let Some(index) = text.rfind(EXIT_PREFIX) {
        let rest = &text[index + EXIT_PREFIX.len()..];
        if let Some(digits) = rest.strip_suffix(']') {
            if !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()) {
                if let Ok(code) = digits.parse::<i32>() {
                    return ParsedExitStatus {
                        body: text[..index].to_string(),
                        status: ExitStatusMarker::Code(code),
                    };
                }
            }
        }
    }
    ParsedExitStatus {
        body: text.to_string(),
        status: ExitStatusMarker::Code(0),
    }
}
