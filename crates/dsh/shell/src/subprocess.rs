//! Managed subprocess capability (port of upstream
//! `packages/subprocess/subprocess` + `subprocess-local`, collapsed: the
//! local provider is the only backend ported, so it registers directly as
//! the `subprocess` service and its methods are the seam contract). Each
//! spawn is a detached POSIX process group with fully explicit stdio, a
//! credential-scrubbed environment, bounded tail-keep output collection, and
//! tree-scoped SIGTERM→grace→SIGKILL termination driven by `terminate()` or
//! the spec's abort signal.
//!
//! Divergences: unix-only (`setsid`/`killpg` via libc; no Windows taskkill
//! tier); collect is the only output disposition ported (raw `pipe`,
//! `inherit`, and spill files are unused by the ported consumers — a
//! truncated stream keeps its in-memory tail and reports the loss); the
//! terminal/PTY primitive is out of scope; service disposal force-kills
//! still-live groups synchronously instead of awaiting whole-tree
//! quiescence; `resolveExecutable` is not ported (the bash executor spawns a
//! PATH name directly).

use dsh_cordis::{Context, Service};
use dsh_timeout::{AbortSignal, MAX_TIMER_DELAY_MS};
use futures::FutureExt;
use futures::future::{LocalBoxFuture, Shared};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::os::unix::process::ExitStatusExt;
use std::rc::Rc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Namespace prefix reserved for harness-managed child environment facts.
pub const DSH_ENV_PREFIX: &str = "DSH_";

/// Whether an environment name is credential-shaped and must not be
/// forwarded to children implicitly. One heuristic for every in-repo
/// spawner; a deliberately supplied entry survives because explicit env
/// layers merge after the scrub.
pub fn is_sensitive_env_key(key: &str) -> bool {
    let upper = key.to_uppercase();
    ["KEY", "PASSWORD", "SECRET", "TOKEN"]
        .iter()
        .any(|marker| upper.contains(marker))
}

/// The ambient parent environment minus credential-shaped names and minus
/// all `DSH_*` names — the base every harness child starts from. `PATH`,
/// `HOME`, locale, and proxy variables survive so child CLIs run normally;
/// harness identity and secrets never leak implicitly.
pub fn scrubbed_parent_env() -> BTreeMap<String, String> {
    std::env::vars()
        .filter(|(key, _)| {
            !is_sensitive_env_key(key) && !key.to_uppercase().starts_with(DSH_ENV_PREFIX)
        })
        .collect()
}

/// Merge explicit caller entries onto the scrubbed parent base: a string
/// deliberately restores or overrides an entry, a `None` tombstone removes
/// an ordinary ambient one.
pub fn child_env(extra: &BTreeMap<String, Option<String>>) -> BTreeMap<String, String> {
    let mut env = scrubbed_parent_env();
    for (key, value) in extra {
        match value {
            Some(value) => {
                env.insert(key.clone(), value.clone());
            }
            None => {
                env.remove(key);
            }
        }
    }
    env
}

/// stdin disposition: leave fd 0 closed, or write the bytes and close (the
/// batch shape). The streaming `pipe` mode is not ported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubprocessStdinMode {
    Ignore,
    Data(String),
}

/// Bounded in-memory tail collection for one output stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubprocessCollect {
    /// In-memory cap in bytes; overflow keeps the TAIL (errors and final
    /// results cluster at the end of command output).
    pub max_bytes: usize,
}

/// Per-stream stdio dispositions, all explicit — this seam applies no
/// defaults (the caller's own config decides every budget).
#[derive(Debug, Clone, PartialEq)]
pub struct SubprocessStdio {
    pub stdin: SubprocessStdinMode,
    pub stdout: SubprocessCollect,
    pub stderr: SubprocessCollect,
}

/// A fully-specified spawn request; nothing here is defaulted.
pub struct SubprocessSpawnSpec {
    /// Executable and arguments; `argv[0]` is the program. Never
    /// shell-interpreted at this seam.
    pub argv: Vec<String>,
    /// Working directory for the child.
    pub cwd: String,
    pub stdio: SubprocessStdio,
    /// Positive finite grace period in milliseconds (at most
    /// [`MAX_TIMER_DELAY_MS`]) for the terminate escalation and for
    /// draining still-open pipes after exit.
    pub grace_ms: f64,
    /// Starts the terminate escalation on the group when it fires; the
    /// caller owns deadlines and cause classification.
    pub signal: Option<AbortSignal>,
    /// Explicit entries merged onto the scrubbed parent base; `None` is a
    /// tombstone removing an ambient entry.
    pub env: BTreeMap<String, Option<String>>,
}

/// Exit facts of one closed process. Deliberately carries no timeout or
/// cancellation classification (the caller reads the signal it owns) and no
/// output — collected streams stay readable through the handle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubprocessOutcome {
    /// Exit code; `None` when the process died from a signal.
    pub exit_code: Option<i32>,
    /// Terminating signal name (e.g. `SIGTERM`); `None` on normal exit.
    pub signal: Option<String>,
}

/// One incremental read from a collected stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubprocessOutputRead {
    /// Stream text from the requested offset (the whole retained tail when
    /// lossy).
    pub text: String,
    /// Whole-stream offset to resume from on the next read.
    pub next_offset: u64,
    /// True when the requested offset slid out of the in-memory tail window.
    pub lossy: bool,
}

/// Bounded tail-keep collector state shared with its reader.
struct CollectorState {
    retained: Vec<u8>,
    /// Total bytes ever pushed (not just retained).
    total: u64,
    max_bytes: usize,
}

impl CollectorState {
    fn push(&mut self, chunk: &[u8]) {
        self.total += chunk.len() as u64;
        self.retained.extend_from_slice(chunk);
        if self.retained.len() > self.max_bytes {
            // Byte-exact tail window regardless of chunking.
            let excess = self.retained.len() - self.max_bytes;
            self.retained.drain(..excess);
        }
    }
}

/// Cursor-free incremental access to one collected stream. Offsets are
/// whole-stream byte coordinates owned by the caller, so independent readers
/// never consume one another's output; `read_from(0)` after settlement is
/// the batch result (`lossy` then means the tail lost its head).
#[derive(Clone)]
pub struct SubprocessOutputReader {
    state: Rc<RefCell<CollectorState>>,
}

impl SubprocessOutputReader {
    /// Read everything captured since `from_byte`; a read whose offset slid
    /// out of the tail window is `lossy` and returns the whole retained
    /// tail.
    pub fn read_from(&self, from_byte: u64) -> SubprocessOutputRead {
        let state = self.state.borrow();
        let window_start = state.total - state.retained.len() as u64;
        let lossy = from_byte < window_start;
        let slice = if lossy {
            &state.retained[..]
        } else {
            &state.retained[(from_byte - window_start) as usize..]
        };
        SubprocessOutputRead {
            text: String::from_utf8_lossy(slice).into_owned(),
            next_offset: state.total,
            lossy,
        }
    }
}

fn signal_name(number: i32) -> String {
    match number {
        1 => "SIGHUP".to_string(),
        2 => "SIGINT".to_string(),
        9 => "SIGKILL".to_string(),
        13 => "SIGPIPE".to_string(),
        15 => "SIGTERM".to_string(),
        other => format!("SIG{other}"),
    }
}

/// Send `sig` to a detached process group. Never fails: delivery races
/// process exit, so a gone group and a non-positive pid are no-ops.
pub fn kill_group(pid: i32, sig: i32) {
    if pid <= 0 {
        return;
    }
    unsafe {
        libc::kill(-pid, sig);
    }
}

/// Whether the detached group still has members (a `kill(…, 0)` probe).
fn group_alive(pid: i32) -> bool {
    if pid <= 0 {
        return false;
    }
    unsafe { libc::kill(-pid, 0) == 0 }
}

struct HandleState {
    pid: i32,
    grace_ms: f64,
    settled: Cell<bool>,
    terminating: Cell<bool>,
}

impl HandleState {
    /// SIGTERM the group now, SIGKILL it after the grace period if it is
    /// still alive. Idempotent; a no-op once the group is gone.
    fn terminate(self: &Rc<Self>) {
        if self.terminating.replace(true) {
            return;
        }
        if !group_alive(self.pid) {
            return;
        }
        kill_group(self.pid, libc::SIGTERM);
        let state = self.clone();
        tokio::task::spawn_local(async move {
            tokio::time::sleep(std::time::Duration::from_millis(state.grace_ms as u64)).await;
            // The escalation survives direct-child settlement: a
            // TERM-trapping descendant must still be reachable.
            if group_alive(state.pid) {
                kill_group(state.pid, libc::SIGKILL);
            }
        });
    }
}

type DoneFuture = Shared<LocalBoxFuture<'static, SubprocessOutcome>>;

/// A live child process rooted in its own process group. Collected output
/// remains readable after exit; termination is group-scoped, so helpers the
/// command spawned cannot outlive the handle unnoticed.
#[derive(Clone)]
pub struct SubprocessHandle {
    state: Rc<HandleState>,
    stdout: SubprocessOutputReader,
    stderr: SubprocessOutputReader,
    done: DoneFuture,
}

impl std::fmt::Debug for SubprocessHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubprocessHandle")
            .field("pid", &self.state.pid)
            .field("settled", &self.state.settled.get())
            .finish_non_exhaustive()
    }
}

impl SubprocessHandle {
    /// Process id (the group leader).
    pub fn pid(&self) -> i32 {
        self.state.pid
    }

    /// Resolves at process close with exit facts; collected output is
    /// complete once this settles.
    pub fn done(&self) -> impl std::future::Future<Output = SubprocessOutcome> + 'static {
        self.done.clone()
    }

    /// Offset-based reader for collected stdout.
    pub fn stdout(&self) -> SubprocessOutputReader {
        self.stdout.clone()
    }

    /// Offset-based reader for collected stderr.
    pub fn stderr(&self) -> SubprocessOutputReader {
        self.stderr.clone()
    }

    /// Begin the SIGTERM → grace → SIGKILL escalation on the process group —
    /// the seam's only termination verb. Idempotent; also triggered by the
    /// spec's abort signal.
    pub fn terminate(&self) {
        self.state.terminate();
    }
}

/// Local subprocess service (`ctx.subprocess`): detached process groups,
/// scrubbed environment, bounded collection, group-scoped escalation.
/// Disposal force-kills every group it still owns.
pub struct LocalSubprocessRuntime {
    live: Rc<RefCell<Vec<Rc<HandleState>>>>,
}

impl Service for LocalSubprocessRuntime {
    const NAME: &'static str = "subprocess";
}

impl LocalSubprocessRuntime {
    /// Build and register the service; the registered effect's disposer
    /// force-kills still-live groups.
    pub fn provide(ctx: &Context) -> anyhow::Result<Rc<LocalSubprocessRuntime>> {
        let runtime = Rc::new(LocalSubprocessRuntime {
            live: Rc::default(),
        });
        let live = runtime.live.clone();
        ctx.effect_labeled("local subprocess teardown", move |_ctx| {
            Ok(dsh_cordis::Effect::One(dsh_cordis::Disposer::sync(
                move || {
                    for state in live.borrow_mut().drain(..) {
                        if !state.settled.get() {
                            kill_group(state.pid, libc::SIGKILL);
                        }
                    }
                },
            )))
        })?;
        ctx.provide_service(runtime.clone())?;
        Ok(runtime)
    }

    /// Start one managed child process from a fully-specified spec. Spawn
    /// failures fail here synchronously; runtime exits resolve the handle's
    /// `done` with exit facts.
    pub fn spawn(&self, spec: SubprocessSpawnSpec) -> anyhow::Result<SubprocessHandle> {
        if !spec.grace_ms.is_finite() || spec.grace_ms <= 0.0 || spec.grace_ms > MAX_TIMER_DELAY_MS
        {
            anyhow::bail!(
                "subprocess grace_ms must be a positive finite number no greater than {MAX_TIMER_DELAY_MS}"
            );
        }
        if let Some(signal) = &spec.signal {
            if signal.aborted() {
                anyhow::bail!("aborted before spawn");
            }
        }
        let Some((program, args)) = spec.argv.split_first() else {
            anyhow::bail!("invalid argv: expected a non-empty program name at argv[0]");
        };
        if program.is_empty() {
            anyhow::bail!("invalid argv: expected a non-empty program name at argv[0]");
        }

        let mut command = tokio::process::Command::new(program);
        command
            .args(args)
            .current_dir(&spec.cwd)
            .env_clear()
            .envs(child_env(&spec.env))
            .stdin(match &spec.stdio.stdin {
                SubprocessStdinMode::Ignore => std::process::Stdio::null(),
                SubprocessStdinMode::Data(_) => std::process::Stdio::piped(),
            })
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        // A fresh session makes the child the leader of its own process
        // group — the tree root every termination tier signals.
        unsafe {
            command.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }
        let mut child = command.spawn()?;
        let pid = child.id().map(|id| id as i32).unwrap_or(-1);

        let stdout_state = Rc::new(RefCell::new(CollectorState {
            retained: Vec::new(),
            total: 0,
            max_bytes: spec.stdio.stdout.max_bytes,
        }));
        let stderr_state = Rc::new(RefCell::new(CollectorState {
            retained: Vec::new(),
            total: 0,
            max_bytes: spec.stdio.stderr.max_bytes,
        }));
        let state = Rc::new(HandleState {
            pid,
            grace_ms: spec.grace_ms,
            settled: Cell::new(false),
            terminating: Cell::new(false),
        });
        self.live.borrow_mut().push(state.clone());

        let stdin_data = match spec.stdio.stdin {
            SubprocessStdinMode::Data(data) => Some(data),
            SubprocessStdinMode::Ignore => None,
        };
        let mut child_stdin = child.stdin.take();
        let mut child_stdout = child.stdout.take();
        let mut child_stderr = child.stderr.take();

        let done_state = state.clone();
        let done_stdout = stdout_state.clone();
        let done_stderr = stderr_state.clone();
        let live = self.live.clone();
        let grace = std::time::Duration::from_millis(spec.grace_ms as u64);
        let done: DoneFuture = async move {
            // All stream I/O runs concurrently with the exit wait so a full
            // pipe can never deadlock the child.
            let io = async {
                let write_stdin = async {
                    if let (Some(data), Some(stdin)) = (stdin_data, child_stdin.as_mut()) {
                        // Best-effort batch write: exit facts and captured
                        // output stay authoritative over an EPIPE here.
                        let _ = stdin.write_all(data.as_bytes()).await;
                    }
                    drop(child_stdin.take());
                };
                let read_stdout = async {
                    if let Some(stream) = child_stdout.as_mut() {
                        let mut buffer = [0u8; 8192];
                        while let Ok(count) = stream.read(&mut buffer).await {
                            if count == 0 {
                                break;
                            }
                            done_stdout.borrow_mut().push(&buffer[..count]);
                        }
                    }
                };
                let read_stderr = async {
                    if let Some(stream) = child_stderr.as_mut() {
                        let mut buffer = [0u8; 8192];
                        while let Ok(count) = stream.read(&mut buffer).await {
                            if count == 0 {
                                break;
                            }
                            done_stderr.borrow_mut().push(&buffer[..count]);
                        }
                    }
                };
                futures::join!(write_stdin, read_stdout, read_stderr);
            };
            let mut io = std::pin::pin!(io);
            let mut wait = std::pin::pin!(child.wait());
            let status = futures::select! {
                status = (&mut wait).fuse() => {
                    // The child exited but a surviving descendant may hold a
                    // pipe open; the same grace that bounds kills bounds the
                    // drain.
                    let _ = tokio::time::timeout(grace, &mut io).await;
                    status
                }
                () = (&mut io).fuse() => wait.await,
            };
            done_state.settled.set(true);
            live.borrow_mut()
                .retain(|entry| !Rc::ptr_eq(entry, &done_state));
            match status {
                Ok(status) => SubprocessOutcome {
                    exit_code: status.code(),
                    signal: status.signal().map(signal_name),
                },
                Err(_) => SubprocessOutcome {
                    exit_code: None,
                    signal: None,
                },
            }
        }
        .boxed_local()
        .shared();

        // The abort watcher reacts to the caller's signal for the process's
        // lifetime and stops with it.
        if let Some(signal) = spec.signal {
            let watch_state = state.clone();
            let watch_done = done.clone();
            tokio::task::spawn_local(async move {
                futures::select! {
                    _ = watch_done.fuse() => {}
                    _ = signal.wait().fuse() => watch_state.terminate(),
                }
            });
        }

        Ok(SubprocessHandle {
            state,
            stdout: SubprocessOutputReader {
                state: stdout_state,
            },
            stderr: SubprocessOutputReader {
                state: stderr_state,
            },
            done,
        })
    }
}
