//! Shell capability for the harness, ported at the contract level from the
//! upstream TypeScript packages and collapsed into ONE crate (upstream
//! splits them across five):
//!
//! - `packages/subprocess/subprocess` + `subprocess-local` — the
//!   `ctx.subprocess` seam and its local provider ([`subprocess`]): managed
//!   detached process groups, credential-scrubbed environment (secrets and
//!   ambient `DSH_*` never forwarded implicitly), bounded tail-keep output
//!   capture with offset-based incremental reads, group-scoped
//!   SIGTERM→grace→SIGKILL termination on cancel.
//! - `packages/shell/shell` + `packages/shell/bash-local` — the `ctx.shell`
//!   seam and the local bash executor ([`shell`]): the request/spec split
//!   (defaulting is the explicit `resolve` step, never a hidden fallback in
//!   `run`), `bash -c` execution with the model-friendly terminal
//!   environment and managed `DSH_*` merge order, foreground timeout/abort
//!   cause classification, background process handles.
//! - `packages/shell/tool-bash` — the model-facing `bash` tool ([`tool`])
//!   with terminal presentation, `timeoutMs`, and session-relative `workdir`
//!   handling.
//!
//! Out of scope (per the port plan): pwsh executors, sandboxing executors,
//! the persistent-bash tool, the terminal/PTY tier, background jobs
//! (`run_in_background`), and spill files. Per-module docs carry the
//! finer-grained divergences. Unix-only.

mod shell;
mod subprocess;
mod tool;

pub use crate::shell::{
    BASH_TIMEOUT_CODE, BashConfig, CollectedOutput, ENV_OVERRIDES, ExitStatusMarker,
    LocalBashExecutor, ParsedExitStatus, ShellExecRequest, ShellExecSpec, ShellProcess,
    ShellProcessRead, ShellProcessStatus, ShellRunResult, parse_exit_status,
};
pub use crate::subprocess::{
    DSH_ENV_PREFIX, LocalSubprocessRuntime, SubprocessCollect, SubprocessHandle, SubprocessOutcome,
    SubprocessOutputRead, SubprocessOutputReader, SubprocessSpawnSpec, SubprocessStdinMode,
    SubprocessStdio, child_env, is_sensitive_env_key, kill_group, scrubbed_parent_env,
};
pub use crate::tool::{register_bash_tool, render_result};
