//! Port of `packages/boot/cmdline` (`@deepseek-ai/dsh-cmdline`): the command
//! line a dsh launcher hands to the app it boots.
//!
//! The launcher parses only its own flags and hands everything after them to
//! the app verbatim through [`CmdlineArgs`], so an app owns its flag family,
//! its help text, and its parse errors instead of the launcher knowing them.
//! [`AppExit`] is the launcher-owned bounded exit request the app calls once
//! its tree can be disposed. Both are launcher facts, not config: an
//! embedding host with no command line hands over an empty argument list.
//!
//! Divergences from the TS original:
//! - The Cordis context provisioning (`provideCmdline`, the `cmdlineArgs` /
//!   `appExit` context slots) and the commander adapter (`parseCmdline`,
//!   `internals`, the action/exit-routing guards) are not ported. This crate
//!   is the immutable handoff structure; a Rust app parses the snapshot with
//!   its own parser (e.g. clap) and reports rejection through [`AppExit`].
//! - Upstream freezes the argument array; here immutability is structural:
//!   [`CmdlineArgs`] snapshots on construction and exposes no mutation.

use std::sync::Arc;

/// The invocation's inner arguments: everything after the launcher's own
/// flags, verbatim and in argv order. `dsh --profile tui --resume abc`
/// yields `["--resume", "abc"]`. Clones share one immutable snapshot, so
/// every reader observes the same command line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CmdlineArgs {
    args: Arc<[String]>,
}

impl CmdlineArgs {
    /// Snapshot the invocation's inner arguments; later changes to the
    /// caller's collection cannot reach the snapshot.
    pub fn new<I, S>(args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            args: args.into_iter().map(Into::into).collect(),
        }
    }

    /// Read the inner arguments, in argv order; empty when the invocation
    /// carried none.
    pub fn get(&self) -> &[String] {
        &self.args
    }
}

impl Default for CmdlineArgs {
    /// The embedding-host case: no command line at all.
    fn default() -> Self {
        Self::new(std::iter::empty::<String>())
    }
}

/// Request bounded process exit once the app's tree has been disposed; the
/// launcher wires it to its shutdown controller. The argument is the process
/// exit code.
pub type AppExit = Arc<dyn Fn(i32) + Send + Sync>;

/// The launcher facts an app needs, handed over before anything mounts.
#[derive(Clone)]
pub struct CmdlineHost {
    /// The invocation's inner arguments, in argv order.
    pub args: CmdlineArgs,
    /// Bounded process-exit request.
    pub exit: AppExit,
}
