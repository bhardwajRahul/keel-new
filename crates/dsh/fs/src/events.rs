//! The `fs/*` events the tools dispatch and the observation policy answers
//! (port of the event declarations in upstream `packages/fs/fs/src/index.ts`).
//!
//! Divergence: upstream keys observed state by the live `agent.session`
//! object held in a `WeakMap`; object-identity weak keys do not translate,
//! so the events carry an explicit [`FsOwnerKey`] (the owning session id)
//! and the policy's state is dropped when its listeners unwind.

use crate::types::{FsObservation, FsTarget, FsVersion, FsWriteIntent};
use dsh_cordis::Event;
use dsh_tools::ToolExecution;

/// The observed-state owner of one tool call: the calling agent's session
/// id. `None` (a call with no agent) reads freely but can never satisfy the
/// prior-observation policy.
pub type FsOwnerKey = String;

/// Derive the observed-state owner from a tool execution.
pub fn owner_of(exec: &ToolExecution) -> Option<FsOwnerKey> {
    exec.agent()
        .map(|agent| agent.session().id().as_str().to_string())
}

/// Single-slot decision for the next `write_text`: the default (`next`)
/// yields the bare provider's unconditional write; the first listener that
/// answers owns the decision. `@mode waterfall`.
pub struct FsWriteIntentSlot;
impl Event for FsWriteIntentSlot {
    const NAME: &'static str = "fs/write-intent";
    type Args = (FsTarget, Option<FsOwnerKey>);
    type Ret = Option<FsWriteIntent>;
}

/// Single-slot decision for the next `edit_text`: the default yields an
/// unconditional edit; the first returned version guard wins, and the slot
/// may fail the call outright (`FS_NOT_OBSERVED`). `@mode waterfall`.
pub struct FsEditIntentSlot;
impl Event for FsEditIntentSlot {
    const NAME: &'static str = "fs/edit-intent";
    type Args = (FsTarget, Option<FsOwnerKey>);
    type Ret = Option<FsVersion>;
}

/// Record one authoritative presence/absence observation. Listeners are
/// synchronous recorders; the tools dispatch this awaited (`ctx.parallel`)
/// because the Rust bus's `emit` defers spawned listener futures, which
/// would let a follow-up mutation race the recording. `@mode emit` upstream.
pub struct FsObservedEvent;
impl Event for FsObservedEvent {
    const NAME: &'static str = "fs/observed";
    type Args = (FsTarget, FsObservation, Option<FsOwnerKey>);
    type Ret = ();
}
