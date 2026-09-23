//! Read-before-edit observation policy (port of upstream
//! `packages/fs/fs-observation-policy`). Event-only: it registers no
//! service. A per-owner map records every authoritative presence/absence
//! observation; the two single-slot intent listeners derive mutation guards
//! from that state, and the provider performs the atomic freshness /
//! no-clobber check. Without this plugin the tools keep the bare provider's
//! unconditional mutation behavior.

use crate::events::{FsEditIntentSlot, FsObservedEvent, FsOwnerKey, FsWriteIntentSlot};
use crate::types::{FsError, FsErrorCode, FsObservation, FsTarget, FsVersion, FsWriteIntent};
use dsh_cordis::{Context, EventOptions};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Observed-file state keyed by owner, then by target key. An entry's
/// presence is the prior-observation record; its variant keeps confirmed
/// absence distinct from an unseen target.
type ObservedState = RefCell<HashMap<FsOwnerKey, HashMap<String, FsObservation>>>;

fn prior(
    state: &ObservedState,
    owner: &Option<FsOwnerKey>,
    target: &FsTarget,
) -> Option<FsObservation> {
    let owner = owner.as_ref()?;
    state
        .borrow()
        .get(owner)?
        .get(target.target_key.as_str())
        .cloned()
}

/// Decide a write intent: unseen or confirmed absent means a guarded create;
/// confirmed present means a guarded replace at the observed version.
fn write_intent(
    state: &ObservedState,
    target: &FsTarget,
    owner: &Option<FsOwnerKey>,
) -> FsWriteIntent {
    match prior(state, owner, target) {
        Some(FsObservation::Present { version }) => FsWriteIntent::ReplaceIfVersion(version),
        _ => FsWriteIntent::CreateIfAbsent,
    }
}

/// Decide an edit guard: unseen rejects with `FS_NOT_OBSERVED`, confirmed
/// absence with `FS_NOT_FOUND`, and presence supplies the observed version
/// as the compare-and-swap basis.
fn edit_intent(
    state: &ObservedState,
    target: &FsTarget,
    owner: &Option<FsOwnerKey>,
) -> Result<FsVersion, FsError> {
    match prior(state, owner, target) {
        None => Err(FsError::new(
            format!("edit requires reading \"{}\" first", target.display_path),
            FsErrorCode::NotObserved,
        )),
        Some(FsObservation::Absent) => Err(FsError::new(
            format!("cannot edit \"{}\": not found", target.display_path),
            FsErrorCode::NotFound,
        )),
        Some(FsObservation::Present { version }) => Ok(version),
    }
}

/// Register the three `fs/*` listeners on this context. State is owned by
/// the listeners themselves, so disposing the registering fiber drops it
/// (the upstream HMR-teardown effect is Rust drop semantics here).
pub fn install_observation_policy(ctx: &Context) -> anyhow::Result<()> {
    let state: Rc<ObservedState> = Rc::default();

    // Occupy the single write-intent decision slot: deliberately does NOT
    // call next().
    let write_state = state.clone();
    ctx.on_waterfall::<FsWriteIntentSlot, _, _>(
        EventOptions::default(),
        move |_ctx, (target, owner), _next| {
            let intent = write_intent(&write_state, &target, &owner);
            async move { Ok(Some(intent)) }
        },
    )?;

    // Occupy the single edit-intent decision slot; an unread target fails
    // the call here, before the provider runs.
    let edit_state = state.clone();
    ctx.on_waterfall::<FsEditIntentSlot, _, _>(
        EventOptions::default(),
        move |_ctx, (target, owner), _next| {
            let decision = edit_intent(&edit_state, &target, &owner);
            async move { Ok(Some(decision?)) }
        },
    )?;

    // Record observations synchronously (mutations already committed; a
    // recorder must not fail the call).
    let observe_state = state.clone();
    ctx.on::<FsObservedEvent, _, _>(
        EventOptions::default(),
        move |_ctx, (target, observation, owner)| {
            if let Some(owner) = owner {
                observe_state
                    .borrow_mut()
                    .entry(owner.clone())
                    .or_default()
                    .insert(target.target_key.as_str().to_string(), observation.clone());
            }
            async { None }
        },
    )?;
    Ok(())
}
