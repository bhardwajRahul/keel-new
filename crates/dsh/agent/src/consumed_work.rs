//! Consumed-work accounting over one agent log, ported from
//! `packages/core/agent/src/consumed-work.ts`.
//!
//! Turn vocabulary alone cannot say whether work ran: a turn that stops
//! before its first step looks exactly like a balanced no-op turn. The
//! missing fact is the inbox's own record — each splice logs `removedCount`
//! and marks cancellations — which separates a turn claiming its input from
//! work dropped unrun.

use crate::types::{InboxSplice, SpliceOutcome};
use dsh_session::{SessionEvent, SessionEventData, TurnEndReason};
use std::collections::HashSet;

/// How one agent log accounts for the work it consumed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConsumedWork {
    /// The latest closed turn that accounts for consumed work: one that
    /// entered a model step, or one that claimed input and then failed, was
    /// stopped, or was rejected. `None` when no turn closed over any work.
    pub end: Option<SessionEvent>,
    /// Whether accepted work was cancelled out of the inbox, unrun, after
    /// that turn — the only account of input a cancellation took before any
    /// turn could open over it.
    pub dropped_unrun: bool,
}

/// Whether a turn that consumed input but never reached a step ends in a way
/// that accounts for that input. Only a `Completed` end does not (its claim
/// was rewritten away); an unknown ending over consumed input must not read
/// as success.
fn accounts_for_claim(reason: &TurnEndReason) -> bool {
    !matches!(reason, TurnEndReason::Completed)
}

/// Fold one agent log, or an owned suffix of one, into its account of
/// consumed work. Single pass; every input is the log itself, so a
/// cancellation issued by anyone reads the same.
pub fn fold_consumed_work(events: &[SessionEvent]) -> ConsumedWork {
    let mut stepped: HashSet<u64> = HashSet::new();
    let mut claimed: HashSet<u64> = HashSet::new();
    let mut open: Option<u64> = None;
    let mut end: Option<SessionEvent> = None;
    let mut dropped_unrun = false;
    for event in events {
        match &event.data {
            SessionEventData::TurnStart { turn } => {
                open = Some(*turn);
            }
            SessionEventData::StepStart { turn, .. } => {
                stepped.insert(*turn);
            }
            SessionEventData::TurnEnd { turn, reason } => {
                open = None;
                let was_stepped = stepped.remove(turn);
                let was_claimed = claimed.remove(turn);
                if was_stepped || (was_claimed && accounts_for_claim(reason)) {
                    end = Some(event.clone());
                    // Anything dropped before this turn closed is what its own
                    // ending reports; only a later drop stays unaccounted.
                    dropped_unrun = false;
                }
            }
            _ => {
                if let Some(splice) = InboxSplice::from_event(event) {
                    if splice.removed_count.is_none() {
                        continue;
                    }
                    if splice.outcome == Some(SpliceOutcome::Canceled) {
                        // A replacement keeps the work pending under a new
                        // identity; only a cancellation leaving nothing drops it.
                        dropped_unrun |= splice.inserted.is_empty();
                    } else if let Some(turn) = open {
                        // Claims are the loop's own step-boundary reads,
                        // always inside a turn.
                        claimed.insert(turn);
                    }
                }
            }
        }
    }
    ConsumedWork { end, dropped_unrun }
}
