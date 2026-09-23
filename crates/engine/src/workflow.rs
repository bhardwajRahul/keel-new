//! Host-owned prepared actions for bounded workflow decisions.
//!
//! A selector sees descriptions and opaque IDs. The host keeps the exact
//! payload, checks eligibility before selection, and checks it again just
//! before dispatch through the existing executor. This module does not enter
//! or control an external ACP provider's internal agent loop.

use std::collections::BTreeSet;

use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Default)]
pub struct TaskState {
    pub revision: u64,
    pub cancelled: bool,
    completed: BTreeSet<String>,
}

impl TaskState {
    pub fn new(revision: u64) -> Self {
        Self {
            revision,
            ..Self::default()
        }
    }

    /// Call when user instructions or other material task state changes.
    pub fn invalidate(&mut self) {
        self.revision = self
            .revision
            .checked_add(1)
            .expect("task revision exhausted");
    }

    pub fn is_completed(&self, id: &str) -> bool {
        self.completed.contains(id)
    }

    /// Record a real host outcome. Only independently verified execution
    /// satisfies a dependency or advances the task revision.
    pub fn record_outcome<T>(
        &mut self,
        action: &PreparedAction<T>,
        validation: &ValidationTrace,
        outcome: ExecutionOutcome,
    ) -> OutcomeTrace {
        let status = if !validation.accepted
            || validation.action_id != action.id
            || validation.task_revision != self.revision
            || self.revision != action.prepared_at_revision
            || self.cancelled
        {
            OutcomeStatus::Rejected
        } else {
            match outcome {
                ExecutionOutcome::Verified => {
                    self.completed.insert(action.id.clone());
                    self.invalidate();
                    OutcomeStatus::Verified
                }
                ExecutionOutcome::Failed => OutcomeStatus::ExecutionFailed,
                ExecutionOutcome::Unverified => OutcomeStatus::VerificationFailed,
            }
        };
        OutcomeTrace {
            action_id: action.id.clone(),
            task_revision: self.revision,
            status,
            rejection: if status == OutcomeStatus::Rejected {
                validation.rejection.or(Some(RejectReason::StaleRevision))
            } else {
                None
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct PreparedAction<T> {
    /// Opaque selector ID. The payload is never derived from this string.
    pub id: String,
    /// Bounded, non-secret text supplied to the selector.
    pub description: String,
    /// Exact host-owned arguments or route to dispatch after revalidation.
    pub payload: T,
    pub prepared_at_revision: u64,
    pub preconditions: Vec<Precondition>,
    /// Hash of the relevant observed files, registry entries, or other reads.
    /// `None` is valid only when the action has no read-set dependency.
    pub read_set_fingerprint: Option<String>,
    /// Milliseconds since Unix epoch; the action is invalid at this instant.
    pub expires_at_ms: Option<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Precondition {
    Completed(String),
}

#[derive(Clone, Debug)]
pub struct ValidationContext {
    pub now_ms: i64,
    /// Recomputed by the host immediately before selection or dispatch.
    pub current_read_set_fingerprint: Option<String>,
    /// The existing permission and policy gate's result for this exact payload.
    pub authorized: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RejectReason {
    Cancelled,
    InvalidId,
    WrongSelection,
    StaleRevision,
    Expired,
    UnmetPrecondition,
    StaleReadSet,
    Unauthorized,
}

/// Loggable evidence only: no payload, user text, key, or read-set content.
#[derive(Clone, Debug, Serialize)]
pub struct ValidationTrace {
    pub action_id: String,
    pub task_revision: u64,
    pub accepted: bool,
    pub rejection: Option<RejectReason>,
}

/// Call before offering an action to Laya or Jev, and again after selection.
pub fn eligible<T>(
    state: &TaskState,
    action: &PreparedAction<T>,
    context: &ValidationContext,
) -> bool {
    validate_selected(state, action, &action.id, context).accepted
}

/// Resolve a selector's opaque ID against a stored action, then recheck the
/// task, read-set, expiry, dependencies, and host permission result. The host
/// must call its own permission gate again at dispatch time.
pub fn validate_selected<T>(
    state: &TaskState,
    action: &PreparedAction<T>,
    selected_id: &str,
    context: &ValidationContext,
) -> ValidationTrace {
    let rejection = if state.cancelled {
        Some(RejectReason::Cancelled)
    } else if action.id.trim().is_empty() || action.id == "escalate" {
        Some(RejectReason::InvalidId)
    } else if action.id != selected_id {
        Some(RejectReason::WrongSelection)
    } else if state.revision != action.prepared_at_revision {
        Some(RejectReason::StaleRevision)
    } else if action
        .expires_at_ms
        .is_some_and(|deadline| context.now_ms >= deadline)
    {
        Some(RejectReason::Expired)
    } else if action
        .preconditions
        .iter()
        .any(|precondition| match precondition {
            Precondition::Completed(id) => !state.is_completed(id),
        })
    {
        Some(RejectReason::UnmetPrecondition)
    } else if action.read_set_fingerprint != context.current_read_set_fingerprint {
        Some(RejectReason::StaleReadSet)
    } else if !context.authorized {
        Some(RejectReason::Unauthorized)
    } else {
        None
    };
    ValidationTrace {
        action_id: action.id.clone(),
        task_revision: state.revision,
        accepted: rejection.is_none(),
        rejection,
    }
}

/// Hash an ordered, serializable read-set snapshot. The caller decides which
/// reads matter and must recompute the same snapshot at dispatch time.
pub fn fingerprint_of<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    let bytes = serde_json::to_vec(value)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionOutcome {
    Verified,
    Failed,
    Unverified,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeStatus {
    Verified,
    ExecutionFailed,
    VerificationFailed,
    Rejected,
}

#[derive(Clone, Debug, Serialize)]
pub struct OutcomeTrace {
    pub action_id: String,
    pub task_revision: u64,
    pub status: OutcomeStatus,
    pub rejection: Option<RejectReason>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action() -> PreparedAction<&'static str> {
        PreparedAction {
            id: "inspect".into(),
            description: "Read the relevant source".into(),
            payload: "exact-host-payload",
            prepared_at_revision: 7,
            preconditions: Vec::new(),
            read_set_fingerprint: Some("files-v1".into()),
            expires_at_ms: Some(100),
        }
    }

    fn context() -> ValidationContext {
        ValidationContext {
            now_ms: 99,
            current_read_set_fingerprint: Some("files-v1".into()),
            authorized: true,
        }
    }

    #[test]
    fn selected_action_is_bound_to_fresh_state_and_host_authority() {
        let state = TaskState::new(7);
        let action = action();
        assert!(eligible(&state, &action, &context()));
        assert_eq!(
            validate_selected(&state, &action, "other", &context()).rejection,
            Some(RejectReason::WrongSelection)
        );
        let mut changed = context();
        changed.current_read_set_fingerprint = Some("files-v2".into());
        assert_eq!(
            validate_selected(&state, &action, "inspect", &changed).rejection,
            Some(RejectReason::StaleReadSet)
        );
        changed = context();
        changed.authorized = false;
        assert_eq!(
            validate_selected(&state, &action, "inspect", &changed).rejection,
            Some(RejectReason::Unauthorized)
        );
        changed = context();
        changed.now_ms = 100;
        assert_eq!(
            validate_selected(&state, &action, "inspect", &changed).rejection,
            Some(RejectReason::Expired)
        );
    }

    #[test]
    fn only_verified_outcome_advances_dependencies() {
        let mut state = TaskState::new(7);
        let action = action();
        let accepted = validate_selected(&state, &action, "inspect", &context());
        let failed = state.record_outcome(&action, &accepted, ExecutionOutcome::Unverified);
        assert_eq!(failed.status, OutcomeStatus::VerificationFailed);
        assert_eq!(state.revision, 7);
        assert!(!state.is_completed("inspect"));

        let mut other = action.clone();
        other.id = "mutate".into();
        let mismatched = state.record_outcome(&other, &accepted, ExecutionOutcome::Verified);
        assert_eq!(mismatched.status, OutcomeStatus::Rejected);
        assert!(!state.is_completed("mutate"));

        let passed = state.record_outcome(&action, &accepted, ExecutionOutcome::Verified);
        assert_eq!(passed.status, OutcomeStatus::Verified);
        assert_eq!(state.revision, 8);
        assert!(state.is_completed("inspect"));
        let replay = state.record_outcome(&action, &accepted, ExecutionOutcome::Verified);
        assert_eq!(replay.status, OutcomeStatus::Rejected);
    }

    #[test]
    fn unmet_dependency_and_user_revision_stop_dispatch() {
        let mut state = TaskState::new(7);
        let mut action = action();
        action
            .preconditions
            .push(Precondition::Completed("plan".into()));
        assert_eq!(
            validate_selected(&state, &action, "inspect", &context()).rejection,
            Some(RejectReason::UnmetPrecondition)
        );
        action.preconditions.clear();
        state.invalidate();
        assert_eq!(
            validate_selected(&state, &action, "inspect", &context()).rejection,
            Some(RejectReason::StaleRevision)
        );
    }

    #[test]
    fn read_set_hash_is_stable_for_same_ordered_snapshot() {
        let snapshot = vec![("src/lib.rs", "rev-a"), ("Cargo.toml", "rev-b")];
        assert_eq!(
            fingerprint_of(&snapshot).unwrap(),
            fingerprint_of(&snapshot).unwrap()
        );
        assert_ne!(
            fingerprint_of(&snapshot).unwrap(),
            fingerprint_of(&vec![("src/lib.rs", "rev-c")]).unwrap()
        );
    }
}
