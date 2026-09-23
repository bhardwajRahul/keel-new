//! Incremental projection of durable agent inbox events, ported from
//! `packages/core/agent/src/inbox.ts`: the live pending-message lists are a
//! replay-once projection of `agent/inbox/spliced` session events, and every
//! mutation commits its durable splice before the projection changes.

use crate::types::{InboxSplice, InboxTarget, SpliceOutcome};
use dsh_llm::{Message, MessageId};
use dsh_session::{Session, SurfaceIntent};
use std::cell::RefCell;
use std::rc::Rc;

/// Live notifications committed by inbox mutations (upstream
/// `InboxNotifications`).
pub trait InboxNotifications {
    fn inserted(&self, message: &Message);
    fn discarded(&self, message: &Message);
    fn claimed(&self, message: &Message, turn: u64);
}

/// No-op notifications for detached or test use.
pub struct SilentNotifications;
impl InboxNotifications for SilentNotifications {
    fn inserted(&self, _message: &Message) {}
    fn discarded(&self, _message: &Message) {}
    fn claimed(&self, _message: &Message, _turn: u64) {}
}

/// A replay-once projection that incrementally consumes later inbox splices
/// (upstream `Inbox`).
pub struct Inbox {
    session: Rc<Session>,
    notifications: Box<dyn InboxNotifications>,
    next_turn: RefCell<Vec<Message>>,
    next_step: RefCell<Vec<Message>>,
}

#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct InboxError(pub String);

impl Inbox {
    /// Build the projection by replaying persisted splices past the durable
    /// seed boundary; a malformed persisted splice is a load failure.
    pub fn new(
        session: Rc<Session>,
        notifications: Box<dyn InboxNotifications>,
    ) -> Result<Inbox, InboxError> {
        let inbox = Inbox {
            session: session.clone(),
            notifications,
            next_turn: RefCell::new(Vec::new()),
            next_step: RefCell::new(Vec::new()),
        };
        let seed_boundary = session.header.seed_length.unwrap_or(0) as usize;
        for event in session.events().iter().skip(seed_boundary) {
            if let Some(splice) = InboxSplice::from_event(event) {
                inbox.validate(&splice).map_err(|error| {
                    InboxError(format!(
                        "invalid persisted inbox splice at session seq {}: {}",
                        event.seq, error.0
                    ))
                })?;
                inbox.apply(&splice);
            }
        }
        Ok(inbox)
    }

    fn list(&self, target: InboxTarget) -> &RefCell<Vec<Message>> {
        match target {
            InboxTarget::NextTurn => &self.next_turn,
            InboxTarget::NextStep => &self.next_step,
        }
    }

    /// Prompts awaiting individual turns.
    pub fn next_turn(&self) -> Vec<Message> {
        self.next_turn.borrow().clone()
    }

    /// Input awaiting the next step boundary.
    pub fn next_step(&self) -> Vec<Message> {
        self.next_step.borrow().clone()
    }

    /// Whether either pending list contains work.
    pub fn has_pending(&self) -> bool {
        !self.next_turn.borrow().is_empty() || !self.next_step.borrow().is_empty()
    }

    /// Durably cancel all pending input, clearing next-step before next-turn.
    pub fn clear(&self) -> Result<(), InboxError> {
        let step_len = self.next_step.borrow().len() as u64;
        self.splice(InboxTarget::NextStep, 0, step_len, vec![])?;
        let turn_len = self.next_turn.borrow().len() as u64;
        self.splice(InboxTarget::NextTurn, 0, turn_len, vec![])?;
        Ok(())
    }

    /// Remove and return the complete batch proposed for one step, publishing
    /// each claimed message. The durable splices are pure deletions (no
    /// `canceled` outcome — a claim is the loop's own read).
    pub fn claim(&self, target: InboxTarget, turn: u64) -> Result<Vec<Message>, InboxError> {
        let step_len = self.next_step.borrow().len() as u64;
        let mut claimed = self.mutate(InboxTarget::NextStep, 0, step_len, vec![], false)?;
        if target == InboxTarget::NextTurn {
            claimed.extend(self.mutate(InboxTarget::NextTurn, 0, 1, vec![], false)?);
        }
        for message in &claimed {
            self.notifications.claimed(message, turn);
        }
        Ok(claimed)
    }

    /// Append one message and durably record the insertion.
    pub fn append(&self, target: InboxTarget, message: Message) -> Result<(), InboxError> {
        let len = self.list(target).borrow().len() as u64;
        self.splice(target, len, 0, vec![message])?;
        Ok(())
    }

    /// Prepend one message and durably record the insertion.
    pub fn prepend(&self, target: InboxTarget, message: Message) -> Result<(), InboxError> {
        self.splice(target, 0, 0, vec![message])?;
        Ok(())
    }

    /// Replace one pending message in place (possibly changing identity),
    /// publishing the old as discarded and the new as inserted.
    pub fn replace(
        &self,
        message_id: &MessageId,
        new_message: Message,
    ) -> Result<bool, InboxError> {
        let Some((target, index)) = self.locate(message_id) else {
            return Ok(false);
        };
        self.splice(target, index as u64, 1, vec![new_message])?;
        Ok(true)
    }

    /// Remove one pending message and durably record its cancellation.
    pub fn remove(&self, message_id: &MessageId) -> Result<bool, InboxError> {
        let Some((target, index)) = self.locate(message_id) else {
            return Ok(false);
        };
        self.splice(target, index as u64, 1, vec![])?;
        Ok(true)
    }

    /// Apply standard splice semantics and durably record the normalized
    /// result; removed messages publish as discarded.
    pub fn splice(
        &self,
        target: InboxTarget,
        start: u64,
        delete_count: u64,
        inserted: Vec<Message>,
    ) -> Result<Vec<Message>, InboxError> {
        self.mutate(target, start, delete_count, inserted, true)
    }

    fn locate(&self, message_id: &MessageId) -> Option<(InboxTarget, usize)> {
        for target in [InboxTarget::NextTurn, InboxTarget::NextStep] {
            if let Some(index) = self
                .list(target)
                .borrow()
                .iter()
                .position(|message| message.id == *message_id)
            {
                return Some((target, index));
            }
        }
        None
    }

    /// Commit one normalized mutation: durable event first, projection second
    /// (synchronous `session/event` observers see the pre-splice lists), then
    /// live notifications.
    fn mutate(
        &self,
        target: InboxTarget,
        start: u64,
        delete_count: u64,
        inserted: Vec<Message>,
        discard_removed: bool,
    ) -> Result<Vec<Message>, InboxError> {
        let len = self.list(target).borrow().len() as u64;
        let actual_start = start.min(len);
        let actual_delete = delete_count.min(len - actual_start);
        if actual_delete == 0 && inserted.is_empty() {
            return Ok(Vec::new());
        }
        let outcome = (discard_removed && actual_delete > 0).then_some(SpliceOutcome::Canceled);
        let splice = InboxSplice {
            target,
            start: actual_start,
            removed_count: (actual_delete > 0).then_some(actual_delete),
            inserted,
            outcome,
        };
        self.validate(&splice)?;
        self.session
            .append(splice.to_event_data(), Some(SurfaceIntent::default()))
            .map_err(|error| InboxError(error.0.clone()))?;
        let removed = self.apply(&splice);
        if discard_removed {
            for message in &removed {
                self.notifications.discarded(message);
            }
        }
        for message in &splice.inserted {
            self.notifications.inserted(message);
        }
        Ok(removed)
    }

    /// Apply one normalized durable splice to the projection.
    fn apply(&self, splice: &InboxSplice) -> Vec<Message> {
        let mut list = self.list(splice.target).borrow_mut();
        let start = splice.start as usize;
        let removed_count = splice.removed_count.unwrap_or(0) as usize;
        list.splice(
            start..start + removed_count,
            splice.inserted.iter().cloned(),
        )
        .collect()
    }

    /// Validate one normalized splice against the current projection: bounds,
    /// and no duplicate pending identity across both lists.
    fn validate(&self, splice: &InboxSplice) -> Result<(), InboxError> {
        let removed_count = splice.removed_count.unwrap_or(0);
        let start = splice.start as usize;
        // Collect owned ids so the borrow ends before the cross-list check.
        let candidate_ids: Vec<String> = {
            let list = self.list(splice.target).borrow();
            if splice.start + removed_count > list.len() as u64 {
                return Err(InboxError("invalid inbox splice".into()));
            }
            list.iter()
                .take(start)
                .chain(splice.inserted.iter())
                .chain(list.iter().skip(start + removed_count as usize))
                .map(|message| message.id.as_str().to_string())
                .collect()
        };
        let other_ids: Vec<String> = match splice.target {
            InboxTarget::NextTurn => self.next_step.borrow(),
            InboxTarget::NextStep => self.next_turn.borrow(),
        }
        .iter()
        .map(|message| message.id.as_str().to_string())
        .collect();
        let mut ids = std::collections::HashSet::new();
        for id in candidate_ids.into_iter().chain(other_ids) {
            if !ids.insert(id.clone()) {
                return Err(InboxError(format!("message \"{id}\" is already pending")));
            }
        }
        Ok(())
    }
}
