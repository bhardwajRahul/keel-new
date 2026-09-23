//! The ordered surface over the session event log, ported from
//! `packages/core/session/src/surface.ts`: the model-visible view of
//! message-producing events. The append-only log remains the source of truth.

use crate::types::{SessionEvent, SessionEventData, SurfaceOp};
use dsh_llm::{ContentBlock, Message};

/// Whether an event type can join the model-visible surface.
pub fn is_surface_eligible_type(event_type: &str) -> bool {
    matches!(
        event_type,
        "user/message" | "assistant/message" | "tool/result"
    )
}

/// Whether the event carries its required surface marker (upstream
/// `isSurfaceEvent`).
pub fn is_surface_event(event: &SessionEvent) -> bool {
    is_surface_eligible_type(event.event_type()) && event.surface_op.is_some()
}

/// Whether the event entered the surface at its own log position (never a
/// replacement copy). Append-origin events are the durable source material for
/// a human transcript; replacement copies stay model-only.
pub fn is_append_surface_event(event: &SessionEvent) -> bool {
    matches!(&event.surface_op, Some(SurfaceOp::Append(_))) && is_surface_event(event)
}

/// Whether the event shadowed an existing surface range instead of appending.
pub fn is_replacement_surface_event(event: &SessionEvent) -> bool {
    matches!(&event.surface_op, Some(SurfaceOp::Replace { .. })) && is_surface_event(event)
}

/// Project a single event into the LLM message it derives to, or `None` when
/// it produces none — a non-surface event, or an empty-content
/// assistant/message (which exists only to host a max-tokens step's usage).
/// THE per-node projection rule: the live session folds it over its surface,
/// and offline reconstructors fold the same function over a log prefix.
/// Content stays verbatim — framing is caller-owned, never re-added here.
pub fn derive_event_message(event: &SessionEvent) -> Option<&Message> {
    match &event.data {
        SessionEventData::UserMessage(message) => Some(message),
        SessionEventData::AssistantMessage { message, .. } => {
            if message.content.is_empty() {
                None
            } else {
                Some(message)
            }
        }
        SessionEventData::ToolResult { message, .. } => Some(message),
        _ => None,
    }
}

/// One replacement operation observed while folding a surface.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceFoldReplacement {
    /// Seq of the event that replaced the prior surface range.
    pub seq: u64,
    /// Declared inclusive start seq of the replaced range.
    pub start: u64,
    /// Declared inclusive end seq of the replaced range.
    pub end: u64,
    /// Actual surface entries removed, in surface order.
    pub shadowed_seqs: Vec<u64>,
}

/// Complete result of replaying the surface operations in a session log.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceFoldResult {
    /// Current surface event sequences in model-visible order.
    pub nodes: Vec<u64>,
    /// Replacement operations in event order.
    pub replacements: Vec<SurfaceFoldReplacement>,
}

#[derive(Default)]
struct FoldState {
    nodes: Vec<u64>,
    replace_generation: u64,
}

enum SurfacePlan {
    Append {
        seq: u64,
    },
    Replace {
        seq: u64,
        start: u64,
        end: u64,
        start_idx: usize,
        end_idx: usize,
        shadowed: Vec<u64>,
    },
}

fn surface_op_of(event: &SessionEvent) -> Result<Option<&SurfaceOp>, String> {
    let event_type = event.event_type();
    if !is_surface_eligible_type(event_type) {
        if event.surface_op.is_some() {
            return Err(format!(
                "session event \"{event_type}\" is not surface-eligible and cannot carry surfaceOp"
            ));
        }
        if event.source_event_seqs.is_some() {
            return Err(format!(
                "session event \"{event_type}\" is not surface-eligible and cannot carry sourceEventSeqs"
            ));
        }
        return Ok(None);
    }
    match &event.surface_op {
        None => Err(format!(
            "session event \"{event_type}\" is surface-eligible and requires a surfaceOp marker"
        )),
        Some(op) => Ok(Some(op)),
    }
}

fn assert_provenance(event: &SessionEvent, shadowed: &[u64]) -> Result<(), String> {
    let mut sources: Vec<u64> = Vec::new();
    if let Some(raw) = &event.source_event_seqs {
        if raw.is_empty() && event.event_type() != "assistant/message" {
            return Err("sourceEventSeqs must not be empty except on assistant/message".into());
        }
        for source in raw {
            if sources.contains(source) {
                return Err("sourceEventSeqs must not contain duplicates".into());
            }
            if *source >= event.seq {
                return Err(format!(
                    "sourceEventSeqs must reference earlier events: {source} >= current seq {}",
                    event.seq
                ));
            }
            sources.push(*source);
        }
    }
    let missing: Vec<String> = shadowed
        .iter()
        .filter(|seq| !sources.contains(seq))
        .map(|seq| seq.to_string())
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "surface replace: sourceEventSeqs must include every shadowed surface node; missing {}",
            missing.join(", ")
        ));
    }
    Ok(())
}

/// Restrict a tool-result replacement to one current result's content: the
/// wrapper (turn/step/error/meta, message identity, call correlation) must
/// stay identical, only the result block's inner content may change.
fn assert_tool_result_rewrite(
    event: &SessionEvent,
    shadowed: &[u64],
    events: &[SessionEvent],
    base_seq: u64,
) -> Result<(), String> {
    let SessionEventData::ToolResult {
        turn,
        step,
        message,
        error,
        meta,
    } = &event.data
    else {
        return Ok(());
    };
    if shadowed.len() != 1 {
        return Err("tool/result surface replacement must rewrite exactly one current node".into());
    }
    let original_seq = shadowed[0];
    let index = (original_seq - base_seq) as usize;
    let Some(original) = events.get(index) else {
        return Err("tool/result surface replacement must target a current tool/result".into());
    };
    let SessionEventData::ToolResult {
        turn: original_turn,
        step: original_step,
        message: original_message,
        error: original_error,
        meta: original_meta,
    } = &original.data
    else {
        return Err("tool/result surface replacement must target a current tool/result".into());
    };
    let strip_content = |message: &Message| -> Message {
        let mut stripped = message.clone();
        if let Some(ContentBlock::ToolResult { content, .. }) = stripped.content.first_mut() {
            content.clear();
        }
        stripped
    };
    if turn != original_turn
        || step != original_step
        || error != original_error
        || meta != original_meta
        || strip_content(message) != strip_content(original_message)
    {
        return Err("tool/result surface replacement may change only content".into());
    }
    Ok(())
}

fn plan_surface_event(
    state: &FoldState,
    event: &SessionEvent,
    expected_seq: u64,
    events: &[SessionEvent],
    base_seq: u64,
) -> Result<Option<SurfacePlan>, String> {
    if event.seq != expected_seq {
        return Err(format!(
            "session event seq {} is not contiguous; expected {expected_seq}",
            event.seq
        ));
    }
    let Some(op) = surface_op_of(event)? else {
        return Ok(None);
    };
    match op {
        SurfaceOp::Append(_) => {
            assert_provenance(event, &[])?;
            Ok(Some(SurfacePlan::Append { seq: event.seq }))
        }
        SurfaceOp::Replace { start, end, .. } => {
            let start_idx = state
                .nodes
                .iter()
                .position(|seq| seq == start)
                .ok_or_else(|| {
                    format!("surface replace: start seq {start} not found in surface")
                })?;
            let end_idx = state
                .nodes
                .iter()
                .position(|seq| seq == end)
                .ok_or_else(|| format!("surface replace: end seq {end} not found in surface"))?;
            if start_idx > end_idx {
                return Err(format!(
                    "surface replace: start seq {start} (index {start_idx}) is after end seq {end} (index {end_idx})"
                ));
            }
            let shadowed: Vec<u64> = state.nodes[start_idx..=end_idx].to_vec();
            assert_provenance(event, &shadowed)?;
            assert_tool_result_rewrite(event, &shadowed, events, base_seq)?;
            Ok(Some(SurfacePlan::Replace {
                seq: event.seq,
                start: *start,
                end: *end,
                start_idx,
                end_idx,
                shadowed,
            }))
        }
    }
}

fn apply_surface_plan(
    state: &mut FoldState,
    plan: Option<SurfacePlan>,
) -> Option<SurfaceFoldReplacement> {
    match plan {
        Some(SurfacePlan::Append { seq }) => {
            state.nodes.push(seq);
            None
        }
        Some(SurfacePlan::Replace {
            seq,
            start,
            end,
            start_idx,
            end_idx,
            shadowed,
        }) => {
            state.nodes.splice(start_idx..=end_idx, [seq]);
            state.replace_generation += 1;
            Some(SurfaceFoldReplacement {
                seq,
                start,
                end,
                shadowed_seqs: shadowed,
            })
        }
        None => None,
    }
}

/// Replay a complete session log through the canonical surface fold (upstream
/// `foldSurface`).
pub fn fold_surface(events: &[SessionEvent]) -> Result<SurfaceFoldResult, String> {
    let mut state = FoldState::default();
    let mut replacements = Vec::new();
    for (index, event) in events.iter().enumerate() {
        let plan = plan_surface_event(&state, event, index as u64, events, 0)?;
        if let Some(replacement) = apply_surface_plan(&mut state, plan) {
            replacements.push(replacement);
        }
    }
    Ok(SurfaceFoldResult {
        nodes: state.nodes,
        replacements,
    })
}

/// Incremental ordered surface view and append-boundary validator (upstream
/// `SurfaceManager`). Owned by [`crate::Session`]; validates each candidate
/// before it enters the log so a failure cannot partially mutate the surface.
#[derive(Default)]
pub struct SurfaceManager {
    state: FoldState,
    processed: usize,
}

impl SurfaceManager {
    /// Validate the next candidate against the current log without mutating
    /// the committed surface; on success the caller appends and calls
    /// [`SurfaceManager::commit`].
    pub fn validate_next(
        &mut self,
        log: &[SessionEvent],
        event: &SessionEvent,
    ) -> Result<(), String> {
        debug_assert_eq!(self.processed, log.len(), "surface fold lags the log");
        let expected_seq = log.len() as u64;
        plan_surface_event(&self.state, event, expected_seq, log, 0).map(|_| ())
    }

    /// Fold the event that was just appended at the log tail.
    pub fn commit(&mut self, log: &[SessionEvent]) {
        while self.processed < log.len() {
            let event = &log[self.processed];
            let plan = plan_surface_event(&self.state, event, self.processed as u64, log, 0)
                .expect("committed events were validated before append");
            apply_surface_plan(&mut self.state, plan);
            self.processed += 1;
        }
    }

    /// Surface event sequences in model-visible order.
    pub fn nodes(&self) -> &[u64] {
        &self.state.nodes
    }

    /// Monotonic count of committed positional replacements; a change
    /// invalidates derived-history caches.
    pub fn replace_generation(&self) -> u64 {
        self.state.replace_generation
    }
}
