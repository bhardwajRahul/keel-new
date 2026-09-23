//! Crash-recovery repair for an interrupted session log, ported from
//! `packages/core/session/src/repair.ts`: preserves a fully written final
//! turn and supplies the missing tool, step, and turn boundaries needed to
//! resume with a provider-valid transcript.

use crate::types::{SessionEvent, SessionEventData, SurfaceOp, ToolErrorRef, TurnEndReason};
use dsh_llm::{CallId, ContentBlock, Message, MessageId, MessageSource, Role};

/// Recovery code for an assistant tool request that never reached a recorded
/// call start.
pub const TOOL_NOT_STARTED: &str = "TOOL_NOT_STARTED";

/// Recovery code for a recorded tool call whose completed outcome was not
/// durably recorded.
pub const TOOL_OUTCOME_UNKNOWN: &str = "TOOL_OUTCOME_UNKNOWN";

struct PendingCall {
    call_id: CallId,
    step: u64,
    call_seq: Option<u64>,
}

/// Return deterministic synthetic events that close an open tail turn.
/// Unmatched calls receive error results first, then an open `step/end`, then
/// an interrupted `turn/end`; seqs continue the log and timestamps reuse the
/// last real event (never inventing a "future" time). A balanced or empty log
/// returns no events.
pub fn interrupted_turn_closers(events: &[SessionEvent]) -> Vec<SessionEvent> {
    let mut open_turn: Option<u64> = None;
    let mut open_step: Option<u64> = None;
    // Reset at each turn boundary so earlier calls cannot leak into tail
    // repair. Assistant blocks register calls; later tool/call events add
    // their seqs for source citation. Insertion order preserved.
    let mut pending: Vec<PendingCall> = Vec::new();
    for event in events {
        match &event.data {
            SessionEventData::TurnStart { turn } => {
                open_turn = Some(*turn);
                open_step = None;
                pending.clear();
            }
            SessionEventData::TurnEnd { .. } => {
                open_turn = None;
                open_step = None;
                pending.clear();
            }
            SessionEventData::StepStart { step, .. } => {
                open_step = Some(*step);
            }
            SessionEventData::StepEnd { .. } => {
                pending.clear();
                open_step = None;
            }
            SessionEventData::AssistantMessage { step, message, .. } => {
                for block in &message.content {
                    if let ContentBlock::ToolCall { id, .. } = block {
                        pending.push(PendingCall {
                            call_id: id.clone(),
                            step: *step,
                            call_seq: None,
                        });
                    }
                }
            }
            SessionEventData::ToolCall { call_id, .. } => {
                if let Some(entry) = pending.iter_mut().find(|entry| entry.call_id == *call_id) {
                    entry.call_seq = Some(event.seq);
                }
            }
            SessionEventData::ToolResult { message, .. } => {
                if let MessageSource::Tool { call_id } = &message.source {
                    pending.retain(|entry| entry.call_id != *call_id);
                }
            }
            _ => {}
        }
    }

    let (Some(turn), Some(last)) = (open_turn, events.last()) else {
        return Vec::new();
    };

    let mut seq = last.seq + 1;
    let time = last.time;
    let mut closers = Vec::new();

    // Close calls before their step: providers reject dangling assistant
    // calls, in transcript order.
    for entry in &pending {
        let started = entry.call_seq.is_some();
        let text = if started {
            "The tool call was interrupted after it was recorded, but no result was durably \
             recorded. Its outcome is unknown. Decide whether to retry from the tool semantics: \
             retry only if the operation is read-only or idempotent; if it may have side effects, \
             first verify external state or ask the user. Do not retry blindly."
        } else {
            "The tool call was interrupted before the Harness recorded it as started. Retry it if \
             it is still needed."
        };
        let message = Message {
            id: MessageId::new(format!(
                "interrupted-tool-result-{}-{seq}",
                entry.call_id.as_str()
            )),
            role: Role::User,
            source: MessageSource::Tool {
                call_id: entry.call_id.clone(),
            },
            content: vec![ContentBlock::ToolResult {
                tool_call_id: entry.call_id.clone(),
                is_error: Some(true),
                content: vec![ContentBlock::Text {
                    text: text.to_string(),
                }],
            }],
        };
        closers.push(SessionEvent {
            seq,
            time,
            data: SessionEventData::ToolResult {
                turn,
                step: entry.step,
                message,
                error: Some(if started {
                    ToolErrorRef {
                        name: "ToolOutcomeUnknownError".into(),
                        code: TOOL_OUTCOME_UNKNOWN.into(),
                    }
                } else {
                    ToolErrorRef {
                        name: "ToolNotStartedError".into(),
                        code: TOOL_NOT_STARTED.into(),
                    }
                }),
                meta: None,
            },
            surface_op: Some(SurfaceOp::append()),
            source_event_seqs: entry.call_seq.map(|call_seq| vec![call_seq]),
            ignorable: None,
        });
        seq += 1;
    }

    // A turn/end while a step is open would violate the boundary invariant, so
    // synthesize the step's closer first.
    if let Some(step) = open_step {
        closers.push(SessionEvent {
            seq,
            time,
            data: SessionEventData::StepEnd { turn, step },
            surface_op: None,
            source_event_seqs: None,
            ignorable: None,
        });
        seq += 1;
    }
    closers.push(SessionEvent {
        seq,
        time,
        data: SessionEventData::TurnEnd {
            turn,
            reason: TurnEndReason::Interrupted,
        },
        surface_op: None,
        source_event_seqs: None,
        ignorable: None,
    });
    closers
}
