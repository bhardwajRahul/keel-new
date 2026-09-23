//! Rust port of `@deepseek-ai/dsh-session` (`packages/core/session`): the
//! append-only session event log, the ordered surface projecting derived LLM
//! history, the in-memory store with its publication events, request-header
//! reconstruction, and crash-recovery turn repair.
//!
//! Crate-level divergences (each module documents its own):
//! - Lossless-JSON validation (`json.ts`) collapses: events are typed serde
//!   structs, and `serde_json` guarantees the round-trip domain.
//! - Deep-freeze machinery collapses into Rust ownership.
//! - The persistence chunk-row packing (`chunk-rows.ts`) ships with the
//!   persistence crate, not here.

mod header_fold;
mod repair;
mod session;
mod surface;
mod types;

pub use header_fold::{canonical_header, fold_request_header, header_equals};
pub use repair::{TOOL_NOT_STARTED, TOOL_OUTCOME_UNKNOWN, interrupted_turn_closers};
pub use session::{
    Session, SessionCreated, SessionDisposed, SessionError, SessionEventPublished, SessionFlush,
    SessionForkError, SessionForkErrorCode, SessionStore,
};
pub use surface::{
    SurfaceFoldReplacement, SurfaceFoldResult, SurfaceManager, derive_event_message, fold_surface,
    is_append_surface_event, is_replacement_surface_event, is_surface_eligible_type,
    is_surface_event,
};
pub use types::{
    AgentCancelCause, AppendTag, EpochHeader, ReplaceTag, RequestContext, RequestHeaderReason,
    SESSION_FORMAT_VERSION, SessionEvent, SessionEventData, SessionHeader, SessionId, SessionMeta,
    SessionOrigin, SurfaceIntent, SurfaceOp, TodoItem, TodoStatus, ToolErrorRef, TurnEndReason,
};

/// Every session event type this build understands — the persistence read
/// path refuses to interpret a log containing a type outside this set unless
/// the event carries the envelope's `ignorable` marker: such a log was likely
/// written by a newer harness, and silently skipping a required event would
/// reconstruct a wrong session (upstream `KNOWN_SESSION_EVENT_TYPES`).
pub const KNOWN_SESSION_EVENT_TYPES: &[&str] = &[
    "agent-preset/selected",
    "agent/inbox/spliced",
    "approval/asked",
    "approval/decided",
    "approval/policy",
    "assistant/chunk",
    "assistant/message",
    "command/done",
    "command/run",
    "compaction/end",
    "compaction/prune",
    "compaction/start",
    "compaction/summary",
    "decision/receipt",
    "feedback/record",
    "goal/change",
    "hook/invoked",
    "hook/result",
    "llm/retry",
    "llm/retry-started",
    "permission/preset",
    "plan/mode",
    "request/context",
    "request/header",
    "sandbox/mode",
    "schedule/change",
    "session/end-seed",
    "session/title",
    "session/title-llm-request",
    "step/end",
    "step/start",
    "subagent/descriptor",
    "todo/write",
    "tool-workflow/agent-end",
    "tool-workflow/agent-start",
    "tool-workflow/run-end",
    "tool-workflow/run-start",
    "tool/call",
    "tool/code-dispatch",
    "tool/code-dispatch-start",
    "tool/result",
    "turn/end",
    "turn/start",
    "user/message",
    "web/deepseek-search-llm-request",
];

/// Whether this build's persistence readers may interpret an event of the
/// given type (see [`KNOWN_SESSION_EVENT_TYPES`]).
pub fn is_known_session_event_type(event_type: &str) -> bool {
    KNOWN_SESSION_EVENT_TYPES.contains(&event_type)
}
