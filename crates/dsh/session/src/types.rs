//! Session vocabulary, ported from `packages/core/session/src/types.ts`:
//! ids, the durable header, turn-end reasons, and the event envelope.
//!
//! Divergences:
//! - `SessionEventMap` is merge-extensible upstream; here [`SessionEventData`]
//!   is a closed enum over the core vocabulary plus an `Extension` variant
//!   carrying `(type, JSON data, ignorable)` for plugin events — the Rust
//!   counterpart of declaration merging that still round-trips unknown types.
//! - Events serialize with the same field names (`type`, `seq`, `time`,
//!   `data`, `surfaceOp`, `sourceEventSeqs`, `ignorable`).

use dsh_llm::{
    CallId, LlmCallConfig, LlmCallConfigAdapterDefaults, LlmFailure, Message, StreamChunk,
    TokenUsage, ToolSchema,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use dsh_llm::SessionId;

/// The on-disk session format version, stamped into every new header and
/// enforced on load. Pinned at 0 while unreleased: incompatible logs are
/// rejected, no migration is provided.
pub const SESSION_FORMAT_VERSION: u64 = 0;

/// Immutable validated storage metadata, kept outside the conversation event
/// log (upstream `SessionHeader`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionHeader {
    /// On-disk format version; backends reject any other version on load.
    pub version: u64,
    /// The session's id.
    pub id: SessionId,
    /// Non-negative Unix epoch milliseconds when the session was created.
    pub created_at: u64,
    /// Absolute working directory the session was created in, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    /// The session this one was forked from (seed lineage), if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session: Option<SessionId>,
    /// How many leading events were inherited through a seed — the durable
    /// fork-lineage boundary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed_length: Option<u64>,
    /// Coarse product classification for a subagent child (presentation
    /// metadata only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<SessionOrigin>,
    /// Delegation depth: absent (zero) for a top-level session, parent + 1 for
    /// a subagent child. Persisted so recursion budgets survive resume.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delegation_depth: Option<u64>,
    /// Id of the agent preset this session's agent was composed from, when the
    /// deployment composes per session.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_preset: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionOrigin {
    Subagent,
}

/// Caller-supplied storage fields folded into a [`SessionHeader`] at creation
/// (upstream `CreateSessionOptions.meta`).
#[derive(Debug, Clone, Default)]
pub struct SessionMeta {
    pub cwd: Option<String>,
    pub parent_session: Option<SessionId>,
    pub created_at: Option<u64>,
    pub seed_length: Option<u64>,
    pub origin: Option<SessionOrigin>,
    pub delegation_depth: Option<u64>,
    pub agent_preset: Option<String>,
}

/// Why an active agent driver was cancelled (upstream `AgentCancelCause`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum AgentCancelCause {
    User,
    Parent,
    Hook {
        reason: String,
    },
    Disposed,
    /// Durable import whose original coarse record carried no cause
    /// (upstream `TurnEndCancelCause`'s extra variant).
    Legacy,
}

/// Why a turn ended (upstream `TurnEndReasonMap`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum TurnEndReason {
    Completed,
    /// A cancellation request interrupted the live turn.
    Aborted {
        reason: AgentCancelCause,
    },
    Blocked,
    /// The turn failed; `error` is always structured failure facts.
    Error {
        error: LlmFailure,
    },
    /// At least one step reached its output-token ceiling.
    MaxTokens,
    /// A persistence backend closed a crash-orphaned turn on reload. The loop
    /// never emits this marker.
    Interrupted,
}

/// One entry in an agent's todo list — the unit of the `todo/write`
/// whole-list snapshot. Deliberately minimal: the list is replaced wholesale
/// on every write, so entries need no stable identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TodoItem {
    /// Short imperative line shown in the UI.
    pub content: String,
    pub status: TodoStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

/// Logged request state outside derived history: call config, system prompt,
/// and tools; the latest full `request/header` snapshot reconstructs it
/// (upstream `EpochHeader`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EpochHeader {
    /// The conversation's call configuration.
    pub config: LlmCallConfig,
    /// Config fields materialized from the exact adapter rather than proposed
    /// by a caller.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapter_defaults: Option<LlmCallConfigAdapterDefaults>,
    /// Rendered system prompt text; absent for a system-less request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Assembled tool schemas; absent for a tool-less request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolSchema>>,
}

/// Registration-bound metadata for one resolved model route (upstream
/// `RequestContext`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestContext {
    pub provider: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
}

/// Why a `request/header` snapshot was appended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RequestHeaderReason {
    /// The log's first header (a new conversation).
    Initial,
    /// A loop instance's first request over a log that already has headers.
    Resume,
    /// A later request used a different header.
    Change,
}

/// The core append-only event vocabulary (upstream `SessionEventMap`).
///
/// Merge-extensible upstream; plugin events land in [`SessionEventData::Extension`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum SessionEventData {
    /// Opens a turn before the loop claims queued input or runs pre-step.
    #[serde(rename = "turn/start")]
    TurnStart { turn: u64 },
    /// Closes a turn with the reason that ended it.
    #[serde(rename = "turn/end")]
    TurnEnd { turn: u64, reason: TurnEndReason },
    /// Opens one step — one model call plus the tool executions it requested.
    #[serde(rename = "step/start")]
    StepStart { turn: u64, step: u64 },
    /// Closes one step.
    #[serde(rename = "step/end")]
    StepEnd { turn: u64, step: u64 },
    /// A user-role message on the model-visible surface: a direct prompt, an
    /// injected context, or a goal continuation; `source` tells them apart.
    #[serde(rename = "user/message")]
    UserMessage(Message),
    /// Raw stream chunk — token-level replay fidelity.
    #[serde(rename = "assistant/chunk")]
    AssistantChunk {
        turn: u64,
        step: u64,
        chunk: StreamChunk,
    },
    /// Assembled assistant message for one step; carries the step's usage so
    /// output and accounting travel together.
    #[serde(rename = "assistant/message", rename_all = "camelCase")]
    AssistantMessage {
        turn: u64,
        step: u64,
        message: Message,
        #[serde(skip_serializing_if = "Option::is_none")]
        usage: Option<TokenUsage>,
    },
    /// The model requested one tool invocation, arguments verbatim (unparsed).
    #[serde(rename = "tool/call", rename_all = "camelCase")]
    ToolCall {
        turn: u64,
        step: u64,
        call_id: CallId,
        name: String,
        arguments: String,
    },
    /// A completed tool call's model-facing result, optional failure identity,
    /// and optional tool-private JSON `meta` presentation payload.
    #[serde(rename = "tool/result", rename_all = "camelCase")]
    ToolResult {
        turn: u64,
        step: u64,
        message: Message,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<ToolErrorRef>,
        #[serde(skip_serializing_if = "Option::is_none")]
        meta: Option<Value>,
    },
    /// Whole-list snapshot; latest write wins on replay. Log-only UI state.
    #[serde(rename = "todo/write")]
    TodoWrite { todos: Vec<TodoItem> },
    /// Full header for the next request, appended inside its step before
    /// dispatch. Log-only; the latest snapshot reconstructs the header.
    #[serde(rename = "request/header")]
    RequestHeader {
        header: EpochHeader,
        reason: RequestHeaderReason,
    },
    /// Route metadata for the next request, logged only when the route or
    /// capacity changes.
    #[serde(rename = "request/context")]
    RequestContext(RequestContext),
    /// Marks the end of a constructor seed: events before it came from the
    /// seed (resume, fork, replay). `Session`'s constructor is the only
    /// legitimate writer.
    #[serde(rename = "session/end-seed")]
    SessionEndSeed {},
    /// A plugin-owned event outside the core vocabulary: type string plus raw
    /// JSON data (the Rust face of upstream declaration merging).
    #[serde(untagged)]
    Extension {
        #[serde(rename = "type")]
        event_type: String,
        data: Value,
    },
}

/// Internal failure identity attached to a failed tool result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolErrorRef {
    pub name: String,
    pub code: String,
}

impl SessionEventData {
    /// The event's `type` tag string (matches the wire encoding).
    pub fn event_type(&self) -> &str {
        match self {
            SessionEventData::TurnStart { .. } => "turn/start",
            SessionEventData::TurnEnd { .. } => "turn/end",
            SessionEventData::StepStart { .. } => "step/start",
            SessionEventData::StepEnd { .. } => "step/end",
            SessionEventData::UserMessage(_) => "user/message",
            SessionEventData::AssistantChunk { .. } => "assistant/chunk",
            SessionEventData::AssistantMessage { .. } => "assistant/message",
            SessionEventData::ToolCall { .. } => "tool/call",
            SessionEventData::ToolResult { .. } => "tool/result",
            SessionEventData::TodoWrite { .. } => "todo/write",
            SessionEventData::RequestHeader { .. } => "request/header",
            SessionEventData::RequestContext(_) => "request/context",
            SessionEventData::SessionEndSeed {} => "session/end-seed",
            SessionEventData::Extension { event_type, .. } => event_type,
        }
    }
}

/// How a session event entered the ordered surface (upstream `SurfaceOp`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SurfaceOp {
    /// `"append"`: added to the tail — the normal path.
    Append(AppendTag),
    /// Replaces surface nodes from `start` through `end` (both inclusive,
    /// both must exist as surface nodes) with this node. Used by compaction.
    Replace {
        op: ReplaceTag,
        start: u64,
        end: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AppendTag {
    #[serde(rename = "append")]
    Append,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplaceTag {
    #[serde(rename = "replace")]
    Replace,
}

impl SurfaceOp {
    pub fn append() -> SurfaceOp {
        SurfaceOp::Append(AppendTag::Append)
    }

    pub fn replace(start: u64, end: u64) -> SurfaceOp {
        SurfaceOp::Replace {
            op: ReplaceTag::Replace,
            start,
            end,
        }
    }
}

/// Surface placement and cited source-event seqs for `Session::append`
/// (upstream `SurfaceIntent`). Required on message-producing events and
/// forbidden on log-only events.
#[derive(Debug, Clone, Default)]
pub struct SurfaceIntent {
    pub surface_op: Option<SurfaceOp>,
    /// Complete set of known source-event seqs. `assistant/message` may carry
    /// a present empty set for a known empty provider stream; other surface
    /// events require a non-empty set when present.
    pub source_event_seqs: Option<Vec<u64>>,
}

impl SurfaceIntent {
    pub fn append() -> SurfaceIntent {
        SurfaceIntent {
            surface_op: Some(SurfaceOp::append()),
            source_event_seqs: None,
        }
    }

    pub fn append_with_sources(sources: Vec<u64>) -> SurfaceIntent {
        SurfaceIntent {
            surface_op: Some(SurfaceOp::append()),
            source_event_seqs: Some(sources),
        }
    }
}

/// One immutable entry in the session log (upstream `SessionEvent`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionEvent {
    /// Monotonic sequence number within the session (`seq = log length`).
    pub seq: u64,
    /// Unix epoch milliseconds.
    pub time: u64,
    /// The typed payload, tagged by `type` on the wire.
    #[serde(flatten)]
    pub data: SessionEventData,
    /// Seq numbers of earlier events this event cites as sources.
    #[serde(rename = "sourceEventSeqs", skip_serializing_if = "Option::is_none")]
    pub source_event_seqs: Option<Vec<u64>>,
    /// How this event entered the surface; absent for non-surface events.
    #[serde(rename = "surfaceOp", skip_serializing_if = "Option::is_none")]
    pub surface_op: Option<SurfaceOp>,
    /// Marks an event a reader may safely skip when it does not recognize the
    /// type. Absent means required: a reader meeting an unrecognized type
    /// without this marker MUST refuse to reconstruct the session.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignorable: Option<bool>,
}

impl SessionEvent {
    /// The event's `type` tag string.
    pub fn event_type(&self) -> &str {
        self.data.event_type()
    }
}
