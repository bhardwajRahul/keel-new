//! Message value types, identity, and construction helpers, ported from
//! `packages/llm/llm/src/message.ts`.
//!
//! Divergence: upstream deep-freezes messages before publication; Rust owned
//! values are immutable by construction wherever they are shared, so the
//! freeze/clone machinery has no counterpart.

use crate::brand::{CallId, MessageId};
use crate::types::ContentBlock;
use serde::{Deserialize, Serialize};

/// Provider/model identity and adapter-private replay data for an assistant
/// message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantProvenance {
    /// Provider route that produced the message.
    pub provider: String,
    /// Provider model id that produced the message.
    pub model: String,
    /// Lossless-JSON adapter state needed to replay the provider response.
    /// `LlmRuntime` exposes it to a target adapter only when that adapter
    /// instance currently owns both this historical provider and the target.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replay_state: Option<serde_json::Value>,
}

/// The kind of information in producer-supplied context, declared by the
/// producer beside its provenance (upstream `ContextForm` + `ContextFormed`).
///
/// The vocabulary is SEMANTIC, never visual: a value states what kind of
/// thing the content is, and a consumer decides what that looks like.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "form", rename_all = "kebab-case")]
pub enum ContextForm {
    /// Instructions read out of workspace files the model is expected to
    /// follow.
    Instructions,
    /// A catalog of items available in this session, republished as it
    /// changes.
    Catalog,
    /// Current state, where a later snapshot from the same producer supersedes
    /// an earlier one.
    Snapshot {
        /// The named contributions this snapshot assembled, in order.
        sections: Vec<ContextSnapshotSection>,
    },
    /// A one-off account of something that just happened; supersedes nothing.
    Notice {
        /// One-line account of what happened, shown without expanding the row.
        summary: String,
    },
    /// A message another agent addressed to this one.
    Relay,
    /// Material lifted out of another session's log, possibly reduced on the
    /// way in.
    Recall,
}

/// One named contribution to a `Snapshot`-form context, in assembly order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextSnapshotSection {
    /// The contributing subsystem's name.
    pub name: String,
    /// That contribution's model-facing text, exactly as assembled.
    pub text: String,
}

/// Bound for a `Notice` summary: the account rides a collapsed transcript row
/// and is committed to the durable log.
pub const CONTEXT_SUMMARY_MAX_CHARS: usize = 120;

/// Bound one `Notice` summary to [`CONTEXT_SUMMARY_MAX_CHARS`] characters,
/// ellipsizing when it exceeds the bound.
pub fn bound_context_summary(summary: &str) -> String {
    let count = summary.chars().count();
    if count <= CONTEXT_SUMMARY_MAX_CHARS {
        summary.to_string()
    } else {
        let mut bounded: String = summary
            .chars()
            .take(CONTEXT_SUMMARY_MAX_CHARS - 1)
            .collect();
        bounded.push('…');
        bounded
    }
}

/// Where a message (or injected content) came from (upstream
/// `MessageSourceMap`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum MessageSource {
    User,
    Plugin {
        plugin: String,
        /// Producer-declared context form; `None` is the documented default
        /// (opaque content).
        #[serde(flatten, skip_serializing_if = "Option::is_none")]
        form: Option<ContextForm>,
    },
    Model(AssistantProvenance),
    Tool {
        call_id: CallId,
    },
}

/// Provider-neutral conversation role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

/// One message representation shared by delivery, durable history, and model
/// requests.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Stable identity preserved across every representation boundary.
    pub id: MessageId,
    /// Provider-neutral conversation role.
    pub role: Role,
    /// Exact model-facing blocks.
    pub content: Vec<ContentBlock>,
    /// Required source fields supplied by the producer.
    pub source: MessageSource,
}

/// Create one identified message (upstream `createMessage`).
pub fn create_message(role: Role, content: Vec<ContentBlock>, source: MessageSource) -> Message {
    Message {
        id: MessageId::new(uuid::Uuid::new_v4().to_string()),
        role,
        content,
        source,
    }
}

/// Create one identified user-role message (upstream `createUserMessage`).
pub fn create_user_message(content: Vec<ContentBlock>, source: MessageSource) -> Message {
    create_message(Role::User, content, source)
}

/// Create one identified model-produced assistant message (upstream
/// `createAssistantMessage`).
pub fn create_assistant_message(
    content: Vec<ContentBlock>,
    source: AssistantProvenance,
) -> Message {
    create_message(Role::Assistant, content, MessageSource::Model(source))
}

/// Input whose acceptance creates one tool-result message.
pub struct ToolResultMessageInput {
    pub call_id: CallId,
    pub content: Vec<ContentBlock>,
    pub is_error: bool,
}

/// Create one identified tool-result message: a user-role message whose single
/// block retains call correlation (upstream `createToolResultMessage`).
pub fn create_tool_result_message(input: ToolResultMessageInput) -> Message {
    create_user_message(
        vec![ContentBlock::ToolResult {
            tool_call_id: input.call_id.clone(),
            content: input.content,
            is_error: Some(input.is_error),
        }],
        MessageSource::Tool {
            call_id: input.call_id,
        },
    )
}
