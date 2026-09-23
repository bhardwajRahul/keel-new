//! Provider-neutral message and streaming vocabulary, ported from
//! `packages/llm/llm/src/types.ts`. Adapters alone translate provider wire
//! messages into these types.
//!
//! Divergences:
//! - Upstream unions are merge-extensible via declaration merging; Rust enums
//!   are closed, so extension means adding a variant here (with adapter, UI,
//!   and compaction support landing together, as upstream requires anyway).
//! - `ImageBlock::attachment` is temporarily a raw JSON value until the
//!   dsh-attachment port's `ImageAttachmentRef` is wired in.
//! - `AbortSignal` is dsh-timeout's port of the Web signal.

use crate::brand::{CallId, ProviderRequestId, ReasoningEffortId, SessionId};
use crate::message::Message;
use serde::{Deserialize, Serialize};

/// Serializable provider or transport failure facts; policy decides whether
/// they are retryable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmFailure {
    /// Human-readable provider or transport failure.
    pub message: String,
    /// Stable provider-neutral machine-routing code.
    pub code: String,
    /// HTTP status returned by the provider, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    /// Provider-requested delay in milliseconds, when valid and available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_retry_after_ms: Option<f64>,
    /// Opaque provider-issued request identifier for diagnostics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<ProviderRequestId>,
}

impl LlmFailure {
    /// A failure carrying only a message and code.
    pub fn new(message: impl Into<String>, code: impl Into<String>) -> Self {
        LlmFailure {
            message: message.into(),
            code: code.into(),
            status: None,
            provider_retry_after_ms: None,
            request_id: None,
        }
    }
}

/// Exact model-facing content blocks (upstream `ContentBlockMap` union).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ContentBlock {
    /// Plain text visible to the end user.
    Text { text: String },
    /// Reasoning / thinking content, distinct from visible text.
    Reasoning { text: String },
    /// A durable raster image reference, valid in user or assistant content.
    Image { attachment: serde_json::Value },
    /// A tool invocation requested by the model.
    ToolCall {
        /// Provider-issued call id; correlates with the matching tool result.
        id: CallId,
        name: String,
        /// Raw JSON string as produced by the model.
        arguments: String,
    },
    /// The result of a tool invocation, sent back to the model.
    ToolResult {
        tool_call_id: CallId,
        content: Vec<ContentBlock>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

impl ContentBlock {
    /// The block `type` tag (upstream `ContentBlockType`).
    pub fn block_type(&self) -> BlockType {
        match self {
            ContentBlock::Text { .. } => BlockType::Text,
            ContentBlock::Reasoning { .. } => BlockType::Reasoning,
            ContentBlock::Image { .. } => BlockType::Image,
            ContentBlock::ToolCall { .. } => BlockType::ToolCall,
            ContentBlock::ToolResult { .. } => BlockType::ToolResult,
        }
    }
}

/// The block `type` tag vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BlockType {
    Text,
    Reasoning,
    Image,
    ToolCall,
    ToolResult,
}

/// True when typed model content contains an image block, walking nested
/// tool-result content — the one recursive image walk shared by every image
/// policy (upstream `contentHasImage`).
pub fn content_has_image(content: &[ContentBlock]) -> bool {
    content.iter().any(|block| match block {
        ContentBlock::Image { .. } => true,
        ContentBlock::ToolResult { content, .. } => content_has_image(content),
        _ => false,
    })
}

/// Why a model response stopped (upstream `FinishReasonMap`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum FinishReason {
    Stop,
    ToolCalls,
    MaxTokens,
    Aborted { failure: LlmFailure },
    Error { failure: LlmFailure },
}

/// Token accounting for one model call.
///
/// Counts are DISJOINT: `input_tokens` is uncached input only; cached input is
/// reported separately (billed input = sum of the three). Adapters whose
/// providers fold cache hits into a total prompt count subtract them out.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
}

/// Display metadata for one registered provider route.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmProviderInfo {
    /// Provider route key used by `GenerateOptions::provider`.
    pub id: String,
    /// Human-readable provider name for selectors and diagnostics.
    pub name: String,
}

/// Accepted request modality (upstream `ModelModality`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelModality {
    Text,
    Image,
}

/// One provider route an adapter plugin can activate through configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmConfigurableProvider {
    /// Provider route key this entry activates when configured.
    pub provider: String,
    /// Human-readable provider name for configuration surfaces.
    pub display_name: String,
    /// User-settings namespace whose section configures this provider.
    pub settings_ns: String,
    /// Path from that namespace's section root to this provider's profile
    /// object; empty when the whole section is the profile.
    pub settings_path: Vec<String>,
    /// Whether the owning adapter knows this route only because configuration
    /// declared it. Absent means the adapter draws no such distinction.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub declared: Option<bool>,
}

/// One interrogation of a provider endpoint that configuration has not stored
/// yet (a draft the user is still editing).
#[derive(Default)]
pub struct LlmModelDiscoveryRequest {
    /// Route the draft is editing, when it edits an existing one.
    pub provider: Option<String>,
    /// Endpoint to interrogate.
    pub base_url: Option<String>,
    /// Wire protocol the endpoint speaks, when the draft names one.
    pub api: Option<String>,
    /// Credential for this interrogation alone; the harness never stores it.
    pub api_key: Option<String>,
    /// Caller cancellation; implementations must settle promptly after abort.
    pub signal: Option<dsh_timeout::AbortSignal>,
}

/// One model an endpoint reports about itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmDiscoveredModel {
    /// Model id the endpoint accepts.
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
}

/// One adapter-discovered model; catalog membership is advisory, not request
/// validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmModelInfo {
    /// Provider route that owns this model entry.
    pub provider: String,
    /// Model id passed to `GenerateOptions::model`.
    pub id: String,
    /// Human-readable model name for selectors.
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Accepted request modalities; `None` means unknown, while an explicit
    /// empty list is negative capability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_modalities: Option<Vec<ModelModality>>,
}

/// Provider-owned context capacity for one exact provider/model route.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmModelContext {
    /// Maximum combined request and response context in tokens.
    pub context_window: u64,
}

/// Display metadata for one adapter-owned reasoning effort.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmReasoningEffortInfo {
    /// Opaque stable value accepted by `GenerateOptions::reasoning_effort`.
    pub id: ReasoningEffortId,
    /// Human-readable effort name for selectors and diagnostics.
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Selectable reasoning efforts for one exact provider/model route.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmModelReasoningInfo {
    /// Supported efforts in adapter-preferred display order.
    pub efforts: Vec<LlmReasoningEffortInfo>,
    /// Adapter-configured default materialized into requests when callers omit
    /// an effort. Absence preserves the provider's own default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_effort: Option<ReasoningEffortId>,
}

/// Exact-route model metadata resolved by its owning adapter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmResolvedModelInfo {
    pub provider: String,
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_modalities: Option<Vec<ModelModality>>,
    /// Provider-owned context capacity when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<LlmModelContext>,
    /// Adapter-configured per-request output cap materialized when callers
    /// omit one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_max_tokens: Option<u64>,
    /// Adapter-owned selectable reasoning levels when exposed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<LlmModelReasoningInfo>,
}

impl LlmResolvedModelInfo {
    /// The minimal answer: identity only (upstream `resolveModel` default).
    pub fn bare(provider: impl Into<String>, model: impl Into<String>) -> Self {
        let id: String = model.into();
        LlmResolvedModelInfo {
            provider: provider.into(),
            name: id.clone(),
            id,
            description: None,
            input_modalities: None,
            context: None,
            default_max_tokens: None,
            reasoning: None,
        }
    }
}

/// Raw streaming protocol emitted by adapters (upstream `StreamChunk`).
///
/// Block indexes correlate interleaved deltas, and `BlockEnd` carries the
/// assembled block. Adapters emit usage before the terminal finish and nothing
/// afterward; tool arguments remain raw JSON strings. An adapter may fail, but
/// `LlmRuntime::stream` normalizes that failure to a terminal `Error` or
/// `Aborted` finish before exposing it to consumers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum StreamChunk {
    BlockStart {
        index: u64,
        block_type: BlockType,
    },
    TextDelta {
        index: u64,
        text: String,
    },
    ReasoningDelta {
        index: u64,
        text: String,
    },
    ToolCallDelta {
        index: u64,
        id: CallId,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        arguments_delta: String,
    },
    BlockEnd {
        index: u64,
        block: ContentBlock,
    },
    Usage {
        usage: TokenUsage,
    },
    Finish {
        reason: FinishReason,
        /// Adapter-private lossless-JSON state for replaying a successful
        /// response.
        #[serde(skip_serializing_if = "Option::is_none")]
        replay_state: Option<serde_json::Value>,
    },
}

/// Whether a stream chunk carries visible model output — the first-token
/// boundary shared by step timing and stats (upstream `isTokenDelta`). Empty
/// deltas (heartbeats, empty tool-call frames) do not count.
pub fn is_token_delta(chunk: &StreamChunk) -> bool {
    match chunk {
        StreamChunk::TextDelta { text, .. } | StreamChunk::ReasoningDelta { text, .. } => {
            !text.is_empty()
        }
        StreamChunk::ToolCallDelta {
            arguments_delta,
            name,
            ..
        } => !arguments_delta.is_empty() || name.is_some(),
        _ => false,
    }
}

/// JSON-schema description of a tool, as sent to the model. Declared here (not
/// in dsh-tools) because it is part of `GenerateOptions`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    /// JSON Schema object for the arguments.
    pub parameters: serde_json::Map<String, serde_json::Value>,
}

/// Provider-neutral classification for an auxiliary model call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CallPurpose {
    Compaction,
    SessionTitle,
}

/// A single model request, fully assembled (upstream `GenerateOptions`).
#[derive(Clone, Default)]
pub struct GenerateOptions {
    /// Registered provider route selecting the adapter instance.
    pub provider: String,
    pub model: String,
    /// Adapter-owned reasoning effort selected for this exact model.
    pub reasoning_effort: Option<ReasoningEffortId>,
    /// Ordered conversation messages, exactly as the provider sees them.
    pub messages: Vec<Message>,
    /// System prompt text (adapters map to the provider's system slot).
    pub system: Option<String>,
    /// Tool schemas (adapters map to the provider's `tools` field).
    pub tools: Option<Vec<ToolSchema>>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    /// Stop sequences: generation halts on any of these strings.
    pub stop: Option<Vec<String>>,
    /// Caller cancellation channel.
    pub signal: Option<dsh_timeout::AbortSignal>,
    /// Session identity stamped by the loop for request routing.
    pub session_id: Option<SessionId>,
    /// Auxiliary-call classification; ordinary conversation requests leave it
    /// unset.
    pub purpose: Option<CallPurpose>,
    /// Marks a request assembled by dsh-agent-loop (upstream tracks this with
    /// a process-local WeakSet of request identities; a field is the Rust
    /// equivalent). Loop-built requests are pure functions of the session log:
    /// listeners read them, never rewrite them.
    pub agent_loop: bool,
}

/// Mark one request as assembled by dsh-agent-loop (upstream
/// `markAgentLoopRequest`).
pub fn mark_agent_loop_request(mut request: GenerateOptions) -> GenerateOptions {
    request.agent_loop = true;
    request
}

/// Whether the request was assembled by dsh-agent-loop (upstream
/// `isAgentLoopRequest`).
pub fn is_agent_loop_request(request: &GenerateOptions) -> bool {
    request.agent_loop
}
