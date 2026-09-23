//! Rust port of `@deepseek-ai/dsh-llm` (`packages/llm/llm`): the
//! provider-neutral message and streaming vocabulary, the `llm` adapter
//! registry service with its waterfall-interceptable streaming call API, and
//! the canonical chunk-to-message assembler.
//!
//! Crate-level divergences (each module documents its own):
//! - Closed Rust enums replace merge-extensible TS unions.
//! - `Result` returns replace synchronous throws at registration APIs.
//! - The freeze/clone immutability machinery has no counterpart — Rust
//!   ownership provides it.

mod adapter_failure;
mod api_key;
mod assembler;
mod attribution;
mod brand;
mod call_config;
mod error;
mod message;
mod retry;
mod runtime;
mod types;

pub use adapter_failure::normalize_llm_failure;
pub use api_key::{ApiKeyCheck, ApiKeyRejection, assert_usable_api_key, normalize_api_key};
pub use assembler::BlockAssembler;
pub use attribution::{APP_IDENTITY, AppIdentity, attribution_headers, user_agent};
pub use brand::{
    CallId, CallIdMark, MessageId, MessageIdMark, ProviderRequestId, ProviderRequestIdMark,
    ReasoningEffortId, ReasoningEffortIdMark, SessionId, SessionIdMark,
};
pub use call_config::{
    LlmCallConfig, LlmCallConfigAdapterDefaults, call_config_equals, options_match_config,
};
pub use error::{
    CONTEXT_WINDOW_EXCEEDED_CODE, EMPTY_RESPONSE_CODE, HarnessError, INVALID_CREDENTIAL_CODE,
    LlmError, QUOTA_EXCEEDED_CODE, error_chain, is_context_window_exceeded_error,
    is_quota_exceeded_error,
};
pub use message::{
    AssistantProvenance, CONTEXT_SUMMARY_MAX_CHARS, ContextForm, ContextSnapshotSection, Message,
    MessageSource, Role, ToolResultMessageInput, bound_context_summary, create_assistant_message,
    create_message, create_tool_result_message, create_user_message,
};
pub use retry::{
    BackoffConfig, ResolvedRetryBackoff, ResolvedRetryPolicy, RetryPolicyConfig,
    default_retryable_codes, resolve_retry_policy,
};
pub use runtime::{
    AdapterRegistrationHandle, AdapterStream, AdaptersUpdated, ChunkStream,
    DirectoryRegistrationHandle, LlmAdapter, LlmRuntime, LlmStream, PreparedLlmCall,
};
pub use types::{
    BlockType, CallPurpose, ContentBlock, FinishReason, GenerateOptions, LlmConfigurableProvider,
    LlmDiscoveredModel, LlmFailure, LlmModelContext, LlmModelDiscoveryRequest, LlmModelInfo,
    LlmModelReasoningInfo, LlmProviderInfo, LlmReasoningEffortInfo, LlmResolvedModelInfo,
    ModelModality, StreamChunk, TokenUsage, ToolSchema, content_has_image, is_agent_loop_request,
    is_token_delta, mark_agent_loop_request,
};
