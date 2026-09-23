//! Branded ids owned by dsh-llm, ported from `packages/llm/llm/src/brand.ts`:
//! tool-call correlation and provider request diagnostics.
//!
//! Divergence: upstream brands are structural (`Branded<'SessionId'>` written
//! in any package is the same type). Rust markers are nominal, so `SessionId`
//! is declared here — the lowest crate in the dependency order that names it —
//! and `dsh-session` re-exports it as the owning package.

use dsh_brand::Branded;

/// Stable identity carried by one message across inbox, log, and
/// model-request boundaries.
pub enum MessageIdMark {}
pub type MessageId = Branded<MessageIdMark>;

/// Correlates a model-issued tool call with its result. Provider-issued for
/// real adapters; synthesized by mocks and assembler fallbacks.
pub enum CallIdMark {}
pub type CallId = Branded<CallIdMark>;

/// Provider-issued request identifier retained for diagnostics.
pub enum ProviderRequestIdMark {}
pub type ProviderRequestId = Branded<ProviderRequestIdMark>;

/// Adapter-owned identifier for one model's selectable reasoning effort.
pub enum ReasoningEffortIdMark {}
pub type ReasoningEffortId = Branded<ReasoningEffortIdMark>;

/// Session identity stamped by the loop for request routing (owned by
/// dsh-session upstream; declared here for dependency order, re-exported
/// there).
pub enum SessionIdMark {}
pub type SessionId = Branded<SessionIdMark>;
