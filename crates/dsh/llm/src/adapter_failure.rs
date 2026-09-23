//! Normalization for failures escaping a final LLM adapter boundary, ported
//! from `packages/llm/llm/src/adapter-failure.ts`.
//!
//! Divergence: upstream defends against hostile JS accessors on foreign error
//! objects; Rust errors are plain data, so normalization reduces to downcast
//! (trust an `LlmError`'s carried facts) or render (anything else becomes an
//! `UNKNOWN`-coded fact set).

use crate::error::LlmError;
use crate::types::LlmFailure;

/// Detach serializable provider facts from a failure raised during adapter
/// dispatch or iteration, suitable for a terminal finish chunk.
pub fn normalize_llm_failure(error: &anyhow::Error) -> LlmFailure {
    if let Some(llm_error) = error.downcast_ref::<LlmError>() {
        return llm_error.failure.clone();
    }
    if let Some(harness) = error.downcast_ref::<crate::error::HarnessError>() {
        return LlmFailure::new(harness.message.clone(), harness.code.clone());
    }
    let message = error.to_string();
    LlmFailure::new(
        if message.is_empty() {
            "LLM adapter failed".to_string()
        } else {
            message
        },
        "UNKNOWN",
    )
}
