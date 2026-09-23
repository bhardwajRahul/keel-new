//! Conversation call configuration, ported from
//! `packages/llm/llm/src/call-config.ts`. Provider routing, model, reasoning
//! effort, and sampling values are request-header state that can affect cache
//! reuse; request waterfalls replace them and the loop logs changed snapshots
//! instead of allowing silent per-call drift.

use crate::brand::ReasoningEffortId;
use crate::types::GenerateOptions;
use serde::{Deserialize, Serialize};

/// Provider, model, reasoning effort, and sampling scalars of one
/// conversation's requests. Every field maps 1:1 onto the same-named
/// `GenerateOptions` field.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmCallConfig {
    pub provider: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffortId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

impl LlmCallConfig {
    /// The call-config projection of one full request.
    pub fn of(options: &GenerateOptions) -> LlmCallConfig {
        LlmCallConfig {
            provider: options.provider.clone(),
            model: options.model.clone(),
            reasoning_effort: options.reasoning_effort.clone(),
            temperature: options.temperature,
            max_tokens: options.max_tokens,
            stop: options.stop.clone(),
        }
    }

    /// Overwrite a request's call-config fields with this configuration.
    pub fn apply_to(&self, options: &mut GenerateOptions) {
        options.provider = self.provider.clone();
        options.model = self.model.clone();
        options.reasoning_effort = self.reasoning_effort.clone();
        options.temperature = self.temperature;
        options.max_tokens = self.max_tokens;
        options.stop = self.stop.clone();
    }
}

/// Effective config fields supplied by exact-model adapter resolution rather
/// than by the caller's request proposal.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmCallConfigAdapterDefaults {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<bool>,
}

/// Field-wise equality over [`LlmCallConfig`] — the comparison a caller runs
/// to decide whether a proposed configuration is a real change (worth a
/// logged header snapshot) or the held one restated (upstream
/// `callConfigEquals`).
pub fn call_config_equals(a: &LlmCallConfig, b: &LlmCallConfig) -> bool {
    a == b
}

/// Equality between a full request's call-config projection and a held
/// configuration.
pub fn options_match_config(options: &GenerateOptions, config: &LlmCallConfig) -> bool {
    &LlmCallConfig::of(options) == config
}
