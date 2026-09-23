//! Provider-owned request-retry policy configuration and resolution, ported
//! from `packages/llm/llm/src/retry-policy.ts`. Adapters expose one resolved
//! policy per registered provider route; the retry plugin executes it on the
//! agent's failed-step extension point.

use crate::error::EMPTY_RESPONSE_CODE;
use dsh_timeout::MAX_TIMER_DELAY_MS;
use serde::{Deserialize, Serialize};

const DEFAULT_MAX_RETRIES: u64 = 2;
const DEFAULT_INITIAL_DELAY_MS: f64 = 500.0;
const DEFAULT_MAX_DELAY_MS: f64 = 10_000.0;
const DEFAULT_JITTER_RATIO: f64 = 0.1;

/// Stable failure codes retried by default under `normal` mode.
pub fn default_retryable_codes() -> Vec<String> {
    [
        EMPTY_RESPONSE_CODE,
        "RATE_LIMIT",
        "SERVER",
        "TIMEOUT",
        "TRANSPORT",
    ]
    .iter()
    .map(|code| code.to_string())
    .collect()
}

/// Bounded exponential backoff with symmetric jitter around each local delay.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BackoffConfig {
    /// Initial local exponential-backoff delay in milliseconds (default 500).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initial_delay_ms: Option<f64>,
    /// Maximum locally scheduled or accepted provider delay in milliseconds
    /// (default 10000).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_delay_ms: Option<f64>,
    /// Symmetric random multiplier range around one (default 0.1).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jitter_ratio: Option<f64>,
}

/// Provider-owned model-request retry policy configuration (upstream
/// `RetryPolicyConfig`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "camelCase", deny_unknown_fields)]
pub enum RetryPolicyConfig {
    /// Retry only configured transient failure codes.
    #[serde(rename_all = "camelCase")]
    Normal {
        /// Maximum eligible retries after the first request (default 2).
        #[serde(skip_serializing_if = "Option::is_none")]
        max_retries: Option<u64>,
        /// Stable failure codes eligible for this policy.
        #[serde(skip_serializing_if = "Option::is_none")]
        retryable_codes: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        backoff: Option<BackoffConfig>,
    },
    /// Retry every model-request failure until success, cancellation, or
    /// disposal.
    #[serde(rename_all = "camelCase")]
    Always {
        #[serde(skip_serializing_if = "Option::is_none")]
        backoff: Option<BackoffConfig>,
    },
}

/// Fully resolved backoff shared by both retry modes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedRetryBackoff {
    pub initial_delay_ms: f64,
    pub max_delay_ms: f64,
    pub jitter_ratio: f64,
}

/// Immutable provider policy captured when its adapter route is registered.
#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedRetryPolicy {
    Normal {
        max_retries: u64,
        retryable_codes: Vec<String>,
        backoff: ResolvedRetryBackoff,
    },
    Always {
        backoff: ResolvedRetryBackoff,
    },
}

impl ResolvedRetryPolicy {
    pub fn backoff(&self) -> &ResolvedRetryBackoff {
        match self {
            ResolvedRetryPolicy::Normal { backoff, .. } => backoff,
            ResolvedRetryPolicy::Always { backoff } => backoff,
        }
    }
}

fn resolve_backoff(
    config: Option<&BackoffConfig>,
    path: &str,
) -> Result<ResolvedRetryBackoff, String> {
    let initial_delay_ms = config
        .and_then(|c| c.initial_delay_ms)
        .unwrap_or(DEFAULT_INITIAL_DELAY_MS);
    let max_delay_ms = config
        .and_then(|c| c.max_delay_ms)
        .unwrap_or(DEFAULT_MAX_DELAY_MS);
    let jitter_ratio = config
        .and_then(|c| c.jitter_ratio)
        .unwrap_or(DEFAULT_JITTER_RATIO);

    if !initial_delay_ms.is_finite()
        || initial_delay_ms <= 0.0
        || initial_delay_ms > MAX_TIMER_DELAY_MS
    {
        return Err(format!(
            "{path}.initialDelayMs must be a positive finite number no greater than {MAX_TIMER_DELAY_MS}"
        ));
    }
    if !max_delay_ms.is_finite() || max_delay_ms <= 0.0 || max_delay_ms > MAX_TIMER_DELAY_MS {
        return Err(format!(
            "{path}.maxDelayMs must be a positive finite number no greater than {MAX_TIMER_DELAY_MS}"
        ));
    }
    if initial_delay_ms > max_delay_ms {
        return Err(format!(
            "{path}.initialDelayMs must be less than or equal to maxDelayMs"
        ));
    }
    if !jitter_ratio.is_finite() || !(0.0..=1.0).contains(&jitter_ratio) {
        return Err(format!("{path}.jitterRatio must be between 0 and 1"));
    }
    Ok(ResolvedRetryBackoff {
        initial_delay_ms,
        max_delay_ms,
        jitter_ratio,
    })
}

/// Validate, default, and detach one provider-owned retry policy (upstream
/// `resolveRetryPolicy`). Omission selects normal defaults. `path` names the
/// provider config that owns the value, for diagnostics.
pub fn resolve_retry_policy(
    config: Option<&RetryPolicyConfig>,
    path: &str,
) -> Result<ResolvedRetryPolicy, String> {
    match config {
        None => Ok(ResolvedRetryPolicy::Normal {
            max_retries: DEFAULT_MAX_RETRIES,
            retryable_codes: default_retryable_codes(),
            backoff: resolve_backoff(None, &format!("{path}.backoff"))?,
        }),
        Some(RetryPolicyConfig::Normal {
            max_retries,
            retryable_codes,
            backoff,
        }) => {
            let max_retries = max_retries.unwrap_or(DEFAULT_MAX_RETRIES);
            let retryable_codes = retryable_codes
                .clone()
                .unwrap_or_else(default_retryable_codes);
            if retryable_codes.is_empty() {
                return Err(format!("{path}.retryableCodes must not be empty"));
            }
            if retryable_codes.iter().any(|code| code.is_empty()) {
                return Err(format!(
                    "{path}.retryableCodes must contain only non-empty strings"
                ));
            }
            let unique: std::collections::HashSet<&String> = retryable_codes.iter().collect();
            if unique.len() != retryable_codes.len() {
                return Err(format!("{path}.retryableCodes must not contain duplicates"));
            }
            Ok(ResolvedRetryPolicy::Normal {
                max_retries,
                retryable_codes,
                backoff: resolve_backoff(backoff.as_ref(), &format!("{path}.backoff"))?,
            })
        }
        Some(RetryPolicyConfig::Always { backoff }) => Ok(ResolvedRetryPolicy::Always {
            backoff: resolve_backoff(backoff.as_ref(), &format!("{path}.backoff"))?,
        }),
    }
}
