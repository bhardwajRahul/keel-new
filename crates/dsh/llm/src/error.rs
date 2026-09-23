//! Harness error base with a stable machine-routable code, ported from
//! `packages/llm/llm/src/error.ts` (plus `LlmError` from `index.ts`).
//!
//! Route on [`HarnessError::code`], never by parsing the message.

use crate::types::LlmFailure;
use regex::Regex;
use std::sync::LazyLock;

/// Base error for the harness: a human-readable message plus a stable
/// machine-routable failure class (e.g. `RATE_LIMIT`, `NO_ADAPTER`,
/// `INVARIANT`) and an optional rendered cause chain.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct HarnessError {
    pub message: String,
    /// Stable machine-routable failure class; route on this.
    pub code: String,
    /// Rendered cause chain, outermost first (upstream keeps live `cause`
    /// objects; the port renders them at construction).
    pub cause: Option<String>,
}

impl HarnessError {
    pub fn new(message: impl Into<String>, code: impl Into<String>) -> Self {
        HarnessError {
            message: message.into(),
            code: code.into(),
            cause: None,
        }
    }

    pub fn with_cause(mut self, cause: impl Into<String>) -> Self {
        self.cause = Some(cause.into());
        self
    }
}

/// Typed error for LLM-related failures: a [`HarnessError`] carrying
/// serializable provider facts (upstream `LlmError`).
#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct LlmError {
    pub message: String,
    pub code: String,
    /// Serializable facts retained beside this live error.
    pub failure: LlmFailure,
}

impl LlmError {
    /// Build an error whose facts carry only message and code. Panics on an
    /// empty message or code — those are caller bugs upstream too.
    pub fn new(message: impl Into<String>, code: impl Into<String>) -> Self {
        let message: String = message.into();
        let code: String = code.into();
        assert!(
            !message.is_empty(),
            "LlmError message must be a non-empty string"
        );
        assert!(!code.is_empty(), "LlmError code must be a non-empty string");
        let failure = LlmFailure::new(message.clone(), code.clone());
        LlmError {
            message,
            code,
            failure,
        }
    }

    /// Attach validated provider facts. Panics on out-of-range values,
    /// mirroring upstream constructor validation.
    pub fn with_facts(
        mut self,
        status: Option<u16>,
        provider_retry_after_ms: Option<f64>,
        request_id: Option<crate::brand::ProviderRequestId>,
    ) -> Self {
        if let Some(status) = status {
            assert!(
                (100..=599).contains(&status),
                "LlmError status must be an integer from 100 through 599"
            );
        }
        if let Some(delay) = provider_retry_after_ms {
            assert!(
                delay.is_finite() && delay > 0.0,
                "LlmError providerRetryAfterMs must be a positive finite number"
            );
        }
        if let Some(request_id) = &request_id {
            assert!(
                !request_id.as_str().is_empty(),
                "LlmError requestId must be a non-empty string"
            );
        }
        self.failure.status = status;
        self.failure.provider_retry_after_ms = provider_retry_after_ms;
        self.failure.request_id = request_id;
        self
    }
}

/// Canonical code for a model request rejected because its context window was
/// exceeded.
pub const CONTEXT_WINDOW_EXCEEDED_CODE: &str = "CONTEXT_WINDOW_EXCEEDED";

/// Canonical code for an exhausted account quota or balance.
pub const QUOTA_EXCEEDED_CODE: &str = "QUOTA";

/// Canonical code for a response that completed normally but carried no
/// content blocks at all. The attempt produced nothing durable, so retry
/// policy treats it as safe to repeat.
pub const EMPTY_RESPONSE_CODE: &str = "EMPTY_RESPONSE";

/// Canonical code for a credential that was supplied but cannot be used —
/// malformed rather than absent. Deliberately outside the default retryable
/// set: a malformed credential fails identically on every attempt.
pub const INVALID_CREDENTIAL_CODE: &str = "INVALID_CREDENTIAL";

static STRUCTURED_CONTEXT_OVERFLOW: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(?:^|[^a-z0-9])context[\s_-](?:length|window)[\s_-](?:exceed(?:ed|s)?|overflow(?:ed)?|limit[\s_-]exceeded)(?:$|[^a-z0-9])",
    )
    .expect("static pattern")
});

static MAX_CONTEXT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(?:maximum|max)(?:\s+(?:allowed|supported))?\s+context\s+(?:length|window)\b",
    )
    .expect("static pattern")
});

static TOO_LARGE_FOR_CONTEXT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(?:request|prompt|input|messages?)\s+(?:is\s+|are\s+)?too\s+(?:large|long)\s+for\s+(?:(?:this|the)\s+)?(?:model(?:'s)?\s+)?context(?:\s+window)?\b",
    )
    .expect("static pattern")
});

static TOO_LONG_FOR_MODEL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(?:input|prompt|request)\s+(?:is\s+)?too\s+(?:long|large)\s+for\s+(?:this|the)\s+model\b",
    )
    .expect("static pattern")
});

static EXCEEDS_MODEL_CONTEXT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(?:input|prompt|request|messages?)\b.{0,40}\b(?:exceed(?:s|ed)?|overflows?|is\s+larger\s+than)\b.{0,40}\b(?:the\s+)?(?:model(?:'s)?\s+)?context(?:\s+(?:length|window))?\b",
    )
    .expect("static pattern")
});

/// Recognize the context-overflow wording used by OpenAI-compatible providers
/// and library adapters. Adapters pass all available provider code, type, and
/// message text joined into one string, so thrown and in-band delivery styles
/// share one classifier.
pub fn is_context_window_exceeded_error(detail: &str) -> bool {
    STRUCTURED_CONTEXT_OVERFLOW.is_match(detail)
        || MAX_CONTEXT.is_match(detail)
        || TOO_LARGE_FOR_CONTEXT.is_match(detail)
        || TOO_LONG_FOR_MODEL.is_match(detail)
        || EXCEEDS_MODEL_CONTEXT.is_match(detail)
}

static INSUFFICIENT_QUOTA: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\binsufficient[\s_-]+(?:quota|balance|credits?)\b").expect("static pattern")
});
static QUOTA_EXHAUSTED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:quota|usage[\s_-]+limit)[\s_-]+(?:exceeded|exhausted|reached)\b")
        .expect("static pattern")
});
static EXCEEDED_QUOTA: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\bexceed(?:ed|s)?[\s_-]+(?:(?:your|the)[\s_-]+)?(?:current[\s_-]+)?quota\b")
        .expect("static pattern")
});
static BALANCE_EXHAUSTED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:balance|credits?)[\s_-]+(?:exhausted|depleted)\b")
        .expect("static pattern")
});
static OUT_OF_CREDITS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\bout[\s_-]+of[\s_-]+(?:credits?|budget)\b").expect("static pattern")
});

/// Recognize provider wording that identifies an exhausted account quota
/// rather than a transient request-rate limit.
pub fn is_quota_exceeded_error(detail: &str) -> bool {
    INSUFFICIENT_QUOTA.is_match(detail)
        || QUOTA_EXHAUSTED.is_match(detail)
        || EXCEEDED_QUOTA.is_match(detail)
        || BALANCE_EXHAUSTED.is_match(detail)
        || OUT_OF_CREDITS.is_match(detail)
}

/// Render an error with its full source chain, outermost message first, each
/// cause appended with `: ` (skipped when it repeats the wrapper verbatim) —
/// upstream `errorChain`, for diagnostic surfaces only.
pub fn error_chain(error: &anyhow::Error) -> String {
    let mut rendered = String::new();
    let mut previous: Option<String> = None;
    for cause in error.chain() {
        let message = cause.to_string();
        if previous.as_deref() == Some(message.as_str()) {
            continue;
        }
        if !rendered.is_empty() {
            rendered.push_str(": ");
        }
        rendered.push_str(&message);
        previous = Some(message);
    }
    rendered
}
