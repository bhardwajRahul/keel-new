//! The one definition of a well-formed provider API key, ported from
//! `packages/llm/llm/src/api-key.ts` (plus `assertUsableApiKey` from
//! `index.ts`). Shared by every adapter that puts a key in an HTTP header.

use crate::error::{INVALID_CREDENTIAL_CODE, LlmError};

/// Why a supplied API key cannot be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiKeyRejection {
    Empty,
    IllegalCharacters,
}

/// The verdict on one supplied API key.
pub type ApiKeyCheck = Result<String, ApiKeyRejection>;

/// Judge one *supplied* API key, trimming surrounding whitespace first.
///
/// Trimming is silent because a padded key has one unambiguous reading; every
/// other defect is reported. The legal alphabet is printable ASCII with space
/// excluded — a transport invariant (an HTTP header cannot carry anything
/// else), not one provider's policy.
pub fn normalize_api_key(raw: &str) -> ApiKeyCheck {
    let value = raw.trim();
    if value.is_empty() {
        return Err(ApiKeyRejection::Empty);
    }
    if !value.bytes().all(|byte| (0x21..=0x7E).contains(&byte)) {
        return Err(ApiKeyRejection::IllegalCharacters);
    }
    Ok(value.to_string())
}

/// Accept one supplied credential, or refuse it as unusable with a diagnosis
/// that names where to fix it. The key itself never enters the message —
/// echoing any part of a secret into a log or UI is the failure this avoids.
pub fn assert_usable_api_key(raw: &str, pkg: &str, reference: &str) -> Result<String, LlmError> {
    match normalize_api_key(raw) {
        Ok(value) => Ok(value),
        Err(ApiKeyRejection::Empty) => Err(LlmError::new(
            format!(
                "{pkg}: the API key resolved from {reference} is blank; set {reference} to the raw key \
                 (the web Models page writes it) or export it in the launching environment"
            ),
            INVALID_CREDENTIAL_CODE,
        )),
        Err(ApiKeyRejection::IllegalCharacters) => Err(LlmError::new(
            format!(
                "{pkg}: the API key resolved from {reference} contains characters no HTTP header can carry; \
                 set {reference} to the raw key alone (the web Models page writes it)"
            ),
            INVALID_CREDENTIAL_CODE,
        )),
    }
}
