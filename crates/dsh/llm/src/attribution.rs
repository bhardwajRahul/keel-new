//! App-attribution identity every provider request sends as `User-Agent`,
//! ported from `packages/llm/llm/src/attribution.ts`. Centralized so adapters
//! cannot drift.

/// Static public application identity sent to LLM providers.
///
/// Every field is a public product fact, safe on every request: no secrets,
/// local paths, session ids, prompt text, or per-user identifiers belong
/// here, and nothing per-request may influence the values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppIdentity {
    /// `User-Agent` product token (lowercase, hyphenated).
    pub product: &'static str,
    /// Product version; sourced from crate metadata, never hand-copied.
    pub version: &'static str,
    /// Repository home URL of the app, used as the `User-Agent` comment.
    pub url: &'static str,
}

/// The harness's own identity: the default every adapter sends. Deployments
/// needing a white-label identity pass their own [`AppIdentity`]; omission
/// falls back to this default — nothing can suppress attribution entirely.
pub const APP_IDENTITY: AppIdentity = AppIdentity {
    product: "deepseek-harness-rs",
    version: env!("CARGO_PKG_VERSION"),
    url: "https://github.com/deepseek-ai/deepseek-harness",
};

/// The standard `User-Agent` value: `product/version (+url)` (RFC 9110
/// §10.1.5 product + comment syntax).
pub fn user_agent(identity: &AppIdentity) -> String {
    format!(
        "{}/{} (+{})",
        identity.product, identity.version, identity.url
    )
}

/// Build the attribution headers an adapter must send on every provider
/// request (currently just lowercase `user-agent`).
pub fn attribution_headers(identity: Option<&AppIdentity>) -> Vec<(String, String)> {
    vec![(
        "user-agent".to_string(),
        user_agent(identity.unwrap_or(&APP_IDENTITY)),
    )]
}
