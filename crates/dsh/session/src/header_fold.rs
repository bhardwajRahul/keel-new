//! Request-header reconstruction over `request/header` events, ported from
//! `packages/core/session/src/request-header.ts`. Anyone holding a log
//! reconstructs the header any request was built under by taking the latest
//! canonical snapshot; the loop uses the same equality to avoid logging
//! unchanged headers.

use crate::types::{EpochHeader, SessionEvent, SessionEventData};
use dsh_llm::call_config_equals;

/// Normalize a header to canonical form: an empty system prompt, empty tool
/// list, and all-absent adapter defaults become absent fields, matching how
/// requests are built.
pub fn canonical_header(header: &EpochHeader) -> EpochHeader {
    let adapter_defaults = header.adapter_defaults.filter(|defaults| {
        defaults.reasoning_effort == Some(true) || defaults.max_tokens == Some(true)
    });
    EpochHeader {
        config: header.config.clone(),
        adapter_defaults,
        system: header.system.clone().filter(|system| !system.is_empty()),
        tools: header.tools.clone().filter(|tools| !tools.is_empty()),
    }
}

/// Field-wise equality over canonical headers. Tool schemas compare in order
/// by their canonical JSON encoding.
pub fn header_equals(a: &EpochHeader, b: &EpochHeader) -> bool {
    if !call_config_equals(&a.config, &b.config)
        || a.adapter_defaults.and_then(|d| d.reasoning_effort)
            != b.adapter_defaults.and_then(|d| d.reasoning_effort)
        || a.adapter_defaults.and_then(|d| d.max_tokens)
            != b.adapter_defaults.and_then(|d| d.max_tokens)
        || a.system != b.system
    {
        return false;
    }
    let empty = Vec::new();
    let at = a.tools.as_ref().unwrap_or(&empty);
    let bt = b.tools.as_ref().unwrap_or(&empty);
    at.len() == bt.len()
        && at
            .iter()
            .zip(bt.iter())
            .all(|(a, b)| serde_json::to_string(a).ok() == serde_json::to_string(b).ok())
}

/// Fold the header events of a log (or any prefix) into the header in force
/// after the last snapshot; `from` continues a previous fold. Pure offline
/// reconstruction — the live session tracks the same fold incrementally.
pub fn fold_request_header(
    events: &[SessionEvent],
    from: Option<EpochHeader>,
) -> Option<EpochHeader> {
    let mut state = from;
    for event in events {
        if let SessionEventData::RequestHeader { header, .. } = &event.data {
            state = Some(canonical_header(header));
        }
    }
    state
}
