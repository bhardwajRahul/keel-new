//! Read-only access to persisted decision receipts for evaluation.
//!
//! Decision receipts live inside each chat's session doc (see
//! `DocHost::write_decision_event`). This module opens the snapshot database
//! READ-ONLY, so it is safe while the app or the daemon holds the engine lock.
//! It never writes, migrates, or creates the database.
//!
//! Two consumers use it: `keel decisions export` writes one replay case per
//! receipt (improvement-loop step 1), and `keel decisions report` prints
//! aggregate counts for a baseline comparison.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use keel_doc::{MessagePart, SessionDoc};
use keel_proto::{DecisionEvent, DecisionResult};
use rusqlite::{Connection, OpenFlags};
use serde::Serialize;

use crate::EngineError;

/// One persisted receipt with the chat that owns it.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatDecision {
    pub chat_id: String,
    pub event: DecisionEvent,
}

/// A replay case: the recorded receipt plus an empty slot for the expected
/// candidate. A human fills `expectedCandidateId` (or leaves it null for an
/// expected abstention) before the case goes into a comparison run.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReplayCase<'a> {
    pub chat_id: &'a str,
    pub decision: &'a DecisionEvent,
    pub expected_candidate_id: Option<String>,
}

impl ChatDecision {
    pub fn replay_case(&self) -> ReplayCase<'_> {
        ReplayCase {
            chat_id: &self.chat_id,
            decision: &self.event,
            expected_candidate_id: None,
        }
    }
}

/// Every decision receipt in `{store_root}/docs.sqlite3`, oldest first.
/// A missing database is an empty result, not an error.
pub fn read_decisions(store_root: &Path) -> Result<Vec<ChatDecision>, EngineError> {
    let db = store_root.join("docs.sqlite3");
    if !db.exists() {
        return Ok(Vec::new());
    }
    let conn = Connection::open_with_flags(
        &db,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(sqlite_error)?;
    conn.busy_timeout(std::time::Duration::from_secs(5))
        .map_err(sqlite_error)?;
    let mut stmt = conn
        .prepare("SELECT bytes FROM snapshots")
        .map_err(sqlite_error)?;
    let rows = stmt
        .query_map([], |row| row.get::<_, Vec<u8>>(0))
        .map_err(sqlite_error)?;

    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for bytes in rows {
        let bytes = bytes.map_err(sqlite_error)?;
        let raw = loro::LoroDoc::new();
        // Registry and workspace docs share the table; a row that does not
        // import or has no chat id is not a session doc.
        if raw.import(&bytes).is_err() {
            continue;
        }
        let doc = SessionDoc::from_doc(raw);
        let Some(chat_id) = doc.chat_id() else {
            continue;
        };
        let Ok(entries) = doc.read_entries() else {
            continue;
        };
        for entry in entries {
            for part in entry.parts {
                if let MessagePart::Decision { event, .. } = part
                    // `.pre-chat2` rollback copies repeat the same receipts.
                    && seen.insert((chat_id.clone(), event.id.clone()))
                {
                    out.push(ChatDecision {
                        chat_id: chat_id.clone(),
                        event,
                    });
                }
            }
        }
    }
    out.sort_by(|a, b| {
        (a.event.created_at, &a.chat_id, &a.event.id).cmp(&(
            b.event.created_at,
            &b.chat_id,
            &b.event.id,
        ))
    });
    Ok(out)
}

fn sqlite_error(error: rusqlite::Error) -> EngineError {
    EngineError::Other(format!("decision store: {error}"))
}

/// Aggregate counts over a set of receipts. Every map is keyed by the
/// serialized enum name so the report matches the exported JSON.
#[derive(Debug, Default, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DecisionReport {
    pub total: usize,
    pub chats: usize,
    pub selected: usize,
    pub abstained: usize,
    pub with_fallback: usize,
    pub with_outcome: usize,
    pub by_backend: BTreeMap<String, usize>,
    pub by_stage: BTreeMap<String, usize>,
    pub by_validation: BTreeMap<String, usize>,
    /// Mean of reported confidences; `None` when no receipt reported one.
    pub mean_confidence: Option<f64>,
    pub first_at: Option<i64>,
    pub last_at: Option<i64>,
}

pub fn report(decisions: &[ChatDecision]) -> DecisionReport {
    let mut report = DecisionReport {
        total: decisions.len(),
        ..Default::default()
    };
    let mut chats = HashSet::new();
    let mut confidence_sum = 0.0;
    let mut confidence_count = 0usize;
    for decision in decisions {
        let event = &decision.event;
        chats.insert(decision.chat_id.as_str());
        match event.result {
            DecisionResult::Selected { .. } => report.selected += 1,
            DecisionResult::Abstained => report.abstained += 1,
        }
        report.with_fallback += usize::from(event.fallback.is_some());
        report.with_outcome += usize::from(event.observed_outcome.is_some());
        *report
            .by_backend
            .entry(enum_name(&event.backend))
            .or_default() += 1;
        *report.by_stage.entry(enum_name(&event.stage)).or_default() += 1;
        *report
            .by_validation
            .entry(enum_name(&event.validation))
            .or_default() += 1;
        if let Some(confidence) = event.confidence {
            confidence_sum += confidence;
            confidence_count += 1;
        }
        report.first_at = Some(
            report
                .first_at
                .map_or(event.created_at, |t| t.min(event.created_at)),
        );
        report.last_at = Some(
            report
                .last_at
                .map_or(event.created_at, |t| t.max(event.created_at)),
        );
    }
    report.chats = chats.len();
    report.mean_confidence =
        (confidence_count > 0).then(|| confidence_sum / confidence_count as f64);
    report
}

fn enum_name(value: &impl Serialize) -> String {
    match serde_json::to_value(value) {
        Ok(serde_json::Value::String(name)) => name,
        _ => "unknown".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use keel_doc::{MessageRole, MessageStatus, SessionMessageEntry};
    use keel_proto::{DecisionBackend, DecisionCandidate, DecisionStage, DecisionValidation};

    fn event(id: &str, backend: DecisionBackend, result: DecisionResult, at: i64) -> DecisionEvent {
        DecisionEvent::new(
            id,
            1,
            backend,
            DecisionStage::Intake,
            vec![
                DecisionCandidate::new("a", "route a"),
                DecisionCandidate::new("b", "route b"),
            ],
            result,
            DecisionValidation::Accepted,
            at,
        )
    }

    fn save(store: &keel_sync::DocsStore, doc_id: &str, chat_id: &str, events: &[DecisionEvent]) {
        let doc = SessionDoc::init(chat_id).unwrap();
        for event in events {
            doc.push_message(&SessionMessageEntry {
                id: format!("decision-{}", event.id),
                role: MessageRole::System,
                parts: vec![MessagePart::Decision {
                    id: "decision".into(),
                    event: event.clone(),
                }],
                created_at: event.created_at,
                device_id: "test".into(),
                status: Some(MessageStatus::Complete),
                continuation_of: None,
            })
            .unwrap();
        }
        let bytes = doc.doc().export(loro::ExportMode::Snapshot).unwrap();
        store.save_snapshot(doc_id, &bytes).unwrap();
    }

    #[test]
    fn reads_receipts_across_chats_and_skips_rollback_duplicates() {
        let dir = tempfile::tempdir().unwrap();
        let store = keel_sync::DocsStore::open(dir.path()).unwrap();
        let laya = event(
            "d1",
            DecisionBackend::Laya,
            DecisionResult::Selected {
                candidate_id: "a".into(),
            },
            20,
        )
        .with_confidence(0.8);
        let jev = event("d2", DecisionBackend::Jev, DecisionResult::Abstained, 10)
            .with_fallback("normal harness");
        save(&store, "chat-1", "chat-1", std::slice::from_ref(&laya));
        save(
            &store,
            "chat-1.pre-chat2",
            "chat-1",
            std::slice::from_ref(&laya),
        );
        save(&store, "chat-2", "chat-2", std::slice::from_ref(&jev));
        store.save_snapshot("registry1", b"not a loro doc").unwrap();
        drop(store);

        let decisions = read_decisions(dir.path()).unwrap();
        let ids: Vec<_> = decisions.iter().map(|d| d.event.id.as_str()).collect();
        assert_eq!(ids, ["d2", "d1"], "deduped and oldest first");

        let report = report(&decisions);
        assert_eq!(report.total, 2);
        assert_eq!(report.chats, 2);
        assert_eq!((report.selected, report.abstained), (1, 1));
        assert_eq!(report.with_fallback, 1);
        assert_eq!(report.by_backend.get("laya"), Some(&1));
        assert_eq!(report.by_backend.get("jev"), Some(&1));
        assert_eq!(report.by_validation.get("accepted"), Some(&2));
        assert_eq!(report.mean_confidence, Some(0.8));
        assert_eq!((report.first_at, report.last_at), (Some(10), Some(20)));

        let case = serde_json::to_value(decisions[1].replay_case()).unwrap();
        assert_eq!(case["chatId"], "chat-1");
        assert_eq!(case["decision"]["id"], "d1");
        assert!(case["expectedCandidateId"].is_null());
    }

    #[test]
    fn missing_database_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_decisions(dir.path()).unwrap().is_empty());
        assert_eq!(report(&[]), DecisionReport::default());
    }
}
