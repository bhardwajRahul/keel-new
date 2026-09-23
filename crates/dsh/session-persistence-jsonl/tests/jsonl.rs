//! Behavior tests for the JSONL backend and the write-behind pipeline:
//! artifact round-trip, path-segment injectivity, contiguity enforcement,
//! torn-tail truncation, cold interrupted-turn recovery, watermark reads,
//! listing, and the session-store wiring end to end.

use dsh_cordis::App;
use dsh_llm::{ContentBlock, MessageSource, create_user_message};
use dsh_session::{
    Session, SessionEventData, SessionId, SessionMeta, SessionStore, SurfaceIntent, TurnEndReason,
};
use dsh_session_persistence::{PersistenceService, SessionPersistence, install_write_behind};
use dsh_session_persistence_jsonl::{JsonlPersistence, encode_path_segment};
use std::rc::Rc;

fn user_event(text: &str) -> SessionEventData {
    SessionEventData::UserMessage(create_user_message(
        vec![ContentBlock::Text { text: text.into() }],
        MessageSource::User,
    ))
}

fn seeded_session(id: &str) -> Rc<Session> {
    let session = Session::create(SessionId::new(id), vec![], None).unwrap();
    session
        .append(SessionEventData::TurnStart { turn: 1 }, None)
        .unwrap();
    session
        .append(user_event("hello"), Some(SurfaceIntent::append()))
        .unwrap();
    session
        .append(
            SessionEventData::TurnEnd {
                turn: 1,
                reason: TurnEndReason::Completed,
            },
            None,
        )
        .unwrap();
    session
}

#[test]
fn path_segment_encoding_neutralizes_traversal() {
    assert_eq!(encode_path_segment("session-1"), "session-1");
    // Traversal and separators cannot survive.
    assert!(!encode_path_segment("../evil").contains(".."));
    assert!(!encode_path_segment("a/b\\c").contains('/'));
    assert!(!encode_path_segment("a/b\\c").contains('\\'));
    // Whole-segment dots escape.
    assert!(!encode_path_segment(".").contains('.'));
    assert!(!encode_path_segment("..").contains('.'));
    // Injective: distinct inputs, distinct outputs.
    assert_ne!(encode_path_segment("a~b"), encode_path_segment("a b"));
}

#[test]
fn create_append_load_round_trip() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlPersistence::new(dir.path()).unwrap();
        let session = seeded_session("s1");
        store.create(&session.header).await.unwrap();
        store.append(&session.id(), session.events()).await.unwrap();

        let loaded = store.load(&session.id()).await.unwrap();
        assert_eq!(loaded.meta.id.as_str(), "s1");
        assert_eq!(loaded.events, session.events());

        // Non-contiguous batch rejected.
        let mut stale = session.events();
        stale[0].seq = 99;
        let error = store
            .append(&session.id(), vec![stale.remove(0)])
            .await
            .err()
            .unwrap();
        assert!(error.to_string().contains("stored next-seq"));

        // Appending to a never-created session rejected.
        let error = store
            .append(&SessionId::new("ghost"), vec![])
            .await
            .err()
            .unwrap();
        assert!(error.to_string().contains("never created"));
    });
}

#[test]
fn torn_tail_is_discarded_and_committed_corruption_rejects() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlPersistence::new(dir.path()).unwrap();
        let session = seeded_session("s2");
        store.create(&session.header).await.unwrap();
        store.append(&session.id(), session.events()).await.unwrap();
        let path = store.locate(&session.header).unwrap().path;

        // A torn final record (no trailing newline) is truncation, not error.
        let mut content = std::fs::read_to_string(&path).unwrap();
        content.push_str("{\"seq\":3,\"time\":1,\"type\":\"turn/sta");
        std::fs::write(&path, &content).unwrap();
        let loaded = store.load(&session.id()).await.unwrap();
        assert_eq!(loaded.events.len(), 3);

        // An unknown required event type in the committed prefix refuses the
        // log (a newer writer's vocabulary must not be silently skipped).
        let unknown_type = content.replace("turn/start", "future/event");
        std::fs::write(&path, unknown_type).unwrap();
        let error = store.load(&session.id()).await.err().unwrap();
        assert!(error.to_string().contains("unknown required event type"));

        // Structural corruption inside the committed prefix rejects.
        let corrupt = content.replace("\"seq\":0", "\"seq\":!!");
        std::fs::write(&path, corrupt).unwrap();
        assert!(store.load(&session.id()).await.is_err());
    });
}

#[test]
fn load_commits_interrupted_turn_recovery_and_inspect_does_not() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlPersistence::new(dir.path()).unwrap();
        // A log crashing mid-turn: turn/start then a user message, no end.
        let session = Session::create(SessionId::new("s3"), vec![], None).unwrap();
        session
            .append(SessionEventData::TurnStart { turn: 1 }, None)
            .unwrap();
        session
            .append(user_event("crashing"), Some(SurfaceIntent::append()))
            .unwrap();
        store.create(&session.header).await.unwrap();
        store.append(&session.id(), session.events()).await.unwrap();

        // Inspect: balanced in memory, artifact untouched.
        let inspected = store.inspect(&session.id()).await.unwrap();
        assert!(matches!(
            inspected.events.last().unwrap().data,
            SessionEventData::TurnEnd {
                reason: TurnEndReason::Interrupted,
                ..
            }
        ));
        let before = std::fs::read_to_string(store.locate(&session.header).unwrap().path).unwrap();

        // Load: recovery committed durably, exactly once.
        let loaded = store.load(&session.id()).await.unwrap();
        assert!(matches!(
            loaded.events.last().unwrap().data,
            SessionEventData::TurnEnd {
                reason: TurnEndReason::Interrupted,
                ..
            }
        ));
        let after = std::fs::read_to_string(store.locate(&session.header).unwrap().path).unwrap();
        assert!(after.len() > before.len());
        let again = store.load(&session.id()).await.unwrap();
        assert_eq!(again.events.len(), loaded.events.len());
    });
}

#[test]
fn read_from_returns_watermark_suffix() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlPersistence::new(dir.path()).unwrap();
        let session = seeded_session("s4");
        store.create(&session.header).await.unwrap();
        store.append(&session.id(), session.events()).await.unwrap();

        let (_, suffix) = store.read_from(&session.id(), 1).await.unwrap();
        assert_eq!(suffix.len(), 2);
        assert_eq!(suffix[0].seq, 1);
        // At/beyond the prefix: empty, never an error.
        let (_, empty) = store.read_from(&session.id(), 99).await.unwrap();
        assert!(empty.is_empty());
    });
}

#[test]
fn list_and_snapshots_read_headers_without_full_parse() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonlPersistence::new(dir.path()).unwrap();
        for id in ["a", "b"] {
            let session = seeded_session(id);
            store.create(&session.header).await.unwrap();
            store.append(&session.id(), session.events()).await.unwrap();
        }
        let headers = store.list().await.unwrap();
        assert_eq!(headers.len(), 2);
        let snapshots = store.list_snapshots().await.unwrap();
        assert_eq!(snapshots.len(), 2);
        // Revision stable while unchanged, changed after an append.
        let first = snapshots
            .iter()
            .find(|s| s.header.id.as_str() == "a")
            .unwrap()
            .revision
            .clone();
        let again = store.list_snapshots().await.unwrap();
        let second = again
            .iter()
            .find(|s| s.header.id.as_str() == "a")
            .unwrap()
            .revision
            .clone();
        assert_eq!(first, second);
    });
}

/// Regression: a resumed session appends its `session/end-seed` marker
/// inside the constructor, before persistence attaches, so the live log
/// starts one event ahead of disk. Without a tail reconcile at attach, every
/// later append fails contiguity and the session silently stops persisting.
#[test]
fn resumed_sessions_reconcile_the_durable_tail() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let app = App::new();
        let ctx = app.root();
        let sessions = SessionStore::provide(&ctx).unwrap();
        let backend = Rc::new(JsonlPersistence::new(dir.path()).unwrap());
        let service = PersistenceService::provide(&ctx, backend.clone()).unwrap();
        install_write_behind(&ctx, backend.clone(), 5).unwrap();

        // A first life: three durable events, then the session leaves the
        // store (a process exit, in production).
        let first = sessions
            .prepare(
                Some(SessionId::new("resumed")),
                vec![],
                SessionMeta::default(),
            )
            .unwrap();
        let first_attachment = sessions.enter(first.clone()).unwrap();
        sessions.announce(&first).unwrap();
        first
            .append(SessionEventData::TurnStart { turn: 1 }, None)
            .unwrap();
        first
            .append(user_event("first"), Some(SurfaceIntent::append()))
            .unwrap();
        first
            .append(
                SessionEventData::TurnEnd {
                    turn: 1,
                    reason: TurnEndReason::Completed,
                },
                None,
            )
            .unwrap();
        tokio::task::yield_now().await;
        sessions.flush(&first).await.unwrap();
        let durable_before = backend
            .load(&SessionId::new("resumed"))
            .await
            .unwrap()
            .events
            .len();
        assert_eq!(durable_before, 3);
        first_attachment.dispose().await;

        // A second life over the same artifact: prepare, then publish.
        let resumed = service.prepare(&SessionId::new("resumed")).await.unwrap();
        // The constructor added the end-seed marker the disk has never seen.
        assert_eq!(resumed.seq(), 4);
        sessions.enter(resumed.clone()).unwrap();
        sessions.announce(&resumed).unwrap();
        tokio::task::yield_now().await;

        // New live work must still persist — the pre-fix bug made every
        // append here fail with a seq gap.
        resumed
            .append(SessionEventData::TurnStart { turn: 2 }, None)
            .unwrap();
        resumed
            .append(user_event("second"), Some(SurfaceIntent::append()))
            .unwrap();
        tokio::task::yield_now().await;
        sessions.flush(&resumed).await.unwrap();

        // Raw durable state (no recovery commit): disk matches memory
        // exactly — no gap, and no duplicated seed.
        let (_, stored) = backend
            .read_from(&SessionId::new("resumed"), 0)
            .await
            .unwrap();
        assert_eq!(
            stored.len(),
            resumed.seq() as usize,
            "disk caught up with the live log"
        );
        let seqs: Vec<u64> = stored.iter().map(|event| event.seq).collect();
        assert_eq!(
            seqs,
            (0..resumed.seq()).collect::<Vec<u64>>(),
            "contiguous, no repeats"
        );
        assert!(
            stored
                .iter()
                .any(|event| matches!(event.data, SessionEventData::TurnStart { turn: 2 }))
        );
    });
}

#[test]
fn write_behind_pipeline_persists_live_sessions() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let app = App::new();
        let ctx = app.root();
        let sessions = SessionStore::provide(&ctx).unwrap();
        let backend = Rc::new(JsonlPersistence::new(dir.path()).unwrap());
        let service = PersistenceService::provide(&ctx, backend.clone()).unwrap();
        install_write_behind(&ctx, backend.clone(), 10).unwrap();

        let session = sessions
            .create(Some(SessionId::new("live")), vec![], SessionMeta::default())
            .unwrap();
        session
            .append(SessionEventData::TurnStart { turn: 1 }, None)
            .unwrap();
        session
            .append(user_event("persist me"), Some(SurfaceIntent::append()))
            .unwrap();
        session
            .append(
                SessionEventData::TurnEnd {
                    turn: 1,
                    reason: TurnEndReason::Completed,
                },
                None,
            )
            .unwrap();
        // Let emit listeners run, then flush the durability checkpoint.
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;
        sessions.flush(&session).await.unwrap();

        let loaded = backend.load(&SessionId::new("live")).await.unwrap();
        assert_eq!(loaded.events.len(), 3);

        // Resume through the seam: the prepared session replays the log.
        let resumed = service.prepare(&SessionId::new("live")).await.unwrap();
        assert_eq!(resumed.first_live_seq, 3);
        assert_eq!(resumed.derive_messages().len(), 1);
    });
}
