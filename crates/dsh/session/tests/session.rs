//! Behavior tests mirroring the upstream dsh-session package suites: append
//! and surface invariants, seed validation, derived history with replacement,
//! fork boundaries, request-header folding, and interrupted-turn repair.

use dsh_cordis::App;
use dsh_llm::{
    AssistantProvenance, CallId, ContentBlock, LlmCallConfig, MessageSource,
    create_assistant_message, create_user_message,
};
use dsh_session::*;
use std::cell::RefCell;
use std::rc::Rc;

fn user_event_data(text: &str) -> SessionEventData {
    SessionEventData::UserMessage(create_user_message(
        vec![ContentBlock::Text { text: text.into() }],
        MessageSource::User,
    ))
}

fn assistant_event_data(turn: u64, step: u64, text: &str) -> SessionEventData {
    SessionEventData::AssistantMessage {
        turn,
        step,
        message: create_assistant_message(
            vec![ContentBlock::Text { text: text.into() }],
            AssistantProvenance {
                provider: "p".into(),
                model: "m".into(),
                replay_state: None,
            },
        ),
        usage: None,
    }
}

#[test]
fn append_assigns_contiguous_seqs_and_projects_surface() {
    let session = Session::create(SessionId::new("s1"), vec![], None).unwrap();
    session
        .append(SessionEventData::TurnStart { turn: 1 }, None)
        .unwrap();
    let user = session
        .append(user_event_data("hello"), Some(SurfaceIntent::append()))
        .unwrap();
    assert_eq!(user.seq, 1);
    session
        .append(
            assistant_event_data(1, 1, "hi"),
            Some(SurfaceIntent::append()),
        )
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
    assert_eq!(session.seq(), 4);
    assert_eq!(session.surface_nodes(), vec![1, 2]);
    let messages = session.derive_messages();
    assert_eq!(messages.len(), 2);
}

#[test]
fn surface_metadata_is_required_and_forbidden_by_type() {
    let session = Session::create(SessionId::new("s2"), vec![], None).unwrap();
    // Message-producing event without a marker: rejected.
    let missing = session.append(user_event_data("x"), None);
    assert!(missing.is_err());
    // Log-only event with a marker: rejected.
    let forbidden = session.append(
        SessionEventData::TurnStart { turn: 1 },
        Some(SurfaceIntent::append()),
    );
    assert!(forbidden.is_err());
    // Failed appends never entered the log.
    assert_eq!(session.seq(), 0);
}

#[test]
fn empty_assistant_message_is_not_derived() {
    let session = Session::create(SessionId::new("s3"), vec![], None).unwrap();
    session
        .append(user_event_data("prompt"), Some(SurfaceIntent::append()))
        .unwrap();
    session
        .append(
            SessionEventData::AssistantMessage {
                turn: 1,
                step: 1,
                message: create_assistant_message(
                    vec![],
                    AssistantProvenance {
                        provider: "p".into(),
                        model: "m".into(),
                        replay_state: None,
                    },
                ),
                usage: None,
            },
            Some(SurfaceIntent::append()),
        )
        .unwrap();
    // Both are surface nodes, but the empty assistant message derives to nothing.
    assert_eq!(session.surface_nodes().len(), 2);
    assert_eq!(session.derive_messages().len(), 1);
}

#[test]
fn replacement_shadows_range_and_requires_cited_sources() {
    let session = Session::create(SessionId::new("s4"), vec![], None).unwrap();
    session
        .append(user_event_data("one"), Some(SurfaceIntent::append()))
        .unwrap();
    session
        .append(user_event_data("two"), Some(SurfaceIntent::append()))
        .unwrap();

    // Replacement without citing shadowed nodes: rejected.
    let uncited = session.append(
        user_event_data("summary"),
        Some(SurfaceIntent {
            surface_op: Some(SurfaceOp::replace(0, 1)),
            source_event_seqs: None,
        }),
    );
    assert!(uncited.is_err());

    // Correctly cited replacement collapses the surface to one node.
    session
        .append(
            user_event_data("summary"),
            Some(SurfaceIntent {
                surface_op: Some(SurfaceOp::replace(0, 1)),
                source_event_seqs: Some(vec![0, 1]),
            }),
        )
        .unwrap();
    assert_eq!(session.surface_nodes(), vec![2]);
    assert_eq!(session.replace_generation(), 1);
    let messages = session.derive_messages();
    assert_eq!(messages.len(), 1);
}

#[test]
fn seed_must_be_contiguous_and_marks_end_seed() {
    let donor = Session::create(SessionId::new("donor"), vec![], None).unwrap();
    donor
        .append(user_event_data("seeded"), Some(SurfaceIntent::append()))
        .unwrap();
    let events = donor.events();

    let seeded = Session::create(SessionId::new("child"), events.clone(), None).unwrap();
    assert_eq!(seeded.first_live_seq, 1);
    // The seed did not end in a marker, so construction appended one.
    let last = seeded.events().last().cloned().unwrap();
    assert!(matches!(last.data, SessionEventData::SessionEndSeed {}));
    // Reopening with the marker present does not grow the log.
    let reopened = Session::create(SessionId::new("child2"), seeded.events(), None).unwrap();
    assert_eq!(reopened.seq(), seeded.seq());

    // Non-contiguous seed rejected.
    let mut broken = events;
    broken[0].seq = 5;
    assert!(Session::create(SessionId::new("bad"), broken, None).is_err());
}

#[test]
fn store_lifecycle_publishes_events_and_disposal() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let store = SessionStore::provide(&ctx).unwrap();

        let seen: Rc<RefCell<Vec<String>>> = Rc::default();
        let seen2 = seen.clone();
        ctx.on::<SessionEventPublished, _, _>(Default::default(), move |_ctx, (_, event)| {
            seen2.borrow_mut().push(event.event_type().to_string());
            async { None }
        })
        .unwrap();
        let disposed: Rc<RefCell<u32>> = Rc::default();
        let disposed2 = disposed.clone();
        ctx.on::<SessionDisposed, _, _>(Default::default(), move |_ctx, _session| {
            *disposed2.borrow_mut() += 1;
            async { None }
        })
        .unwrap();

        let session = store.create(None, vec![], SessionMeta::default()).unwrap();
        assert_eq!(session.id().as_str(), "session-1");
        session
            .append(user_event_data("hi"), Some(SurfaceIntent::append()))
            .unwrap();
        // Emit listeners run on the local set; yield so they fire.
        tokio::task::yield_now().await;
        assert_eq!(*seen.borrow(), vec!["user/message".to_string()]);

        assert_eq!(store.list().len(), 1);
        app.shutdown().await;
        tokio::task::yield_now().await;
        assert_eq!(store.list().len(), 0);
        assert_eq!(*disposed.borrow(), 1);
    });
}

#[test]
fn fork_rejects_open_turn_and_copies_prefix() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let store = SessionStore::provide(&ctx).unwrap();
        let source = store
            .create(Some(SessionId::new("src")), vec![], SessionMeta::default())
            .unwrap();
        source
            .append(SessionEventData::TurnStart { turn: 1 }, None)
            .unwrap();
        source
            .append(user_event_data("q"), Some(SurfaceIntent::append()))
            .unwrap();

        // Boundary inside the open turn: rejected.
        let open = store
            .fork(&SessionId::new("src"), Some(1), None)
            .err()
            .unwrap();
        assert_eq!(open.code, SessionForkErrorCode::OpenTurn);

        source
            .append(
                SessionEventData::TurnEnd {
                    turn: 1,
                    reason: TurnEndReason::Completed,
                },
                None,
            )
            .unwrap();
        let child = store
            .fork(&SessionId::new("src"), None, Some(SessionId::new("kid")))
            .unwrap();
        assert_eq!(
            child.header.parent_session.as_ref().unwrap().as_str(),
            "src"
        );
        assert_eq!(child.header.seed_length, Some(3));
        // Child log = seed + end-seed marker.
        assert_eq!(child.seq(), 4);

        let missing = store
            .fork(&SessionId::new("nope"), None, None)
            .err()
            .unwrap();
        assert_eq!(missing.code, SessionForkErrorCode::SessionNotFound);
    });
}

#[test]
fn request_header_folds_latest_canonical_snapshot() {
    let session = Session::create(SessionId::new("s5"), vec![], None).unwrap();
    assert!(session.request_header().is_none());
    let header = EpochHeader {
        config: LlmCallConfig {
            provider: "p".into(),
            model: "m".into(),
            ..Default::default()
        },
        adapter_defaults: None,
        system: Some(String::new()), // canonicalizes to absent
        tools: Some(vec![]),         // canonicalizes to absent
    };
    session
        .append(
            SessionEventData::RequestHeader {
                header,
                reason: RequestHeaderReason::Initial,
            },
            None,
        )
        .unwrap();
    let folded = session.request_header().unwrap();
    assert!(folded.system.is_none());
    assert!(folded.tools.is_none());
    assert_eq!(folded.config.model, "m");

    // headerEquals ignores canonically-absent fields.
    let other = EpochHeader {
        config: LlmCallConfig {
            provider: "p".into(),
            model: "m".into(),
            ..Default::default()
        },
        adapter_defaults: None,
        system: None,
        tools: None,
    };
    assert!(header_equals(&folded, &other));
}

#[test]
fn interrupted_turn_closers_synthesize_boundaries() {
    let session = Session::create(SessionId::new("s6"), vec![], None).unwrap();
    session
        .append(SessionEventData::TurnStart { turn: 3 }, None)
        .unwrap();
    session
        .append(SessionEventData::StepStart { turn: 3, step: 1 }, None)
        .unwrap();
    let assistant = create_assistant_message(
        vec![ContentBlock::ToolCall {
            id: CallId::new("call-7"),
            name: "bash".into(),
            arguments: "{}".into(),
        }],
        AssistantProvenance {
            provider: "p".into(),
            model: "m".into(),
            replay_state: None,
        },
    );
    session
        .append(
            SessionEventData::AssistantMessage {
                turn: 3,
                step: 1,
                message: assistant,
                usage: None,
            },
            Some(SurfaceIntent::append()),
        )
        .unwrap();
    let call = session
        .append(
            SessionEventData::ToolCall {
                turn: 3,
                step: 1,
                call_id: CallId::new("call-7"),
                name: "bash".into(),
                arguments: "{}".into(),
            },
            None,
        )
        .unwrap();

    let closers = interrupted_turn_closers(&session.events());
    assert_eq!(closers.len(), 3);
    // First: the synthetic error result citing the recorded call.
    match &closers[0].data {
        SessionEventData::ToolResult { error, message, .. } => {
            assert_eq!(error.as_ref().unwrap().code, TOOL_OUTCOME_UNKNOWN);
            assert!(
                matches!(&message.source, MessageSource::Tool { call_id } if call_id.as_str() == "call-7")
            );
        }
        other => panic!("expected tool/result closer, got {other:?}"),
    }
    assert_eq!(closers[0].source_event_seqs, Some(vec![call.seq]));
    // Then step/end, then interrupted turn/end.
    assert!(matches!(
        closers[1].data,
        SessionEventData::StepEnd { turn: 3, step: 1 }
    ));
    assert!(matches!(
        &closers[2].data,
        SessionEventData::TurnEnd {
            turn: 3,
            reason: TurnEndReason::Interrupted
        }
    ));
    // Closers continue the log contiguously; appending them yields a balanced log.
    for closer in &closers {
        assert_eq!(closer.time, session.events().last().unwrap().time);
    }
    let balanced: Vec<SessionEvent> = session.events().into_iter().chain(closers).collect();
    assert!(interrupted_turn_closers(&balanced).is_empty());
}

#[test]
fn events_round_trip_through_json() {
    let session = Session::create(SessionId::new("s7"), vec![], None).unwrap();
    session
        .append(SessionEventData::TurnStart { turn: 1 }, None)
        .unwrap();
    session
        .append(user_event_data("hello"), Some(SurfaceIntent::append()))
        .unwrap();
    session
        .append(
            SessionEventData::TodoWrite {
                todos: vec![TodoItem {
                    content: "port".into(),
                    status: TodoStatus::InProgress,
                }],
            },
            None,
        )
        .unwrap();
    for event in session.events() {
        let encoded = serde_json::to_string(&event).unwrap();
        let decoded: SessionEvent = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, event);
    }
    // The wire tag matches upstream type strings.
    let encoded = serde_json::to_value(&session.events()[0]).unwrap();
    assert_eq!(encoded["type"], "turn/start");
}
