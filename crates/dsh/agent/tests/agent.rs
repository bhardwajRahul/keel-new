//! Behavior tests mirroring upstream dsh-agent suites: inbox splice
//! durability and replay, duplicate-identity rejection, claim semantics,
//! consumed-work accounting, and registry lifecycle.

use dsh_agent::*;
use dsh_cordis::App;
use dsh_llm::{ContentBlock, MessageSource, create_user_message};
use dsh_session::{Session, SessionEventData, SessionId, SessionMeta, TurnEndReason};
use std::cell::RefCell;
use std::rc::Rc;

fn message(text: &str) -> dsh_llm::Message {
    create_user_message(
        vec![ContentBlock::Text { text: text.into() }],
        MessageSource::User,
    )
}

fn fresh_inbox() -> (Rc<Session>, Inbox) {
    let session = Session::create(SessionId::new("s"), vec![], None).unwrap();
    let inbox = Inbox::new(session.clone(), Box::new(SilentNotifications)).unwrap();
    (session, inbox)
}

#[test]
fn inbox_append_claims_and_replays_from_log() {
    let (session, inbox) = fresh_inbox();
    let first = message("one");
    let second = message("two");
    inbox.append(InboxTarget::NextTurn, first.clone()).unwrap();
    inbox.append(InboxTarget::NextStep, second.clone()).unwrap();
    assert!(inbox.has_pending());

    // Claim with a next-turn target: next-step batch first, then one queued turn.
    let claimed = inbox.claim(InboxTarget::NextTurn, 1).unwrap();
    assert_eq!(claimed.len(), 2);
    assert_eq!(claimed[0].id, second.id);
    assert_eq!(claimed[1].id, first.id);
    assert!(!inbox.has_pending());

    // Every mutation is durable: replaying the log reproduces the projection.
    let replayed = Inbox::new(session, Box::new(SilentNotifications)).unwrap();
    assert!(!replayed.has_pending());
}

#[test]
fn inbox_rejects_duplicate_pending_identity() {
    let (_session, inbox) = fresh_inbox();
    let original = message("dup");
    inbox
        .append(InboxTarget::NextTurn, original.clone())
        .unwrap();
    let error = inbox.append(InboxTarget::NextStep, original).err().unwrap();
    assert!(error.0.contains("already pending"));
}

#[test]
fn inbox_replace_and_remove_by_identity() {
    let (_session, inbox) = fresh_inbox();
    let original = message("original");
    inbox
        .append(InboxTarget::NextTurn, original.clone())
        .unwrap();
    let replacement = message("replacement");
    assert!(inbox.replace(&original.id, replacement.clone()).unwrap());
    assert_eq!(inbox.next_turn()[0].id, replacement.id);
    assert!(inbox.remove(&replacement.id).unwrap());
    assert!(!inbox.has_pending());
    // Missing identity: false, no error.
    assert!(!inbox.remove(&original.id).unwrap());
}

#[test]
fn consumed_work_separates_claims_from_drops() {
    let (session, inbox) = fresh_inbox();
    // No work yet: empty account.
    let account = fold_consumed_work(&session.events());
    assert!(account.end.is_none());
    assert!(!account.dropped_unrun);

    // A claimed turn that ends completed does NOT account (its claim was
    // rewritten away).
    inbox.append(InboxTarget::NextTurn, message("a")).unwrap();
    session
        .append(SessionEventData::TurnStart { turn: 1 }, None)
        .unwrap();
    inbox.claim(InboxTarget::NextTurn, 1).unwrap();
    session
        .append(
            SessionEventData::TurnEnd {
                turn: 1,
                reason: TurnEndReason::Completed,
            },
            None,
        )
        .unwrap();
    let account = fold_consumed_work(&session.events());
    assert!(account.end.is_none());

    // A claimed turn that ends blocked accounts for its input.
    inbox.append(InboxTarget::NextTurn, message("b")).unwrap();
    session
        .append(SessionEventData::TurnStart { turn: 2 }, None)
        .unwrap();
    inbox.claim(InboxTarget::NextTurn, 2).unwrap();
    session
        .append(
            SessionEventData::TurnEnd {
                turn: 2,
                reason: TurnEndReason::Blocked,
            },
            None,
        )
        .unwrap();
    let account = fold_consumed_work(&session.events());
    let end = account.end.unwrap();
    assert!(matches!(
        end.data,
        SessionEventData::TurnEnd {
            turn: 2,
            reason: TurnEndReason::Blocked
        }
    ));

    // A cancellation clearing pending input marks dropped-unrun.
    inbox.append(InboxTarget::NextTurn, message("c")).unwrap();
    inbox.clear().unwrap();
    let account = fold_consumed_work(&session.events());
    assert!(account.dropped_unrun);
}

#[test]
fn consumed_work_stepped_turn_accounts() {
    let session = Session::create(SessionId::new("s2"), vec![], None).unwrap();
    session
        .append(SessionEventData::TurnStart { turn: 1 }, None)
        .unwrap();
    session
        .append(SessionEventData::StepStart { turn: 1, step: 1 }, None)
        .unwrap();
    session
        .append(SessionEventData::StepEnd { turn: 1, step: 1 }, None)
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
    let account = fold_consumed_work(&session.events());
    // A stepped turn accounts regardless of its ending.
    assert!(account.end.is_some());
}

struct TestAgent {
    session: Rc<Session>,
    ctx: dsh_cordis::Context,
}

impl Agent for TestAgent {
    fn id(&self) -> SessionId {
        self.session.id()
    }
    fn options(&self) -> AgentOptions {
        AgentOptions::default()
    }
    fn session(&self) -> Rc<Session> {
        self.session.clone()
    }
    fn status(&self) -> AgentStatus {
        AgentStatus::Idle
    }
    fn ctx(&self) -> dsh_cordis::Context {
        self.ctx.clone()
    }
    fn cancel(&self, _cause: AgentCancelCause, _options: CancelOptions) {}
    fn when_idle(&self) -> futures::future::LocalBoxFuture<'static, ()> {
        Box::pin(async {})
    }
    fn send(&self, _message: dsh_llm::Message, _target: InboxTarget, _wakeup: bool) {}
    fn followup(&self, _message: dsh_llm::Message) {}
    fn steer(&self, _message: dsh_llm::Message) {}
    fn inject(&self, _message: dsh_llm::Message) {}
}

#[test]
fn registry_lifecycle_and_initiators() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let registry = AgentRegistry::provide(&ctx).unwrap();

        let created: Rc<RefCell<u32>> = Rc::default();
        let created2 = created.clone();
        ctx.on::<AgentCreated, _, _>(Default::default(), move |_ctx, _agent| {
            *created2.borrow_mut() += 1;
            async { None }
        })
        .unwrap();
        let disposed: Rc<RefCell<u32>> = Rc::default();
        let disposed2 = disposed.clone();
        ctx.on::<AgentDisposed, _, _>(Default::default(), move |_ctx, _agent| {
            *disposed2.borrow_mut() += 1;
            async { None }
        })
        .unwrap();

        let session = Session::create(SessionId::new("a1"), vec![], None).unwrap();
        let agent: AgentRef = Rc::new(TestAgent {
            session,
            ctx: ctx.clone(),
        });
        let handle = registry.register(agent.clone()).unwrap();
        tokio::task::yield_now().await;
        assert_eq!(*created.borrow(), 1);
        assert!(registry.get(&SessionId::new("a1")).is_some());
        assert_eq!(registry.list().len(), 1);
        assert_eq!(registry.roots().len(), 1);

        // Duplicate registration rejected.
        assert!(registry.register(agent.clone()).is_err());

        // Initiator attribution: inherited inside, absent outside.
        let seen = registry
            .with_initiator(&agent, async {
                registry
                    .current_initiator()
                    .unwrap()
                    .map(|agent| agent.id())
            })
            .await
            .unwrap();
        assert_eq!(seen.unwrap().as_str(), "a1");
        assert!(registry.current_initiator().unwrap().is_none());
        assert!(registry.require_initiator().is_err());
        // A clearing boundary hides an inherited initiator.
        let hidden = registry
            .with_initiator(&agent, async {
                registry
                    .without_initiator(async {
                        registry
                            .current_initiator()
                            .unwrap()
                            .map(|agent| agent.id())
                    })
                    .await
                    .unwrap()
            })
            .await
            .unwrap();
        assert!(hidden.is_none());

        // No factory registered: create fails with the canonical message.
        let error = registry
            .create(CreateAgentOptions {
                session_id: SessionId::new("x"),
                meta: SessionMeta::default(),
                seed: vec![],
                agent_options: AgentOptions::default(),
            })
            .await
            .err()
            .unwrap();
        assert!(error.to_string().contains("no agent factory registered"));

        // Disposal emits the paired edge.
        handle.dispose().await;
        tokio::task::yield_now().await;
        assert_eq!(*disposed.borrow(), 1);
        assert!(registry.get(&SessionId::new("a1")).is_none());

        // Disposed initiator scope rejects boundaries.
        registry.dispose_initiators();
        assert!(registry.current_initiator().is_err());
    });
}

#[test]
fn inbox_splice_round_trips_as_extension_event() {
    let splice = InboxSplice {
        target: InboxTarget::NextStep,
        start: 0,
        removed_count: Some(2),
        inserted: vec![message("kept")],
        outcome: Some(SpliceOutcome::Canceled),
    };
    let data = splice.to_event_data();
    assert_eq!(data.event_type(), INBOX_SPLICED_EVENT);
    let event = dsh_session::SessionEvent {
        seq: 0,
        time: 0,
        data,
        source_event_seqs: None,
        surface_op: None,
        ignorable: None,
    };
    assert_eq!(InboxSplice::from_event(&event).unwrap(), splice);
    // Wire encoding keeps upstream field names.
    let value = serde_json::to_value(&event).unwrap();
    assert_eq!(value["type"], "agent/inbox/spliced");
    assert_eq!(value["data"]["removedCount"], 2);
    assert_eq!(value["data"]["target"], "next-step");
}
