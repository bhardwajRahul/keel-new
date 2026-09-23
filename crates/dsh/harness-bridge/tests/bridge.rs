//! Headless integration test over the real bridge: boots the actor thread and
//! the full dsh composition, runs one prompt without a credential, and
//! asserts the comet-facing event contract (SessionStarted → Error →
//! Done{Errored} carrying the session id), plus interrupt handling.
//!
//! One test function on purpose: the bridge thread reads $DSH_HOME once at
//! composition build, and cargo runs tests in threads.

use dsh_harness_bridge::DshHarness;
use futures::StreamExt;
use keel_harness::{Harness, RunControls};
use keel_proto::{AgentEvent, DoneStatus, HarnessId, RunRequest, SandboxLevel};
use tokio_util::sync::CancellationToken;

fn run_request(prompt: &str, resume: Option<String>) -> RunRequest {
    RunRequest {
        prompt: prompt.into(),
        harness: Some(HarnessId::Dsh),
        model: Some("deepseek-chat".into()),
        reasoning: None,
        model_options: serde_json::Map::new(),
        cwd: std::env::temp_dir().to_string_lossy().into_owned(),
        sandbox: SandboxLevel::WorkspaceWrite,
        auto_approve: false,
        resume,
        attachments: vec![],
    }
}

fn controls() -> (
    RunControls,
    tokio::sync::mpsc::Sender<keel_harness::SteerMessage>,
    CancellationToken,
) {
    let (steer_tx, steer_rx) = tokio::sync::mpsc::channel(4);
    let interrupt = CancellationToken::new();
    let controls = RunControls {
        request_input: Box::new(|_questions| {
            let (tx, rx) = tokio::sync::oneshot::channel();
            let _ = tx.send(vec![]);
            rx
        }),
        steering: steer_rx,
        interrupt: interrupt.clone(),
    };
    (controls, steer_tx, interrupt)
}

#[tokio::test(flavor = "multi_thread")]
async fn bridge_runs_sessions_end_to_end() {
    // Isolated home; no credential so the turn fails structurally instead of
    // hitting the network.
    let home = tempfile::tempdir().unwrap();
    unsafe {
        std::env::set_var("DSH_HOME", home.path());
        std::env::remove_var("DEEPSEEK_API_KEY");
    }

    let harness = DshHarness::new();
    assert_eq!(harness.id(), HarnessId::Dsh);
    // The harness stays hidden from production pickers without a credential,
    // while direct runs still surface a structured missing-key error.
    assert!(!harness.installed());
    assert!(!harness.models().await.unwrap().is_empty());

    // Run 1: no credential → SessionStarted, Error, Done{Errored, session_id}.
    let (controls1, _steer1, _interrupt1) = controls();
    let mut stream = harness
        .run(run_request("say hi", None), controls1)
        .await
        .unwrap();
    let mut session_id: Option<String> = None;
    let mut saw_error = false;
    let mut done: Option<(DoneStatus, Option<String>)> = None;
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            AgentEvent::SessionStarted {
                session_id: id,
                harness,
                ..
            } => {
                assert_eq!(harness, HarnessId::Dsh);
                session_id = Some(id);
            }
            AgentEvent::Error { message } => {
                assert!(
                    message.contains("MISSING_CREDENTIAL"),
                    "unexpected error: {message}"
                );
                saw_error = true;
            }
            AgentEvent::Done {
                status,
                session_id: done_session,
                ..
            } => {
                done = Some((status, done_session));
                break;
            }
            _ => {}
        }
    }
    let started_id = session_id.expect("SessionStarted arrived");
    assert!(saw_error, "structured error surfaced");
    let (status, done_session) = done.expect("Done arrived");
    assert_eq!(status, DoneStatus::Errored);
    assert_eq!(done_session.as_deref(), Some(started_id.as_str()));

    // The session artifact persisted for resume.
    let artifacts: Vec<_> = std::fs::read_dir(home.path().join("sessions"))
        .unwrap()
        .flatten()
        .collect();
    assert_eq!(artifacts.len(), 1);
    assert!(
        std::fs::read_to_string(artifacts[0].path())
            .unwrap()
            .contains("decision/receipt"),
        "decision receipts survive the session log roundtrip"
    );

    // Run 2: resume the same session id — the bridge reloads the persisted
    // log and answers with the same identity.
    let (controls2, _steer2, _interrupt2) = controls();
    let mut stream = harness
        .run(run_request("continue", Some(started_id.clone())), controls2)
        .await
        .unwrap();
    let mut resumed_id: Option<String> = None;
    let mut resume_error: Option<String> = None;
    let mut resume_done: Option<String> = None;
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            AgentEvent::SessionStarted { session_id: id, .. } => resumed_id = Some(id),
            AgentEvent::Error { message } => resume_error = Some(message),
            AgentEvent::Done { session_id, .. } => {
                resume_done = session_id;
                break;
            }
            _ => {}
        }
    }
    assert_eq!(
        resumed_id.as_deref(),
        Some(started_id.as_str()),
        "resume error: {resume_error:?}"
    );
    assert_eq!(resume_done.as_deref(), Some(started_id.as_str()));
}
