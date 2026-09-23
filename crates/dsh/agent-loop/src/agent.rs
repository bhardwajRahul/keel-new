//! The default agent driver over queued turns and step-boundary input,
//! ported from `packages/core/agent-loop/src/agent.ts`. Every request is
//! derived from the session log.
//!
//! Divergences:
//! - Phase state lives in a RefCell'd struct; the driver runs as a spawned
//!   local task holding an Rc of the loop state.
//! - Scope wiring on the agent context is deferred with the registry's
//!   scope-filtered dispatch (events carry the agent handle).

use crate::runtime_context::RuntimeContextProjection;
use crate::tool_calls::{PlannedCall, execute_tool_calls};
use dsh_agent::{
    Agent, AgentInboxClaimed, AgentInboxDiscarded, AgentInboxInserted, AgentOptions, AgentPreStep,
    AgentRef, AgentRequest, AgentRequestError, AgentStatus, AgentStatusChanged, AgentTurnStopping,
    CancelOptions, Inbox, InboxNotifications, InboxTarget, PreStepDecision, PreStepPayload,
    RequestErrorAction, RequestErrorPayload, RequestPayload,
};
use dsh_cordis::{Context, Event};
use dsh_llm::{
    AssistantProvenance, BlockAssembler, ContentBlock, FinishReason, GenerateOptions,
    LlmCallConfig, LlmError, LlmRuntime, Message, MessageSource, Role, create_assistant_message,
    error_chain,
};
use dsh_session::{
    AgentCancelCause, EpochHeader, RequestContext, RequestHeaderReason, Session, SessionEventData,
    SessionId, SurfaceIntent, TurnEndReason, canonical_header, header_equals,
};
use dsh_system_prompt::{
    AssembleContext, SystemPrompt, join_context_sections, render_context_sections, render_prompt,
};
use dsh_timeout::{AbortController, AbortReason, AbortSignal};
use dsh_tools::ToolRuntime;
use futures::FutureExt;
use futures::future::{LocalBoxFuture, Shared};
use jev_core::{Candidate, DecisionBudget, DecisionState, JevConfig, JevSelector, SelectionInput};
use keel_proto::{
    DecisionActivity, DecisionBackend, DecisionCandidate, DecisionEvent, DecisionPhase,
    DecisionResult, DecisionStage, DecisionValidation,
};
use laya_local::LayaSelector;
use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::rc::Rc;

/// Driver phase (upstream `Phase`).
enum Phase {
    Idle {
        last_turn: u64,
    },
    Maintenance {
        abort: AbortController,
        wake_requested: Cell<bool>,
    },
    Running {
        abort: AbortController,
        turn: Cell<u64>,
        step: Cell<u64>,
        wake_requested: Cell<bool>,
    },
}

impl Phase {
    fn status(&self) -> AgentStatus {
        match self {
            Phase::Idle { .. } | Phase::Maintenance { .. } => AgentStatus::Idle,
            Phase::Running { .. } => AgentStatus::Running,
        }
    }
}

/// Remove adapter-derived values before plugins propose the next request
/// config (upstream `requestProposal`).
fn request_proposal(header: &EpochHeader) -> LlmCallConfig {
    let mut proposal = header.config.clone();
    if let Some(defaults) = &header.adapter_defaults {
        if defaults.reasoning_effort == Some(true) {
            proposal.reasoning_effort = None;
        }
        if defaults.max_tokens == Some(true) {
            proposal.max_tokens = None;
        }
    }
    proposal
}

fn cancel_cause_of(signal: &AbortSignal) -> AgentCancelCause {
    match signal.reason() {
        Some(AbortReason::Cancelled(reason)) => {
            serde_json::from_str(&reason).unwrap_or(AgentCancelCause::User)
        }
        _ => AgentCancelCause::User,
    }
}

fn encode_cause(cause: &AgentCancelCause) -> AbortReason {
    AbortReason::Cancelled(serde_json::to_string(cause).unwrap_or_else(|_| "user".into()))
}

/// Only send a bounded real user request to the selected decision backend.
/// Tool output and plugin context are never used as the decision task.
fn latest_user_task(messages: &[Message]) -> Option<String> {
    let message = messages.iter().rev().find(|message| {
        message.role == Role::User && matches!(message.source, MessageSource::User)
    })?;
    let text: String = message
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    (!text.trim().is_empty()).then(|| text.chars().take(3_000).collect())
}

fn focus_text(id: &str) -> Option<&'static str> {
    match id {
        "inspect" => Some(
            "Inspect relevant evidence before making a change. Choose exact reads or searches yourself.",
        ),
        "implement" => Some(
            "Continue the requested implementation with the admitted file tools. Formulate exact edits yourself.",
        ),
        "verify" => Some(
            "Check the current change with focused shell commands. Existing shell permissions still govern execution.",
        ),
        "answer" => Some(
            "Respond based on verified results, or ask for missing information when necessary. Do not claim unperformed checks.",
        ),
        _ => None,
    }
}

/// The host offers only bundles it can bind to current tool schemas. Shell is
/// excluded from implementation and admitted for verification, but its
/// commands still require the existing runtime permission checks.
fn admitted_tool_names(focus: Option<&str>, tools: &[dsh_llm::ToolSchema]) -> Option<Vec<String>> {
    match focus {
        Some("inspect") => {
            let names: Vec<String> = tools
                .iter()
                .filter(|tool| matches!(tool.name.as_str(), "read" | "grep" | "glob"))
                .map(|tool| tool.name.clone())
                .collect();
            (!names.is_empty()).then_some(names)
        }
        Some("implement") => {
            let names: Vec<String> = tools
                .iter()
                .filter(|tool| {
                    matches!(
                        tool.name.as_str(),
                        "read" | "grep" | "glob" | "write" | "edit"
                    )
                })
                .map(|tool| tool.name.clone())
                .collect();
            names
                .iter()
                .any(|name| matches!(name.as_str(), "write" | "edit"))
                .then_some(names)
        }
        Some("verify") => {
            let names: Vec<String> = tools
                .iter()
                .filter(|tool| matches!(tool.name.as_str(), "read" | "grep" | "glob" | "bash"))
                .map(|tool| tool.name.clone())
                .collect();
            names.iter().any(|name| name == "bash").then_some(names)
        }
        Some("answer") => Some(Vec::new()),
        _ => None,
    }
}

const FOCUS_CHOICES: [(&str, &str); 4] = [
    (
        "inspect",
        "Read or search available evidence; no file edits or shell.",
    ),
    (
        "implement",
        "Edit files with available write/edit tools; no shell.",
    ),
    (
        "verify",
        "Run shell checks and read evidence; write/edit tools are hidden.",
    ),
    (
        "answer",
        "Respond without tools when enough evidence is available.",
    ),
];

#[derive(Clone)]
struct PreparedFocus {
    id: &'static str,
    description: &'static str,
    admitted: Vec<String>,
}

fn prepare_focus_choices(tools: &[dsh_llm::ToolSchema]) -> Vec<PreparedFocus> {
    FOCUS_CHOICES
        .iter()
        .filter_map(|(id, description)| {
            admitted_tool_names(Some(id), tools).map(|admitted| PreparedFocus {
                id,
                description,
                admitted,
            })
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FocusMode {
    Laya,
    Jev,
    Normal,
}

fn focus_mode(path: Option<&PathBuf>) -> FocusMode {
    let Some(path) = path else {
        return FocusMode::Laya;
    };
    match std::fs::read_to_string(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => FocusMode::Laya,
        Err(_) => FocusMode::Normal,
        Ok(value) => match value.trim() {
            "laya" => FocusMode::Laya,
            "jev" => FocusMode::Jev,
            _ => FocusMode::Normal,
        },
    }
}

struct FocusSelection {
    backend: DecisionBackend,
    revision: u64,
    candidates: Vec<PreparedFocus>,
    selected: Option<PreparedFocus>,
    confidence: Option<f64>,
    selected_probability: Option<f64>,
    fit: Option<f64>,
    fallback: Option<String>,
}

/// Live, nonpersistent selector activity. The bridge forwards it to Keel's
/// session status stream; it never enters the dsh session log.
pub struct DecisionActivityChanged;

impl Event for DecisionActivityChanged {
    const NAME: &'static str = "decision/activity";
    type Args = (String, Option<DecisionActivity>);
    type Ret = ();
}

struct FocusActivityGuard {
    ctx: Context,
    session_id: String,
}

impl FocusActivityGuard {
    fn start(ctx: &Context, session_id: &str, backend: DecisionBackend) -> Self {
        let session_id = session_id.to_string();
        ctx.emit::<DecisionActivityChanged>(&(
            session_id.clone(),
            Some(DecisionActivity {
                backend,
                phase: DecisionPhase::ChoosingFocus,
            }),
        ));
        Self {
            ctx: ctx.clone(),
            session_id,
        }
    }
}

impl Drop for FocusActivityGuard {
    fn drop(&mut self) {
        self.ctx
            .emit::<DecisionActivityChanged>(&(self.session_id.clone(), None));
    }
}

/// One model choice is recorded after the host's step outcome is observed.
/// A drop guard also records a bounded incomplete outcome if sampling fails.
struct FocusReceipt {
    session: Rc<Session>,
    turn: u64,
    step: u64,
    selection: FocusSelection,
    observed: String,
}

impl FocusReceipt {
    fn new(session: Rc<Session>, turn: u64, step: u64, selection: FocusSelection) -> Self {
        Self {
            session,
            turn,
            step,
            selection,
            observed: "bundle_bound_step_incomplete".into(),
        }
    }

    fn observed(&mut self, observed: impl Into<String>) {
        self.observed = observed.into();
    }
}

impl Drop for FocusReceipt {
    fn drop(&mut self) {
        let selected = self.selection.selected.as_ref();
        let mut event = DecisionEvent::new(
            format!(
                "dsh-{}-{}-{}",
                self.turn,
                self.step,
                uuid::Uuid::new_v4().simple()
            ),
            self.selection.revision,
            self.selection.backend,
            DecisionStage::PreStep,
            self.selection
                .candidates
                .iter()
                .map(|choice| DecisionCandidate {
                    id: choice.id.into(),
                    summary: choice.description.into(),
                })
                .collect(),
            match selected {
                Some(choice) => DecisionResult::Selected {
                    candidate_id: choice.id.into(),
                },
                None => DecisionResult::Abstained,
            },
            if selected.is_some() {
                DecisionValidation::Accepted
            } else {
                DecisionValidation::Rejected
            },
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_millis().min(i64::MAX as u128) as i64)
                .unwrap_or(0),
        )
        .with_observed_outcome(self.observed.clone());
        if let Some(fallback) = &self.selection.fallback {
            event = event.with_fallback(fallback.clone());
        }
        if let Some(confidence) = self.selection.confidence {
            event = event.with_confidence(confidence);
        }
        if let Some(probability) = self.selection.selected_probability {
            event = event.with_selected_probability(probability);
        }
        if let Some(fit) = self.selection.fit {
            event = event.with_fit(fit);
        }
        let Ok(data) = serde_json::to_value(event) else {
            return;
        };
        if let Err(error) = self.session.append(
            SessionEventData::Extension {
                event_type: "decision/receipt".into(),
                data,
            },
            None,
        ) {
            tracing::warn!(error = %error, "failed to record bounded decision receipt");
        }
    }
}

struct LoopState {
    ctx: Context,
    id: SessionId,
    options: AgentOptions,
    session: Rc<Session>,
    llm: Rc<LlmRuntime>,
    tools: Rc<ToolRuntime>,
    system_prompt: Rc<SystemPrompt>,
    inbox: RefCell<Option<Rc<Inbox>>>,
    phase: RefCell<Phase>,
    activity_done: RefCell<Shared<LocalBoxFuture<'static, ()>>>,
    request_header_logged: Cell<bool>,
    runtime_context: RuntimeContextProjection,
    laya: Option<LayaSelector>,
    jev: Option<JevSelector>,
    jev_budget: DecisionBudget,
    decision_mode_path: Option<PathBuf>,
    focus_mode: Cell<FocusMode>,
    self_ref: RefCell<Option<std::rc::Weak<ReactLoopAgent>>>,
}

/// Drives one session through turn and step boundaries (upstream
/// `ReactLoopAgent`).
pub struct ReactLoopAgent {
    state: Rc<LoopState>,
}

type AgentBackRef = Rc<RefCell<Option<std::rc::Weak<ReactLoopAgent>>>>;

struct LoopInbox {
    ctx: Context,
    agent: AgentBackRef,
}

impl LoopInbox {
    fn agent(&self) -> Option<AgentRef> {
        self.agent
            .borrow()
            .as_ref()
            .and_then(|weak| weak.upgrade())
            .map(|agent| agent as AgentRef)
    }
}

impl InboxNotifications for LoopInbox {
    fn inserted(&self, message: &Message) {
        if let Some(agent) = self.agent() {
            self.ctx
                .emit::<AgentInboxInserted>(&(agent, message.clone()));
        }
    }
    fn discarded(&self, message: &Message) {
        if let Some(agent) = self.agent() {
            self.ctx
                .emit::<AgentInboxDiscarded>(&(agent, message.clone()));
        }
    }
    fn claimed(&self, message: &Message, turn: u64) {
        if let Some(agent) = self.agent() {
            self.ctx
                .emit::<AgentInboxClaimed>(&(agent, message.clone(), turn));
        }
    }
}

impl ReactLoopAgent {
    /// Build the loop agent over a live session; `ctx` carries the llm,
    /// tools, and systemPrompt services.
    pub fn new(
        ctx: Context,
        id: SessionId,
        options: AgentOptions,
        session: Rc<Session>,
        llm: Rc<LlmRuntime>,
        tools: Rc<ToolRuntime>,
        system_prompt: Rc<SystemPrompt>,
    ) -> anyhow::Result<Rc<ReactLoopAgent>> {
        Self::new_with_decisions(
            ctx,
            id,
            options,
            session,
            llm,
            tools,
            system_prompt,
            None,
            None,
        )
    }

    pub fn new_with_decisions(
        ctx: Context,
        id: SessionId,
        options: AgentOptions,
        session: Rc<Session>,
        llm: Rc<LlmRuntime>,
        tools: Rc<ToolRuntime>,
        system_prompt: Rc<SystemPrompt>,
        decision_mode_path: Option<PathBuf>,
        laya: Option<LayaSelector>,
    ) -> anyhow::Result<Rc<ReactLoopAgent>> {
        let last_turn = session.with_events(|events| {
            events
                .iter()
                .rev()
                .find_map(|event| match &event.data {
                    SessionEventData::TurnStart { turn } => Some(*turn),
                    _ => None,
                })
                .unwrap_or(0)
        });
        let back_ref: AgentBackRef = Rc::default();
        let notifications = Box::new(LoopInbox {
            ctx: ctx.clone(),
            agent: back_ref.clone(),
        });
        let inbox = Rc::new(
            Inbox::new(session.clone(), notifications).map_err(|error| anyhow::anyhow!(error.0))?,
        );
        let jev = jev_core::protected_local_key_path()
            .and_then(|path| JevSelector::new(JevConfig::app_owned(path)).ok());
        let selected_mode = focus_mode(decision_mode_path.as_ref());
        let state = Rc::new(LoopState {
            ctx,
            id,
            options,
            session: session.clone(),
            llm,
            tools,
            system_prompt,
            inbox: RefCell::new(Some(inbox)),
            phase: RefCell::new(Phase::Idle { last_turn }),
            activity_done: RefCell::new(futures::future::ready(()).boxed_local().shared()),
            request_header_logged: Cell::new(false),
            runtime_context: RuntimeContextProjection::new(session),
            laya,
            jev,
            jev_budget: DecisionBudget::new(8),
            decision_mode_path,
            focus_mode: Cell::new(selected_mode),
            self_ref: RefCell::new(None),
        });
        let agent = Rc::new(ReactLoopAgent { state });
        *agent.state.self_ref.borrow_mut() = Some(Rc::downgrade(&agent));
        *back_ref.borrow_mut() = Some(Rc::downgrade(&agent));
        Ok(agent)
    }

    fn inbox(&self) -> Rc<Inbox> {
        self.state
            .inbox
            .borrow()
            .as_ref()
            .expect("inbox lives with the agent")
            .clone()
    }

    fn as_agent_ref(&self) -> AgentRef {
        self.state
            .self_ref
            .borrow()
            .as_ref()
            .and_then(|weak| weak.upgrade())
            .expect("self reference set in constructor") as AgentRef
    }

    fn set_phase(&self, next: Phase) {
        let previous = self.state.phase.borrow().status();
        *self.state.phase.borrow_mut() = next;
        let status = self.state.phase.borrow().status();
        if status != previous {
            self.state
                .ctx
                .emit::<AgentStatusChanged>(&(self.as_agent_ref(), status));
        }
    }

    /// Start one driver, or latch its wake behind maintenance or an aborted
    /// activity (upstream `wakeDriver`). A wake sent while idle always opens
    /// its turn boundary.
    fn wake_driver(&self, wake_after_abort: bool) {
        {
            let phase = self.state.phase.borrow();
            match &*phase {
                Phase::Maintenance {
                    abort,
                    wake_requested,
                    ..
                } => {
                    let disposed = matches!(
                        abort.signal().reason(),
                        Some(AbortReason::Cancelled(reason))
                            if reason.contains("disposed")
                    );
                    if !disposed {
                        wake_requested.set(true);
                    }
                    return;
                }
                Phase::Running {
                    abort,
                    wake_requested,
                    ..
                } => {
                    let disposed = matches!(
                        abort.signal().reason(),
                        Some(AbortReason::Cancelled(reason))
                            if reason.contains("disposed")
                    );
                    if !disposed && wake_after_abort {
                        wake_requested.set(true);
                    }
                    return;
                }
                Phase::Idle { .. } => {}
            }
        }
        let last_turn = match &*self.state.phase.borrow() {
            Phase::Idle { last_turn } => *last_turn,
            _ => unreachable!("checked idle above"),
        };
        self.set_phase(Phase::Running {
            abort: AbortController::new(),
            turn: Cell::new(last_turn),
            step: Cell::new(0),
            wake_requested: Cell::new(false),
        });
        let agent = self.as_agent_ref();
        let this = self
            .state
            .self_ref
            .borrow()
            .as_ref()
            .and_then(|weak| weak.upgrade())
            .expect("self ref");
        let (done_tx, done_rx) = futures::channel::oneshot::channel::<()>();
        *self.state.activity_done.borrow_mut() = async move {
            let _ = done_rx.await;
        }
        .boxed_local()
        .shared();
        tokio::task::spawn_local(async move {
            let _agent = agent;
            this.kick().await;
            let _ = done_tx.send(());
        });
    }

    /// Driver pump: turns until nothing is owed (upstream `kick`).
    async fn kick(self: &Rc<Self>) {
        loop {
            match self.turn().await {
                Ok(true) => continue,
                Ok(false) | Err(_) => break,
            }
        }
        let (last_turn, wake_requested) = {
            let phase = self.state.phase.borrow();
            match &*phase {
                Phase::Running {
                    turn,
                    wake_requested,
                    ..
                } => (turn.get(), wake_requested.get()),
                _ => return,
            }
        };
        self.set_phase(Phase::Idle { last_turn });
        if wake_requested && self.inbox().has_pending() {
            self.wake_driver(false);
        }
    }

    fn emit_error(&self, turn: u64, step: u64, error: &anyhow::Error) {
        self.state.ctx.emit::<dsh_agent::AgentError>(&(
            self.as_agent_ref(),
            turn,
            step,
            format!("{error:#}"),
        ));
    }

    /// Claim the batch, assemble the prompt, and run the pre-step waterfall
    /// (upstream `preStep`).
    async fn pre_step(
        self: &Rc<Self>,
        target: InboxTarget,
        turn: u64,
        step: u64,
        signal: &AbortSignal,
    ) -> anyhow::Result<Option<(Vec<Message>, dsh_system_prompt::PromptAssembly)>> {
        let claimed = self
            .inbox()
            .claim(target, turn)
            .map_err(|error| anyhow::anyhow!(error.0))?;
        let assembly = self
            .state
            .system_prompt
            .assemble(AssembleContext::default())
            .await?;
        anyhow::ensure!(!signal.aborted(), "aborted");
        let sections = render_context_sections(&assembly)?;
        let context = self
            .state
            .runtime_context
            .project(&join_context_sections(&sections), &sections);
        let mut proposed = claimed;
        if let Some(context) = context {
            proposed.push(context);
        }
        let decision = self
            .state
            .ctx
            .waterfall::<AgentPreStep, _, _>(
                PreStepPayload {
                    agent: self.as_agent_ref(),
                    messages: proposed,
                    turn,
                    step,
                    signal: signal.clone(),
                },
                |payload| async move {
                    Ok(PreStepDecision::Enter {
                        messages: payload.messages,
                    })
                },
            )
            .await?;
        anyhow::ensure!(!signal.aborted(), "aborted");
        Ok(match decision {
            PreStepDecision::Reject => None,
            PreStepDecision::Enter { messages } => Some((messages, assembly)),
        })
    }

    /// Open one turn and drive its steps (upstream `turn`). Returns whether
    /// pending work warrants another turn.
    async fn turn(self: &Rc<Self>) -> anyhow::Result<bool> {
        // One backend owns every decision in this turn, even if Settings
        // changes while a tool or model call is in flight.
        self.state
            .focus_mode
            .set(focus_mode(self.state.decision_mode_path.as_ref()));
        let (signal, turn) = {
            let phase = self.state.phase.borrow();
            match &*phase {
                Phase::Running { abort, turn, .. } => (abort.signal(), turn.get() + 1),
                _ => anyhow::bail!("turn without driver reservation"),
            }
        };
        anyhow::ensure!(!signal.aborted(), "aborted");
        self.state
            .session
            .append(SessionEventData::TurnStart { turn }, None)
            .map_err(|error| anyhow::anyhow!(error.0))?;
        if let Phase::Running {
            turn: turn_cell, ..
        } = &*self.state.phase.borrow()
        {
            turn_cell.set(turn);
        }

        let mut turn_ends: Option<TurnEndReason> = None;
        let mut target = InboxTarget::NextTurn;
        let outcome: anyhow::Result<()> = async {
            loop {
                anyhow::ensure!(!signal.aborted(), "aborted");
                let step = {
                    let phase = self.state.phase.borrow();
                    match &*phase {
                        Phase::Running { step, .. } => step.get() + 1,
                        _ => anyhow::bail!("step outside running phase"),
                    }
                };
                let decision = self.pre_step(target, turn, step, &signal).await?;
                let Some((messages, assembly)) = decision else {
                    turn_ends = Some(TurnEndReason::Blocked);
                    return Ok(());
                };
                if turn_ends.is_some() && messages.is_empty() {
                    return Ok(());
                }
                // A removed waking message still owns the initial turn
                // boundary but spends no model call.
                let current_step = {
                    let phase = self.state.phase.borrow();
                    match &*phase {
                        Phase::Running { step, .. } => step.get(),
                        _ => 0,
                    }
                };
                if current_step == 0 && messages.is_empty() {
                    turn_ends = Some(TurnEndReason::Completed);
                    return Ok(());
                }
                anyhow::ensure!(!signal.aborted(), "aborted");
                self.state
                    .session
                    .append(SessionEventData::StepStart { turn, step }, None)
                    .map_err(|error| anyhow::anyhow!(error.0))?;
                if let Phase::Running {
                    step: step_cell, ..
                } = &*self.state.phase.borrow()
                {
                    step_cell.set(step);
                }
                let step_outcome: anyhow::Result<Option<TurnEndReason>> = async {
                    for message in &messages {
                        self.state
                            .session
                            .append(
                                SessionEventData::UserMessage(message.clone()),
                                Some(SurfaceIntent::append()),
                            )
                            .map_err(|error| anyhow::anyhow!(error.0))?;
                    }
                    self.step(turn, step, &assembly, &signal).await
                }
                .await;
                let _ = self
                    .state
                    .session
                    .append(SessionEventData::StepEnd { turn, step }, None);
                let step_end = step_outcome?;
                // max-tokens is sticky: a later completed step must not
                // downgrade the turn outcome.
                if !matches!(turn_ends, Some(TurnEndReason::MaxTokens)) {
                    turn_ends = step_end;
                }
                anyhow::ensure!(!signal.aborted(), "aborted");
                if turn_ends.is_some() && self.inbox().next_step().is_empty() {
                    self.state
                        .ctx
                        .serial::<AgentTurnStopping>(&(self.as_agent_ref(), turn, signal.clone()))
                        .await;
                    anyhow::ensure!(!signal.aborted(), "aborted");
                }
                if turn_ends.is_some() && self.inbox().next_step().is_empty() {
                    return Ok(());
                }
                target = InboxTarget::NextStep;
            }
        }
        .await;

        if let Err(error) = &outcome {
            if signal.aborted() {
                turn_ends = Some(TurnEndReason::Aborted {
                    reason: cancel_cause_of(&signal),
                });
            } else {
                let failure = error
                    .downcast_ref::<LlmError>()
                    .map(|llm| llm.failure.clone())
                    .unwrap_or_else(|| dsh_llm::LlmFailure::new(error_chain(error), "UNKNOWN"));
                turn_ends = Some(TurnEndReason::Error { error: failure });
                let step = match &*self.state.phase.borrow() {
                    Phase::Running { step, .. } => step.get(),
                    _ => 0,
                };
                self.emit_error(turn, step, error);
            }
        }
        let _ = self.state.session.append(
            SessionEventData::TurnEnd {
                turn,
                reason: turn_ends.unwrap_or(TurnEndReason::Completed),
            },
            None,
        );
        if outcome.is_err() {
            return Ok(false);
        }
        if !self.inbox().has_pending() {
            return Ok(false);
        }
        // Fresh controller for the next turn; a latch on the old one is stale.
        if let Phase::Running {
            abort,
            step,
            wake_requested,
            ..
        } = &mut *self.state.phase.borrow_mut()
        {
            *abort = AbortController::new();
            step.set(0);
            wake_requested.set(false);
        }
        Ok(true)
    }

    /// One model call plus the tool executions it requested (upstream
    /// `step`). Returns the step-end reason, or `None` when tools owe another
    /// request.
    async fn step(
        self: &Rc<Self>,
        turn: u64,
        step: u64,
        assembly: &dsh_system_prompt::PromptAssembly,
        signal: &AbortSignal,
    ) -> anyhow::Result<Option<TurnEndReason>> {
        let mut system = render_prompt(assembly)?;
        let mut selection = self
            .select_decision_focus(turn, step, &assembly.tools, signal)
            .await;
        // Reprepare at the boundary before binding schemas. The selected ID
        // never becomes a tool name or command; only its stored bundle does.
        if let Some(choice) = selection.as_mut()
            && let Some(selected) = choice.selected.as_ref()
            && admitted_tool_names(Some(selected.id), &assembly.tools)
                != Some(selected.admitted.clone())
        {
            choice.selected = None;
            choice.fallback = Some("stale_tool_bundle".into());
        }
        let focus = selection
            .as_ref()
            .and_then(|choice| choice.selected.as_ref())
            .map(|selected| selected.id);
        let admitted = selection
            .as_ref()
            .and_then(|choice| choice.selected.as_ref())
            .map(|selected| selected.admitted.clone());
        let step_tools: Vec<dsh_llm::ToolSchema> = match &admitted {
            Some(names) => assembly
                .tools
                .iter()
                .filter(|tool| names.contains(&tool.name))
                .cloned()
                .collect(),
            None => assembly.tools.clone(),
        };
        if let Some(focus) = focus {
            // The exact schema set is committed in RequestHeader and checked
            // again at tool dispatch, including calls the model invents.
            system.push_str("\n\nHost-selected step (tool availability is enforced): ");
            system.push_str(focus_text(focus).expect("prepared focus ID"));
        }
        let mut receipt = selection
            .map(|selection| FocusReceipt::new(self.state.session.clone(), turn, step, selection));
        anyhow::ensure!(!signal.aborted(), "aborted");
        loop {
            let (request, prepared) = self
                .build_request(turn, step, &step_tools, &system, signal)
                .await?;
            let mut assembler = BlockAssembler::new();
            let mut chunk_seqs: Vec<u64> = Vec::new();
            let mut stream = match &prepared {
                Some(prepared) => prepared
                    .stream(request.clone())
                    .map_err(anyhow::Error::new)?,
                None => self.state.llm.stream(request.clone()),
            };
            anyhow::ensure!(!signal.aborted(), "aborted");
            use futures::StreamExt;
            while let Some(chunk) = stream.next().await {
                anyhow::ensure!(!signal.aborted(), "aborted");
                let event = self
                    .state
                    .session
                    .append(
                        SessionEventData::AssistantChunk {
                            turn,
                            step,
                            chunk: chunk.clone(),
                        },
                        None,
                    )
                    .map_err(|error| anyhow::anyhow!(error.0))?;
                chunk_seqs.push(event.seq);
                assembler.push(&chunk);
            }
            anyhow::ensure!(!signal.aborted(), "aborted");
            let finish = assembler.finish();
            if let FinishReason::Error { failure } | FinishReason::Aborted { failure } = &finish {
                let action = self
                    .state
                    .ctx
                    .waterfall::<AgentRequestError, _, _>(
                        RequestErrorPayload {
                            agent: self.as_agent_ref(),
                            turn,
                            step,
                            provider: request.provider.clone(),
                            failure: failure.clone(),
                            retry_policy: prepared.as_ref().map(|p| p.retry_policy.clone()),
                            signal: signal.clone(),
                        },
                        |_payload| async move { Ok(None::<RequestErrorAction>) },
                    )
                    .await?;
                anyhow::ensure!(!signal.aborted(), "aborted");
                if action != Some(RequestErrorAction::Retry) {
                    return Err(anyhow::Error::new(LlmError::new(
                        failure.message.clone(),
                        failure.code.clone(),
                    )));
                }
                continue;
            }

            let message = create_assistant_message(
                assembler.blocks(),
                AssistantProvenance {
                    provider: request.provider.clone(),
                    model: request.model.clone(),
                    replay_state: assembler.replay_state().cloned(),
                },
            );
            let calls: Vec<PlannedCall> = message
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                    } => Some(PlannedCall {
                        call_id: id.clone(),
                        name: name.clone(),
                        raw_arguments: arguments.clone(),
                    }),
                    _ => None,
                })
                .collect();
            self.state
                .session
                .append(
                    SessionEventData::AssistantMessage {
                        turn,
                        step,
                        message,
                        usage: assembler.usage(),
                    },
                    Some(dsh_session::SurfaceIntent {
                        surface_op: Some(dsh_session::SurfaceOp::append()),
                        source_event_seqs: Some(chunk_seqs),
                    }),
                )
                .map_err(|error| anyhow::anyhow!(error.0))?;
            if matches!(finish, FinishReason::MaxTokens) {
                if let Some(receipt) = &mut receipt {
                    receipt.observed("model_max_tokens");
                }
                return Ok(Some(TurnEndReason::MaxTokens));
            }
            if calls.is_empty() {
                if let Some(receipt) = &mut receipt {
                    receipt.observed("completed_without_tool_calls");
                }
                return Ok(Some(TurnEndReason::Completed));
            }
            let inbox = self.inbox();
            let outcome = execute_tool_calls(
                &self.state.tools,
                &self.as_agent_ref(),
                &self.state.session,
                turn,
                step,
                calls,
                admitted.as_deref(),
                signal,
                |context| {
                    let _ = inbox.append(InboxTarget::NextStep, context);
                },
            )
            .await;
            if let Some(receipt) = &mut receipt {
                receipt.observed(format!(
                    "tools_dispatched={},denied={},failed={},aborted={},concluded={}",
                    outcome.dispatched,
                    outcome.denied,
                    outcome.failed,
                    outcome.aborted,
                    outcome.concluded
                ));
            }
            return Ok(outcome.concluded.then_some(TurnEndReason::Completed));
        }
    }

    async fn select_decision_focus(
        &self,
        turn: u64,
        step: u64,
        tools: &[dsh_llm::ToolSchema],
        signal: &AbortSignal,
    ) -> Option<FocusSelection> {
        let mode = self.state.focus_mode.get();
        if mode == FocusMode::Normal {
            return None;
        }
        let candidates = prepare_focus_choices(tools);
        let mut selection = FocusSelection {
            backend: match mode {
                FocusMode::Laya => DecisionBackend::Laya,
                FocusMode::Jev => DecisionBackend::Jev,
                FocusMode::Normal => unreachable!(),
            },
            revision: (turn << 32) | (step & 0xffff_ffff),
            candidates,
            selected: None,
            confidence: None,
            selected_probability: None,
            fit: None,
            fallback: None,
        };
        if selection.candidates.is_empty()
            || (tools.len() > 0
                && selection.candidates.len() == 1
                && selection.candidates[0].id == "answer")
        {
            selection.fallback = Some("no_enforceable_tool_bundle".into());
            return Some(selection);
        }
        let Some(task) = latest_user_task(&self.state.session.derive_messages()) else {
            selection.fallback = Some("no_user_task".into());
            return Some(selection);
        };
        let tool_names = tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let local = mode == FocusMode::Laya;
        let input = SelectionInput {
            state: DecisionState {
                task: if local {
                    task.chars().take(900).collect()
                } else {
                    task
                },
                context: if local {
                    format!(
                        "Coding turn {turn}, step {step}. Tools: {}. Choose focus only.",
                        tool_names.chars().take(120).collect::<String>()
                    )
                } else {
                    format!(
                        "Current coding loop turn {turn}, step {step}. Available tool names: {}. Select one admitted tool bundle; do not create a command or patch.",
                        tool_names.chars().take(800).collect::<String>()
                    )
                },
                state_version: selection.revision,
            },
            candidates: selection
                .candidates
                .iter()
                .map(|choice| Candidate {
                    id: choice.id.into(),
                    description: choice.description.into(),
                })
                .collect(),
        };
        let (selected, confidence, selected_probability, fit, selector_fallback) = match mode {
            FocusMode::Laya => {
                let Some(selector) = self.state.laya.as_ref() else {
                    selection.fallback = Some("selector_unavailable".into());
                    return Some(selection);
                };
                let _activity = FocusActivityGuard::start(
                    &self.state.ctx,
                    self.state.id.as_str(),
                    selection.backend,
                );
                let outcome = tokio::select! {
                    result = selector.select(input) => result,
                    _ = signal.wait() => return None,
                };
                tracing::debug!(trace = ?outcome.trace, "Local Laya focus decision");
                (
                    outcome.selected_id,
                    outcome.trace.confidence,
                    outcome.trace.selected_probability,
                    None,
                    outcome.trace.fallback.map(|reason| format!("{reason:?}")),
                )
            }
            FocusMode::Jev => {
                let Some(selector) = self.state.jev.as_ref() else {
                    selection.fallback = Some("selector_unavailable".into());
                    return Some(selection);
                };
                let _activity = FocusActivityGuard::start(
                    &self.state.ctx,
                    self.state.id.as_str(),
                    selection.backend,
                );
                let outcome = tokio::select! {
                    result = selector.select(input, &self.state.jev_budget) => result,
                    _ = signal.wait() => return None,
                };
                tracing::debug!(trace = ?outcome.trace, "TypeSafe Jev focus decision");
                (
                    outcome.selected_id,
                    outcome.trace.confidence,
                    None,
                    outcome.trace.fit,
                    outcome.trace.fallback.map(|reason| format!("{reason:?}")),
                )
            }
            FocusMode::Normal => (None, None, None, None, None),
        };
        selection.confidence = confidence;
        selection.selected_probability = selected_probability;
        selection.fit = fit;
        selection.selected = selected
            .as_deref()
            .and_then(|id| selection.candidates.iter().find(|choice| choice.id == id))
            .cloned();
        if selection.selected.is_none() {
            selection.fallback = Some(if selected.is_some() {
                "unprepared_choice".into()
            } else {
                selector_fallback.unwrap_or_else(|| "selector_abstained".into())
            });
        }
        Some(selection)
    }

    /// Compose one request and bind it to the adapter registration that
    /// resolved its exact-model defaults (upstream `buildRequest`).
    async fn build_request(
        self: &Rc<Self>,
        turn: u64,
        step: u64,
        tools: &[dsh_llm::ToolSchema],
        system: &str,
        signal: &AbortSignal,
    ) -> anyhow::Result<(GenerateOptions, Option<dsh_llm::PreparedLlmCall>)> {
        let session = &self.state.session;
        // A loop instance starts from its declared route, restoring only an
        // explicit effort owned by that exact model; later steps re-resolve
        // marked defaults.
        let persisted = session.request_header();
        let route_provider = self.state.options.provider.clone().unwrap_or_default();
        let route_model = self.state.options.model.clone().unwrap_or_default();
        let seed_config = if self.state.request_header_logged.get() {
            persisted.as_ref().map(request_proposal).unwrap_or_default()
        } else {
            let reasoning_effort = persisted
                .as_ref()
                .filter(|header| {
                    header.config.provider == route_provider
                        && header.config.model == route_model
                        && header.adapter_defaults.and_then(|d| d.reasoning_effort) != Some(true)
                })
                .and_then(|header| header.config.reasoning_effort.clone());
            LlmCallConfig {
                provider: route_provider,
                model: route_model,
                reasoning_effort,
                max_tokens: self.state.options.max_tokens,
                ..Default::default()
            }
        };
        let proposed = self
            .state
            .ctx
            .waterfall::<AgentRequest, _, _>(
                RequestPayload {
                    agent: self.as_agent_ref(),
                    turn,
                    step,
                    signal: signal.clone(),
                },
                {
                    let seed = seed_config.clone();
                    |_payload| async move { Ok(seed) }
                },
            )
            .await?;
        anyhow::ensure!(!signal.aborted(), "aborted");
        anyhow::ensure!(
            !proposed.provider.is_empty() && !proposed.model.is_empty(),
            "agent \"{}\" has no provider/model: set AgentOptions.provider and AgentOptions.model \
             or supply both via the agent/request waterfall",
            self.state.id.as_str()
        );
        let (config, prepared) = match self
            .state
            .llm
            .prepare_call(&proposed, Some(signal.clone()))
            .await
        {
            Ok(prepared) => (prepared.config.clone(), Some(prepared)),
            // Middleware may serve an unregistered route; terminal dispatch
            // still requires an adapter.
            Err(error) if error.code == "NO_ADAPTER" => (proposed, None),
            Err(error) => return Err(anyhow::Error::new(error)),
        };
        anyhow::ensure!(!signal.aborted(), "aborted");

        let header = canonical_header(&EpochHeader {
            config: config.clone(),
            adapter_defaults: prepared.as_ref().map(|p| p.adapter_defaults),
            system: (!system.is_empty()).then(|| system.to_string()),
            tools: (!tools.is_empty()).then(|| tools.to_vec()),
        });
        let baseline = session.request_header();
        if !self.state.request_header_logged.get() {
            let reason = if baseline.is_none() {
                RequestHeaderReason::Initial
            } else {
                RequestHeaderReason::Resume
            };
            session
                .append(
                    SessionEventData::RequestHeader {
                        header: header.clone(),
                        reason,
                    },
                    None,
                )
                .map_err(|error| anyhow::anyhow!(error.0))?;
            self.state.request_header_logged.set(true);
        } else if baseline
            .as_ref()
            .map(|baseline| !header_equals(baseline, &header))
            .unwrap_or(true)
        {
            session
                .append(
                    SessionEventData::RequestHeader {
                        header: header.clone(),
                        reason: RequestHeaderReason::Change,
                    },
                    None,
                )
                .map_err(|error| anyhow::anyhow!(error.0))?;
        }

        let request_context = RequestContext {
            provider: config.provider.clone(),
            model: config.model.clone(),
            context_window: prepared
                .as_ref()
                .and_then(|p| p.context.as_ref())
                .map(|c| c.context_window),
        };
        let previous = session.request_context();
        if previous.as_ref() != Some(&request_context) {
            session
                .append(SessionEventData::RequestContext(request_context), None)
                .map_err(|error| anyhow::anyhow!(error.0))?;
        }
        anyhow::ensure!(!signal.aborted(), "aborted");

        let mut request = GenerateOptions {
            messages: session.derive_messages(),
            system: header.system.clone(),
            tools: header.tools.clone(),
            session_id: Some(self.state.id.clone()),
            signal: Some(signal.clone()),
            ..Default::default()
        };
        header.config.apply_to(&mut request);
        Ok((dsh_llm::mark_agent_loop_request(request), prepared))
    }
}

#[cfg(test)]
mod jev_focus_tests {
    use super::*;
    use dsh_llm::{create_message, create_user_message};

    #[test]
    fn live_focus_activity_clears_on_drop_without_a_session_event() {
        dsh_cordis::run(async {
            let app = dsh_cordis::App::new();
            let ctx = app.root();
            let seen: Rc<RefCell<Vec<(String, Option<DecisionActivity>)>>> = Rc::default();
            let observed = seen.clone();
            ctx.on::<DecisionActivityChanged, _, _>(Default::default(), move |_ctx, activity| {
                observed.borrow_mut().push(activity.clone());
                async { None }
            })
            .unwrap();
            let session = Session::create(SessionId::new("focus-live"), vec![], None).unwrap();
            let before = session.events().len();
            {
                let _activity = FocusActivityGuard::start(&ctx, "focus-live", DecisionBackend::Jev);
                assert_eq!(
                    seen.borrow().as_slice(),
                    &[(
                        "focus-live".into(),
                        Some(DecisionActivity {
                            backend: DecisionBackend::Jev,
                            phase: DecisionPhase::ChoosingFocus,
                        })
                    )]
                );
            }
            assert_eq!(seen.borrow()[1], ("focus-live".into(), None));
            assert_eq!(session.events().len(), before);
        });
    }

    #[test]
    fn one_explicit_mode_selects_one_backend_for_a_turn() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mode");
        assert_eq!(focus_mode(Some(&path)), FocusMode::Laya);
        std::fs::write(&path, "jev\n").unwrap();
        assert_eq!(focus_mode(Some(&path)), FocusMode::Jev);
        std::fs::write(&path, "laya\n").unwrap();
        assert_eq!(focus_mode(Some(&path)), FocusMode::Laya);
        std::fs::write(&path, "normal\n").unwrap();
        assert_eq!(focus_mode(Some(&path)), FocusMode::Normal);
    }

    #[test]
    fn type_safe_state_uses_only_bounded_real_user_text() {
        let user = create_user_message(
            vec![ContentBlock::Text {
                text: "fix the failing test".into(),
            }],
            MessageSource::User,
        );
        let plugin = create_message(
            Role::User,
            vec![ContentBlock::Text {
                text: "private plugin context".into(),
            }],
            MessageSource::Plugin {
                plugin: "test".into(),
                form: None,
            },
        );
        assert_eq!(
            latest_user_task(&[user, plugin]),
            Some("fix the failing test".into())
        );
        let long = create_user_message(
            vec![ContentBlock::Text {
                text: "x".repeat(4_000),
            }],
            MessageSource::User,
        );
        assert_eq!(latest_user_task(&[long]).unwrap().chars().count(), 3_000);
    }

    #[test]
    fn unprepared_action_id_cannot_become_guidance() {
        assert!(focus_text("implement").is_some());
        assert_eq!(focus_text("bash -c rm"), None);
        assert_eq!(focus_text("escalate"), None);
    }

    #[test]
    fn selected_bundles_admit_only_current_named_tools() {
        let tools = ["read", "grep", "glob", "write", "edit", "bash", "unknown"]
            .into_iter()
            .map(|name| dsh_llm::ToolSchema {
                name: name.into(),
                description: String::new(),
                parameters: Default::default(),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            admitted_tool_names(Some("inspect"), &tools),
            Some(vec!["read".into(), "grep".into(), "glob".into()])
        );
        assert_eq!(admitted_tool_names(Some("answer"), &tools), Some(vec![]));
        assert_eq!(
            admitted_tool_names(Some("implement"), &tools),
            Some(vec![
                "read".into(),
                "grep".into(),
                "glob".into(),
                "write".into(),
                "edit".into()
            ])
        );
        assert_eq!(
            admitted_tool_names(Some("verify"), &tools),
            Some(vec![
                "read".into(),
                "grep".into(),
                "glob".into(),
                "bash".into()
            ])
        );
        assert_eq!(admitted_tool_names(Some("inspect"), &tools[3..]), None);
        assert_eq!(admitted_tool_names(Some("implement"), &tools[..3]), None);
        assert_eq!(admitted_tool_names(Some("verify"), &tools[..5]), None);
        assert_eq!(
            prepare_focus_choices(&tools[..3])
                .iter()
                .map(|choice| choice.id)
                .collect::<Vec<_>>(),
            vec!["inspect", "answer"]
        );
    }

    #[test]
    fn one_pre_step_receipt_records_selected_bundle_after_observed_outcome() {
        let session = Session::create(SessionId::new("receipt"), vec![], None).unwrap();
        let tools = ["read", "write", "bash"]
            .into_iter()
            .map(|name| dsh_llm::ToolSchema {
                name: name.into(),
                description: String::new(),
                parameters: Default::default(),
            })
            .collect::<Vec<_>>();
        let candidates = prepare_focus_choices(&tools);
        let selected = candidates
            .iter()
            .find(|candidate| candidate.id == "implement")
            .cloned();
        {
            let mut receipt = FocusReceipt::new(
                session.clone(),
                2,
                3,
                FocusSelection {
                    backend: DecisionBackend::Jev,
                    revision: (2 << 32) | 3,
                    candidates,
                    selected,
                    confidence: Some(0.9),
                    selected_probability: None,
                    fit: Some(0.95),
                    fallback: None,
                },
            );
            receipt.observed("tools_dispatched=1,denied=0,failed=0,aborted=0,concluded=false");
        }
        let receipts = session
            .events()
            .into_iter()
            .filter_map(|event| match event.data {
                SessionEventData::Extension { event_type, data }
                    if event_type == "decision/receipt" =>
                {
                    serde_json::from_value::<DecisionEvent>(data).ok()
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].stage, DecisionStage::PreStep);
        assert_eq!(
            receipts[0].result,
            DecisionResult::Selected {
                candidate_id: "implement".into()
            }
        );
        assert_eq!(receipts[0].confidence, Some(0.9));
        assert_eq!(receipts[0].fit, Some(0.95));
        assert_eq!(
            receipts[0].observed_outcome.as_deref(),
            Some("tools_dispatched=1,denied=0,failed=0,aborted=0,concluded=false")
        );
    }
}

impl Agent for ReactLoopAgent {
    fn id(&self) -> SessionId {
        self.state.id.clone()
    }

    fn options(&self) -> AgentOptions {
        self.state.options.clone()
    }

    fn session(&self) -> Rc<Session> {
        self.state.session.clone()
    }

    fn status(&self) -> AgentStatus {
        self.state.phase.borrow().status()
    }

    fn ctx(&self) -> Context {
        self.state.ctx.clone()
    }

    fn cancel(&self, cause: AgentCancelCause, options: CancelOptions) {
        if !options.keep_inbox {
            let _ = self.inbox().clear();
            match &*self.state.phase.borrow() {
                Phase::Maintenance { wake_requested, .. }
                | Phase::Running { wake_requested, .. } => wake_requested.set(false),
                Phase::Idle { .. } => {}
            }
        }
        match &*self.state.phase.borrow() {
            Phase::Maintenance { abort, .. } | Phase::Running { abort, .. } => {
                abort.abort(encode_cause(&cause));
            }
            Phase::Idle { .. } => {}
        }
    }

    fn when_idle(&self) -> LocalBoxFuture<'static, ()> {
        let state = self.state.clone();
        async move {
            // Follow replacement work started before the observed activity
            // retired: keep waiting while the stored activity changes.
            loop {
                let observed = state.activity_done.borrow().clone();
                observed.clone().await;
                let current = state.activity_done.borrow().clone();
                if observed.ptr_eq(&current) {
                    break;
                }
            }
        }
        .boxed_local()
    }

    fn send(&self, message: Message, target: InboxTarget, wakeup: bool) {
        // Waking input cannot join an aborted activity: classify before the
        // insertion so a reentrant cancel cannot reclassify it.
        let waking_after_abort = wakeup
            && match &*self.state.phase.borrow() {
                Phase::Idle { .. } => false,
                Phase::Maintenance { abort, .. } | Phase::Running { abort, .. } => {
                    abort.signal().aborted()
                }
            };
        let resolved = if waking_after_abort {
            InboxTarget::NextTurn
        } else {
            target
        };
        let _ = self.inbox().append(resolved, message);
        if wakeup {
            self.wake_driver_entry(waking_after_abort);
        }
    }

    fn followup(&self, message: Message) {
        self.send(message, InboxTarget::NextTurn, true);
    }

    fn steer(&self, message: Message) {
        self.send(message, InboxTarget::NextStep, true);
    }

    fn inject(&self, message: Message) {
        self.send(message, InboxTarget::NextStep, false);
    }
}

impl ReactLoopAgent {
    fn wake_driver_entry(&self, wake_after_abort: bool) {
        self.wake_driver(wake_after_abort);
    }

    /// Run one non-turn maintenance task from the true idle phase (upstream
    /// `runMaintenance`): the task claims the phase synchronously, public
    /// status stays idle, waking input latches until it settles, and a
    /// latched wake with pending work restarts the driver afterwards.
    pub async fn run_maintenance<T>(
        self: &Rc<Self>,
        task: impl AsyncFnOnce(AbortSignal) -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        let (signal, last_turn) = {
            let mut phase = self.state.phase.borrow_mut();
            let Phase::Idle { last_turn } = &*phase else {
                anyhow::bail!(
                    "agent \"{}\" already has active work",
                    self.state.id.as_str()
                );
            };
            let last_turn = *last_turn;
            let abort = AbortController::new();
            let signal = abort.signal();
            *phase = Phase::Maintenance {
                abort,
                wake_requested: Cell::new(false),
            };
            (signal, last_turn)
        };
        let (done_tx, done_rx) = futures::channel::oneshot::channel::<()>();
        *self.state.activity_done.borrow_mut() = async move {
            let _ = done_rx.await;
        }
        .boxed_local()
        .shared();
        let result = task(signal).await;
        let wake_requested = match &*self.state.phase.borrow() {
            Phase::Maintenance { wake_requested, .. } => wake_requested.get(),
            _ => false,
        };
        self.set_phase(Phase::Idle { last_turn });
        let _ = done_tx.send(());
        if wake_requested && self.inbox().has_pending() {
            self.wake_driver(false);
        }
        result
    }
}
