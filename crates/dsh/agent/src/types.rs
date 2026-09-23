//! Agent vocabulary and events, ported from
//! `packages/core/agent/src/{types,runtime-types}.ts`.
//!
//! Divergences:
//! - The `agent/inbox/spliced` session event lands in
//!   `dsh_session::SessionEventData::Extension` on the wire (the session
//!   crate's core vocabulary is closed); [`InboxSplice`] provides the typed
//!   encode/decode over that extension payload.
//! - Scope-filtered dispatch is deferred to the loop tier: events carry the
//!   agent in the payload, and listeners filter by id until dsh-scope wiring
//!   lands.

use dsh_llm::{LlmCallConfig, LlmFailure, Message, ReasoningEffortId, ResolvedRetryPolicy};
use dsh_session::{AgentCancelCause, SessionEvent, SessionEventData, SessionId};
use serde::{Deserialize, Serialize};
use std::rc::Rc;

/// Merge-extensible agent creation options (upstream `AgentOptions`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentOptions {
    /// Provider route (must have a registered adapter at call time).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Model id interpreted by the selected provider adapter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Maximum output tokens for each conversation-model request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
}

/// An agent's lifecycle state: `Idle` means no driver is active; `Running`
/// lasts while the driver drains, closes, or checkpoints turns. Disposal is
/// not a third status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AgentStatus {
    Idle,
    Running,
}

/// One of the two ordered pending-message lists owned by an agent
/// (upstream `InboxTarget`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InboxTarget {
    NextTurn,
    NextStep,
}

/// Whether and with which messages the loop enters a proposed step
/// (upstream `PreStepDecision`).
#[derive(Debug, Clone, PartialEq)]
pub enum PreStepDecision {
    Reject,
    Enter { messages: Vec<Message> },
}

/// Action returned by a listener that owns model-request recovery
/// (upstream `RequestErrorAction`; `None` leaves the failure terminal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestErrorAction {
    Retry,
}

/// Why a session lifecycle began (upstream `SessionStartSource`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionStartSource {
    Startup,
    Resume,
    Clear,
    Compact,
}

/// Options for `Agent::cancel` (upstream `CancelOptions`).
#[derive(Debug, Clone, Copy, Default)]
pub struct CancelOptions {
    /// Preserve queued and steering inbox items instead of discarding them;
    /// the active turn is still aborted.
    pub keep_inbox: bool,
}

/// Public live-agent handle (upstream `Agent` interface). The concrete
/// implementation ships with the agent-loop crate; the registry and every
/// extension point program against this trait.
pub trait Agent: 'static {
    /// The single identity shared with the session.
    fn id(&self) -> SessionId;
    /// The provider route and model this agent's requests use.
    fn options(&self) -> AgentOptions;
    /// The live session this agent drives; its log is the durable source of
    /// truth.
    fn session(&self) -> Rc<dsh_session::Session>;
    /// The current lifecycle state.
    fn status(&self) -> AgentStatus;
    /// Agent-scoped context; contributions unwind on disposal.
    fn ctx(&self) -> dsh_cordis::Context;
    /// Clear pending work (unless `keep_inbox`) and abort the active turn or
    /// between-turn task; the first cause wins. A no-op without activity.
    fn cancel(&self, cause: AgentCancelCause, options: CancelOptions);
    /// Resolve after the current whole-agent activity reaches quiescence.
    fn when_idle(&self) -> futures::future::LocalBoxFuture<'static, ()>;
    /// Route identified input to an inbox boundary, optionally waking the
    /// driver.
    fn send(&self, message: Message, target: InboxTarget, wakeup: bool);
    /// Queue an ordinary follow-up turn and wake the driver.
    fn followup(&self, message: Message);
    /// Submit steering for the nearest step.
    fn steer(&self, message: Message);
    /// Queue model-facing context for the next pre-step without waking.
    fn inject(&self, message: Message);
}

/// Shared, clonable agent handle.
pub type AgentRef = Rc<dyn Agent>;

/// One normalized mutation of an agent's durable pending-message lists
/// (upstream session event `agent/inbox/spliced`). Live dispatch precedes
/// projection mutation, so synchronous observers may read the pre-splice
/// inbox to recover the removed messages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InboxSplice {
    pub target: InboxTarget,
    pub start: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub removed_count: Option<u64>,
    pub inserted: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<SpliceOutcome>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SpliceOutcome {
    Canceled,
}

/// The extension event type string for inbox splices.
pub const INBOX_SPLICED_EVENT: &str = "agent/inbox/spliced";

impl InboxSplice {
    /// Encode as the session extension event payload.
    pub fn to_event_data(&self) -> SessionEventData {
        SessionEventData::Extension {
            event_type: INBOX_SPLICED_EVENT.to_string(),
            data: serde_json::to_value(self).expect("inbox splice serializes"),
        }
    }

    /// Decode from a session event when it is an inbox splice.
    pub fn from_event(event: &SessionEvent) -> Option<InboxSplice> {
        match &event.data {
            SessionEventData::Extension { event_type, data }
                if event_type == INBOX_SPLICED_EVENT =>
            {
                serde_json::from_value(data.clone()).ok()
            }
            _ => None,
        }
    }
}

/// Complete provider, model, and optional reasoning effort selected for one
/// live agent (upstream `ModelSelection`).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSelection {
    pub provider: String,
    pub model: String,
    pub reasoning_effort: Option<ReasoningEffortId>,
}

impl ModelSelection {
    /// Apply this selection onto a resolved call config: provider/model always
    /// switch, and an absent selected effort clears any inherited effort
    /// (restoring the model's provider/default behavior) — upstream
    /// `installModelSelection`'s request-side rule.
    pub fn apply_to(&self, mut config: LlmCallConfig) -> LlmCallConfig {
        config.provider = self.provider.clone();
        config.model = self.model.clone();
        config.reasoning_effort = self.reasoning_effort.clone();
        config
    }
}

// ---- live-runtime events (upstream cordis Events declarations) ----

macro_rules! agent_event {
    ($(#[$doc:meta])* $name:ident, $event:literal, $args:ty, $ret:ty) => {
        $(#[$doc])*
        pub struct $name;
        impl dsh_cordis::Event for $name {
            const NAME: &'static str = $event;
            type Args = $args;
            type Ret = $ret;
        }
    };
}

agent_event!(
    /// A fully configured agent and live session were published.
    AgentCreated, "agent/created", AgentRef, ());
agent_event!(
    /// An agent left the registry, after driver quiescence, before session
    /// detachment.
    AgentDisposed, "agent/disposed", AgentRef, ());
agent_event!(
    /// Agent status changed (`idle` ⇄ `running`).
    AgentStatusChanged, "agent/status", (AgentRef, AgentStatus), ());
agent_event!(
    /// One message entered the live inbox.
    AgentInboxInserted, "agent/inbox/inserted", (AgentRef, Message), ());
agent_event!(
    /// One message left the inbox inside its open turn (a rejected step's
    /// claim ends here — neither discarded nor re-emitted).
    AgentInboxClaimed, "agent/inbox/claimed", (AgentRef, Message, u64), ());
agent_event!(
    /// One message was discarded from the live inbox.
    AgentInboxDiscarded, "agent/inbox/discarded", (AgentRef, Message), ());
agent_event!(
    /// The session lifecycle began, once before the first turn; seed
    /// model-facing context with `agent.inject()` here.
    AgentSessionStart, "agent/session-start", (AgentRef, SessionStartSource), ());
agent_event!(
    /// A step or turn errored (also for failures with no in-turn position).
    AgentError, "agent/error", (AgentRef, u64, u64, String), ());

/// Waterfall: reject a proposed step or replace the messages that enter it;
/// calling `next` preserves the current messages (upstream `agent/pre-step`).
pub struct AgentPreStep;
impl dsh_cordis::Event for AgentPreStep {
    const NAME: &'static str = "agent/pre-step";
    type Args = PreStepPayload;
    type Ret = PreStepDecision;
}

pub struct PreStepPayload {
    pub agent: AgentRef,
    pub messages: Vec<Message>,
    pub turn: u64,
    pub step: u64,
    pub signal: dsh_timeout::AbortSignal,
}

/// Waterfall: replace the frozen call configuration for one step's request
/// (upstream `agent/request`). Model-visible content must use logged
/// channels; this cannot mutate messages.
pub struct AgentRequest;
impl dsh_cordis::Event for AgentRequest {
    const NAME: &'static str = "agent/request";
    type Args = RequestPayload;
    type Ret = LlmCallConfig;
}

pub struct RequestPayload {
    pub agent: AgentRef,
    pub turn: u64,
    pub step: u64,
    pub signal: dsh_timeout::AbortSignal,
}

/// Waterfall: handle one failed model-request attempt before the loop retries
/// or closes its step; return `Some(Retry)` without calling `next` to own
/// recovery (upstream `agent/request-error`).
pub struct AgentRequestError;
impl dsh_cordis::Event for AgentRequestError {
    const NAME: &'static str = "agent/request-error";
    type Args = RequestErrorPayload;
    type Ret = Option<RequestErrorAction>;
}

pub struct RequestErrorPayload {
    pub agent: AgentRef,
    pub turn: u64,
    pub step: u64,
    pub provider: String,
    pub failure: LlmFailure,
    pub retry_policy: Option<ResolvedRetryPolicy>,
    pub signal: dsh_timeout::AbortSignal,
}

/// Serial: the turn is about to close (no owed response). A listener that
/// objects steers and the machine re-reads its inbox — data decides, so
/// listener order cannot change the outcome (upstream `agent/turn-stopping`).
pub struct AgentTurnStopping;
impl dsh_cordis::Event for AgentTurnStopping {
    const NAME: &'static str = "agent/turn-stopping";
    type Args = (AgentRef, u64, dsh_timeout::AbortSignal);
    type Ret = ();
}
