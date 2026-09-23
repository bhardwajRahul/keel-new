//! Rust port of `@deepseek-ai/dsh-agent` (`packages/core/agent`): the public
//! agent trait and event vocabulary, the live registry with factory
//! delegation and initiator attribution, the durable inbox projection, the
//! consumed-work fold, and model selection.
//!
//! Concrete agent creation and driving belong to the loop crate; this crate
//! owns the contracts every extension point programs against.

mod consumed_work;
mod inbox;
mod registry;
mod types;

pub use consumed_work::{ConsumedWork, fold_consumed_work};
pub use inbox::{Inbox, InboxError, InboxNotifications, SilentNotifications};
pub use registry::{
    AgentFactory, AgentHandle, AgentRegistry, AgentRegistryError, CreateAgentOptions,
    ResumeAgentOptions,
};
pub use types::{
    Agent, AgentCreated, AgentDisposed, AgentError, AgentInboxClaimed, AgentInboxDiscarded,
    AgentInboxInserted, AgentOptions, AgentPreStep, AgentRef, AgentRequest, AgentRequestError,
    AgentSessionStart, AgentStatus, AgentStatusChanged, AgentTurnStopping, CancelOptions,
    INBOX_SPLICED_EVENT, InboxSplice, InboxTarget, ModelSelection, PreStepDecision, PreStepPayload,
    RequestErrorAction, RequestErrorPayload, RequestPayload, SessionStartSource, SpliceOutcome,
};

pub use dsh_session::AgentCancelCause;
