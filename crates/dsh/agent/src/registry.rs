//! The agent registry service, ported from
//! `packages/core/agent/src/index.ts`: tracks live agents, delegates
//! creation to the loop-provided factory, and carries the initiating agent
//! through one process-local asynchronous driver chain.
//!
//! Divergences:
//! - Node `AsyncLocalStorage` → `tokio::task_local!` for the async initiator
//!   chain, plus a synchronous nesting stack for sync operations. Both share
//!   the same closing/disposed gate.
//! - The typert wire-lookup registration is deferred to the API tier.
//! - `announce` veto semantics: the port's event bus has no synchronous-throw
//!   channel, so a failing `agent/created` listener is contained (upstream
//!   contains async rejections the same way).

use crate::types::{AgentCreated, AgentDisposed, AgentOptions, AgentRef, SessionStartSource};
use dsh_cordis::{Context, Service};
use dsh_session::{SessionEvent, SessionId, SessionMeta};
use futures::future::LocalBoxFuture;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct AgentRegistryError(pub String);

const NO_FACTORY_MESSAGE: &str = "no agent factory registered (load an agent-loop plugin)";
const NO_INITIATOR_MESSAGE: &str = "no initiating agent is active";
const DISPOSED_INITIATOR_MESSAGE: &str = "agent initiator scope is disposed";

/// Options for creating an agent through the registry factory (upstream
/// `CreateAgentOptions`, minus the setup callback, which the factory models
/// directly).
pub struct CreateAgentOptions {
    /// The live agent/session identity.
    pub session_id: SessionId,
    /// Durable session creation metadata (validated at the session boundary).
    pub meta: SessionMeta,
    /// Initial replay/fork history: a balanced completed-turn prefix,
    /// contiguous from seq 0.
    pub seed: Vec<SessionEvent>,
    /// Per-agent options (model, …).
    pub agent_options: AgentOptions,
}

/// Options for resuming an agent on a persisted session (upstream
/// `ResumeAgentOptions`).
pub struct ResumeAgentOptions {
    /// The persisted session id to load and use as the live identity.
    pub resume_session_id: SessionId,
    pub agent_options: AgentOptions,
}

/// An owned agent plus its disposer (upstream `AgentHandle`). The disposer is
/// a capability: only the holder can tear this agent down. Dispose stops the
/// loop, awaits its exit, unregisters the agent, removes its session, and
/// unwinds its scoped world.
pub struct AgentHandle {
    pub agent: AgentRef,
    dispose: Box<dyn FnOnce() -> LocalBoxFuture<'static, ()>>,
}

impl AgentHandle {
    pub fn new(
        agent: AgentRef,
        dispose: impl FnOnce() -> LocalBoxFuture<'static, ()> + 'static,
    ) -> AgentHandle {
        AgentHandle {
            agent,
            dispose: Box::new(dispose),
        }
    }

    pub async fn dispose(self) {
        (self.dispose)().await;
    }
}

/// The agent-creation factory the loop provides via
/// [`AgentRegistry::set_factory`] (upstream `AgentFactory`). Consumers
/// program against `ctx.agents` without depending on the loop crate.
pub trait AgentFactory: 'static {
    /// Create a new agent on a caller-supplied session id: await setup while
    /// unpublished, insert and announce session then agent, emit
    /// `agent/session-start`, then start the loop — rollback-covered.
    fn create_agent(
        &self,
        owner_ctx: Context,
        options: CreateAgentOptions,
    ) -> LocalBoxFuture<'static, anyhow::Result<AgentHandle>>;

    /// Prepare a persisted session and resume an agent on it; same ordered
    /// publication boundary as `create_agent`.
    fn resume(
        &self,
        owner_ctx: Context,
        options: ResumeAgentOptions,
    ) -> LocalBoxFuture<'static, anyhow::Result<AgentHandle>>;
}

struct AgentEntry {
    agent: AgentRef,
    /// Runtime creator-agent ownership; independent of durable lineage.
    owner: Option<SessionId>,
    announced: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum InitiatorState {
    Active,
    Closing,
    Disposed,
}

tokio::task_local! {
    static INITIATOR: Option<SessionId>;
}

/// Agent service (`ctx.agents`): live registry + factory delegation +
/// process-local initiator attribution. Ambient initiator presence is causal
/// attribution only — neither liveness proof nor authorization.
pub struct AgentRegistry {
    ctx: Context,
    store: RefCell<HashMap<SessionId, AgentEntry>>,
    order: RefCell<Vec<SessionId>>,
    factory: RefCell<Option<Rc<dyn AgentFactory>>>,
    initiator_state: Cell<InitiatorState>,
    /// Synchronous nesting stack for `with_initiator_sync`.
    sync_initiators: RefCell<Vec<Option<SessionId>>>,
}

impl Service for AgentRegistry {
    const NAME: &'static str = "agents";
}

impl AgentRegistry {
    /// Create and register the service in `ctx`.
    pub fn provide(ctx: &Context) -> dsh_cordis::Result<Rc<AgentRegistry>> {
        let registry = Rc::new(AgentRegistry {
            ctx: ctx.clone(),
            store: RefCell::new(HashMap::new()),
            order: RefCell::new(Vec::new()),
            factory: RefCell::new(None),
            initiator_state: Cell::new(InitiatorState::Active),
            sync_initiators: RefCell::new(Vec::new()),
        });
        ctx.provide_service(registry.clone())?;
        Ok(registry)
    }

    /// Register the agent-creation factory (the loop calls this on
    /// construction). Fails if one is already registered; the returned effect
    /// clears the slot on dispose.
    pub fn set_factory(
        self: &Rc<Self>,
        factory: Rc<dyn AgentFactory>,
    ) -> Result<dsh_cordis::EffectHandle, AgentRegistryError> {
        if self.factory.borrow().is_some() {
            return Err(AgentRegistryError(
                "an agent factory is already registered".into(),
            ));
        }
        *self.factory.borrow_mut() = Some(factory);
        let registry = self.clone();
        self.ctx
            .effect_labeled("agents.setFactory()", move |_| {
                Ok(dsh_cordis::Effect::One(dsh_cordis::Disposer::sync(
                    move || {
                        *registry.factory.borrow_mut() = None;
                    },
                )))
            })
            .map_err(|error| AgentRegistryError(error.to_string()))
    }

    fn require_factory(&self) -> Result<Rc<dyn AgentFactory>, AgentRegistryError> {
        self.factory
            .borrow()
            .clone()
            .ok_or_else(|| AgentRegistryError(NO_FACTORY_MESSAGE.into()))
    }

    /// Create and publish a new agent through the registered factory.
    pub async fn create(&self, options: CreateAgentOptions) -> anyhow::Result<AgentHandle> {
        let factory = self.require_factory()?;
        factory.create_agent(self.ctx.clone(), options).await
    }

    /// Load a persisted session and resume an agent through the factory.
    pub async fn resume(&self, options: ResumeAgentOptions) -> anyhow::Result<AgentHandle> {
        let factory = self.require_factory()?;
        factory.resume(self.ctx.clone(), options).await
    }

    /// Register a live agent: enter + announce, unregistered when the calling
    /// fiber unloads (upstream `register`).
    pub fn register(
        self: &Rc<Self>,
        agent: AgentRef,
    ) -> Result<dsh_cordis::EffectHandle, AgentRegistryError> {
        let handle = self.enter(agent.clone(), None)?;
        self.announce(&agent)?;
        Ok(handle)
    }

    /// Insert an already-constructed agent without announcing it — the
    /// ordered-lifecycle primitive the async factory uses (upstream `enter`).
    pub fn enter(
        self: &Rc<Self>,
        agent: AgentRef,
        owner: Option<SessionId>,
    ) -> Result<dsh_cordis::EffectHandle, AgentRegistryError> {
        let id = agent.id();
        if id != agent.session().id() {
            return Err(AgentRegistryError(format!(
                "agent id \"{}\" does not match session id \"{}\"",
                id.as_str(),
                agent.session().id().as_str()
            )));
        }
        if self.store.borrow().contains_key(&id) {
            return Err(AgentRegistryError(format!(
                "agent \"{}\" is already registered",
                id.as_str()
            )));
        }
        self.store.borrow_mut().insert(
            id.clone(),
            AgentEntry {
                agent,
                owner,
                announced: false,
            },
        );
        self.order.borrow_mut().push(id.clone());
        let registry = self.clone();
        self.ctx
            .effect_labeled("agents.register()", move |_| {
                Ok(dsh_cordis::Effect::One(dsh_cordis::Disposer::sync(
                    move || {
                        registry.detach(&id);
                    },
                )))
            })
            .map_err(|error| AgentRegistryError(error.to_string()))
    }

    fn detach(&self, id: &SessionId) {
        let entry = self.store.borrow_mut().remove(id);
        self.order.borrow_mut().retain(|existing| existing != id);
        if let Some(entry) = entry {
            // An insertion rolled back before announce was never externally
            // created; emitting disposed would invent a lifecycle edge.
            if entry.announced {
                self.ctx.emit::<AgentDisposed>(&entry.agent);
            }
        }
    }

    /// Announce an agent previously inserted with [`AgentRegistry::enter`].
    pub fn announce(&self, agent: &AgentRef) -> Result<(), AgentRegistryError> {
        let id = agent.id();
        let mut store = self.store.borrow_mut();
        let entry = store.get_mut(&id).ok_or_else(|| {
            AgentRegistryError(format!(
                "agent \"{}\" is not live in this registry",
                id.as_str()
            ))
        })?;
        if entry.announced {
            return Err(AgentRegistryError(format!(
                "agent \"{}\" was already announced",
                id.as_str()
            )));
        }
        entry.announced = true;
        let agent = entry.agent.clone();
        drop(store);
        self.ctx.emit::<AgentCreated>(&agent);
        Ok(())
    }

    /// Emit `agent/session-start` for one live agent (the factory calls this
    /// once before the first turn).
    pub fn emit_session_start(&self, agent: &AgentRef, source: SessionStartSource) {
        self.ctx
            .emit::<crate::types::AgentSessionStart>(&(agent.clone(), source));
    }

    /// Look up a live agent.
    pub fn get(&self, id: &SessionId) -> Option<AgentRef> {
        self.store.borrow().get(id).map(|entry| entry.agent.clone())
    }

    /// Whether a live agent was created through one exact parent agent's
    /// scoped context (runtime ownership, not durable lineage).
    pub fn is_owned_by(&self, id: &SessionId, owner: &SessionId) -> bool {
        self.store
            .borrow()
            .get(id)
            .map(|entry| entry.owner.as_ref() == Some(owner))
            .unwrap_or(false)
    }

    /// All live agents, in registration order.
    pub fn list(&self) -> Vec<AgentRef> {
        let store = self.store.borrow();
        self.order
            .borrow()
            .iter()
            .filter_map(|id| store.get(id).map(|entry| entry.agent.clone()))
            .collect()
    }

    /// All live top-level agents (created without an owning agent context).
    pub fn roots(&self) -> Vec<AgentRef> {
        let store = self.store.borrow();
        self.order
            .borrow()
            .iter()
            .filter_map(|id| store.get(id))
            .filter(|entry| entry.owner.is_none())
            .map(|entry| entry.agent.clone())
            .collect()
    }

    // ---- initiator attribution ----

    fn assert_initiators_open(&self) -> Result<(), AgentRegistryError> {
        if self.initiator_state.get() != InitiatorState::Active {
            return Err(AgentRegistryError(DISPOSED_INITIATOR_MESSAGE.into()));
        }
        Ok(())
    }

    /// Read the agent id that initiated the inherited driver chain: the async
    /// task-local first, else the synchronous nesting stack. `None` outside
    /// any boundary or inside a clearing boundary.
    pub fn current_initiator(&self) -> Result<Option<AgentRef>, AgentRegistryError> {
        if self.initiator_state.get() == InitiatorState::Disposed {
            return Err(AgentRegistryError(DISPOSED_INITIATOR_MESSAGE.into()));
        }
        let id = INITIATOR
            .try_with(|current| current.clone())
            .ok()
            .flatten()
            .or_else(|| self.sync_initiators.borrow().last().cloned().flatten());
        Ok(id.and_then(|id| self.get(&id)))
    }

    /// Read the initiating agent and fail when no boundary is active.
    pub fn require_initiator(&self) -> Result<AgentRef, AgentRegistryError> {
        self.current_initiator()?
            .ok_or_else(|| AgentRegistryError(NO_INITIATOR_MESSAGE.into()))
    }

    /// Run an async operation with one exact agent as its process-local
    /// initiator (upstream `withInitiator`).
    pub async fn with_initiator<T>(
        &self,
        agent: &AgentRef,
        operation: impl std::future::Future<Output = T>,
    ) -> Result<T, AgentRegistryError> {
        self.assert_initiators_open()?;
        Ok(INITIATOR.scope(Some(agent.id()), operation).await)
    }

    /// Run an async operation inside a boundary that hides any inherited
    /// initiator (upstream `withoutInitiator`) — for shared timers, queue
    /// pumps, and pool maintenance that must not inherit the first agent that
    /// initializes them.
    pub async fn without_initiator<T>(
        &self,
        operation: impl std::future::Future<Output = T>,
    ) -> Result<T, AgentRegistryError> {
        self.assert_initiators_open()?;
        Ok(INITIATOR.scope(None, operation).await)
    }

    /// Synchronous initiator boundary.
    pub fn with_initiator_sync<T>(
        &self,
        agent: Option<&AgentRef>,
        operation: impl FnOnce() -> T,
    ) -> Result<T, AgentRegistryError> {
        self.assert_initiators_open()?;
        self.sync_initiators
            .borrow_mut()
            .push(agent.map(|agent| agent.id()));
        let result = operation();
        self.sync_initiators.borrow_mut().pop();
        Ok(result)
    }

    /// Reject new initiator boundaries (teardown begins).
    pub fn close_initiators(&self) {
        if self.initiator_state.get() == InitiatorState::Active {
            self.initiator_state.set(InitiatorState::Closing);
        }
    }

    /// Invalidate initiator reads entirely (teardown complete).
    pub fn dispose_initiators(&self) {
        self.close_initiators();
        self.initiator_state.set(InitiatorState::Disposed);
    }
}
