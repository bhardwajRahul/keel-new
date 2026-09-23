//! Rust port of `@deepseek-ai/dsh-agent-loop` (`packages/core/agent-loop`):
//! the default ReAct driver over queued turns and step-boundary input, its
//! per-step tool-call scheduling, the runtime-context projection, and the
//! agent factory that publishes loop agents through the registries.
//!
//! Crate-level divergences (modules document their own):
//! - Tool calls dispatch sequentially (model-order contracts all hold; the
//!   bounded parallel pool is an upgrade path noted in `tool_calls`).
//! - Declarative config-started agents and launcher identities are deferred
//!   to the bundle tier; the factory covers create/resume.
//! - Scope wiring on the agent context follows the registry's deferral.

mod agent;
mod runtime_context;
mod tool_calls;

pub use agent::{DecisionActivityChanged, ReactLoopAgent};
pub use runtime_context::RuntimeContextProjection;
pub use tool_calls::{PlannedCall, ToolCallsOutcome, execute_tool_calls};

use dsh_agent::{
    AgentFactory, AgentHandle, AgentRef, AgentRegistry, CreateAgentOptions, ResumeAgentOptions,
    SessionStartSource,
};
use dsh_cordis::Context;
use dsh_llm::LlmRuntime;
use dsh_session::{AgentCancelCause, SessionStore};
use dsh_session_persistence::PersistenceService;
use dsh_system_prompt::SystemPrompt;
use dsh_tools::ToolRuntime;
use futures::FutureExt;
use futures::future::LocalBoxFuture;
use laya_local::{LayaConfig, LayaSelector};
use std::path::{Path, PathBuf};
use std::rc::Rc;

/// Default maximum parallel-safe tool calls in flight per agent step.
pub const DEFAULT_MAX_PARALLEL_TOOL_CALLS: u64 = 10;

/// Concrete agent factory and driver service (upstream `AgentLoop`). Creates
/// scoped loop agents, publishes them through the agent/session registries in
/// order, and owns their ordered teardown.
pub struct AgentLoop {
    ctx: Context,
    agents: Rc<AgentRegistry>,
    sessions: Rc<SessionStore>,
    llm: Rc<LlmRuntime>,
    tools: Rc<ToolRuntime>,
    system_prompt: Rc<SystemPrompt>,
    decision_mode_path: Option<PathBuf>,
    laya_selector: Option<LayaSelector>,
}

fn local_laya_assets(data_dir: Option<&Path>) -> Option<(PathBuf, PathBuf)> {
    let executable = std::env::current_exe().ok()?;
    let resources = executable.parent()?.parent()?.join("Resources");
    let worker = std::env::var_os("KEEL_LAYA_WORKER")
        .map(PathBuf::from)
        .unwrap_or_else(|| resources.join("laya-worker"));
    let model = std::env::var_os("KEEL_LAYA_MODEL")
        .map(PathBuf::from)
        .or_else(|| {
            resources
                .join("laya-model/coreml_config.json")
                .is_file()
                .then(|| resources.join("laya-model"))
        })
        .or_else(|| {
            data_dir
                .map(|dir| dir.join("laya/model"))
                .filter(|path| laya_local::model_is_installed(path))
        })?;
    if !worker.is_file() || !model.join("coreml_config.json").is_file() {
        return None;
    }
    Some((worker, model))
}

impl AgentLoop {
    /// Construct the factory over the injected services and register it on
    /// the agent registry (upstream constructor + `setFactory`).
    pub fn install(
        ctx: &Context,
        agents: Rc<AgentRegistry>,
        sessions: Rc<SessionStore>,
        llm: Rc<LlmRuntime>,
        tools: Rc<ToolRuntime>,
        system_prompt: Rc<SystemPrompt>,
    ) -> anyhow::Result<Rc<AgentLoop>> {
        Self::install_with_decision_data_dir(ctx, agents, sessions, llm, tools, system_prompt, None)
    }

    /// The preference is local; no remote selector is initialized here.
    pub fn install_with_decision_data_dir(
        ctx: &Context,
        agents: Rc<AgentRegistry>,
        sessions: Rc<SessionStore>,
        llm: Rc<LlmRuntime>,
        tools: Rc<ToolRuntime>,
        system_prompt: Rc<SystemPrompt>,
        decision_data_dir: Option<PathBuf>,
    ) -> anyhow::Result<Rc<AgentLoop>> {
        let decision_mode_path = decision_data_dir
            .as_ref()
            .map(|data_dir| data_dir.join("decisions/mode"));
        let laya_selector =
            local_laya_assets(decision_data_dir.as_deref()).and_then(|(worker, model)| {
                LayaSelector::new(LayaConfig::standalone(worker, model)).ok()
            });
        let factory = Rc::new(AgentLoop {
            ctx: ctx.clone(),
            agents: agents.clone(),
            sessions,
            llm,
            tools,
            system_prompt,
            decision_mode_path,
            laya_selector,
        });
        agents
            .set_factory(factory.clone())
            .map_err(|error| anyhow::anyhow!(error.0))?;
        Ok(factory)
    }

    /// The ordered creation transaction (upstream `createAgent` body):
    /// prepare the session, build the unpublished agent, enter + announce
    /// session then agent, emit session-start, and hand back the owned
    /// handle whose disposer runs the reverse chain.
    async fn publish(
        self: Rc<Self>,
        session: Rc<dsh_session::Session>,
        agent_options: dsh_agent::AgentOptions,
        source: SessionStartSource,
    ) -> anyhow::Result<AgentHandle> {
        let id = session.id();
        let session_effect = self
            .sessions
            .enter(session.clone())
            .map_err(|error| anyhow::anyhow!(error.0))?;
        let laya_selector = self.laya_selector.clone().or_else(|| {
            let data_dir = self
                .decision_mode_path
                .as_ref()
                .and_then(|path| path.parent()?.parent());
            local_laya_assets(data_dir).and_then(|(worker, model)| {
                LayaSelector::new(LayaConfig::standalone(worker, model)).ok()
            })
        });
        let loop_agent = ReactLoopAgent::new_with_decisions(
            self.ctx.clone(),
            id.clone(),
            agent_options,
            session.clone(),
            self.llm.clone(),
            self.tools.clone(),
            self.system_prompt.clone(),
            self.decision_mode_path.clone(),
            laya_selector,
        )?;
        let agent_ref: AgentRef = loop_agent.clone();
        let agent_effect = self
            .agents
            .enter(agent_ref.clone(), None)
            .map_err(|error| anyhow::anyhow!(error.0))?;
        self.sessions
            .announce(&session)
            .map_err(|error| anyhow::anyhow!(error.0))?;
        self.agents
            .announce(&agent_ref)
            .map_err(|error| anyhow::anyhow!(error.0))?;
        self.agents.emit_session_start(&agent_ref, source);

        let dispose_agent = agent_ref.clone();
        Ok(AgentHandle::new(agent_ref, move || {
            async move {
                // Reverse teardown: stop the driver, await quiescence,
                // unregister the agent, then detach the session.
                dispose_agent.cancel(
                    AgentCancelCause::Disposed,
                    dsh_agent::CancelOptions::default(),
                );
                dispose_agent.when_idle().await;
                agent_effect.dispose().await;
                session_effect.dispose().await;
            }
            .boxed_local()
        }))
    }
}

impl AgentFactory for AgentLoop {
    fn create_agent(
        &self,
        _owner_ctx: Context,
        options: CreateAgentOptions,
    ) -> LocalBoxFuture<'static, anyhow::Result<AgentHandle>> {
        let factory = Rc::new(AgentLoop {
            ctx: self.ctx.clone(),
            agents: self.agents.clone(),
            sessions: self.sessions.clone(),
            llm: self.llm.clone(),
            tools: self.tools.clone(),
            system_prompt: self.system_prompt.clone(),
            decision_mode_path: self.decision_mode_path.clone(),
            laya_selector: self.laya_selector.clone(),
        });
        async move {
            if factory.agents.get(&options.session_id).is_some() {
                anyhow::bail!(
                    "agent \"{}\" is already registered",
                    options.session_id.as_str()
                );
            }
            let session = factory
                .sessions
                .prepare(Some(options.session_id.clone()), options.seed, options.meta)
                .map_err(|error| anyhow::anyhow!(error.0))?;
            factory
                .clone()
                .publish(session, options.agent_options, SessionStartSource::Startup)
                .await
        }
        .boxed_local()
    }

    fn resume(
        &self,
        _owner_ctx: Context,
        options: ResumeAgentOptions,
    ) -> LocalBoxFuture<'static, anyhow::Result<AgentHandle>> {
        let ctx = self.ctx.clone();
        let factory = Rc::new(AgentLoop {
            ctx: self.ctx.clone(),
            agents: self.agents.clone(),
            sessions: self.sessions.clone(),
            llm: self.llm.clone(),
            tools: self.tools.clone(),
            system_prompt: self.system_prompt.clone(),
            decision_mode_path: self.decision_mode_path.clone(),
            laya_selector: self.laya_selector.clone(),
        });
        async move {
            let persistence = ctx
                .service::<PersistenceService>()
                .map_err(|_| anyhow::anyhow!("session persistence is not configured"))?;
            let session = persistence.prepare(&options.resume_session_id).await?;
            factory
                .publish(session, options.agent_options, SessionStartSource::Resume)
                .await
        }
        .boxed_local()
    }
}
