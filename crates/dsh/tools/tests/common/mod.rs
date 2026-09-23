//! Shared fixtures for the runtime behavior tests.
#![allow(dead_code)] // each test binary uses a different subset

use dsh_agent::{
    Agent, AgentCancelCause, AgentOptions, AgentRef, AgentStatus, CancelOptions, InboxTarget,
};
use dsh_cordis::{App, Context};
use dsh_llm::{CallId, ContentBlock, Message, MessageSource};
use dsh_scope::{Scope, ScopeKey, create_scope};
use dsh_session::{Session, SessionId};
use dsh_timeout::AbortSignal;
use dsh_tools::{
    Config, ParameterSchemaSpec, ToolDefinition, ToolExecutionInput, ToolOutputDefinition,
    ToolRuntime, ValueSchemaSpec,
};
use futures::FutureExt;
use futures::future::LocalBoxFuture;
use serde_json::{Value, json};
use std::cell::Cell;
use std::rc::Rc;

pub fn setup() -> (App, Context, Rc<ToolRuntime>) {
    let app = App::new();
    let ctx = app.root();
    let tools = ToolRuntime::provide(&ctx, Config::default()).unwrap();
    (app, ctx, tools)
}

/// A raw string-output tool answering with `reply` (the upstream scoped-suite
/// fixture shape).
pub fn tool(name: &str, reply: &str) -> ToolDefinition {
    let reply = reply.to_string();
    ToolDefinition {
        name: name.into(),
        description: format!("tool {name}"),
        parameters: json!({ "type": "object", "properties": {} })
            .as_object()
            .unwrap()
            .clone(),
        output: ToolOutputDefinition {
            schema: json!({ "type": "string" }),
            render: Rc::new(|_args, value| {
                Ok(vec![ContentBlock::Text {
                    text: value.as_str().unwrap_or_default().into(),
                }])
            }),
            presentation_meta: None,
        },
        execute: Rc::new(move |_args, _exec| {
            let reply = reply.clone();
            async move { Ok(json!(reply)) }.boxed_local()
        }),
        finalize_content: None,
        timeout_ms: None,
        is_concurrency_safe: None,
        present_call: None,
        present_result: None,
    }
}

/// The same fixture with a custom body.
pub fn tool_with_execute(
    name: &str,
    execute: impl Fn(Value, dsh_tools::ToolRunContext) -> LocalBoxFuture<'static, anyhow::Result<Value>>
    + 'static,
) -> ToolDefinition {
    ToolDefinition {
        execute: Rc::new(execute),
        ..tool(name, "unused")
    }
}

pub fn input(name: &str, arguments: Value) -> ToolExecutionInput {
    ToolExecutionInput {
        call_id: CallId::new("c1"),
        root_call_id: None,
        name: name.into(),
        arguments,
        agent: None,
        parent: None,
        signal: AbortSignal::never(),
    }
}

pub fn input_for(name: &str, agent: Option<AgentRef>) -> ToolExecutionInput {
    ToolExecutionInput {
        agent,
        ..input(name, json!({}))
    }
}

/// Execute and render the first content block as text (the upstream `run`
/// helper).
pub async fn run(tools: &Rc<ToolRuntime>, name: &str, agent: Option<AgentRef>) -> String {
    let result = tools.execute(input_for(name, agent)).await;
    match result.content().first() {
        Some(ContentBlock::Text { text }) => text.clone(),
        other => format!("{other:?}"),
    }
}

pub fn plugin_message(text: &str, plugin: &str) -> Message {
    dsh_llm::create_user_message(
        vec![ContentBlock::Text { text: text.into() }],
        MessageSource::Plugin {
            plugin: plugin.into(),
            form: None,
        },
    )
}

/// Minimal Agent stand-in whose `ctx()` is a scoped context, giving the
/// registry its scope key (upstream uses the agent object itself as the key).
pub struct FakeAgent {
    pub id: SessionId,
    pub session: Rc<Session>,
    pub ctx: Context,
}

impl Agent for FakeAgent {
    fn id(&self) -> SessionId {
        self.id.clone()
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
    fn ctx(&self) -> Context {
        self.ctx.clone()
    }
    fn cancel(&self, _cause: AgentCancelCause, _options: CancelOptions) {}
    fn when_idle(&self) -> LocalBoxFuture<'static, ()> {
        async {}.boxed_local()
    }
    fn send(&self, _message: Message, _target: InboxTarget, _wakeup: bool) {}
    fn followup(&self, _message: Message) {}
    fn steer(&self, _message: Message) {}
    fn inject(&self, _message: Message) {}
}

/// Mint an agent scope: the scope's context registers into its layer, and
/// the returned agent routes executions to the same key.
pub fn mint_agent(
    ctx: &Context,
    name: &str,
    parent: Option<&ScopeKey>,
) -> (Scope, AgentRef, ScopeKey) {
    let key = ScopeKey::new();
    let scope = create_scope(ctx, key.clone(), parent).unwrap();
    let session = Session::create(SessionId::new(name), vec![], None).unwrap();
    let agent: AgentRef = Rc::new(FakeAgent {
        id: SessionId::new(name),
        session,
        ctx: scope.ctx(),
    });
    (scope, agent, key)
}

/// Busy-yield until `flag` is set (test-only synchronization).
pub async fn until(flag: &Rc<Cell<bool>>) {
    while !flag.get() {
        tokio::task::yield_now().await;
    }
}

/// Compile a parameter spec from the author syntax.
pub fn params(spec: Value) -> ParameterSchemaSpec {
    ParameterSchemaSpec::from_author_value(&spec, "parameters").unwrap()
}

/// Compile a value spec from the author syntax.
pub fn value_spec(spec: Value) -> ValueSchemaSpec {
    ValueSchemaSpec::from_author_value(&spec, "schema").unwrap()
}

pub fn provide_config(ctx: &Context, config: Config) -> anyhow::Result<Rc<ToolRuntime>> {
    ToolRuntime::provide(ctx, config)
}
