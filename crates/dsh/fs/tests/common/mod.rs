//! Shared fixtures for the dsh-fs behavior tests: a real-tempdir workspace,
//! the tool registry, and a minimal agent whose session carries the
//! workspace cwd (the observed-state owner).
#![allow(dead_code)] // each test binary uses a different subset

use dsh_agent::{
    Agent, AgentCancelCause, AgentOptions, AgentRef, AgentStatus, CancelOptions, InboxTarget,
};
use dsh_cordis::{App, Context};
use dsh_fs::{LocalFileSystem, LocalFileSystemConfig};
use dsh_llm::{CallId, ContentBlock, Message};
use dsh_session::{SESSION_FORMAT_VERSION, Session, SessionHeader, SessionId};
use dsh_timeout::AbortSignal;
use dsh_tools::{Config as ToolsConfig, ToolExecutionInput, ToolExecutionResult, ToolRuntime};
use futures::FutureExt;
use futures::future::LocalBoxFuture;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::rc::Rc;

/// One test workspace: app, root context, tool registry, local fs rooted at
/// a canonicalized tempdir.
pub struct Fixture {
    pub app: App,
    pub ctx: Context,
    pub tools: Rc<ToolRuntime>,
    pub fs: Rc<LocalFileSystem>,
    pub dir: tempfile::TempDir,
    pub root: PathBuf,
}

pub fn fixture() -> Fixture {
    let app = App::new();
    let ctx = app.root();
    let tools = ToolRuntime::provide(&ctx, ToolsConfig::default()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    // Canonicalize so display paths and target keys agree in assertions
    // (macOS aliases /var to /private/var).
    let root = std::fs::canonicalize(dir.path()).unwrap();
    let fs = LocalFileSystem::provide(
        &ctx,
        LocalFileSystemConfig {
            cwd: Some(root.clone()),
            diff_basis_max_bytes: None,
        },
    )
    .unwrap();
    Fixture {
        app,
        ctx,
        tools,
        fs,
        dir,
        root,
    }
}

pub fn write(root: &Path, name: &str, content: &str) -> PathBuf {
    let path = root.join(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(&path, content).unwrap();
    path
}

/// Minimal agent whose `session().header.cwd` is the given workspace and
/// whose session id is the observed-state owner.
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

pub fn agent_with_cwd(ctx: &Context, name: &str, cwd: &Path) -> AgentRef {
    let id = SessionId::new(name);
    let header = SessionHeader {
        version: SESSION_FORMAT_VERSION,
        id: id.clone(),
        created_at: 0,
        cwd: Some(cwd.to_string_lossy().into_owned()),
        parent_session: None,
        seed_length: None,
        origin: None,
        delegation_depth: None,
        agent_preset: None,
    };
    let session = Session::create(id.clone(), vec![], Some(header)).unwrap();
    Rc::new(FakeAgent {
        id,
        session,
        ctx: ctx.clone(),
    })
}

pub fn input(name: &str, arguments: Value, agent: Option<AgentRef>) -> ToolExecutionInput {
    ToolExecutionInput {
        call_id: CallId::new("c1"),
        root_call_id: None,
        name: name.into(),
        arguments,
        agent,
        parent: None,
        signal: AbortSignal::never(),
    }
}

/// The single text block of a result, for assertions.
pub fn text_of(result: &ToolExecutionResult) -> String {
    match result.content() {
        [ContentBlock::Text { text }] => text.clone(),
        other => panic!("expected one text block, got {other:?}"),
    }
}

/// The stable machine-routable code carried by an error result.
pub fn code_of(result: &ToolExecutionResult) -> String {
    result
        .error()
        .and_then(|failure| failure.info.as_ref())
        .map(|info| info.code.clone())
        .unwrap_or_default()
}
