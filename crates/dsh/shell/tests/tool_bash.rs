//! Behavior tests for the model-facing bash tool, mirroring the portable
//! parts of upstream `tool-bash/tests/tools.spec.ts` + `integration.spec.ts`
//! over a real executor. Not ported: background/job suites
//! (`run_in_background` is out of scope) and sandbox-escalation fixtures.

use dsh_agent::{
    Agent, AgentCancelCause, AgentOptions, AgentRef, AgentStatus, CancelOptions, InboxTarget,
};
use dsh_cordis::{App, Context};
use dsh_llm::{CallId, ContentBlock, Message};
use dsh_session::{SESSION_FORMAT_VERSION, Session, SessionHeader, SessionId};
use dsh_shell::{
    BashConfig, CollectedOutput, LocalBashExecutor, LocalSubprocessRuntime, ShellRunResult,
    register_bash_tool, render_result,
};
use dsh_timeout::{AbortController, AbortSignal};
use dsh_tools::{
    Config as ToolsConfig, ToolCallView, ToolExecutionInput, ToolExecutionResult, ToolResult,
    ToolResultView, ToolRuntime,
};
use futures::FutureExt;
use futures::future::LocalBoxFuture;
use serde_json::{Value, json};
use std::path::Path;
use std::rc::Rc;

struct Fixture {
    _app: App,
    ctx: Context,
    tools: Rc<ToolRuntime>,
}

fn fixture_with(config: BashConfig) -> Fixture {
    let app = App::new();
    let ctx = app.root();
    let tools = ToolRuntime::provide(&ctx, ToolsConfig::default()).unwrap();
    let subprocess = LocalSubprocessRuntime::provide(&ctx).unwrap();
    let shell = LocalBashExecutor::provide(&ctx, subprocess, config).unwrap();
    register_bash_tool(&ctx, &tools, &shell).unwrap();
    Fixture {
        _app: app,
        ctx,
        tools,
    }
}

fn fixture() -> Fixture {
    fixture_with(BashConfig::default())
}

fn input(arguments: Value, agent: Option<AgentRef>) -> ToolExecutionInput {
    ToolExecutionInput {
        call_id: CallId::new("c1"),
        root_call_id: None,
        name: "bash".into(),
        arguments,
        agent,
        parent: None,
        signal: AbortSignal::never(),
    }
}

fn text_of(result: &ToolExecutionResult) -> String {
    match result.content() {
        [ContentBlock::Text { text }] => text.clone(),
        other => panic!("expected one text block, got {other:?}"),
    }
}

struct FakeAgent {
    id: SessionId,
    session: Rc<Session>,
    ctx: Context,
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

fn agent_with_cwd(ctx: &Context, cwd: &Path) -> AgentRef {
    let id = SessionId::new("bash-session");
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

#[test]
fn registers_the_bash_schema_without_leaking_internals() {
    dsh_cordis::run(async {
        let fixture = fixture();
        let schemas = fixture.tools.schemas(None);
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0].name, "bash");
        let params = &schemas[0].parameters;
        let properties = params.get("properties").unwrap().as_object().unwrap();
        assert!(properties.contains_key("command"));
        assert!(properties.contains_key("description"));
        assert!(properties.contains_key("timeoutMs"));
        assert!(properties.contains_key("workdir"));
        assert_eq!(
            params.get("required").unwrap(),
            &json!(["command", "description"]),
        );
    });
}

#[test]
fn executes_a_command_and_returns_clean_output() {
    dsh_cordis::run(async {
        let fixture = fixture();
        let result = fixture
            .tools
            .execute(input(
                json!({ "command": "echo hello", "description": "Print hello" }),
                None,
            ))
            .await;
        assert!(!result.is_error(), "{:?}", result.error());
        assert_eq!(text_of(&result), "hello\n");
    });
}

#[test]
fn nonzero_exits_are_reported_not_errored() {
    dsh_cordis::run(async {
        let fixture = fixture();
        let result = fixture
            .tools
            .execute(input(
                json!({ "command": "exit 3", "description": "Exit nonzero" }),
                None,
            ))
            .await;
        assert!(!result.is_error());
        assert_eq!(text_of(&result), "(no output)\n[exit code: 3]");
    });
}

#[test]
fn stderr_rides_in_a_marked_section() {
    dsh_cordis::run(async {
        let fixture = fixture();
        let result = fixture
            .tools
            .execute(input(
                json!({ "command": "echo out; echo err 1>&2", "description": "Mixed streams" }),
                None,
            ))
            .await;
        assert_eq!(text_of(&result), "out\n[stderr]\nerr\n");
    });
}

#[test]
fn argument_validation_rejects_blank_and_bad_values() {
    dsh_cordis::run(async {
        let fixture = fixture();
        let blank_command = fixture
            .tools
            .execute(input(json!({ "command": "  ", "description": "d" }), None))
            .await;
        assert!(text_of(&blank_command).contains("invalid command"));
        let blank_description = fixture
            .tools
            .execute(input(
                json!({ "command": "true", "description": " " }),
                None,
            ))
            .await;
        assert!(text_of(&blank_description).contains("invalid description"));
        let bad_timeout = fixture
            .tools
            .execute(input(
                json!({ "command": "true", "description": "d", "timeoutMs": -5 }),
                None,
            ))
            .await;
        assert!(text_of(&bad_timeout).contains("invalid timeoutMs"));
        let missing_required = fixture
            .tools
            .execute(input(json!({ "command": "true" }), None))
            .await;
        assert!(missing_required.is_error());
    });
}

#[test]
fn workdir_defaults_to_the_session_and_resolves_relative_overrides() {
    dsh_cordis::run(async {
        let fixture = fixture();
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        std::fs::create_dir(root.join("sub")).unwrap();
        let agent = agent_with_cwd(&fixture.ctx, &root);

        let session_default = fixture
            .tools
            .execute(input(
                json!({ "command": "pwd", "description": "Print cwd" }),
                Some(agent.clone()),
            ))
            .await;
        assert_eq!(text_of(&session_default).trim(), root.to_string_lossy());

        let relative = fixture
            .tools
            .execute(input(
                json!({ "command": "pwd", "description": "Print cwd", "workdir": "sub" }),
                Some(agent),
            ))
            .await;
        assert_eq!(
            text_of(&relative).trim(),
            root.join("sub").to_string_lossy()
        );
    });
}

#[test]
fn the_timeout_marker_reports_the_interruption() {
    dsh_cordis::run(async {
        let fixture = fixture_with(BashConfig {
            grace_ms: 500.0,
            ..BashConfig::default()
        });
        let result = fixture
            .tools
            .execute(input(
                json!({ "command": "sleep 10", "description": "Sleep", "timeoutMs": 200 }),
                None,
            ))
            .await;
        assert!(
            !result.is_error(),
            "a timeout is a reported outcome, not an error"
        );
        let text = text_of(&result);
        assert!(text.contains("[timed out after 200ms]"), "{text}");
        assert!(text.contains("[killed by signal: SIGTERM]"), "{text}");
    });
}

#[test]
fn a_caller_abort_becomes_the_canonical_aborted_error() {
    dsh_cordis::run(async {
        let fixture = fixture();
        let controller = AbortController::new();
        let mut call = input(
            json!({ "command": "sleep 10", "description": "Sleep" }),
            None,
        );
        call.signal = controller.signal();
        tokio::task::spawn_local(async move {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            controller.abort("user cancelled");
        });
        let result = fixture.tools.execute(call).await;
        assert!(result.is_error());
        let failure = result.error().unwrap();
        assert_eq!(failure.message, "tool call aborted");
        assert_eq!(failure.info.as_ref().unwrap().code, "ABORTED");
    });
}

#[test]
fn truncated_output_reports_the_loss() {
    dsh_cordis::run(async {
        let fixture = fixture_with(BashConfig {
            max_output_bytes: 8,
            ..BashConfig::default()
        });
        let result = fixture
            .tools
            .execute(input(
                json!({ "command": "printf 1234567890ABCDEF", "description": "Long output" }),
                None,
            ))
            .await;
        let text = text_of(&result);
        assert!(text.starts_with("90ABCDEF"), "{text}");
        assert!(
            text.contains("[output truncated; full output: (unavailable)]"),
            "{text}"
        );
    });
}

#[test]
fn present_call_is_a_terminal_card_titled_by_the_command() {
    dsh_cordis::run(async {
        let fixture = fixture();
        let definition = fixture.tools.get("bash", None).unwrap();
        let view = (definition.present_call.unwrap())(&json!({
            "command": "ls -la",
            "description": "List files",
            "workdir": "/tmp",
        }))
        .unwrap();
        match view {
            ToolCallView::Terminal(card) => {
                assert_eq!(card.title, "ls -la");
                assert_eq!(card.description.as_deref(), Some("List files"));
                assert_eq!(card.cwd.as_deref(), Some("/tmp"));
            }
            other => panic!("expected terminal card, got {other:?}"),
        }
    });
}

#[test]
fn present_result_turns_the_exit_marker_into_the_pill() {
    dsh_cordis::run(async {
        let fixture = fixture();
        let definition = fixture.tools.get("bash", None).unwrap();
        let present = definition.present_result.unwrap();
        let args = json!({ "command": "x", "description": "d" });

        let ok = present(
            &args,
            &ToolResult {
                content: vec![ContentBlock::Text {
                    text: "out\n[exit code: 2]".into(),
                }],
                is_error: false,
                meta: None,
            },
        );
        match ok.unwrap() {
            ToolResultView::Terminal(card) => {
                assert_eq!(card.output.as_deref(), Some("out"));
                assert_eq!(card.exit_code, Some(2));
                assert_eq!(card.signal, None);
            }
            other => panic!("expected terminal card, got {other:?}"),
        }

        let killed = present(
            &args,
            &ToolResult {
                content: vec![ContentBlock::Text {
                    text: "out\n[killed by signal: SIGKILL]".into(),
                }],
                is_error: false,
                meta: None,
            },
        );
        match killed.unwrap() {
            ToolResultView::Terminal(card) => {
                assert_eq!(card.signal.as_deref(), Some("SIGKILL"));
                assert_eq!(card.exit_code, None);
            }
            other => panic!("expected terminal card, got {other:?}"),
        }

        // Errors keep fenced generic output without an exit pill.
        let error = present(
            &args,
            &ToolResult {
                content: vec![ContentBlock::Text {
                    text: "Error: boom\n".into(),
                }],
                is_error: true,
                meta: None,
            },
        );
        match error.unwrap() {
            ToolResultView::Generic(card) => match card.content.unwrap().as_slice() {
                [ContentBlock::Text { text }] => {
                    assert_eq!(text, "```console\nError: boom\n```")
                }
                other => panic!("expected text, got {other:?}"),
            },
            other => panic!("expected generic card, got {other:?}"),
        }
    });
}

#[test]
fn render_result_orders_sections_and_markers() {
    let result = ShellRunResult {
        exit_code: Some(0),
        signal: Some("SIGTERM".into()),
        timed_out: true,
        aborted: false,
        timeout_ms: 100.0,
        stdout: CollectedOutput {
            text: "partial".into(),
            truncated: true,
        },
        stderr: CollectedOutput {
            text: "note\n".into(),
            truncated: false,
        },
    };
    assert_eq!(
        render_result(&result),
        "partial\n[output truncated; full output: (unavailable)]\n[stderr]\nnote\n[timed out after 100ms]\n[killed by signal: SIGTERM]"
    );
    let quiet = ShellRunResult {
        exit_code: Some(0),
        signal: None,
        timed_out: false,
        aborted: false,
        timeout_ms: 100.0,
        stdout: CollectedOutput {
            text: String::new(),
            truncated: false,
        },
        stderr: CollectedOutput {
            text: String::new(),
            truncated: false,
        },
    };
    assert_eq!(render_result(&quiet), "(no output)");
}
