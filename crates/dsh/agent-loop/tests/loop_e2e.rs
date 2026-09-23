//! End-to-end loop behavior over a scripted mock adapter: a first step that
//! requests a tool call, a second step that answers in text. Asserts the
//! durable transcript shape (turn/step boundaries, chunks cited by the
//! assembled message, tool call/result pairs, request-header anchoring) and
//! the derived history the next request would use.

use dsh_agent::{AgentOptions, AgentRegistry, CreateAgentOptions};
use dsh_agent_loop::AgentLoop;
use dsh_cordis::App;
use dsh_llm::{
    AdapterStream, BlockType, ContentBlock, FinishReason, GenerateOptions, LlmAdapter, LlmRuntime,
    StreamChunk,
};
use dsh_session::{SessionEventData, SessionId, SessionMeta, SessionStore, TurnEndReason};
use dsh_system_prompt::{Config as PromptConfig, PromptSection, PromptText, SystemPrompt};
use dsh_tools::{
    Config as ToolsConfig, DefineToolOptions, DefineToolOutput, ParameterPropertySpec,
    ParameterSchemaSpec, StringValueSchemaSpec, ToolRuntime, ValueSchemaSpec, define_tool,
};
use futures::FutureExt;
use futures::StreamExt;
use std::cell::RefCell;
use std::rc::Rc;

/// Scripted adapter: each call pops the next chunk script.
struct ScriptedAdapter {
    scripts: RefCell<Vec<Vec<StreamChunk>>>,
    seen_requests: Rc<RefCell<Vec<GenerateOptions>>>,
}

impl LlmAdapter for ScriptedAdapter {
    fn stream(&self, options: GenerateOptions) -> AdapterStream {
        self.seen_requests.borrow_mut().push(options);
        let script = if self.scripts.borrow().is_empty() {
            vec![StreamChunk::Finish {
                reason: FinishReason::Stop,
                replay_state: None,
            }]
        } else {
            self.scripts.borrow_mut().remove(0)
        };
        futures::stream::iter(script.into_iter().map(Ok)).boxed_local()
    }
}

fn tool_call_step() -> Vec<StreamChunk> {
    vec![
        StreamChunk::BlockStart {
            index: 0,
            block_type: BlockType::ToolCall,
        },
        StreamChunk::ToolCallDelta {
            index: 0,
            id: dsh_llm::CallId::new("call-1"),
            name: Some("echo".into()),
            arguments_delta: "{\"text\":\"ping\"}".into(),
        },
        StreamChunk::BlockEnd {
            index: 0,
            block: ContentBlock::ToolCall {
                id: dsh_llm::CallId::new("call-1"),
                name: "echo".into(),
                arguments: "{\"text\":\"ping\"}".into(),
            },
        },
        StreamChunk::Finish {
            reason: FinishReason::ToolCalls,
            replay_state: None,
        },
    ]
}

fn text_step(text: &str) -> Vec<StreamChunk> {
    vec![
        StreamChunk::BlockStart {
            index: 0,
            block_type: BlockType::Text,
        },
        StreamChunk::TextDelta {
            index: 0,
            text: text.into(),
        },
        StreamChunk::BlockEnd {
            index: 0,
            block: ContentBlock::Text { text: text.into() },
        },
        StreamChunk::Finish {
            reason: FinishReason::Stop,
            replay_state: None,
        },
    ]
}

#[test]
fn loop_drives_tool_step_then_answer() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();

        let sessions = SessionStore::provide(&ctx).unwrap();
        let agents = AgentRegistry::provide(&ctx).unwrap();
        let llm = LlmRuntime::provide(&ctx).unwrap();
        let system_prompt = SystemPrompt::new(&ctx, PromptConfig::default()).unwrap();
        let tools = ToolRuntime::provide(&ctx, ToolsConfig::default()).unwrap();

        system_prompt
            .section(
                &ctx,
                PromptSection {
                    name: "persona".into(),
                    order: 0.0,
                    text: PromptText::fixed("You are a test harness."),
                    complete: false,
                },
            )
            .unwrap();

        // One echo tool.
        let echo = define_tool(DefineToolOptions {
            name: "echo".into(),
            description: "Echo the text back".into(),
            parameters: ParameterSchemaSpec(vec![(
                "text".into(),
                ParameterPropertySpec {
                    required: true,
                    spec: ValueSchemaSpec::String(StringValueSchemaSpec::default()),
                },
            )]),
            output: DefineToolOutput {
                schema: ValueSchemaSpec::String(StringValueSchemaSpec::default()),
                render: Rc::new(|_args, value| {
                    Ok(vec![ContentBlock::Text {
                        text: value.as_str().unwrap_or_default().to_string(),
                    }])
                }),
                presentation_meta: None,
            },
            timeout_ms: None,
            is_concurrency_safe: None,
            execute: Rc::new(|args, _exec| {
                let text = args["text"].as_str().unwrap_or_default().to_string();
                async move { Ok(serde_json::Value::String(format!("echo: {text}"))) }.boxed_local()
            }),
            finalize_content: None,
            present_call: None,
            present_result: None,
        })
        .unwrap();
        tools.register(&ctx, echo).unwrap();

        // Composition glue: the tools registry feeds prompt assembly (the
        // upstream ToolRuntime constructor does this itself; the Rust crates
        // are deliberately decoupled, so the composition wires them).
        {
            let provider = tools.schemas_provider();
            system_prompt
                .tools(&ctx, move |context| {
                    let result = provider(context.scope.as_ref());
                    dsh_system_prompt::ToolProviderResult {
                        schemas: result.schemas,
                        known_names: Some(result.known_names),
                    }
                })
                .unwrap();
        }

        let seen_requests: Rc<RefCell<Vec<GenerateOptions>>> = Rc::default();
        llm.register_adapter(
            &["mock".to_string()],
            Rc::new(ScriptedAdapter {
                scripts: RefCell::new(vec![tool_call_step(), text_step("pong")]),
                seen_requests: seen_requests.clone(),
            }),
        )
        .unwrap();

        let key_dir = tempfile::tempdir().unwrap();
        AgentLoop::install_with_decision_data_dir(
            &ctx,
            agents.clone(),
            sessions.clone(),
            llm.clone(),
            tools.clone(),
            system_prompt.clone(),
            Some(key_dir.path().to_path_buf()),
        )
        .unwrap();

        let handle = agents
            .create(CreateAgentOptions {
                session_id: SessionId::new("e2e"),
                meta: SessionMeta::default(),
                seed: vec![],
                agent_options: AgentOptions {
                    provider: Some("mock".into()),
                    model: Some("mock-model".into()),
                    ..Default::default()
                },
            })
            .await
            .unwrap();

        handle.agent.followup(dsh_llm::create_user_message(
            vec![ContentBlock::Text {
                text: "please ping".into(),
            }],
            dsh_llm::MessageSource::User,
        ));
        handle.agent.when_idle().await;

        let session = handle.agent.session();
        let types: Vec<String> = session
            .events()
            .iter()
            .map(|event| event.event_type().to_string())
            .collect();

        // Turn shape: one turn, two steps (tool step + answer step).
        assert_eq!(types.iter().filter(|t| *t == "turn/start").count(), 1);
        assert_eq!(types.iter().filter(|t| *t == "step/start").count(), 2);
        assert_eq!(types.iter().filter(|t| *t == "tool/call").count(), 1);
        assert_eq!(types.iter().filter(|t| *t == "tool/result").count(), 1);
        assert_eq!(
            types.iter().filter(|t| *t == "assistant/message").count(),
            2
        );
        assert_eq!(types.iter().filter(|t| *t == "request/header").count(), 1);
        let turn_end = session
            .events()
            .iter()
            .rev()
            .find_map(|event| match &event.data {
                SessionEventData::TurnEnd { reason, .. } => Some(reason.clone()),
                _ => None,
            })
            .unwrap();
        assert_eq!(turn_end, TurnEndReason::Completed);

        // The tool actually ran: its result text is in the derived history.
        let derived = session.derive_messages();
        let tool_result_text = derived
            .iter()
            .find_map(|message| {
                message.content.iter().find_map(|block| match block {
                    ContentBlock::ToolResult { content, .. } => {
                        content.iter().find_map(|inner| match inner {
                            ContentBlock::Text { text } => Some(text.clone()),
                            _ => None,
                        })
                    }
                    _ => None,
                })
            })
            .unwrap();
        assert_eq!(tool_result_text, "echo: ping");

        // The second request carried the system prompt and the tool schema,
        // and its history included the tool result.
        let requests = seen_requests.borrow();
        assert_eq!(requests.len(), 2);
        assert!(
            requests[1]
                .system
                .as_deref()
                .unwrap()
                .contains("test harness")
        );
        assert!(
            !requests[1]
                .system
                .as_deref()
                .unwrap()
                .contains("Next-step focus")
        );
        assert!(
            requests[1]
                .tools
                .as_ref()
                .unwrap()
                .iter()
                .any(|tool| tool.name == "echo")
        );
        assert!(requests[1].messages.len() > requests[0].messages.len());

        handle.dispose().await;
        assert!(agents.get(&SessionId::new("e2e")).is_none());
        assert_eq!(sessions.list().len(), 0);
    });
}

#[test]
fn error_finish_closes_turn_with_structured_error() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sessions = SessionStore::provide(&ctx).unwrap();
        let agents = AgentRegistry::provide(&ctx).unwrap();
        let llm = LlmRuntime::provide(&ctx).unwrap();
        let system_prompt = SystemPrompt::new(&ctx, PromptConfig::default()).unwrap();
        let tools = ToolRuntime::provide(&ctx, ToolsConfig::default()).unwrap();

        llm.register_adapter(
            &["mock".to_string()],
            Rc::new(ScriptedAdapter {
                scripts: RefCell::new(vec![vec![StreamChunk::Finish {
                    reason: FinishReason::Error {
                        failure: dsh_llm::LlmFailure::new("boom", "SERVER"),
                    },
                    replay_state: None,
                }]]),
                seen_requests: Rc::default(),
            }),
        )
        .unwrap();
        AgentLoop::install(&ctx, agents.clone(), sessions, llm, tools, system_prompt).unwrap();

        let handle = agents
            .create(CreateAgentOptions {
                session_id: SessionId::new("err"),
                meta: SessionMeta::default(),
                seed: vec![],
                agent_options: AgentOptions {
                    provider: Some("mock".into()),
                    model: Some("m".into()),
                    ..Default::default()
                },
            })
            .await
            .unwrap();
        handle.agent.followup(dsh_llm::create_user_message(
            vec![ContentBlock::Text { text: "hi".into() }],
            dsh_llm::MessageSource::User,
        ));
        handle.agent.when_idle().await;

        let turn_end = handle
            .agent
            .session()
            .events()
            .iter()
            .rev()
            .find_map(|event| match &event.data {
                SessionEventData::TurnEnd { reason, .. } => Some(reason.clone()),
                _ => None,
            })
            .unwrap();
        match turn_end {
            TurnEndReason::Error { error } => assert_eq!(error.code, "SERVER"),
            other => panic!("expected error turn end, got {other:?}"),
        }
        handle.dispose().await;
    });
}
