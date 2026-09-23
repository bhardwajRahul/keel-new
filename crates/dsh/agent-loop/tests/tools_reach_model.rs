//! The composition contract the CLI and the UI bridge both depend on: fs and
//! shell tools registered on the runtime must reach the model's request as
//! schemas AND be executable by the loop — the model can read, search, and
//! run commands, not merely see the names.

use dsh_agent::{AgentOptions, AgentRegistry, CreateAgentOptions};
use dsh_agent_loop::AgentLoop;
use dsh_cordis::App;
use dsh_llm::{
    AdapterStream, BlockType, ContentBlock, FinishReason, GenerateOptions, LlmAdapter, LlmRuntime,
    StreamChunk,
};
use dsh_session::{SessionEventData, SessionId, SessionMeta, SessionStore};
use dsh_system_prompt::{Config as PromptConfig, SystemPrompt};
use dsh_tools::{Config as ToolsConfig, ToolRuntime};
use futures::StreamExt;
use std::cell::RefCell;
use std::rc::Rc;

struct ScriptedAdapter {
    scripts: RefCell<Vec<Vec<StreamChunk>>>,
    seen: Rc<RefCell<Vec<GenerateOptions>>>,
}

impl LlmAdapter for ScriptedAdapter {
    fn stream(&self, options: GenerateOptions) -> AdapterStream {
        self.seen.borrow_mut().push(options);
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

fn read_call(path: &str) -> Vec<StreamChunk> {
    let arguments = serde_json::json!({ "file_path": path }).to_string();
    vec![
        StreamChunk::BlockStart {
            index: 0,
            block_type: BlockType::ToolCall,
        },
        StreamChunk::ToolCallDelta {
            index: 0,
            id: dsh_llm::CallId::new("call-read"),
            name: Some("read".into()),
            arguments_delta: arguments.clone(),
        },
        StreamChunk::BlockEnd {
            index: 0,
            block: ContentBlock::ToolCall {
                id: dsh_llm::CallId::new("call-read"),
                name: "read".into(),
                arguments,
            },
        },
        StreamChunk::Finish {
            reason: FinishReason::ToolCalls,
            replay_state: None,
        },
    ]
}

fn text(text: &str) -> Vec<StreamChunk> {
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
fn fs_and_shell_tools_reach_the_model_and_execute() {
    dsh_cordis::run(async {
        let workspace = tempfile::tempdir().unwrap();
        let target = workspace.path().join("hello.txt");
        std::fs::write(&target, "ported line\n").unwrap();

        let app = App::new();
        let ctx = app.root();
        let sessions = SessionStore::provide(&ctx).unwrap();
        let agents = AgentRegistry::provide(&ctx).unwrap();
        let llm = LlmRuntime::provide(&ctx).unwrap();
        let system_prompt = SystemPrompt::new(&ctx, PromptConfig::default()).unwrap();
        let tools = ToolRuntime::provide(&ctx, ToolsConfig::default()).unwrap();

        // The same registration the CLI and the bridge perform.
        let fs = dsh_fs::LocalFileSystem::provide(
            &ctx,
            dsh_fs::LocalFileSystemConfig {
                cwd: Some(workspace.path().to_path_buf()),
                diff_basis_max_bytes: None,
            },
        )
        .unwrap();
        dsh_fs::install_observation_policy(&ctx).unwrap();
        dsh_fs::register_fs_tools(&ctx, &tools, &fs, dsh_fs::FsToolsConfig::default()).unwrap();
        dsh_fs::register_search_tools(&ctx, &tools, dsh_fs::SearchToolsConfig::default()).unwrap();
        let subprocess = dsh_shell::LocalSubprocessRuntime::provide(&ctx).unwrap();
        let shell = dsh_shell::LocalBashExecutor::provide(
            &ctx,
            subprocess,
            dsh_shell::BashConfig::default(),
        )
        .unwrap();
        dsh_shell::register_bash_tool(&ctx, &tools, &shell).unwrap();

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

        let seen: Rc<RefCell<Vec<GenerateOptions>>> = Rc::default();
        llm.register_adapter(
            &["mock".to_string()],
            Rc::new(ScriptedAdapter {
                scripts: RefCell::new(vec![read_call(target.to_str().unwrap()), text("done")]),
                seen: seen.clone(),
            }),
        )
        .unwrap();
        AgentLoop::install(&ctx, agents.clone(), sessions, llm, tools, system_prompt).unwrap();

        let handle = agents
            .create(CreateAgentOptions {
                session_id: SessionId::new("tools"),
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
            vec![ContentBlock::Text {
                text: "read the file".into(),
            }],
            dsh_llm::MessageSource::User,
        ));
        handle.agent.when_idle().await;

        // Every registered tool reached the model as a schema.
        let requests = seen.borrow();
        let names: Vec<String> = requests[0]
            .tools
            .as_ref()
            .expect("tool schemas ride the request")
            .iter()
            .map(|tool| tool.name.clone())
            .collect();
        for expected in ["read", "write", "edit", "glob", "grep", "bash"] {
            assert!(
                names.contains(&expected.to_string()),
                "missing {expected} in {names:?}"
            );
        }

        // The read actually executed against the real filesystem: the file's
        // content came back through the tool result.
        let transcript = handle.agent.session().derive_messages();
        let result_text = transcript
            .iter()
            .find_map(|message| {
                message.content.iter().find_map(|block| match block {
                    ContentBlock::ToolResult {
                        content, is_error, ..
                    } => {
                        assert_eq!(*is_error, Some(false), "read succeeded: {content:?}");
                        content.iter().find_map(|inner| match inner {
                            ContentBlock::Text { text } => Some(text.clone()),
                            _ => None,
                        })
                    }
                    _ => None,
                })
            })
            .expect("a tool result entered the transcript");
        assert!(
            result_text.contains("ported line"),
            "file content reached the model: {result_text}"
        );

        // The durable log records the call/result pair for replay.
        let types: Vec<String> = handle
            .agent
            .session()
            .events()
            .iter()
            .map(|event| event.event_type().to_string())
            .collect();
        assert_eq!(types.iter().filter(|t| *t == "tool/call").count(), 1);
        assert_eq!(types.iter().filter(|t| *t == "tool/result").count(), 1);
        assert!(handle.agent.session().with_events(|events| {
            events.iter().any(|event| {
                matches!(
                    &event.data,
                    SessionEventData::ToolCall { name, .. } if name == "read"
                )
            })
        }));

        handle.dispose().await;
    });
}
