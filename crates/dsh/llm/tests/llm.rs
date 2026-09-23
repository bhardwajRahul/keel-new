//! Behavior tests ported from the upstream dsh-llm package suites: assembler
//! assembly rules, adapter registry contracts, stream failure normalization,
//! and the `llm/stream` waterfall.

use dsh_cordis::App;
use dsh_llm::*;
use futures::stream::StreamExt;
use std::rc::Rc;

fn text_chunks() -> Vec<StreamChunk> {
    vec![
        StreamChunk::BlockStart {
            index: 0,
            block_type: BlockType::Text,
        },
        StreamChunk::TextDelta {
            index: 0,
            text: "Hello, ".into(),
        },
        StreamChunk::TextDelta {
            index: 0,
            text: "world".into(),
        },
        StreamChunk::BlockEnd {
            index: 0,
            block: ContentBlock::Text {
                text: "Hello, world".into(),
            },
        },
        StreamChunk::Usage {
            usage: TokenUsage {
                input_tokens: 3,
                output_tokens: 2,
                ..Default::default()
            },
        },
        StreamChunk::Finish {
            reason: FinishReason::Stop,
            replay_state: None,
        },
    ]
}

#[test]
fn assembler_builds_blocks_and_message() {
    let mut assembler = BlockAssembler::new();
    for chunk in text_chunks() {
        assembler.push(&chunk);
    }
    assert_eq!(
        assembler.blocks(),
        vec![ContentBlock::Text {
            text: "Hello, world".into()
        }]
    );
    assert_eq!(assembler.finish(), FinishReason::Stop);
    assert_eq!(assembler.usage().unwrap().input_tokens, 3);
    let message = assembler.message(None);
    assert_eq!(message.role, Role::Assistant);
}

#[test]
fn assembler_ignores_stragglers_after_block_end() {
    let mut assembler = BlockAssembler::new();
    assembler.push(&StreamChunk::TextDelta {
        index: 0,
        text: "final".into(),
    });
    assembler.push(&StreamChunk::BlockEnd {
        index: 0,
        block: ContentBlock::Text {
            text: "final".into(),
        },
    });
    assembler.push(&StreamChunk::TextDelta {
        index: 0,
        text: " straggler".into(),
    });
    assert_eq!(
        assembler.blocks(),
        vec![ContentBlock::Text {
            text: "final".into()
        }]
    );
}

#[test]
fn assembler_drops_tool_calls_on_max_tokens_truncation() {
    let mut assembler = BlockAssembler::new();
    assembler.push(&StreamChunk::TextDelta {
        index: 0,
        text: "partial".into(),
    });
    assembler.push(&StreamChunk::ToolCallDelta {
        index: 1,
        id: CallId::new("call-1"),
        name: Some("read_file".into()),
        arguments_delta: "{\"path\":".into(),
    });
    assembler.push(&StreamChunk::Finish {
        reason: FinishReason::MaxTokens,
        replay_state: None,
    });
    assert_eq!(
        assembler.blocks(),
        vec![ContentBlock::Text {
            text: "partial".into()
        }]
    );
}

#[test]
fn assembler_synthesizes_tool_call_from_deltas() {
    let mut assembler = BlockAssembler::new();
    assembler.push(&StreamChunk::ToolCallDelta {
        index: 2,
        id: CallId::new("call-9"),
        name: Some("bash".into()),
        arguments_delta: "{\"cmd\":".into(),
    });
    assembler.push(&StreamChunk::ToolCallDelta {
        index: 2,
        id: CallId::new("call-9"),
        name: None,
        arguments_delta: "\"ls\"}".into(),
    });
    assert_eq!(
        assembler.blocks(),
        vec![ContentBlock::ToolCall {
            id: CallId::new("call-9"),
            name: "bash".into(),
            arguments: "{\"cmd\":\"ls\"}".into(),
        }]
    );
}

struct FixedAdapter {
    chunks: Vec<StreamChunk>,
}

impl LlmAdapter for FixedAdapter {
    fn stream(&self, _options: GenerateOptions) -> AdapterStream {
        futures::stream::iter(self.chunks.clone().into_iter().map(Ok)).boxed_local()
    }
}

struct FailingAdapter;
impl LlmAdapter for FailingAdapter {
    fn stream(&self, _options: GenerateOptions) -> AdapterStream {
        futures::stream::once(async { Err(anyhow::Error::new(LlmError::new("boom", "SERVER"))) })
            .boxed_local()
    }
}

fn options(provider: &str, model: &str) -> GenerateOptions {
    GenerateOptions {
        provider: provider.into(),
        model: model.into(),
        ..Default::default()
    }
}

#[test]
fn runtime_streams_through_registered_adapter() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let runtime = LlmRuntime::provide(&ctx).unwrap();
        runtime
            .register_adapter(
                &["deepseek".to_string()],
                Rc::new(FixedAdapter {
                    chunks: text_chunks(),
                }),
            )
            .unwrap();
        let chunks: Vec<StreamChunk> = runtime
            .stream(options("deepseek", "deepseek-chat"))
            .collect()
            .await;
        assert_eq!(chunks, text_chunks());
    });
}

#[test]
fn runtime_rejects_duplicate_and_unknown_providers() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let runtime = LlmRuntime::provide(&ctx).unwrap();
        runtime
            .register_adapter(
                &["deepseek".to_string()],
                Rc::new(FixedAdapter { chunks: vec![] }),
            )
            .unwrap();
        let duplicate = runtime
            .register_adapter(
                &["deepseek".to_string()],
                Rc::new(FixedAdapter { chunks: vec![] }),
            )
            .err()
            .unwrap();
        assert_eq!(duplicate.code, "DUPLICATE_ADAPTER");

        // Unknown provider becomes a terminal error finish chunk, not a throw.
        let chunks: Vec<StreamChunk> = runtime.stream(options("missing", "m")).collect().await;
        assert_eq!(chunks.len(), 1);
        match &chunks[0] {
            StreamChunk::Finish {
                reason: FinishReason::Error { failure },
                ..
            } => {
                assert_eq!(failure.code, "NO_ADAPTER");
            }
            other => panic!("expected error finish, got {other:?}"),
        }
    });
}

#[test]
fn adapter_iteration_failure_becomes_terminal_finish() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let runtime = LlmRuntime::provide(&ctx).unwrap();
        runtime
            .register_adapter(&["p".to_string()], Rc::new(FailingAdapter))
            .unwrap();
        let chunks: Vec<StreamChunk> = runtime.stream(options("p", "m")).collect().await;
        assert_eq!(chunks.len(), 1);
        match &chunks[0] {
            StreamChunk::Finish {
                reason: FinishReason::Error { failure },
                ..
            } => {
                assert_eq!(failure.code, "SERVER");
                assert_eq!(failure.message, "boom");
            }
            other => panic!("expected error finish, got {other:?}"),
        }
    });
}

#[test]
fn replace_swaps_routes_and_dispose_forbids_replace() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let runtime = LlmRuntime::provide(&ctx).unwrap();
        let handle = runtime
            .register_adapter(&["a".to_string()], Rc::new(FixedAdapter { chunks: vec![] }))
            .unwrap();
        handle.replace(&["b".to_string(), "c".to_string()]).unwrap();
        let providers: Vec<String> = runtime.list_providers().into_iter().map(|p| p.id).collect();
        assert_eq!(providers, vec!["b".to_string(), "c".to_string()]);

        // Empty replacement is legal on a live registration.
        handle.replace(&[]).unwrap();
        assert!(runtime.list_providers().is_empty());

        handle.dispose().await;
        let error = handle.replace(&["d".to_string()]).unwrap_err();
        assert_eq!(error.code, "REGISTRATION_DISPOSED");
    });
}

#[test]
fn llm_stream_waterfall_can_short_circuit() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let runtime = LlmRuntime::provide(&ctx).unwrap();
        runtime
            .register_adapter(
                &["p".to_string()],
                Rc::new(FixedAdapter {
                    chunks: text_chunks(),
                }),
            )
            .unwrap();
        // A listener that never calls next() replaces the adapter stream.
        ctx.on_waterfall::<LlmStream, _, _>(Default::default(), |_ctx, _options, _next| async {
            let replaced: ChunkStream = futures::stream::iter(vec![StreamChunk::Finish {
                reason: FinishReason::Stop,
                replay_state: None,
            }])
            .boxed_local();
            Ok(replaced)
        })
        .unwrap();
        let chunks: Vec<StreamChunk> = runtime.stream(options("p", "m")).collect().await;
        assert_eq!(
            chunks,
            vec![StreamChunk::Finish {
                reason: FinishReason::Stop,
                replay_state: None
            }]
        );
    });
}

#[test]
fn prepared_call_is_one_shot_and_checks_config() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let runtime = LlmRuntime::provide(&ctx).unwrap();
        runtime
            .register_adapter(
                &["p".to_string()],
                Rc::new(FixedAdapter {
                    chunks: text_chunks(),
                }),
            )
            .unwrap();
        let config = LlmCallConfig {
            provider: "p".into(),
            model: "m".into(),
            ..Default::default()
        };
        let prepared = runtime.prepare_call(&config, None).await.unwrap();

        // Mismatched config refuses dispatch.
        let error = prepared.stream(options("p", "other")).err().unwrap();
        assert_eq!(error.code, "INVALID_PREPARED_CALL");

        let chunks: Vec<StreamChunk> = prepared.stream(options("p", "m")).unwrap().collect().await;
        assert_eq!(chunks, text_chunks());

        // Second dispatch refused.
        let error = prepared.stream(options("p", "m")).err().unwrap();
        assert_eq!(error.code, "INVALID_PREPARED_CALL");
    });
}

#[test]
fn api_key_normalization_and_classifiers() {
    assert_eq!(normalize_api_key("  sk-abc  ").unwrap(), "sk-abc");
    assert_eq!(
        normalize_api_key("   ").unwrap_err(),
        ApiKeyRejection::Empty
    );
    assert_eq!(
        normalize_api_key("sk café").unwrap_err(),
        ApiKeyRejection::IllegalCharacters
    );
    let error = assert_usable_api_key("", "dsh-llm-deepseek", "DEEPSEEK_API_KEY").unwrap_err();
    assert_eq!(error.code, INVALID_CREDENTIAL_CODE);
    assert!(!error.message.contains("sk-"));

    assert!(is_context_window_exceeded_error(
        "This model's maximum context length is 65536"
    ));
    assert!(is_context_window_exceeded_error(
        "the prompt is too long for the model context"
    ));
    assert!(!is_context_window_exceeded_error("rate limit exceeded"));
    assert!(is_quota_exceeded_error(
        "insufficient_quota: your balance is depleted"
    ));
    assert!(!is_quota_exceeded_error("429 too many requests"));
}

#[test]
fn retry_policy_defaults_and_validation() {
    let policy = resolve_retry_policy(None, "llm: provider \"p\" retryPolicy").unwrap();
    match policy {
        ResolvedRetryPolicy::Normal {
            max_retries,
            retryable_codes,
            backoff,
        } => {
            assert_eq!(max_retries, 2);
            assert!(retryable_codes.contains(&"RATE_LIMIT".to_string()));
            assert_eq!(backoff.initial_delay_ms, 500.0);
            assert_eq!(backoff.max_delay_ms, 10_000.0);
        }
        other => panic!("expected normal policy, got {other:?}"),
    }
    let invalid = resolve_retry_policy(
        Some(&RetryPolicyConfig::Normal {
            max_retries: None,
            retryable_codes: Some(vec![]),
            backoff: None,
        }),
        "path",
    )
    .unwrap_err();
    assert!(invalid.contains("retryableCodes must not be empty"));
}

#[test]
fn replay_state_stripped_for_foreign_adapters() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let runtime = LlmRuntime::provide(&ctx).unwrap();

        // Echo adapter reports whether replay state survived, via the text.
        struct EchoAdapter;
        impl LlmAdapter for EchoAdapter {
            fn stream(&self, options: GenerateOptions) -> AdapterStream {
                let kept = options.messages.iter().any(|message| {
                    matches!(
                        &message.source,
                        MessageSource::Model(provenance) if provenance.replay_state.is_some()
                    )
                });
                futures::stream::iter(vec![Ok(StreamChunk::TextDelta {
                    index: 0,
                    text: if kept { "kept" } else { "stripped" }.to_string(),
                })])
                .boxed_local()
            }
        }
        runtime
            .register_adapter(&["p".to_string()], Rc::new(EchoAdapter))
            .unwrap();

        // History from a provider this adapter does not own → stripped.
        let mut request = options("p", "m");
        request.messages = vec![create_assistant_message(
            vec![ContentBlock::Text { text: "hi".into() }],
            AssistantProvenance {
                provider: "other-provider".into(),
                model: "m".into(),
                replay_state: Some(serde_json::json!({"cursor": 1})),
            },
        )];
        let chunks: Vec<StreamChunk> = runtime.stream(request).collect().await;
        assert_eq!(
            chunks,
            vec![StreamChunk::TextDelta {
                index: 0,
                text: "stripped".into()
            }]
        );

        // History from the provider this same adapter owns → kept.
        let mut request = options("p", "m");
        request.messages = vec![create_assistant_message(
            vec![ContentBlock::Text { text: "hi".into() }],
            AssistantProvenance {
                provider: "p".into(),
                model: "m".into(),
                replay_state: Some(serde_json::json!({"cursor": 1})),
            },
        )];
        let chunks: Vec<StreamChunk> = runtime.stream(request).collect().await;
        assert_eq!(
            chunks,
            vec![StreamChunk::TextDelta {
                index: 0,
                text: "kept".into()
            }]
        );
    });
}
