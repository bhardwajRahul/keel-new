//! Behavior tests mirroring the upstream llm-deepseek suites: request
//! serialization (thinking rules, tool-result expansion, image rejection),
//! SSE framing, chunk translation (deferred block-ends, empty-response
//! mapping, cache-disjoint usage), and HTTP error-code mapping.

use dsh_llm::{
    AssistantProvenance, BlockType, CallId, CallPurpose, ContentBlock, FinishReason,
    GenerateOptions, MessageSource, ReasoningEffortId, StreamChunk, ToolResultMessageInput,
    create_assistant_message, create_tool_result_message, create_user_message,
};
use dsh_llm_deepseek::*;

fn options(model: &str) -> GenerateOptions {
    GenerateOptions {
        provider: "deepseek".into(),
        model: model.into(),
        ..Default::default()
    }
}

#[test]
fn serializes_conversation_with_tool_results_expanded() {
    let mut request = options("deepseek-chat");
    request.system = Some("be brief".into());
    request.messages = vec![
        create_user_message(
            vec![ContentBlock::Text { text: "hi".into() }],
            MessageSource::User,
        ),
        create_assistant_message(
            vec![
                ContentBlock::Text { text: "".into() },
                ContentBlock::Reasoning {
                    text: "thinking...".into(),
                },
                ContentBlock::ToolCall {
                    id: CallId::new("call-1"),
                    name: "bash".into(),
                    arguments: "{}".into(),
                },
            ],
            AssistantProvenance {
                provider: "deepseek".into(),
                model: "deepseek-chat".into(),
                replay_state: None,
            },
        ),
        create_tool_result_message(ToolResultMessageInput {
            call_id: CallId::new("call-1"),
            content: vec![],
            is_error: false,
        }),
    ];
    let wire = serialize_request(&request, &RequestDefaults::default()).unwrap();
    let value = serde_json::to_value(&wire).unwrap();
    assert_eq!(value["stream"], true);
    assert_eq!(value["stream_options"]["include_usage"], true);
    let messages = value["messages"].as_array().unwrap();
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[1]["role"], "user");
    // Tool-call assistant turn: content "" (never null), reasoning passback
    // present because the turn carried tool calls.
    assert_eq!(messages[2]["role"], "assistant");
    assert_eq!(messages[2]["content"], "");
    assert_eq!(messages[2]["reasoning_content"], "thinking...");
    assert_eq!(messages[2]["tool_calls"][0]["function"]["name"], "bash");
    // Tool result becomes a standalone tool-role message with placeholder
    // content for empty output.
    assert_eq!(messages[3]["role"], "tool");
    assert_eq!(messages[3]["tool_call_id"], "call-1");
    assert_eq!(messages[3]["content"], "(no output)");
}

#[test]
fn reasoning_passback_dropped_on_plain_turns() {
    let mut request = options("m");
    request.messages = vec![create_assistant_message(
        vec![
            ContentBlock::Reasoning { text: "cot".into() },
            ContentBlock::Text {
                text: "answer".into(),
            },
        ],
        AssistantProvenance {
            provider: "p".into(),
            model: "m".into(),
            replay_state: None,
        },
    )];
    let wire = serialize_request(&request, &RequestDefaults::default()).unwrap();
    let value = serde_json::to_value(&wire).unwrap();
    assert_eq!(value["messages"][0]["content"], "answer");
    assert!(value["messages"][0].get("reasoning_content").is_none());
}

#[test]
fn image_content_rejected() {
    let mut request = options("m");
    request.messages = vec![create_user_message(
        vec![ContentBlock::Image {
            attachment: serde_json::json!({}),
        }],
        MessageSource::User,
    )];
    let error = serialize_request(&request, &RequestDefaults::default()).unwrap_err();
    assert_eq!(error.code, "UNSUPPORTED_CONTENT");
}

#[test]
fn thinking_rules() {
    // Explicit high effort → thinking enabled + effort on the wire.
    let mut request = options("m");
    request.reasoning_effort = Some(ReasoningEffortId::new("high"));
    let wire = serialize_request(&request, &RequestDefaults::default()).unwrap();
    let value = serde_json::to_value(&wire).unwrap();
    assert_eq!(value["thinking"]["type"], "enabled");
    assert_eq!(value["reasoning_effort"], "high");

    // Effort "off" → thinking disabled, no wire effort.
    let mut request = options("m");
    request.reasoning_effort = Some(ReasoningEffortId::new("off"));
    let value =
        serde_json::to_value(&serialize_request(&request, &RequestDefaults::default()).unwrap())
            .unwrap();
    assert_eq!(value["thinking"]["type"], "disabled");
    assert!(value.get("reasoning_effort").is_none());

    // Session-title purpose always disables thinking.
    let mut request = options("m");
    request.purpose = Some(CallPurpose::SessionTitle);
    request.reasoning_effort = Some(ReasoningEffortId::new("max"));
    let value =
        serde_json::to_value(&serialize_request(&request, &RequestDefaults::default()).unwrap())
            .unwrap();
    assert_eq!(value["thinking"]["type"], "disabled");

    // Unknown effort rejected.
    let mut request = options("m");
    request.reasoning_effort = Some(ReasoningEffortId::new("medium"));
    let error = serialize_request(&request, &RequestDefaults::default()).unwrap_err();
    assert_eq!(error.code, "UNSUPPORTED_REASONING_EFFORT");

    // Deployment with thinking disabled rejects non-off efforts.
    let defaults = RequestDefaults {
        thinking: Some("disabled".into()),
        reasoning_effort: None,
    };
    let mut request = options("m");
    request.reasoning_effort = Some(ReasoningEffortId::new("high"));
    let error = serialize_request(&request, &defaults).unwrap_err();
    assert_eq!(error.code, "UNSUPPORTED_REASONING_EFFORT");
}

#[test]
fn sse_decoder_frames_events() {
    let mut decoder = SseDecoder::new();
    let mut comments = Vec::new();
    // Split mid-line and mid-event; comment line; CRLF endings.
    let mut payloads = decoder.feed(b": keep-alive\r\ndata: {\"a\":", |c| {
        comments.push(c.to_string())
    });
    assert!(payloads.is_empty());
    payloads.extend(decoder.feed(b"1}\r\n\r\ndata: [DONE]\n\n", |c| {
        comments.push(c.to_string())
    }));
    assert_eq!(
        payloads,
        vec!["{\"a\":1}".to_string(), "[DONE]".to_string()]
    );
    assert_eq!(comments, vec!["keep-alive".to_string()]);

    // Multi-data joining.
    let mut decoder = SseDecoder::new();
    let payloads = decoder.feed(b"data: line1\ndata: line2\n\n", |_| {});
    assert_eq!(payloads, vec!["line1\nline2".to_string()]);

    // Unterminated tail is not dispatched.
    let mut decoder = SseDecoder::new();
    let payloads = decoder.feed(b"data: truncated", |_| {});
    assert!(payloads.is_empty());
}

#[test]
fn translator_defers_ends_and_maps_empty_response() {
    let mut translator = Translator::new();
    // Empty-string first reasoning chunk must not open a block.
    let chunks = translator
        .feed(r#"{"choices":[{"delta":{"reasoning_content":""}}]}"#)
        .unwrap();
    assert!(chunks.is_empty());
    let chunks = translator
        .feed(r#"{"choices":[{"delta":{"reasoning_content":"think","content":"hi"}}]}"#)
        .unwrap();
    assert_eq!(chunks.len(), 4); // reasoning start+delta, text start+delta
    assert!(matches!(
        chunks[0],
        StreamChunk::BlockStart {
            block_type: BlockType::Reasoning,
            ..
        }
    ));
    let chunks = translator
        .feed(r#"{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":3,"prompt_cache_hit_tokens":4}}"#)
        .unwrap();
    assert!(chunks.is_empty()); // finish + usage deferred
    let done = translator.done();
    // block-end ×2, usage, finish — nothing follows finish.
    assert_eq!(done.len(), 4);
    match &done[2] {
        StreamChunk::Usage { usage } => {
            // Disjoint counts: cache hits subtracted from input.
            assert_eq!(usage.input_tokens, 6);
            assert_eq!(usage.cache_read_tokens, Some(4));
        }
        other => panic!("expected usage, got {other:?}"),
    }
    assert!(matches!(
        &done[3],
        StreamChunk::Finish {
            reason: FinishReason::Stop,
            ..
        }
    ));
}

#[test]
fn translator_tool_call_fragments_concatenate() {
    let mut translator = Translator::new();
    translator
        .feed(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"bash","arguments":"{\"cm"}}]}}]}"#)
        .unwrap();
    translator
        .feed(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"d\":1}"}}]},"finish_reason":"tool_calls"}]}"#)
        .unwrap();
    let done = translator.done();
    match &done[0] {
        StreamChunk::BlockEnd {
            block:
                ContentBlock::ToolCall {
                    id,
                    name,
                    arguments,
                },
            ..
        } => {
            assert_eq!(id.as_str(), "c1");
            assert_eq!(name, "bash");
            assert_eq!(arguments, "{\"cmd\":1}");
        }
        other => panic!("expected tool-call block end, got {other:?}"),
    }
    assert!(matches!(
        &done[1],
        StreamChunk::Finish {
            reason: FinishReason::ToolCalls,
            ..
        }
    ));
}

#[test]
fn translator_empty_completion_becomes_empty_response_error() {
    let mut translator = Translator::new();
    translator
        .feed(r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#)
        .unwrap();
    let done = translator.done();
    assert_eq!(done.len(), 1);
    match &done[0] {
        StreamChunk::Finish {
            reason: FinishReason::Error { failure },
            ..
        } => {
            assert_eq!(failure.code, "EMPTY_RESPONSE");
        }
        other => panic!("expected error finish, got {other:?}"),
    }
}

#[test]
fn translator_rejects_malformed_payload_and_maps_unknown_finish() {
    let mut translator = Translator::new();
    let error = translator.feed("not json").unwrap_err();
    assert_eq!(error.code, "MALFORMED_RESPONSE");
    assert!(matches!(
        map_finish_reason("content_filter"),
        FinishReason::Error { failure } if failure.code == "CONTENT_FILTER"
    ));
}

#[test]
fn http_error_codes() {
    assert_eq!(http_error_code(401, None), "AUTH");
    assert_eq!(http_error_code(403, None), "AUTH");
    assert_eq!(http_error_code(429, None), "RATE_LIMIT");
    assert_eq!(http_error_code(400, None), "INVALID_REQUEST");
    assert_eq!(http_error_code(500, None), "SERVER");
    assert_eq!(http_error_code(502, None), "SERVER");
    assert_eq!(http_error_code(418, None), "HTTP_418");
    let context = WireErrorDetail {
        message: Some("This model's maximum context length is 65536 tokens".into()),
        ..Default::default()
    };
    assert_eq!(
        http_error_code(400, Some(&context)),
        "CONTEXT_WINDOW_EXCEEDED"
    );
    let quota = WireErrorDetail {
        message: Some("insufficient_quota: balance depleted".into()),
        ..Default::default()
    };
    assert_eq!(http_error_code(429, Some(&quota)), "QUOTA");
}
