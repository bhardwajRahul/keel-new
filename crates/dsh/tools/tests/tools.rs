//! Behavior tests for the execution pipeline, mirroring the upstream
//! tools/execution suites: policy decisions, output contracts, context
//! plumbing, cancellation codes, and concurrency classification.

mod common;

use common::*;
use dsh_cordis::EventOptions;
use dsh_llm::{ContentBlock, HarnessError};
use dsh_scope::ScopedEvents;
use dsh_timeout::AbortController;
use dsh_tools::{
    DefineToolOptions, DefineToolOutput, PostAcceptReplacement, PostToolDecision, PreToolDecision,
    ScopedWaterfall, TOOL_ABORTED, TOOL_ABORTED_BEFORE_DISPATCH, ToolErrorInfo, ToolExecutionInput,
    ToolExecutionMode, ToolExecutionResult, ToolOutputDefinition, ToolsExecute, ToolsPostExecute,
    ToolsPreExecute, ToolsResult, define_tool,
};
use futures::FutureExt;
use serde_json::{Value, json};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

fn echo_tool() -> dsh_tools::ToolDefinition {
    define_tool(DefineToolOptions {
        name: "echo".into(),
        description: "echo arguments back".into(),
        parameters: params(json!({ "text": { "type": "string" } })),
        output: DefineToolOutput {
            schema: value_spec(json!({ "type": "string" })),
            render: Rc::new(|_args, value| {
                Ok(vec![ContentBlock::Text {
                    text: value.as_str().unwrap_or_default().into(),
                }])
            }),
            presentation_meta: None,
        },
        timeout_ms: None,
        is_concurrency_safe: None,
        execute: Rc::new(|args, _exec| {
            async move { Ok(json!(args["text"].as_str().unwrap_or_default())) }.boxed_local()
        }),
        finalize_content: None,
        present_call: None,
        present_result: None,
    })
    .unwrap()
}

fn abort_error_info(code: &str) -> ToolErrorInfo {
    ToolErrorInfo {
        name: "AbortError".into(),
        code: code.into(),
    }
}

#[test]
fn registers_tools_and_exposes_whitelisted_schemas() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        tools.register(&ctx, echo_tool()).unwrap();

        let schemas = tools.schemas(None);
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0].name, "echo");
        assert_eq!(schemas[0].description, "echo arguments back");
        assert_eq!(
            Value::Object(schemas[0].parameters.clone()),
            json!({ "type": "object", "properties": { "text": { "type": "string" } } })
        );

        // The system-prompt hook projects the same whitelist plus names.
        let provider = tools.schemas_provider();
        let wired = provider(None);
        assert_eq!(wired.known_names, vec!["echo".to_string()]);
        assert_eq!(wired.schemas.len(), 1);
    });
}

#[test]
fn executes_a_tool_and_notifies_the_final_result() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        tools.register(&ctx, echo_tool()).unwrap();
        let observed: Rc<RefCell<Option<ToolExecutionResult>>> = Rc::default();
        let sink = observed.clone();
        ctx.on_scoped::<ToolsResult, _, _>(
            EventOptions::default(),
            move |_ctx, (_exec, result)| {
                *sink.borrow_mut() = Some(result.clone());
                async { None }
            },
        )
        .unwrap();

        let result = tools.execute(input("echo", json!({ "text": "hi" }))).await;
        assert_eq!(
            result,
            ToolExecutionResult::Success {
                value: json!("hi"),
                content: vec![ContentBlock::Text { text: "hi".into() }],
                meta: None,
                additional_contexts: vec![],
                concludes_turn: false,
            }
        );
        assert_eq!(observed.borrow().as_ref(), Some(&result));
    });
}

#[test]
fn projects_presentation_meta_only_for_top_level_calls() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let mut definition = tool("meta-tool", "ok");
        definition.output.presentation_meta =
            Some(Rc::new(|_args, _value| Ok(json!({ "card": true }))));
        tools.register(&ctx, definition).unwrap();
        // Capture a real parent token from a sibling call.
        tools.register(&ctx, tool("parent", "ok")).unwrap();
        let token = Rc::new(Cell::new(None));
        let sink = token.clone();
        let capture = ctx
            .on_waterfall_scoped::<ToolsPreExecute, _, _>(
                EventOptions::default(),
                move |_ctx, exec, next| {
                    sink.set(Some(exec.token()));
                    next(exec)
                },
            )
            .unwrap();
        tools.execute(input("parent", json!({}))).await;
        capture.dispose().await;
        let parent = token.get().unwrap();

        let direct = tools.execute(input("meta-tool", json!({}))).await;
        assert_eq!(direct.meta(), Some(&json!({ "card": true })));

        let nested = tools
            .execute(ToolExecutionInput {
                parent: Some(parent),
                ..input("meta-tool", json!({}))
            })
            .await;
        assert_eq!(nested.meta(), None);
        assert_eq!(nested.value(), Some(&json!("ok")));
    });
}

#[test]
fn rejects_schema_mismatched_body_values() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        tools
            .register(
                &ctx,
                tool_with_execute("wrong-output", |_args, _exec| {
                    async { Ok(json!(42)) }.boxed_local()
                }),
            )
            .unwrap();

        let result = tools.execute(input("wrong-output", json!({}))).await;
        let error = result.error().unwrap();
        assert_eq!(
            error.info,
            Some(ToolErrorInfo {
                name: "ToolOutputError".into(),
                code: "INVALID_TOOL_OUTPUT".into()
            })
        );
        assert!(
            error.message.contains("\"value\" must be a string"),
            "{}",
            error.message
        );
        assert!(result.value().is_none());
    });
}

#[test]
fn contains_a_throwing_render_projector_as_one_failed_call() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let mut definition = tool("throwing-render", "ok");
        definition.output.render = Rc::new(|_args, _value| anyhow::bail!("renderer exploded"));
        tools.register(&ctx, definition).unwrap();

        let result = tools.execute(input("throwing-render", json!({}))).await;
        let error = result.error().unwrap();
        assert!(
            error
                .message
                .contains("output.render failed: renderer exploded")
        );
        assert_eq!(
            error.info,
            Some(ToolErrorInfo {
                name: "ToolOutputError".into(),
                code: "INVALID_TOOL_OUTPUT".into()
            })
        );
    });
}

#[test]
fn returns_is_error_results_for_unknown_and_throwing_tools() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        tools
            .register(
                &ctx,
                tool_with_execute("boom", |_args, _exec| {
                    async { Err(anyhow::anyhow!("exploded")) }.boxed_local()
                }),
            )
            .unwrap();
        tools
            .register(
                &ctx,
                tool_with_execute("coded", |_args, _exec| {
                    async { Err(HarnessError::new("disk full", "ENOSPC").into()) }.boxed_local()
                }),
            )
            .unwrap();

        let unknown = tools.execute(input("nope", json!({}))).await;
        assert_eq!(
            unknown.content(),
            &[ContentBlock::Text {
                text: "Error: unknown tool \"nope\"".into()
            }]
        );
        assert_eq!(
            unknown.error().unwrap(),
            &dsh_tools::ToolFailure {
                message: "unknown tool \"nope\"".into(),
                info: Some(ToolErrorInfo {
                    name: "ToolNotFoundError".into(),
                    code: "UNKNOWN_TOOL".into()
                }),
            }
        );

        let thrown = tools.execute(input("boom", json!({}))).await;
        assert_eq!(
            thrown.content(),
            &[ContentBlock::Text {
                text: "Error: exploded".into()
            }]
        );
        assert_eq!(thrown.error().unwrap().info, None);

        let coded = tools.execute(input("coded", json!({}))).await;
        assert_eq!(
            coded.error().unwrap(),
            &dsh_tools::ToolFailure {
                message: "disk full".into(),
                info: Some(ToolErrorInfo {
                    name: "HarnessError".into(),
                    code: "ENOSPC".into()
                }),
            }
        );
    });
}

#[test]
fn invalid_arguments_surface_the_structured_error() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        tools
            .register(
                &ctx,
                define_tool(DefineToolOptions {
                    name: "reader".into(),
                    description: "reads a path".into(),
                    parameters: params(json!({ "path": { "type": "string", "required": true } })),
                    output: DefineToolOutput {
                        schema: value_spec(json!({ "type": "string" })),
                        render: Rc::new(|_args, value| {
                            Ok(vec![ContentBlock::Text {
                                text: value.as_str().unwrap_or_default().into(),
                            }])
                        }),
                        presentation_meta: None,
                    },
                    timeout_ms: None,
                    is_concurrency_safe: None,
                    execute: Rc::new(|args, _exec| {
                        async move { Ok(args["path"].clone()) }.boxed_local()
                    }),
                    finalize_content: None,
                    present_call: None,
                    present_result: None,
                })
                .unwrap(),
            )
            .unwrap();

        let invalid = tools.execute(input("reader", json!({}))).await;
        assert_eq!(
            invalid.error().unwrap(),
            &dsh_tools::ToolFailure {
                message: "invalid arguments: missing required property \"path\"".into(),
                info: Some(ToolErrorInfo {
                    name: "ToolArgsError".into(),
                    code: "INVALID_ARGS".into()
                }),
            }
        );
        let valid = tools
            .execute(input("reader", json!({ "path": "/x" })))
            .await;
        assert_eq!(valid.value(), Some(&json!("/x")));
    });
}

#[test]
fn pre_execute_deny_short_circuits_before_the_execute_seam() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        tools.register(&ctx, echo_tool()).unwrap();
        let entered = Rc::new(Cell::new(false));
        let entered_probe = entered.clone();
        ctx.on_waterfall_scoped::<ToolsPreExecute, _, _>(
            EventOptions::default(),
            |_ctx, _exec, _next| async {
                Ok(PreToolDecision::Deny {
                    reason: "denied by policy".into(),
                })
            },
        )
        .unwrap();
        ctx.on_waterfall_scoped::<ToolsExecute, _, _>(
            EventOptions::default(),
            move |_ctx, exec, next| {
                entered_probe.set(true);
                next(exec)
            },
        )
        .unwrap();

        let result = tools.execute(input("echo", json!({ "text": "hi" }))).await;
        assert!(result.is_error());
        assert_eq!(
            result.content(),
            &[ContentBlock::Text {
                text: "Error: denied by policy".into()
            }]
        );
        assert_eq!(result.error().unwrap().info, None);
        assert!(!entered.get());
    });
}

#[test]
fn ask_degrades_to_deny_without_an_approval_seam() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        tools.register(&ctx, echo_tool()).unwrap();
        let with_reason = Rc::new(Cell::new(true));
        let flag = with_reason.clone();
        ctx.on_waterfall_scoped::<ToolsPreExecute, _, _>(
            EventOptions::default(),
            move |_ctx, _exec, _next| {
                let reason = flag.get().then(|| "needs approval".to_string());
                async move { Ok(PreToolDecision::Ask { reason }) }
            },
        )
        .unwrap();

        let reasoned = tools.execute(input("echo", json!({}))).await;
        assert_eq!(
            reasoned.content(),
            &[ContentBlock::Text {
                text: "Error: needs approval".into()
            }]
        );

        with_reason.set(false);
        let default = tools.execute(input("echo", json!({}))).await;
        assert_eq!(
            default.content(),
            &[ContentBlock::Text {
                text: "Error: tool \"echo\" requires approval (not yet supported)".into()
            }]
        );
    });
}

#[test]
fn ask_routes_through_the_approval_seam() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        tools.register(&ctx, echo_tool()).unwrap();
        let (_scope, agent, _key) = mint_agent(&ctx, "a", None);

        let seen: Rc<RefCell<Vec<(String, dsh_llm::CallId, Option<String>)>>> = Rc::default();
        let outcome = Rc::new(RefCell::new(dsh_tools::ApprovalOutcome::AllowedOnce));
        let seen_sink = seen.clone();
        let verdict = outcome.clone();
        ctx.provide_service(Rc::new(dsh_tools::ToolApproval::new(move |request| {
            seen_sink.borrow_mut().push((
                request.tool_name.clone(),
                request.call_id.clone(),
                request.reason.clone(),
            ));
            let verdict = *verdict.borrow();
            async move { verdict }.boxed_local()
        })))
        .unwrap();
        ctx.on_waterfall_scoped::<ToolsPreExecute, _, _>(
            EventOptions::default(),
            |_ctx, _exec, _next| async {
                Ok(PreToolDecision::Ask {
                    reason: Some("hook wants a human".into()),
                })
            },
        )
        .unwrap();

        // allowed-once dispatches, forwarding the ask fields.
        let granted = tools.execute(input_for("echo", Some(agent.clone()))).await;
        assert!(!granted.is_error());
        assert_eq!(
            seen.borrow().last().unwrap(),
            &(
                "echo".to_string(),
                dsh_llm::CallId::new("c1"),
                Some("hook wants a human".to_string())
            )
        );

        // The three non-grants deny with distinct reasons.
        *outcome.borrow_mut() = dsh_tools::ApprovalOutcome::Rejected;
        let rejected = tools.execute(input_for("echo", Some(agent.clone()))).await;
        assert_eq!(
            rejected.error().unwrap().message,
            "the user rejected tool \"echo\""
        );

        *outcome.borrow_mut() = dsh_tools::ApprovalOutcome::Cancelled;
        let cancelled = tools.execute(input_for("echo", Some(agent.clone()))).await;
        assert_eq!(
            cancelled.error().unwrap().message,
            "approval for tool \"echo\" was cancelled"
        );

        *outcome.borrow_mut() = dsh_tools::ApprovalOutcome::Unavailable;
        let unavailable = tools.execute(input_for("echo", Some(agent.clone()))).await;
        assert_eq!(
            unavailable.error().unwrap().message,
            "tool \"echo\" requires approval, but no approval channel is available"
        );

        // Agent-less executions deny without asking.
        let calls_before = seen.borrow().len();
        let agentless = tools.execute(input("echo", json!({}))).await;
        assert_eq!(
            agentless.error().unwrap().message,
            "tool \"echo\" requires approval, but the call has no agent to route it through"
        );
        assert_eq!(seen.borrow().len(), calls_before);
    });
}

#[test]
fn post_accept_replaces_content_and_block_turns_into_failure() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        tools.register(&ctx, echo_tool()).unwrap();
        let block = Rc::new(Cell::new(false));
        let mode = block.clone();
        ctx.on_waterfall_scoped::<ToolsPostExecute, _, _>(
            EventOptions::default(),
            move |_ctx, _args, _next| {
                let blocked = mode.get();
                async move {
                    Ok(if blocked {
                        PostToolDecision::Block {
                            feedback: vec![ContentBlock::Text {
                                text: "output rejected: try again".into(),
                            }],
                            additional_contexts: vec![],
                        }
                    } else {
                        PostToolDecision::Accept {
                            replace: Some(PostAcceptReplacement::Content(vec![
                                ContentBlock::Text {
                                    text: "rewritten".into(),
                                },
                            ])),
                            additional_contexts: vec![],
                        }
                    })
                }
            },
        )
        .unwrap();

        let accepted = tools.execute(input("echo", json!({ "text": "hi" }))).await;
        assert!(!accepted.is_error());
        assert_eq!(
            accepted.content(),
            &[ContentBlock::Text {
                text: "rewritten".into()
            }]
        );
        // The canonical value survives a content-only replacement.
        assert_eq!(accepted.value(), Some(&json!("hi")));

        block.set(true);
        let blocked = tools.execute(input("echo", json!({ "text": "hi" }))).await;
        assert!(blocked.is_error());
        assert_eq!(
            blocked.content(),
            &[ContentBlock::Text {
                text: "output rejected: try again".into()
            }]
        );
        assert_eq!(
            blocked.error().unwrap().message,
            "output rejected: try again"
        );
        assert!(blocked.value().is_none());
    });
}

#[test]
fn block_messages_stay_stable_for_empty_and_non_text_feedback() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        tools.register(&ctx, echo_tool()).unwrap();
        let feedback: Rc<RefCell<Vec<ContentBlock>>> = Rc::default();
        let source = feedback.clone();
        ctx.on_waterfall_scoped::<ToolsPostExecute, _, _>(
            EventOptions::default(),
            move |_ctx, _args, _next| {
                let feedback = source.borrow().clone();
                async move {
                    Ok(PostToolDecision::Block {
                        feedback,
                        additional_contexts: vec![],
                    })
                }
            },
        )
        .unwrap();

        let empty = tools.execute(input("echo", json!({}))).await;
        assert_eq!(
            empty.error().unwrap().message,
            "tool result blocked by post-execute policy"
        );

        *feedback.borrow_mut() = vec![ContentBlock::Reasoning {
            text: "private rationale".into(),
        }];
        let reasoning = tools.execute(input("echo", json!({}))).await;
        assert_eq!(reasoning.error().unwrap().message, "[reasoning content]");
    });
}

#[test]
fn value_replacement_recomputes_both_projections() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let mut definition = tool_with_execute("projected", |_args, _exec| {
            async { Ok(json!("body")) }.boxed_local()
        });
        definition.output = ToolOutputDefinition {
            schema: json!({ "type": "string" }),
            render: Rc::new(|_args, value| {
                Ok(vec![ContentBlock::Text {
                    text: format!("render:{}", value.as_str().unwrap_or_default()),
                }])
            }),
            presentation_meta: Some(Rc::new(|_args, value| Ok(json!({ "projected": value })))),
        };
        tools.register(&ctx, definition).unwrap();
        ctx.on_waterfall_scoped::<ToolsPostExecute, _, _>(
            EventOptions::default(),
            |_ctx, _args, _next| async {
                Ok(PostToolDecision::Accept {
                    replace: Some(PostAcceptReplacement::Value(json!("policy value"))),
                    additional_contexts: vec![plugin_message("value context", "test")],
                })
            },
        )
        .unwrap();

        let result = tools.execute(input("projected", json!({}))).await;
        assert_eq!(result.value(), Some(&json!("policy value")));
        assert_eq!(
            result.content(),
            &[ContentBlock::Text {
                text: "render:policy value".into()
            }]
        );
        assert_eq!(result.meta(), Some(&json!({ "projected": "policy value" })));
        assert_eq!(result.additional_contexts().len(), 1);
    });
}

#[test]
fn value_replacement_guards_failed_results_and_output_contracts() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        tools
            .register(
                &ctx,
                tool_with_execute("throw-before-replace", |_args, _exec| {
                    async { Err(anyhow::anyhow!("body failed")) }.boxed_local()
                }),
            )
            .unwrap();
        tools.register(&ctx, echo_tool()).unwrap();
        ctx.on_waterfall_scoped::<ToolsPostExecute, _, _>(
            EventOptions::default(),
            |_ctx, (exec, _result), _next| {
                let failed_call = exec.name() == "throw-before-replace";
                async move {
                    Ok(PostToolDecision::Accept {
                        replace: Some(PostAcceptReplacement::Value(if failed_call {
                            json!("replacement")
                        } else {
                            json!(1)
                        })),
                        additional_contexts: vec![],
                    })
                }
            },
        )
        .unwrap();

        let failed = tools
            .execute(input("throw-before-replace", json!({})))
            .await;
        assert_eq!(
            failed.error().unwrap().message,
            "tools/post-execute cannot replace the value of a failed result"
        );

        // A schema-invalid replacement fails the owning output contract.
        let invalid = tools.execute(input("echo", json!({}))).await;
        assert_eq!(
            invalid.error().unwrap().info.as_ref().unwrap().code,
            "INVALID_TOOL_OUTPUT"
        );
        assert!(invalid.value().is_none());
    });
}

#[test]
fn value_replacement_fails_when_the_owning_tool_disappears() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let handle = Rc::new(tools.register(&ctx, echo_tool()).unwrap());
        let disposer = handle.clone();
        ctx.on_waterfall_scoped::<ToolsPostExecute, _, _>(
            EventOptions::default(),
            move |_ctx, _args, _next| {
                let disposer = disposer.clone();
                async move {
                    disposer.dispose().await;
                    Ok(PostToolDecision::Accept {
                        replace: Some(PostAcceptReplacement::Value(json!("replacement"))),
                        additional_contexts: vec![],
                    })
                }
            },
        )
        .unwrap();

        let result = tools.execute(input("echo", json!({}))).await;
        assert_eq!(
            result.error().unwrap(),
            &dsh_tools::ToolFailure {
                message: "unknown tool \"echo\"".into(),
                info: Some(ToolErrorInfo {
                    name: "ToolNotFoundError".into(),
                    code: "UNKNOWN_TOOL".into()
                }),
            }
        );
    });
}

#[test]
fn preserves_deferred_wrapper_and_post_contexts_in_order() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        tools
            .register(
                &ctx,
                tool_with_execute("composite", |_args, exec| {
                    async move {
                        exec.defer_context(plugin_message("nested-1", "nested-1"));
                        exec.defer_context(plugin_message("nested-2", "nested-2"));
                        Ok(json!("done"))
                    }
                    .boxed_local()
                }),
            )
            .unwrap();
        ctx.on_waterfall_scoped::<ToolsExecute, _, _>(
            EventOptions::default(),
            |_ctx, exec, next| async move {
                let result = next(exec).await?;
                let mut contexts = result.additional_contexts().to_vec();
                contexts.push(plugin_message("wrapper", "wrapper"));
                let ToolExecutionResult::Success {
                    value,
                    content,
                    meta,
                    concludes_turn,
                    ..
                } = result
                else {
                    return Ok(result);
                };
                Ok(ToolExecutionResult::Success {
                    value,
                    content,
                    meta,
                    additional_contexts: contexts,
                    concludes_turn,
                })
            },
        )
        .unwrap();
        ctx.on_waterfall_scoped::<ToolsPostExecute, _, _>(
            EventOptions::default(),
            |_ctx, args, next| async move {
                let downstream = next(args).await?;
                let PostToolDecision::Accept {
                    replace,
                    mut additional_contexts,
                } = downstream
                else {
                    return Ok(downstream);
                };
                additional_contexts.insert(0, plugin_message("post", "post"));
                Ok(PostToolDecision::Accept {
                    replace,
                    additional_contexts,
                })
            },
        )
        .unwrap();

        let result = tools.execute(input("composite", json!({}))).await;
        let plugins: Vec<String> = result
            .additional_contexts()
            .iter()
            .map(|message| match &message.source {
                dsh_llm::MessageSource::Plugin { plugin, .. } => plugin.clone(),
                other => format!("{other:?}"),
            })
            .collect();
        assert_eq!(plugins, vec!["nested-1", "nested-2", "wrapper", "post"]);
    });
}

#[test]
fn keeps_deferred_contexts_on_failure_but_drops_them_on_block() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        tools
            .register(
                &ctx,
                tool_with_execute("failing-composite", |_args, exec| {
                    async move {
                        exec.defer_context(plugin_message("nested", "nested"));
                        Err(anyhow::anyhow!("outer failure"))
                    }
                    .boxed_local()
                }),
            )
            .unwrap();

        let failed = tools.execute(input("failing-composite", json!({}))).await;
        assert!(failed.is_error());
        assert_eq!(failed.additional_contexts().len(), 1);

        ctx.on_waterfall_scoped::<ToolsPostExecute, _, _>(
            EventOptions::default(),
            |_ctx, _args, _next| async {
                Ok(PostToolDecision::Block {
                    feedback: vec![ContentBlock::Text {
                        text: "blocked".into(),
                    }],
                    additional_contexts: vec![plugin_message("block-only", "blocker")],
                })
            },
        )
        .unwrap();
        let blocked = tools.execute(input("failing-composite", json!({}))).await;
        assert!(blocked.is_error());
        let sources: Vec<String> = blocked
            .additional_contexts()
            .iter()
            .map(|message| match &message.source {
                dsh_llm::MessageSource::Plugin { plugin, .. } => plugin.clone(),
                other => format!("{other:?}"),
            })
            .collect();
        assert_eq!(sources, vec!["blocker"]);
    });
}

#[test]
fn composes_the_three_waterfalls_around_dispatch_in_order() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let order: Rc<RefCell<Vec<&'static str>>> = Rc::default();
        let body_order = order.clone();
        tools
            .register(
                &ctx,
                tool_with_execute("traced", move |_args, _exec| {
                    body_order.borrow_mut().push("dispatch");
                    async { Ok(json!("hi")) }.boxed_local()
                }),
            )
            .unwrap();
        let pre = order.clone();
        ctx.on_waterfall_scoped::<ToolsPreExecute, _, _>(
            EventOptions::default(),
            move |_ctx, exec, next| {
                pre.borrow_mut().push("pre");
                next(exec)
            },
        )
        .unwrap();
        let around = order.clone();
        ctx.on_waterfall_scoped::<ToolsExecute, _, _>(
            EventOptions::default(),
            move |_ctx, exec, next| {
                around.borrow_mut().push("execute:before");
                let around = around.clone();
                async move {
                    let result = next(exec).await;
                    around.borrow_mut().push("execute:after");
                    result
                }
            },
        )
        .unwrap();
        let post = order.clone();
        ctx.on_waterfall_scoped::<ToolsPostExecute, _, _>(
            EventOptions::default(),
            move |_ctx, args, next| {
                post.borrow_mut().push("post");
                next(args)
            },
        )
        .unwrap();

        let result = tools.execute(input("traced", json!({}))).await;
        assert!(!result.is_error());
        assert_eq!(
            *order.borrow(),
            vec!["pre", "execute:before", "dispatch", "execute:after", "post"]
        );
    });
}

#[test]
fn wrapper_short_circuit_renormalizes_through_the_output_contract() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let dispatched = Rc::new(Cell::new(false));
        let body = dispatched.clone();
        tools
            .register(
                &ctx,
                tool_with_execute("never-runs", move |_args, _exec| {
                    body.set(true);
                    async { Ok(json!("unreachable")) }.boxed_local()
                }),
            )
            .unwrap();
        ctx.on_waterfall_scoped::<ToolsExecute, _, _>(
            EventOptions::default(),
            |_ctx, _exec, _next| async {
                Ok(ToolExecutionResult::Success {
                    value: json!("short-circuited"),
                    content: vec![ContentBlock::Text {
                        text: "ignored authored content".into(),
                    }],
                    meta: None,
                    additional_contexts: vec![plugin_message("from around dispatch", "test")],
                    concludes_turn: false,
                })
            },
        )
        .unwrap();

        let result = tools.execute(input("never-runs", json!({}))).await;
        assert!(!dispatched.get());
        // Authored content is discarded: the owning contract re-renders.
        assert_eq!(
            result.content(),
            &[ContentBlock::Text {
                text: "short-circuited".into()
            }]
        );
        assert_eq!(result.value(), Some(&json!("short-circuited")));
        assert_eq!(result.additional_contexts().len(), 1);
    });
}

#[test]
fn wrapper_authored_failures_pass_through_with_metadata() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        tools.register(&ctx, echo_tool()).unwrap();
        ctx.on_waterfall_scoped::<ToolsExecute, _, _>(
            EventOptions::default(),
            |_ctx, _exec, _next| async {
                Ok(ToolExecutionResult::Failure {
                    error: dsh_tools::ToolFailure {
                        message: "wrapped failure".into(),
                        info: None,
                    },
                    content: vec![ContentBlock::Text {
                        text: "wrapper content".into(),
                    }],
                    meta: Some(json!({ "wrapped": true })),
                    additional_contexts: vec![plugin_message("wrapper context", "test")],
                })
            },
        )
        .unwrap();

        let result = tools.execute(input("echo", json!({}))).await;
        assert!(result.is_error());
        assert_eq!(result.error().unwrap().message, "wrapped failure");
        assert_eq!(
            result.content(),
            &[ContentBlock::Text {
                text: "wrapper content".into()
            }]
        );
        assert_eq!(result.meta(), Some(&json!({ "wrapped": true })));
        assert_eq!(result.additional_contexts().len(), 1);
    });
}

#[test]
fn listener_errors_normalize_to_failure_results() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        tools.register(&ctx, echo_tool()).unwrap();
        let stage: Rc<RefCell<&'static str>> = Rc::new(RefCell::new("pre"));

        let pre_stage = stage.clone();
        ctx.on_waterfall_scoped::<ToolsPreExecute, _, _>(
            EventOptions::default(),
            move |_ctx, exec, next| {
                let throws = *pre_stage.borrow() == "pre";
                async move {
                    if throws {
                        Err(HarnessError::new("denied", "DENIED").into())
                    } else {
                        next(exec).await
                    }
                }
            },
        )
        .unwrap();
        let around_stage = stage.clone();
        ctx.on_waterfall_scoped::<ToolsExecute, _, _>(
            EventOptions::default(),
            move |_ctx, exec, next| {
                let throws = *around_stage.borrow() == "around";
                async move {
                    if throws {
                        Err(anyhow::anyhow!("wrapper broke"))
                    } else {
                        next(exec).await
                    }
                }
            },
        )
        .unwrap();
        let post_stage = stage.clone();
        ctx.on_waterfall_scoped::<ToolsPostExecute, _, _>(
            EventOptions::default(),
            move |_ctx, args, next| {
                let throws = *post_stage.borrow() == "post";
                async move {
                    if throws {
                        Err(anyhow::anyhow!("post hook broke"))
                    } else {
                        next(args).await
                    }
                }
            },
        )
        .unwrap();

        let pre = tools.execute(input("echo", json!({}))).await;
        assert_eq!(
            pre.error().unwrap(),
            &dsh_tools::ToolFailure {
                message: "denied".into(),
                info: Some(ToolErrorInfo {
                    name: "HarnessError".into(),
                    code: "DENIED".into()
                }),
            }
        );

        *stage.borrow_mut() = "around";
        let around = tools.execute(input("echo", json!({}))).await;
        assert_eq!(around.error().unwrap().message, "wrapper broke");

        *stage.borrow_mut() = "post";
        let post = tools.execute(input("echo", json!({}))).await;
        assert_eq!(post.error().unwrap().message, "post hook broke");
    });
}

#[test]
fn pre_aborted_calls_skip_every_pipeline_phase() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let phases: Rc<RefCell<Vec<&'static str>>> = Rc::default();
        let body = phases.clone();
        tools
            .register(
                &ctx,
                tool_with_execute("domain-abort", move |_args, _exec| {
                    body.borrow_mut().push("body");
                    async { Ok(json!("unreachable")) }.boxed_local()
                }),
            )
            .unwrap();
        let pre = phases.clone();
        ctx.on_waterfall_scoped::<ToolsPreExecute, _, _>(
            EventOptions::default(),
            move |_ctx, exec, next| {
                pre.borrow_mut().push("pre");
                next(exec)
            },
        )
        .unwrap();
        let post = phases.clone();
        ctx.on_waterfall_scoped::<ToolsPostExecute, _, _>(
            EventOptions::default(),
            move |_ctx, args, next| {
                post.borrow_mut().push("post");
                next(args)
            },
        )
        .unwrap();
        let results = phases.clone();
        ctx.on_scoped::<ToolsResult, _, _>(
            EventOptions::default(),
            move |_ctx, (_exec, result)| {
                results.borrow_mut().push(if result.is_error() {
                    "result"
                } else {
                    "result-ok"
                });
                async { None }
            },
        )
        .unwrap();

        let controller = AbortController::new();
        controller.abort("already cancelled");
        let result = tools
            .execute(ToolExecutionInput {
                signal: controller.signal(),
                ..input("domain-abort", json!({ "nested": { "value": 1 } }))
            })
            .await;

        assert_eq!(*phases.borrow(), vec!["result"]);
        assert_eq!(
            result.error().unwrap(),
            &dsh_tools::ToolFailure {
                message: "tool call aborted before dispatch".into(),
                info: Some(abort_error_info(TOOL_ABORTED_BEFORE_DISPATCH)),
            }
        );
        assert_eq!(
            result.content(),
            &[ContentBlock::Text {
                text: "Error: tool call aborted before dispatch".into()
            }]
        );
    });
}

#[test]
fn cancellation_during_pre_execute_skips_dispatch() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let dispatched = Rc::new(Cell::new(false));
        let body = dispatched.clone();
        tools
            .register(
                &ctx,
                tool_with_execute("must-not-run", move |_args, _exec| {
                    body.set(true);
                    async { Ok(json!("")) }.boxed_local()
                }),
            )
            .unwrap();
        let entered = Rc::new(Cell::new(false));
        let release = Rc::new(Cell::new(false));
        let entered_flag = entered.clone();
        let release_flag = release.clone();
        ctx.on_waterfall_scoped::<ToolsPreExecute, _, _>(
            EventOptions::default(),
            move |_ctx, exec, next| {
                let entered = entered_flag.clone();
                let release = release_flag.clone();
                async move {
                    entered.set(true);
                    until(&release).await;
                    next(exec).await
                }
            },
        )
        .unwrap();

        let controller = AbortController::new();
        let runtime = tools.clone();
        let signal = controller.signal();
        let pending = tokio::task::spawn_local(async move {
            runtime
                .execute(ToolExecutionInput {
                    signal,
                    ..input("must-not-run", json!({}))
                })
                .await
        });
        until(&entered).await;
        controller.abort("cancelled in policy");
        release.set(true);

        let result = pending.await.unwrap();
        assert_eq!(
            result.error().unwrap().info,
            Some(abort_error_info(TOOL_ABORTED_BEFORE_DISPATCH))
        );
        assert!(!dispatched.get());
    });
}

#[test]
fn a_denial_that_settles_after_cancellation_is_preserved() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        tools.register(&ctx, echo_tool()).unwrap();
        let entered = Rc::new(Cell::new(false));
        let release = Rc::new(Cell::new(false));
        let entered_flag = entered.clone();
        let release_flag = release.clone();
        ctx.on_waterfall_scoped::<ToolsPreExecute, _, _>(
            EventOptions::default(),
            move |_ctx, _exec, _next| {
                let entered = entered_flag.clone();
                let release = release_flag.clone();
                async move {
                    entered.set(true);
                    until(&release).await;
                    Ok(PreToolDecision::Deny {
                        reason: "policy denied the call".into(),
                    })
                }
            },
        )
        .unwrap();

        let controller = AbortController::new();
        let runtime = tools.clone();
        let signal = controller.signal();
        let pending = tokio::task::spawn_local(async move {
            runtime
                .execute(ToolExecutionInput {
                    signal,
                    ..input("echo", json!({}))
                })
                .await
        });
        until(&entered).await;
        controller.abort("cancelled while policy decided");
        release.set(true);

        let result = pending.await.unwrap();
        assert_eq!(
            result.error().unwrap(),
            &dsh_tools::ToolFailure {
                message: "policy denied the call".into(),
                info: None
            }
        );
    });
}

#[test]
fn a_late_wrapper_success_becomes_aborted_and_keeps_deferred_contexts() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        tools
            .register(
                &ctx,
                tool_with_execute("completed-before-wrapper", |_args, exec| {
                    async move {
                        exec.defer_context(plugin_message("completed child work", "child"));
                        Ok(json!("body complete"))
                    }
                    .boxed_local()
                }),
            )
            .unwrap();
        let entered = Rc::new(Cell::new(false));
        let release = Rc::new(Cell::new(false));
        let entered_flag = entered.clone();
        let release_flag = release.clone();
        ctx.on_waterfall_scoped::<ToolsExecute, _, _>(
            EventOptions::default(),
            move |_ctx, exec, next| {
                let entered = entered_flag.clone();
                let release = release_flag.clone();
                async move {
                    let result = next(exec).await;
                    entered.set(true);
                    until(&release).await;
                    result
                }
            },
        )
        .unwrap();

        let controller = AbortController::new();
        let runtime = tools.clone();
        let signal = controller.signal();
        let pending = tokio::task::spawn_local(async move {
            runtime
                .execute(ToolExecutionInput {
                    signal,
                    ..input("completed-before-wrapper", json!({}))
                })
                .await
        });
        until(&entered).await;
        controller.abort("cancelled while wrapper settled");
        release.set(true);

        let result = pending.await.unwrap();
        assert_eq!(
            result.error().unwrap().info,
            Some(abort_error_info(TOOL_ABORTED))
        );
        assert_eq!(
            result.content(),
            &[ContentBlock::Text {
                text: "Error: tool call aborted".into()
            }]
        );
        assert_eq!(result.additional_contexts().len(), 1);
    });
}

#[test]
fn a_wrapper_supplied_pre_aborted_signal_skips_dispatch() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let dispatched = Rc::new(Cell::new(false));
        let body = dispatched.clone();
        tools
            .register(
                &ctx,
                tool_with_execute("must-not-run", move |_args, _exec| {
                    body.set(true);
                    async { Ok(json!("")) }.boxed_local()
                }),
            )
            .unwrap();
        let replacement = AbortController::new();
        replacement.abort("wrapper cancelled");
        let replacement_signal = replacement.signal();
        ctx.on_waterfall_scoped::<ToolsExecute, _, _>(
            EventOptions::default(),
            move |_ctx, exec, next| {
                let replacement = replacement_signal.clone();
                async move {
                    let upstream = exec.signal();
                    exec.set_signal(replacement);
                    let result = next(exec.clone()).await;
                    exec.set_signal(upstream);
                    result
                }
            },
        )
        .unwrap();

        let result = tools.execute(input("must-not-run", json!({}))).await;
        assert_eq!(
            result.error().unwrap(),
            &dsh_tools::ToolFailure {
                message: "tool call aborted before dispatch".into(),
                info: Some(abort_error_info(TOOL_ABORTED_BEFORE_DISPATCH)),
            }
        );
        assert!(!dispatched.get());
    });
}

#[test]
fn caller_cancellation_is_fused_into_a_wrapper_replacement() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let entered = Rc::new(Cell::new(false));
        let entered_flag = entered.clone();
        tools
            .register(
                &ctx,
                tool_with_execute("cooperative", move |_args, exec| {
                    let entered = entered_flag.clone();
                    async move {
                        entered.set(true);
                        // The body observes the FUSED signal: a caller abort
                        // must reach it through the wrapper's replacement.
                        exec.signal().wait().await;
                        Ok(json!("stopped"))
                    }
                    .boxed_local()
                }),
            )
            .unwrap();
        let replacement = AbortController::new();
        let replacement_signal = replacement.signal();
        ctx.on_waterfall_scoped::<ToolsExecute, _, _>(
            EventOptions::default(),
            move |_ctx, exec, next| {
                let replacement = replacement_signal.clone();
                async move {
                    let upstream = exec.signal();
                    exec.set_signal(replacement);
                    let result = next(exec.clone()).await;
                    exec.set_signal(upstream);
                    result
                }
            },
        )
        .unwrap();

        let controller = AbortController::new();
        let runtime = tools.clone();
        let signal = controller.signal();
        let pending = tokio::task::spawn_local(async move {
            runtime
                .execute(ToolExecutionInput {
                    signal,
                    ..input("cooperative", json!({}))
                })
                .await
        });
        until(&entered).await;
        controller.abort("cancel running body");

        let result = pending.await.unwrap();
        assert_eq!(
            result.error().unwrap().info,
            Some(abort_error_info(TOOL_ABORTED))
        );
        assert!(!replacement.signal().aborted());
    });
}

#[test]
fn execution_mode_classifies_fail_closed() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let mut safe = tool("safe", "ok");
        safe.is_concurrency_safe = Some(Rc::new(|_args| true));
        tools.register(&ctx, safe).unwrap();
        tools.register(&ctx, tool("plain", "ok")).unwrap();
        let mut dependent = tool("rw", "ok");
        dependent.is_concurrency_safe = Some(Rc::new(|args| args["mode"].as_str() == Some("read")));
        tools.register(&ctx, dependent).unwrap();
        // define_tool classifiers fail closed on invalid arguments.
        tools
            .register(
                &ctx,
                define_tool(DefineToolOptions {
                    name: "needs-mode".into(),
                    description: "requires mode".into(),
                    parameters: params(json!({ "mode": { "type": "string", "required": true } })),
                    output: DefineToolOutput {
                        schema: value_spec(json!({ "type": "null" })),
                        render: Rc::new(|_args, _value| Ok(vec![])),
                        presentation_meta: None,
                    },
                    timeout_ms: None,
                    is_concurrency_safe: Some(Rc::new(|_args| true)),
                    execute: Rc::new(|_args, _exec| async { Ok(json!(null)) }.boxed_local()),
                    finalize_content: None,
                    present_call: None,
                    present_result: None,
                })
                .unwrap(),
            )
            .unwrap();

        let mode = |name: &str, args: Value| tools.execution_mode(&input(name, args));
        assert_eq!(mode("safe", json!({})), ToolExecutionMode::Parallel);
        assert_eq!(mode("plain", json!({})), ToolExecutionMode::Exclusive);
        assert_eq!(mode("nonexistent", json!({})), ToolExecutionMode::Exclusive);
        assert_eq!(
            mode("rw", json!({ "mode": "read" })),
            ToolExecutionMode::Parallel
        );
        assert_eq!(
            mode("rw", json!({ "mode": "write" })),
            ToolExecutionMode::Exclusive
        );
        assert_eq!(mode("needs-mode", json!({})), ToolExecutionMode::Exclusive);
        assert_eq!(
            mode("needs-mode", json!({ "mode": "x" })),
            ToolExecutionMode::Parallel
        );
    });
}

#[test]
fn a_nested_conclusion_rides_the_nested_result_for_its_composite() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        tools
            .register(
                &ctx,
                tool_with_execute("terminal-nested", |_args, exec| {
                    async move {
                        exec.conclude_turn();
                        Ok(json!("terminal"))
                    }
                    .boxed_local()
                }),
            )
            .unwrap();
        let runtime = tools.clone();
        let call = Rc::new(Cell::new(0u32));
        tools
            .register(
                &ctx,
                tool_with_execute("composite", move |_args, exec| {
                    let runtime = runtime.clone();
                    let call = call.clone();
                    async move {
                        call.set(call.get() + 1);
                        let nested = runtime
                            .execute(ToolExecutionInput {
                                call_id: dsh_llm::CallId::new(format!("nested-{}", call.get())),
                                parent: Some(exec.token()),
                                signal: exec.signal(),
                                ..input("terminal-nested", json!({}))
                            })
                            .await;
                        if nested.concludes_turn() {
                            exec.conclude_turn();
                        }
                        Ok(json!(if nested.is_error() {
                            "nested failed, composite recovered"
                        } else {
                            "nested succeeded"
                        }))
                    }
                    .boxed_local()
                }),
            )
            .unwrap();

        // A policy blocking the nested success leaves no marker to forward.
        let veto = Rc::new(
            ctx.on_waterfall_scoped::<ToolsPostExecute, _, _>(
                EventOptions::default(),
                |_ctx, (exec, result), next| {
                    let vetoed = exec.name() == "terminal-nested";
                    async move {
                        if vetoed {
                            Ok(PostToolDecision::Block {
                                feedback: vec![ContentBlock::Text {
                                    text: "nested success rejected".into(),
                                }],
                                additional_contexts: vec![],
                            })
                        } else {
                            next((exec, result)).await
                        }
                    }
                },
            )
            .unwrap(),
        );
        let recovered = tools.execute(input("composite", json!({}))).await;
        assert!(!recovered.is_error());
        assert!(!recovered.concludes_turn());
        veto.dispose().await;

        // The same nested call succeeding carries the marker forward.
        let concluded = tools.execute(input("composite", json!({}))).await;
        assert!(!concluded.is_error());
        assert!(concluded.concludes_turn());
    });
}

#[test]
fn finalize_content_survives_disposal_and_runs_once_per_outcome() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let finalize_calls = Rc::new(Cell::new(0u32));
        let counter = finalize_calls.clone();
        let mut bounded = tool("bounded", "body");
        bounded.finalize_content = Some(Rc::new(move |exec, result| {
            counter.set(counter.get() + 1);
            assert_eq!(exec.name(), "bounded");
            assert!(result.is_error());
            Some(vec![ContentBlock::Text {
                text: "bounded failure".into(),
            }])
        }));
        let handle = Rc::new(tools.register(&ctx, bounded).unwrap());
        let disposer = handle.clone();
        ctx.on_waterfall_scoped::<ToolsPreExecute, _, _>(
            EventOptions::default(),
            move |_ctx, _exec, _next| {
                let disposer = disposer.clone();
                async move {
                    disposer.dispose().await;
                    Err(HarnessError::new("policy failed", "POLICY_FAILED").into())
                }
            },
        )
        .unwrap();

        let result = tools.execute(input("bounded", json!({}))).await;
        assert_eq!(
            result,
            ToolExecutionResult::Failure {
                content: vec![ContentBlock::Text {
                    text: "bounded failure".into()
                }],
                error: dsh_tools::ToolFailure {
                    message: "policy failed".into(),
                    info: Some(ToolErrorInfo {
                        name: "HarnessError".into(),
                        code: "POLICY_FAILED".into()
                    }),
                },
                meta: None,
                additional_contexts: vec![],
            }
        );
        assert_eq!(finalize_calls.get(), 1);
    });
}

#[test]
fn finalize_content_returning_none_preserves_content() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let finalized = Rc::new(Cell::new(0u32));
        let counter = finalized.clone();
        let mut identity = tool("identity-finalizer", "kept");
        identity.finalize_content = Some(Rc::new(move |_exec, _result| {
            counter.set(counter.get() + 1);
            None
        }));
        tools.register(&ctx, identity).unwrap();

        let result = tools.execute(input("identity-finalizer", json!({}))).await;
        assert!(!result.is_error());
        assert_eq!(
            result.content(),
            &[ContentBlock::Text {
                text: "kept".into()
            }]
        );
        assert_eq!(finalized.get(), 1);
    });
}

#[test]
fn registration_validates_timeout_and_reserves_run_code() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let mut zero = tool("zero-timeout", "ok");
        zero.timeout_ms = Some(0.0);
        assert!(
            tools
                .register(&ctx, zero)
                .unwrap_err()
                .to_string()
                .contains("timeoutMs must be a positive finite number")
        );
        let mut infinite = tool("infinite-timeout", "ok");
        infinite.timeout_ms = Some(f64::INFINITY);
        assert!(
            tools
                .register(&ctx, infinite)
                .unwrap_err()
                .to_string()
                .contains("timeoutMs must be a positive finite number")
        );

        let reserved = tools.register(&ctx, tool("run_code", "ok")).unwrap_err();
        assert!(
            reserved
                .to_string()
                .contains("reserved for the Code Mode presentation transport")
        );
    });
}

#[test]
fn code_presentation_modes_are_a_clear_unimplemented_error() {
    dsh_cordis::run(async {
        let app = dsh_cordis::App::new();
        let ctx = app.root();
        let error = provide_config(
            &ctx,
            dsh_tools::Config {
                mode: Some(dsh_tools::ToolPresentationMode::Code),
                max_parallel_sub_calls: None,
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("not implemented"), "{error}");

        let zero = provide_config(
            &ctx,
            dsh_tools::Config {
                mode: None,
                max_parallel_sub_calls: Some(0),
            },
        )
        .unwrap_err();
        assert!(
            zero.to_string()
                .contains("maxParallelSubCalls must be a positive integer")
        );

        let tools = provide_config(
            &ctx,
            dsh_tools::Config {
                mode: None,
                max_parallel_sub_calls: Some(3),
            },
        )
        .unwrap();
        assert_eq!(tools.max_parallel_sub_calls(), 3);
    });
}
