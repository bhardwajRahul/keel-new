//! One assistant step's tool-call scheduling, ported from
//! `packages/core/agent-loop/src/tool-calls.ts`.
//!
//! Contract preserved: calls run in model order with exclusive calls as
//! barriers; results, durable events, and result contexts commit in model
//! order; abort records synthetic error results for skipped calls so replay
//! stays valid; a result carrying `concludes_turn` reports turn conclusion.
//!
//! Divergence (ponytail: sequential dispatch, bounded-parallel pool if
//! step latency matters): calls classified `Parallel` still execute one at a
//! time here. Every observable ordering contract holds — only wall-clock
//! overlap is absent.

use dsh_agent::AgentRef;
use dsh_llm::{CallId, ContentBlock, ToolResultMessageInput, create_tool_result_message};
use dsh_session::{Session, SessionEventData, SurfaceIntent};
use dsh_tools::{
    TOOL_ABORTED_BEFORE_DISPATCH, ToolExecutionInput, ToolExecutionResult, ToolRuntime,
};
use std::rc::Rc;

/// One tool-call block lifted from the assistant message.
pub struct PlannedCall {
    pub call_id: CallId,
    pub name: String,
    /// Raw argument JSON exactly as the model produced it.
    pub raw_arguments: String,
}

/// Parse model arguments: invalid JSON is preserved as a text value for the
/// tool's own validation to reject; empty input maps to `{}`.
fn parse_arguments(raw: &str) -> serde_json::Value {
    if raw.is_empty() {
        return serde_json::Value::Object(serde_json::Map::new());
    }
    serde_json::from_str(raw).unwrap_or_else(|_| serde_json::Value::String(raw.to_string()))
}

/// Outcome of one step's tool executions.
pub struct ToolCallsOutcome {
    /// Whether any committed result declared the turn complete.
    pub concluded: bool,
    pub dispatched: u32,
    pub denied: u32,
    pub failed: u32,
    pub aborted: u32,
}

/// Append the durable record of one skipped (never-started) call: its
/// `tool/call` marker and a synthetic aborted-before-dispatch error result.
fn append_skipped(session: &Session, turn: u64, step: u64, call: &PlannedCall) {
    let _ = session.append(
        SessionEventData::ToolCall {
            turn,
            step,
            call_id: call.call_id.clone(),
            name: call.name.clone(),
            arguments: call.raw_arguments.clone(),
        },
        None,
    );
    let message = create_tool_result_message(ToolResultMessageInput {
        call_id: call.call_id.clone(),
        content: vec![ContentBlock::Text {
            text: "The tool call was skipped because the step was cancelled before it started."
                .into(),
        }],
        is_error: true,
    });
    let _ = session.append(
        SessionEventData::ToolResult {
            turn,
            step,
            message,
            error: Some(dsh_session::ToolErrorRef {
                name: "ToolAbortedError".into(),
                code: TOOL_ABORTED_BEFORE_DISPATCH.into(),
            }),
            meta: None,
        },
        Some(SurfaceIntent::append()),
    );
}

/// A selected host step can narrow the advertised schemas, but the model may
/// still emit an unadvertised call. Record a matching error result without
/// executing it so replay and the next model step remain well formed.
fn append_not_admitted(session: &Session, turn: u64, step: u64, call: &PlannedCall) {
    let _ = session.append(
        SessionEventData::ToolCall {
            turn,
            step,
            call_id: call.call_id.clone(),
            name: call.name.clone(),
            arguments: call.raw_arguments.clone(),
        },
        None,
    );
    let message = create_tool_result_message(ToolResultMessageInput {
        call_id: call.call_id.clone(),
        content: vec![ContentBlock::Text {
            text: "The host decision did not admit this tool for this step. Replan from the available tools."
                .into(),
        }],
        is_error: true,
    });
    let _ = session.append(
        SessionEventData::ToolResult {
            turn,
            step,
            message,
            error: Some(dsh_session::ToolErrorRef {
                name: "ToolNotAdmittedError".into(),
                code: "DECISION_TOOL_NOT_ADMITTED".into(),
            }),
            meta: None,
        },
        Some(SurfaceIntent::append()),
    );
}

/// Execute one assistant step's tool calls in model order, committing each
/// call's durable `tool/call` + `tool/result` pair and routing committed
/// result contexts to `accept_context` for the next step boundary. Abort
/// stops new dispatches and records synthetic results for the rest.
pub async fn execute_tool_calls(
    tools: &Rc<ToolRuntime>,
    agent: &AgentRef,
    session: &Rc<Session>,
    turn: u64,
    step: u64,
    calls: Vec<PlannedCall>,
    admitted_tool_names: Option<&[String]>,
    signal: &dsh_timeout::AbortSignal,
    mut accept_context: impl FnMut(dsh_llm::Message),
) -> ToolCallsOutcome {
    let mut concluded = false;
    let mut dispatched = 0u32;
    let mut denied = 0u32;
    let mut failed = 0u32;
    let mut aborted = 0u32;
    let mut index = 0;
    while index < calls.len() {
        if signal.aborted() {
            for call in &calls[index..] {
                append_skipped(session, turn, step, call);
                aborted = aborted.saturating_add(1);
            }
            return ToolCallsOutcome {
                concluded,
                dispatched,
                denied,
                failed,
                aborted,
            };
        }
        let call = &calls[index];
        index += 1;
        if admitted_tool_names.is_some_and(|names| !names.iter().any(|name| name == &call.name)) {
            append_not_admitted(session, turn, step, call);
            denied = denied.saturating_add(1);
            continue;
        }

        // The durable call marker commits when the pipeline is entered.
        let _ = session.append(
            SessionEventData::ToolCall {
                turn,
                step,
                call_id: call.call_id.clone(),
                name: call.name.clone(),
                arguments: call.raw_arguments.clone(),
            },
            None,
        );

        let result = tools
            .execute(ToolExecutionInput {
                call_id: call.call_id.clone(),
                root_call_id: None,
                name: call.name.clone(),
                arguments: parse_arguments(&call.raw_arguments),
                agent: Some(agent.clone()),
                parent: None,
                signal: signal.clone(),
            })
            .await;
        dispatched = dispatched.saturating_add(1);

        let (content, is_error, error_ref, meta, contexts, concludes) = match result {
            ToolExecutionResult::Success {
                content,
                meta,
                additional_contexts,
                concludes_turn,
                ..
            } => (
                content,
                false,
                None,
                meta,
                additional_contexts,
                concludes_turn,
            ),
            ToolExecutionResult::Failure {
                error,
                content,
                meta,
                additional_contexts,
            } => (
                content,
                true,
                error.info.map(|info| dsh_session::ToolErrorRef {
                    name: info.name,
                    code: info.code,
                }),
                meta,
                additional_contexts,
                false,
            ),
        };
        let message = create_tool_result_message(ToolResultMessageInput {
            call_id: call.call_id.clone(),
            content,
            is_error,
        });
        if is_error {
            failed = failed.saturating_add(1);
        }
        let _ = session.append(
            SessionEventData::ToolResult {
                turn,
                step,
                message,
                error: error_ref,
                meta,
            },
            Some(SurfaceIntent::append()),
        );
        for context in contexts {
            accept_context(context);
        }
        concluded |= concludes;
    }
    ToolCallsOutcome {
        concluded,
        dispatched,
        denied,
        failed,
        aborted,
    }
}
