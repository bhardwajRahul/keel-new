//! The model-facing `bash` tool over the `ctx.shell` executor (port of
//! upstream `packages/shell/tool-bash`: `index.ts` + `render.ts`).
//!
//! Divergences: `run_in_background` is not ported (the jobs tier is out of
//! scope), so every call is a foreground run and the output schema keeps
//! only the foreground arm; the sandbox escalation fields, system-prompt
//! section, and `ctx.shellEnv` collection are not ported (the executor still
//! enforces the managed `DSH_*` namespace for trusted callers that pass a
//! snapshot). Truncation reports the loss without a spill path — the spill
//! store is not ported.

use crate::shell::{
    ExitStatusMarker, LocalBashExecutor, ShellExecRequest, ShellRunResult, parse_exit_status,
};
use dsh_cordis::{Context, EffectHandle};
use dsh_llm::{ContentBlock, HarnessError};
use dsh_tools::{
    DefineToolOptions, DefineToolOutput, GenericResultView, ParameterSchemaSpec, TOOL_ABORTED,
    TerminalCallView, TerminalResultView, ToolCallView, ToolExecution, ToolResultView, ToolRuntime,
    ValueSchemaSpec, define_tool,
};
use futures::FutureExt;
use serde_json::{Value, json};
use std::path::Path;
use std::rc::Rc;

fn validate_bash_args(args: &Value) -> anyhow::Result<()> {
    if args["command"]
        .as_str()
        .unwrap_or_default()
        .trim()
        .is_empty()
    {
        anyhow::bail!("invalid command: expected a non-empty string");
    }
    if args["description"]
        .as_str()
        .unwrap_or_default()
        .trim()
        .is_empty()
    {
        anyhow::bail!("invalid description: expected a non-empty string");
    }
    if let Some(timeout) = args.get("timeoutMs").filter(|value| !value.is_null()) {
        let ok = timeout
            .as_f64()
            .is_some_and(|number| number.is_finite() && number > 0.0);
        if !ok {
            anyhow::bail!("invalid timeoutMs: expected a positive number, got {timeout}");
        }
    }
    Ok(())
}

/// An explicit workdir wins (a relative one resolves against the session
/// workspace); otherwise the session cwd; otherwise `None` leaves executor
/// defaulting in place.
fn resolve_workdir(model_workdir: Option<&str>, exec: &ToolExecution) -> Option<String> {
    let session_cwd = exec
        .agent()
        .and_then(|agent| agent.session().header.cwd.clone());
    match model_workdir {
        None => session_cwd,
        Some(workdir) if !Path::new(workdir).is_absolute() => match session_cwd {
            Some(cwd) => Some(Path::new(&cwd).join(workdir).to_string_lossy().into_owned()),
            None => Some(workdir.to_string()),
        },
        Some(workdir) => Some(workdir.to_string()),
    }
}

fn stream_text(text: &str, truncated: bool) -> String {
    if truncated {
        format!("{text}\n[output truncated; full output: (unavailable)]")
    } else {
        text.to_string()
    }
}

/// Shape one finished run into the text the model sees: stdout, a marked
/// stderr section, then timeout/signal/exit markers. Non-zero exits are
/// reported, not errored — the model decides how to react. The exit marker
/// is kept last because [`parse_exit_status`] anchors there.
pub fn render_result(result: &ShellRunResult) -> String {
    let out = stream_text(&result.stdout.text, result.stdout.truncated);
    let err = stream_text(&result.stderr.text, result.stderr.truncated);
    let mut body = out;
    if !err.is_empty() {
        if !body.is_empty() && !body.ends_with('\n') {
            body.push('\n');
        }
        body.push_str(&format!("[stderr]\n{err}"));
    }
    if body.is_empty() {
        body = "(no output)".to_string();
    }
    let mut markers: Vec<String> = Vec::new();
    // A command may trap SIGTERM and exit 0 after timeout; the interruption
    // is still reported.
    if result.timed_out {
        markers.push(format!("[timed out after {}ms]", result.timeout_ms));
    }
    match (&result.signal, result.exit_code) {
        (Some(signal), _) => markers.push(format!("[killed by signal: {signal}]")),
        (None, Some(code)) if code != 0 => markers.push(format!("[exit code: {code}]")),
        _ => {}
    }
    if markers.is_empty() {
        return body;
    }
    if !body.ends_with('\n') {
        body.push('\n');
    }
    format!("{body}{}", markers.join("\n"))
}

fn value_of_run(result: &ShellRunResult) -> Value {
    json!({
        "exitCode": result.exit_code,
        "signal": result.signal,
        "timedOut": result.timed_out,
        "aborted": result.aborted,
        "timeoutMs": result.timeout_ms,
        "stdout": { "text": result.stdout.text, "truncated": result.stdout.truncated },
        "stderr": { "text": result.stderr.text, "truncated": result.stderr.truncated },
    })
}

fn run_of_value(value: &Value) -> ShellRunResult {
    let stream = |node: &Value| crate::shell::CollectedOutput {
        text: node["text"].as_str().unwrap_or_default().to_string(),
        truncated: node["truncated"].as_bool().unwrap_or(false),
    };
    ShellRunResult {
        exit_code: value["exitCode"].as_i64().map(|code| code as i32),
        signal: value["signal"].as_str().map(str::to_string),
        timed_out: value["timedOut"].as_bool().unwrap_or(false),
        aborted: value["aborted"].as_bool().unwrap_or(false),
        timeout_ms: value["timeoutMs"].as_f64().unwrap_or(0.0),
        stdout: stream(&value["stdout"]),
        stderr: stream(&value["stderr"]),
    }
}

/// Register the `bash` tool; the returned handle is the exact registration
/// disposer.
pub fn register_bash_tool(
    ctx: &Context,
    tools: &Rc<ToolRuntime>,
    shell: &Rc<LocalBashExecutor>,
) -> anyhow::Result<EffectHandle> {
    let exec_shell = shell.clone();
    let definition = define_tool(DefineToolOptions {
        name: "bash".into(),
        description: "Execute a bash command (`bash -c`) and return its stdout/stderr. \
            Each call runs in a fresh shell: no state (cwd, variables, functions) persists between calls — \
            pass `workdir` instead of using `cd`. Non-zero exits are reported as `[exit code: N]`. \
            Long output is truncated to its tail. Long-running commands must finish within the timeout."
            .into(),
        parameters: ParameterSchemaSpec::from_author_value(
            &json!({
                "command": { "type": "string", "required": true, "description": "The bash command to execute." },
                "description": {
                    "type": "string",
                    "required": true,
                    "description": "Clear, concise description of what this command does in active voice, 5-10 words (shown in the UI). Examples: \"ls\" → \"List files in current directory\"; \"git status\" → \"Show working tree status\".",
                },
                "timeoutMs": { "type": "number", "description": "Timeout in milliseconds. The executor applies its configured default and cap, and kills the command on expiry." },
                "workdir": { "type": "string", "description": "Working directory for this command. Defaults to the session workspace; a relative path is resolved against it." },
            }),
            "parameters",
        )?,
        output: DefineToolOutput {
            schema: ValueSchemaSpec::from_author_value(
                &json!({
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "exitCode": { "required": true, "oneOf": [{ "type": "integer" }, { "type": "null" }] },
                        "signal": { "required": true, "oneOf": [{ "type": "string" }, { "type": "null" }] },
                        "timedOut": { "type": "boolean", "required": true },
                        "aborted": { "type": "boolean", "required": true },
                        "timeoutMs": { "type": "number", "required": true },
                        "stdout": {
                            "type": "object",
                            "required": true,
                            "additionalProperties": false,
                            "properties": {
                                "text": { "type": "string", "required": true },
                                "truncated": { "type": "boolean", "required": true },
                            },
                        },
                        "stderr": {
                            "type": "object",
                            "required": true,
                            "additionalProperties": false,
                            "properties": {
                                "text": { "type": "string", "required": true },
                                "truncated": { "type": "boolean", "required": true },
                            },
                        },
                    },
                }),
                "schema",
            )?,
            render: Rc::new(|_args, value| {
                Ok(vec![ContentBlock::Text { text: render_result(&run_of_value(value)) }])
            }),
            presentation_meta: None,
        },
        timeout_ms: None,
        is_concurrency_safe: None,
        execute: Rc::new(move |args, exec| {
            let shell = exec_shell.clone();
            async move {
                validate_bash_args(&args)?;
                let workdir = resolve_workdir(args["workdir"].as_str(), &exec);
                let request = ShellExecRequest {
                    command: args["command"].as_str().unwrap_or_default().to_string(),
                    workdir,
                    timeout_ms: args["timeoutMs"].as_f64(),
                    signal: Some(exec.signal()),
                    ..Default::default()
                };
                let result = shell.run(shell.resolve(request)?).await?;
                if result.aborted {
                    return Err(HarnessError::new("tool call aborted", TOOL_ABORTED).into());
                }
                Ok(value_of_run(&result))
            }
            .boxed_local()
        }),
        finalize_content: None,
        // The command is the card title; a capable UI renders a terminal.
        present_call: Some(Rc::new(|args| {
            Some(ToolCallView::Terminal(TerminalCallView {
                title: args["command"].as_str().unwrap_or_default().to_string(),
                description: args["description"].as_str().map(str::to_string),
                cwd: args["workdir"].as_str().map(str::to_string),
            }))
        })),
        // Completed foreground output becomes a terminal card whose exit
        // marker turns into the exit pill; errors keep fenced generic output.
        present_result: Some(Rc::new(|_args, result| {
            let text = match result.content.as_slice() {
                [ContentBlock::Text { text }] => text,
                _ => return None,
            };
            if result.is_error {
                return Some(ToolResultView::Generic(GenericResultView {
                    title: None,
                    content: Some(vec![ContentBlock::Text {
                        text: format!("```console\n{}\n```", text.trim_end_matches('\n')),
                    }]),
                }));
            }
            let parsed = parse_exit_status(text);
            let (exit_code, signal) = match parsed.status {
                ExitStatusMarker::Code(code) => (Some(code), None),
                ExitStatusMarker::Signal(name) => (None, Some(name)),
            };
            Some(ToolResultView::Terminal(TerminalResultView {
                title: None,
                output: Some(parsed.body),
                exit_code,
                signal,
            }))
        })),
    })?;
    tools.register(ctx, definition)
}
