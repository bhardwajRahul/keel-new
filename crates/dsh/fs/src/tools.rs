//! Model-facing `read` / `write` / `edit` tools over the local filesystem
//! (port of upstream `packages/fs/tool-fs`: `read.ts`, `write.ts`,
//! `edit.ts`, `error.ts`, `session-cwd.ts`, `read-target.ts`). This module
//! owns schemas, argument validation, read windows, formatting, and
//! observation events; guarded-mutation intents come from the single-slot
//! `fs/*` events, so without [`crate::install_observation_policy`] the tools
//! keep the provider's unconditional behavior.
//!
//! Divergences: system-prompt sections and the sandbox escalation fields are
//! not ported (no dsh-system-prompt wiring here; the sandbox tier is out of
//! scope); `read` never streams — the provider reads whole text and the
//! window caps still bound what the model sees; `read_image` is not ported
//! (no attachment store).

use crate::diff::{compute_hunk_diffs, diffs_from_meta};
use crate::events::{FsEditIntentSlot, FsObservedEvent, FsWriteIntentSlot, owner_of};
use crate::local::LocalFileSystem;
use crate::read_render::{
    FileReadOutcome, READ_LIMIT, READ_MAX_BYTES, READ_MAX_LINE_LENGTH, ReadWindow, build_window,
    format_read_output, lang_from_path, read_meta_from_meta,
};
use crate::types::{
    FsEditRequest, FsEntryType, FsError, FsErrorCode, FsObservation, FsWriteOperation,
};
use dsh_cordis::{Context, EffectHandle};
use dsh_llm::{ContentBlock, HarnessError};
use dsh_tools::{
    DefineToolOptions, DefineToolOutput, DiffCallView, DiffResultView, FileDiff, FileLocation,
    GenericCallView, ParameterSchemaSpec, ReadResultView, ToolCallKind, ToolCallView,
    ToolExecution, ToolResultView, ToolRuntime, ValueSchemaSpec, define_tool,
};
use futures::FutureExt;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::rc::Rc;

/// Resolved read caps (upstream plugin config after defaulting).
#[derive(Debug, Clone, Copy)]
pub struct FsToolsConfig {
    /// Default and maximum number of lines one `read` call returns.
    pub read_limit: u64,
    /// Per-line character cap before truncation.
    pub read_max_line_length: usize,
    /// Byte cap on the selected lines of one `read` call.
    pub read_max_bytes: usize,
}

impl Default for FsToolsConfig {
    fn default() -> Self {
        FsToolsConfig {
            read_limit: READ_LIMIT,
            read_max_line_length: READ_MAX_LINE_LENGTH,
            read_max_bytes: READ_MAX_BYTES,
        }
    }
}

/// The session workspace cwd for one call: each session's filesystem tools
/// act on ITS workspace, not the server launch directory. `None` (a
/// non-agent call) leaves the fallback to the provider's configured cwd.
pub(crate) fn session_cwd(exec: &ToolExecution) -> Option<PathBuf> {
    exec.agent()
        .and_then(|agent| agent.session().header.cwd.clone())
        .map(PathBuf::from)
}

/// Project a typed filesystem failure into the registry's error vocabulary,
/// preserving the stable `FS_*` code as machine-routable metadata.
fn fs_tool_error(error: FsError) -> anyhow::Error {
    HarnessError::new(error.message.clone(), error.code.as_str()).into()
}

/// Append the model-facing recovery instruction to a guarded-mutation
/// failure: a stale basis recovers only by re-reading, an unobserved target
/// by reading. Codes are preserved; other errors pass through.
fn remediate_fs_error(error: FsError) -> anyhow::Error {
    let remedy = match error.code {
        FsErrorCode::StaleVersion => Some("re-read the file, then retry"),
        FsErrorCode::NotObserved => Some("read the file, then retry"),
        _ => None,
    };
    match remedy {
        Some(remedy) => HarnessError::new(
            format!("{} — {}", error.message, remedy),
            error.code.as_str(),
        )
        .into(),
        None => fs_tool_error(error),
    }
}

/// Remediate an error that may carry an [`FsError`] (the intent slot can
/// raise one through the waterfall).
fn remediate_any(error: anyhow::Error) -> anyhow::Error {
    match error.downcast::<FsError>() {
        Ok(fs_error) => remediate_fs_error(fs_error),
        Err(other) => other,
    }
}

fn params(spec: Value) -> anyhow::Result<ParameterSchemaSpec> {
    Ok(ParameterSchemaSpec::from_author_value(&spec, "parameters")?)
}

fn value_spec(spec: Value) -> anyhow::Result<ValueSchemaSpec> {
    Ok(ValueSchemaSpec::from_author_value(&spec, "schema")?)
}

// --- read -------------------------------------------------------------

struct ReadInput {
    file_path: String,
    offset: u64,
    limit: u64,
}

fn positive_integer(value: &Value, name: &str) -> anyhow::Result<u64> {
    let ok = value
        .as_f64()
        .filter(|number| number.is_finite() && number.fract() == 0.0 && *number >= 1.0);
    match ok {
        Some(number) => Ok(number as u64),
        None => anyhow::bail!("{name} must be a positive integer"),
    }
}

fn parse_read_args(args: &Value, max_limit: u64) -> anyhow::Result<ReadInput> {
    let file_path = args["file_path"].as_str().unwrap_or_default();
    if file_path.trim().is_empty() {
        anyhow::bail!("file_path must be a non-empty string");
    }
    let offset = match args.get("offset") {
        None | Some(Value::Null) => 1,
        Some(value) => positive_integer(value, "offset")?,
    };
    let limit = match args.get("limit") {
        None | Some(Value::Null) => max_limit,
        Some(value) => positive_integer(value, "limit")?,
    };
    if limit > max_limit {
        anyhow::bail!("limit must be less than or equal to {max_limit}");
    }
    Ok(ReadInput {
        file_path: file_path.to_string(),
        offset,
        limit,
    })
}

fn read_lines_of(value: &Value) -> Vec<dsh_tools::ReadFileLine> {
    value["lines"]
        .as_array()
        .map(|lines| {
            lines
                .iter()
                .map(|line| dsh_tools::ReadFileLine {
                    number: line["number"].as_u64().unwrap_or(0),
                    text: line["text"].as_str().unwrap_or_default().to_string(),
                })
                .collect()
        })
        .unwrap_or_default()
}

fn register_read_tool(
    ctx: &Context,
    tools: &Rc<ToolRuntime>,
    fs: &Rc<LocalFileSystem>,
    config: FsToolsConfig,
) -> anyhow::Result<EffectHandle> {
    let exec_ctx = ctx.clone();
    let exec_fs = fs.clone();
    let definition = define_tool(DefineToolOptions {
        name: "read".into(),
        description: "Read a UTF-8 text file and return line-numbered content.".into(),
        parameters: params(json!({
            "file_path": { "type": "string", "required": true, "description": "Path to read, resolved by the filesystem backend." },
            "offset": { "type": "number", "description": "1-based first line to return. Defaults to 1." },
            "limit": { "type": "number", "description": format!("Maximum number of lines to return. Defaults to {}.", config.read_limit) },
        }))?,
        output: DefineToolOutput {
            schema: value_spec(json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "path": { "type": "string", "required": true },
                    "offset": { "type": "integer", "required": true },
                    "lines": {
                        "type": "array",
                        "required": true,
                        "items": {
                            "type": "object",
                            "additionalProperties": false,
                            "properties": {
                                "number": { "type": "integer", "required": true },
                                "text": { "type": "string", "required": true },
                            },
                        },
                    },
                    "totalLines": { "type": "integer", "required": true },
                },
            }))?,
            render: Rc::new(move |args, value| {
                let limit = args["limit"].as_u64().unwrap_or(config.read_limit);
                let lines = read_lines_of(value);
                let offset = value["offset"].as_u64().unwrap_or(1);
                let total_lines = value["totalLines"].as_u64().unwrap_or(0);
                let end_line = lines
                    .last()
                    .map(|line| line.number)
                    .unwrap_or(offset.saturating_sub(1));
                let truncated_by_bytes = (lines.len() as u64) < limit && end_line < total_lines;
                let text = format_read_output(
                    value["path"].as_str().unwrap_or_default(),
                    &FileReadOutcome {
                        offset,
                        lines: &lines,
                        total_lines,
                        truncated_by_bytes,
                    },
                );
                Ok(vec![ContentBlock::Text { text }])
            }),
            // The structured window rides persisted meta so a UI's read card
            // survives replay: only the rendered text is on the wire.
            presentation_meta: Some(Rc::new(|_args, value| {
                let mut meta = json!({
                    "path": value["path"],
                    "offset": value["offset"],
                    "lines": value["lines"],
                    "totalLines": value["totalLines"],
                });
                if let Some(lang) = value["path"].as_str().and_then(lang_from_path) {
                    meta["lang"] = json!(lang);
                }
                Ok(meta)
            })),
        },
        timeout_ms: None,
        // Observation races fail closed: guarded mutations re-check the
        // version atomically, so overlapping reads are safe.
        is_concurrency_safe: Some(Rc::new(|_args| true)),
        execute: Rc::new(move |args, exec| {
            let ctx = exec_ctx.clone();
            let fs = exec_fs.clone();
            async move {
                let input = parse_read_args(&args, config.read_limit)?;
                let signal = exec.signal();
                let owner = owner_of(&exec);
                let cwd = session_cwd(&exec);
                let target = fs
                    .resolve(&input.file_path, cwd.as_deref(), Some(&signal))
                    .map_err(fs_tool_error)?;
                let info = fs.stat(&target, Some(&signal)).map_err(fs_tool_error)?;
                let Some(info) = info else {
                    ctx.parallel::<FsObservedEvent>(&(
                        target.clone(),
                        FsObservation::Absent,
                        owner,
                    ))
                    .await;
                    return Err(fs_tool_error(FsError::new(
                        format!("cannot read \"{}\": not found", target.display_path),
                        FsErrorCode::NotFound,
                    )));
                };
                if info.entry_type != FsEntryType::File {
                    return Err(fs_tool_error(FsError::new(
                        format!(
                            "cannot read \"{}\": not a regular file",
                            target.display_path
                        ),
                        FsErrorCode::NotRegularFile,
                    )));
                }
                let text = fs
                    .read_text(&target, Some(&signal))
                    .map_err(fs_tool_error)?;
                let window = build_window(
                    &text,
                    ReadWindow {
                        offset: input.offset,
                        limit: input.limit,
                        max_line_length: config.read_max_line_length,
                        max_bytes: config.read_max_bytes,
                    },
                    &target.display_path,
                )
                .map_err(fs_tool_error)?;
                let lines: Vec<Value> = window
                    .lines
                    .iter()
                    .map(|line| json!({ "number": line.number, "text": line.text }))
                    .collect();
                ctx.parallel::<FsObservedEvent>(&(
                    target.clone(),
                    FsObservation::Present {
                        version: info.version,
                    },
                    owner,
                ))
                .await;
                Ok(json!({
                    "path": target.display_path,
                    "offset": input.offset,
                    "lines": lines,
                    "totalLines": window.total_lines,
                }))
            }
            .boxed_local()
        }),
        finalize_content: None,
        present_call: Some(Rc::new(|args| {
            let file_path = args["file_path"].as_str().unwrap_or_default().to_string();
            let offset = args["offset"].as_u64();
            let limit = args["limit"].as_u64();
            let window = match (offset, limit) {
                (_, Some(limit)) if limit > 0 => {
                    format!(
                        " ({} - {})",
                        offset.unwrap_or(1),
                        offset.unwrap_or(1) + limit - 1
                    )
                }
                (Some(offset), _) => format!(" (from line {offset})"),
                _ => String::new(),
            };
            Some(ToolCallView::Generic(GenericCallView {
                title: format!("Read {file_path}{window}"),
                kind: Some(ToolCallKind::Read),
                raw_input: None,
                content: None,
                locations: Some(vec![FileLocation {
                    path: file_path,
                    line: Some(offset.unwrap_or(1)),
                }]),
            }))
        })),
        present_result: Some(Rc::new(|_args, result| {
            if result.is_error {
                return None;
            }
            let meta = read_meta_from_meta(result.meta.as_ref()?)?;
            let text = match result.content.as_slice() {
                [ContentBlock::Text { text }] => text,
                _ => return None,
            };
            // Strip the read envelope; a non-envelope result (replayed
            // obsolete output) declines to the generic fallback.
            let rest = text.strip_prefix("<path>")?;
            let (_, rest) = rest.split_once("</path>\n<type>file</type>\n<content>\n")?;
            let body = rest.strip_suffix("\n</content>")?;
            Some(ToolResultView::Read(ReadResultView {
                title: None,
                path: meta.path,
                offset: meta.offset,
                lines: meta.lines,
                total_lines: meta.total_lines,
                lang: meta.lang,
                content: Some(vec![ContentBlock::Text {
                    text: body.to_string(),
                }]),
            }))
        })),
    })?;
    tools.register(ctx, definition)
}

// --- write ------------------------------------------------------------

fn parse_write_args(args: &Value) -> anyhow::Result<(String, String)> {
    let file_path = args["file_path"].as_str().unwrap_or_default();
    if file_path.trim().is_empty() {
        anyhow::bail!("file_path must be a non-empty string");
    }
    // An empty content is legitimate: it writes an empty file.
    let content = args["content"].as_str().unwrap_or_default();
    Ok((file_path.to_string(), content.to_string()))
}

/// Model-facing write confirmation; no file content is echoed back.
pub fn format_write_output(display_path: &str, operation: FsWriteOperation) -> String {
    let verb = match operation {
        FsWriteOperation::Create => "Created",
        FsWriteOperation::Update => "Updated",
    };
    format!("<path>{display_path}</path>\n<type>file</type>\n<content>\n{verb} file\n</content>")
}

fn register_write_tool(
    ctx: &Context,
    tools: &Rc<ToolRuntime>,
    fs: &Rc<LocalFileSystem>,
) -> anyhow::Result<EffectHandle> {
    let exec_ctx = ctx.clone();
    let exec_fs = fs.clone();
    let definition = define_tool(DefineToolOptions {
        name: "write".into(),
        description: "Create or fully replace a UTF-8 text file.".into(),
        parameters: params(json!({
            "file_path": { "type": "string", "required": true, "description": "Path to write, resolved by the filesystem backend." },
            "content": { "type": "string", "required": true, "description": "Full UTF-8 text content to write." },
        }))?,
        output: DefineToolOutput {
            schema: value_spec(json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "path": { "type": "string", "required": true },
                    "operation": { "type": "string", "required": true, "enum": ["create", "update"] },
                    "before": { "required": true, "oneOf": [{ "type": "string" }, { "type": "null" }] },
                    "after": { "type": "string", "required": true },
                },
            }))?,
            render: Rc::new(|_args, value| {
                let operation = if value["operation"] == json!("create") {
                    FsWriteOperation::Create
                } else {
                    FsWriteOperation::Update
                };
                Ok(vec![ContentBlock::Text {
                    text: format_write_output(
                        value["path"].as_str().unwrap_or_default(),
                        operation,
                    ),
                }])
            }),
            presentation_meta: Some(Rc::new(|args, value| {
                let diffs = match value["before"].as_str() {
                    None => Vec::new(),
                    Some(before) => compute_hunk_diffs(
                        args["file_path"].as_str().unwrap_or_default(),
                        before,
                        value["after"].as_str().unwrap_or_default(),
                    ),
                };
                Ok(json!({ "diffs": serde_json::to_value(diffs)? }))
            })),
        },
        timeout_ms: None,
        is_concurrency_safe: None,
        execute: Rc::new(move |args, exec| {
            let ctx = exec_ctx.clone();
            let fs = exec_fs.clone();
            async move {
                let (file_path, content) = parse_write_args(&args)?;
                let signal = exec.signal();
                let owner = owner_of(&exec);
                let cwd = session_cwd(&exec);
                let target = fs
                    .resolve(&file_path, cwd.as_deref(), Some(&signal))
                    .map_err(fs_tool_error)?;
                // Single-slot decision: the policy plugin answers with a
                // guarded intent; the bare default is None (unconditional).
                let intent = ctx
                    .waterfall::<FsWriteIntentSlot, _, _>(
                        (target.clone(), owner.clone()),
                        |_args| async { Ok(None) },
                    )
                    .await
                    .map_err(remediate_any)?;
                let outcome = fs
                    .write_text(&target, &content, intent.as_ref(), Some(&signal))
                    .map_err(remediate_fs_error)?;
                ctx.parallel::<FsObservedEvent>(&(
                    target.clone(),
                    FsObservation::Present {
                        version: outcome.version,
                    },
                    owner,
                ))
                .await;
                Ok(json!({
                    "path": target.display_path,
                    "operation": match outcome.operation {
                        FsWriteOperation::Create => "create",
                        FsWriteOperation::Update => "update",
                    },
                    "before": outcome.before,
                    "after": outcome.after,
                }))
            }
            .boxed_local()
        }),
        finalize_content: None,
        // Call-time display: a diff card; prior content is unavailable at
        // call time, so a create and an overwrite both show old_text None.
        present_call: Some(Rc::new(|args| {
            let file_path = args["file_path"].as_str().unwrap_or_default().to_string();
            Some(ToolCallView::Diff(DiffCallView {
                title: format!("Write {file_path}"),
                diffs: vec![FileDiff {
                    path: file_path.clone(),
                    old_text: None,
                    new_text: args["content"].as_str().unwrap_or_default().to_string(),
                }],
                locations: Some(vec![FileLocation {
                    path: file_path,
                    line: None,
                }]),
            }))
        })),
        // The completed card replaces the pending one, so the diff repeats:
        // applied metadata when present, else the replay-safe args fallback.
        present_result: Some(Rc::new(|args, result| {
            if result.is_error {
                return None;
            }
            let file_path = args["file_path"].as_str().unwrap_or_default().to_string();
            let diffs = result
                .meta
                .as_ref()
                .and_then(diffs_from_meta)
                .unwrap_or_else(|| {
                    vec![FileDiff {
                        path: file_path.clone(),
                        old_text: None,
                        new_text: args["content"].as_str().unwrap_or_default().to_string(),
                    }]
                });
            Some(ToolResultView::Diff(DiffResultView {
                title: Some(format!("Write {file_path}")),
                diffs,
            }))
        })),
    })?;
    tools.register(ctx, definition)
}

// --- edit -------------------------------------------------------------

fn parse_edit_args(args: &Value) -> anyhow::Result<FsEditRequestArgs> {
    let file_path = args["file_path"].as_str().unwrap_or_default();
    if file_path.trim().is_empty() {
        anyhow::bail!("file_path must be a non-empty string");
    }
    let old_string = args["old_string"].as_str().unwrap_or_default();
    if old_string.is_empty() {
        anyhow::bail!("old_string must be a non-empty string");
    }
    let new_string = args["new_string"].as_str().unwrap_or_default();
    if old_string == new_string {
        anyhow::bail!("old_string and new_string must differ");
    }
    Ok(FsEditRequestArgs {
        file_path: file_path.to_string(),
        old_string: old_string.to_string(),
        new_string: new_string.to_string(),
        replace_all: args["replace_all"].as_bool().unwrap_or(false),
    })
}

struct FsEditRequestArgs {
    file_path: String,
    old_string: String,
    new_string: String,
    replace_all: bool,
}

/// Model-facing edit confirmation, distinct wording for replace-all.
pub fn format_edit_output(display_path: &str, replace_all: bool) -> String {
    if replace_all {
        format!(
            "The file {display_path} has been updated. All occurrences were successfully replaced."
        )
    } else {
        format!("The file {display_path} has been updated successfully.")
    }
}

fn register_edit_tool(
    ctx: &Context,
    tools: &Rc<ToolRuntime>,
    fs: &Rc<LocalFileSystem>,
) -> anyhow::Result<EffectHandle> {
    let exec_ctx = ctx.clone();
    let exec_fs = fs.clone();
    let definition = define_tool(DefineToolOptions {
        name: "edit".into(),
        description: "Edit an existing UTF-8 text file by replacing literal text.".into(),
        parameters: params(json!({
            "file_path": { "type": "string", "required": true, "description": "Path to edit, resolved by the filesystem backend." },
            "old_string": { "type": "string", "required": true, "description": "Literal text to replace. Must match exactly." },
            "new_string": { "type": "string", "required": true, "description": "Literal replacement text. Use an empty string to delete the match." },
            "replace_all": { "type": "boolean", "description": "Replace all matches. Defaults to false; when false, old_string must appear exactly once." },
        }))?,
        output: DefineToolOutput {
            schema: value_spec(json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "path": { "type": "string", "required": true },
                    "before": { "type": "string", "required": true },
                    "after": { "type": "string", "required": true },
                },
            }))?,
            render: Rc::new(|args, value| {
                Ok(vec![ContentBlock::Text {
                    text: format_edit_output(
                        value["path"].as_str().unwrap_or_default(),
                        args["replace_all"].as_bool().unwrap_or(false),
                    ),
                }])
            }),
            presentation_meta: Some(Rc::new(|args, value| {
                let diffs = compute_hunk_diffs(
                    args["file_path"].as_str().unwrap_or_default(),
                    value["before"].as_str().unwrap_or_default(),
                    value["after"].as_str().unwrap_or_default(),
                );
                Ok(json!({ "diffs": serde_json::to_value(diffs)? }))
            })),
        },
        timeout_ms: None,
        is_concurrency_safe: None,
        execute: Rc::new(move |args, exec| {
            let ctx = exec_ctx.clone();
            let fs = exec_fs.clone();
            async move {
                let input = parse_edit_args(&args)?;
                let signal = exec.signal();
                let owner = owner_of(&exec);
                let cwd = session_cwd(&exec);
                let target = fs
                    .resolve(&input.file_path, cwd.as_deref(), Some(&signal))
                    .map_err(fs_tool_error)?;
                // The intent slot itself can refuse an unread target
                // (FS_NOT_OBSERVED), so its failure gets the same remedy as
                // the provider's guarded-mutation failures.
                let intent = ctx
                    .waterfall::<FsEditIntentSlot, _, _>(
                        (target.clone(), owner.clone()),
                        |_args| async { Ok(None) },
                    )
                    .await
                    .map_err(remediate_any)?;
                let outcome = fs
                    .edit_text(
                        &target,
                        &FsEditRequest {
                            old_string: input.old_string,
                            new_string: input.new_string,
                            replace_all: input.replace_all,
                        },
                        intent.as_ref(),
                        Some(&signal),
                    )
                    .map_err(remediate_fs_error)?;
                ctx.parallel::<FsObservedEvent>(&(
                    target.clone(),
                    FsObservation::Present {
                        version: outcome.version,
                    },
                    owner,
                ))
                .await;
                Ok(json!({
                    "path": target.display_path,
                    "before": outcome.before,
                    "after": outcome.after,
                }))
            }
            .boxed_local()
        }),
        finalize_content: None,
        // Call-time display: the literal replacement as a diff card; an
        // empty old_string (obsolete replayed args) maps to old_text None.
        present_call: Some(Rc::new(|args| {
            let file_path = args["file_path"].as_str().unwrap_or_default().to_string();
            let old_string = args["old_string"].as_str().unwrap_or_default();
            Some(ToolCallView::Diff(DiffCallView {
                title: format!("Edit {file_path}"),
                diffs: vec![FileDiff {
                    path: file_path.clone(),
                    old_text: (!old_string.is_empty()).then(|| old_string.to_string()),
                    new_text: args["new_string"].as_str().unwrap_or_default().to_string(),
                }],
                locations: Some(vec![FileLocation {
                    path: file_path,
                    line: None,
                }]),
            }))
        })),
        present_result: Some(Rc::new(|args, result| {
            if result.is_error {
                return None;
            }
            let diffs = diffs_from_meta(result.meta.as_ref()?)?;
            Some(ToolResultView::Diff(DiffResultView {
                title: Some(format!(
                    "Edit {}",
                    args["file_path"].as_str().unwrap_or_default()
                )),
                diffs,
            }))
        })),
    })?;
    tools.register(ctx, definition)
}

/// Register the `read` / `write` / `edit` suite; the returned handles are
/// the exact registration disposers.
pub fn register_fs_tools(
    ctx: &Context,
    tools: &Rc<ToolRuntime>,
    fs: &Rc<LocalFileSystem>,
    config: FsToolsConfig,
) -> anyhow::Result<Vec<EffectHandle>> {
    if config.read_limit < 1 || config.read_max_line_length < 1 || config.read_max_bytes < 1 {
        anyhow::bail!("tool-fs: every read cap must be a positive integer");
    }
    Ok(vec![
        register_read_tool(ctx, tools, fs, config)?,
        register_write_tool(ctx, tools, fs)?,
        register_edit_tool(ctx, tools, fs)?,
    ])
}
