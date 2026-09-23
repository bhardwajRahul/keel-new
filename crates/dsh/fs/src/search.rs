//! Model-facing `glob` / `grep` discovery tools (port of upstream
//! `packages/fs/tool-fs-search`). This module owns schemas, argument
//! validation, walking, matching, retention, formatting, and the search-card
//! projections.
//!
//! Divergences from upstream, which shells out to the packaged ripgrep
//! binary through the subprocess seam:
//! - Walking runs in-process on the workspace `ignore` crate and matching on
//!   the `regex` crate (the same regex/glob syntax family ripgrep uses).
//! - Both tools respect `.gitignore` (`require_git` disabled so ignore files
//!   work outside a repo). Upstream `glob` passes `--no-ignore`; here the
//!   ignore files win — the task-mandated trade for not spawning ripgrep.
//!   `glob` still includes hidden files and prunes VCS metadata directories;
//!   `grep` keeps ripgrep's default hidden-file skip.
//! - The formatted-result spill store is not ported: a capped result reports
//!   that the complete result could not be saved (upstream's no-backend arm).
//! - Over-cap `glob` results keep the modification-time head; the optional
//!   top-level sampling mode is not ported.

use crate::tools::session_cwd;
use dsh_cordis::{Context, EffectHandle};
use dsh_llm::{ContentBlock, HarnessError};
use dsh_timeout::AbortSignal;
use dsh_tools::{
    DefineToolOptions, DefineToolOutput, GenericCallView, ParameterSchemaSpec, SearchFileMatches,
    SearchLineMatch, SearchMatchesResultView, SearchPathsResultView, SearchResultView,
    ToolCallKind, ToolCallView, ToolResultView, ToolRuntime, ValueSchemaSpec, define_tool,
};
use futures::FutureExt;
use ignore::WalkBuilder;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::rc::Rc;

/// Default cap on paths one `glob` call returns inline.
pub const GLOB_MAX_RESULTS: usize = 100;
/// Default cap on flat matches one `grep` call returns inline.
pub const GREP_MAX_MATCHES: usize = 250;
/// Default byte cap on one matched-line preview (UTF-8 boundary preserved).
pub const GREP_MAX_LINE_BYTES: usize = 2000;
/// Default byte cap on one search's serialized presentation meta.
pub const SEARCH_META_MAX_BYTES: usize = 65_536;
/// Default cooperative tool-call timeout budget in milliseconds.
pub const SEARCH_TIMEOUT_MS: f64 = 30_000.0;

/// Directory names discovery must never descend into: VCS metadata stores.
pub const GLOB_VCS_EXCLUDES: &[&str] = &[".git", ".svn", ".hg", ".bzr", ".jj", ".sl"];

/// Stable machine-routable codes for search failures — package-owned, not
/// `FS_*`, because discovery is not a `ctx.fs` provider operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchErrorCode {
    /// The regex or glob was rejected.
    InvalidPattern,
    /// The search could not run (inaccessible target, walk failure).
    Failed,
    /// The cooperative timeout or caller cancellation cut the search short.
    Aborted,
}

impl SearchErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            SearchErrorCode::InvalidPattern => "SEARCH_INVALID_PATTERN",
            SearchErrorCode::Failed => "SEARCH_FAILED",
            SearchErrorCode::Aborted => "SEARCH_ABORTED",
        }
    }
}

/// Typed search failure carrying its stable code.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{message}")]
pub struct SearchError {
    pub message: String,
    pub code: SearchErrorCode,
}

impl SearchError {
    pub fn new(message: impl Into<String>, code: SearchErrorCode) -> Self {
        SearchError {
            message: message.into(),
            code,
        }
    }
}

fn search_tool_error(error: SearchError) -> anyhow::Error {
    HarnessError::new(error.message.clone(), error.code.as_str()).into()
}

fn abort_error(tool: &str) -> SearchError {
    SearchError::new(
        format!("{tool} was aborted before completion (tool timeout or caller cancellation)"),
        SearchErrorCode::Aborted,
    )
}

/// Search-tool caps (upstream plugin config after defaulting).
#[derive(Debug, Clone, Copy)]
pub struct SearchToolsConfig {
    pub glob_max_results: usize,
    pub grep_max_matches: usize,
    pub grep_max_line_bytes: usize,
    pub search_meta_max_bytes: usize,
    /// Cooperative timeout budget attached to both tool definitions.
    pub timeout_ms: f64,
}

impl Default for SearchToolsConfig {
    fn default() -> Self {
        SearchToolsConfig {
            glob_max_results: GLOB_MAX_RESULTS,
            grep_max_matches: GREP_MAX_MATCHES,
            grep_max_line_bytes: GREP_MAX_LINE_BYTES,
            search_meta_max_bytes: SEARCH_META_MAX_BYTES,
            timeout_ms: SEARCH_TIMEOUT_MS,
        }
    }
}

/// Display form of one discovered path: workdir-relative when inside the
/// workdir, `.` for the workdir itself, unchanged otherwise.
pub fn to_workdir_relative(path: &Path, workdir: &Path) -> String {
    if !path.is_absolute() {
        return path.to_string_lossy().into_owned();
    }
    match path.strip_prefix(workdir) {
        Ok(rel) if rel.as_os_str().is_empty() => ".".to_string(),
        Ok(rel) => rel.to_string_lossy().into_owned(),
        Err(_) => path.to_string_lossy().into_owned(),
    }
}

/// Bound one matched-line preview to `max_bytes` on a UTF-8 boundary,
/// marking the cut; the complete line stays in the file for `read`.
pub fn preview_line(line: &str, max_bytes: usize) -> String {
    if line.len() <= max_bytes {
        return line.to_string();
    }
    let mut cut = max_bytes;
    while !line.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{} (line truncated)", &line[..cut])
}

/// One parsed content match.
#[derive(Debug, Clone, PartialEq)]
pub struct GrepMatch {
    pub path: String,
    pub line_number: u64,
    pub line: String,
}

// --- shared walking ------------------------------------------------------

fn workdir_of(exec: &dsh_tools::ToolExecution) -> anyhow::Result<PathBuf> {
    match session_cwd(exec) {
        Some(cwd) => Ok(cwd),
        None => Ok(std::env::current_dir()?),
    }
}

fn resolve_root(workdir: &Path, path: Option<&str>) -> PathBuf {
    match path {
        Some(path) if Path::new(path).is_absolute() => PathBuf::from(path),
        Some(path) => workdir.join(path),
        None => workdir.to_path_buf(),
    }
}

struct WalkSpec<'a> {
    root: &'a Path,
    /// Positive override glob (the glob pattern / the include filter).
    include: Option<&'a str>,
    /// Yield hidden files (glob) or keep the default hidden skip (grep).
    include_hidden: bool,
    /// Prune VCS metadata directories explicitly (glob's `--no-ignore
    /// --hidden` analogue needs it; grep's hidden skip already covers them).
    prune_vcs: bool,
}

fn walk_files(
    tool: &str,
    spec: &WalkSpec,
    signal: &AbortSignal,
    mut visit: impl FnMut(&Path) -> Result<(), SearchError>,
) -> Result<(), SearchError> {
    let invalid = |error: ignore::Error| {
        SearchError::new(
            format!("{tool} pattern rejected: {error}"),
            SearchErrorCode::InvalidPattern,
        )
    };
    // The pattern filter is a gitignore-style matcher rather than a walker
    // override: an override WHITELIST would exempt matching files from
    // .gitignore rules, and this port's contract is that ignore files win.
    let matcher = match spec.include {
        Some(pattern) => {
            let mut builder = ignore::gitignore::GitignoreBuilder::new(spec.root);
            builder.add_line(None, pattern).map_err(invalid)?;
            Some(builder.build().map_err(invalid)?)
        }
        None => None,
    };
    let mut builder = WalkBuilder::new(spec.root);
    // Respect ignore files even outside a git repository so the contract is
    // location-independent.
    builder.require_git(false);
    builder.hidden(!spec.include_hidden);
    builder.sort_by_file_name(std::ffi::OsStr::cmp);
    if spec.prune_vcs {
        builder.filter_entry(|entry| {
            let is_dir = entry.file_type().is_some_and(|kind| kind.is_dir());
            !(is_dir
                && GLOB_VCS_EXCLUDES
                    .iter()
                    .any(|name| entry.file_name() == *name))
        });
    }
    for entry in builder.build() {
        if signal.aborted() {
            return Err(abort_error(tool));
        }
        let entry = entry.map_err(|error| {
            SearchError::new(
                format!("{tool} search failed: {error}"),
                SearchErrorCode::Failed,
            )
        })?;
        if !entry.file_type().is_some_and(|kind| kind.is_file()) {
            continue;
        }
        // An explicitly named file root bypasses the filter (the caller
        // already chose it); everything below the root must match.
        if entry.depth() > 0 {
            if let Some(matcher) = &matcher {
                if !matcher.matched(entry.path(), false).is_ignore() {
                    continue;
                }
            }
        }
        visit(entry.path())?;
    }
    Ok(())
}

// --- presentation meta ----------------------------------------------------

fn meta_bytes(meta: &Value) -> usize {
    serde_json::to_string(meta)
        .map(|text| text.len())
        .unwrap_or(usize::MAX)
}

/// Drop trailing top-level items (file groups or paths) until the serialized
/// meta fits the budget, marking `truncated`; `total` keeps counting what
/// the search found. A single over-budget item is kept — the invariant is a
/// bounded payload wherever droppable, never an empty card.
fn cap_meta_bytes(mut meta: Value, list_key: &str, max_bytes: usize) -> Value {
    if meta_bytes(&meta) <= max_bytes {
        return meta;
    }
    meta["truncated"] = json!(true);
    while meta[list_key]
        .as_array()
        .is_some_and(|items| items.len() > 1)
        && meta_bytes(&meta) > max_bytes
    {
        meta[list_key].as_array_mut().expect("checked array").pop();
    }
    meta
}

fn group_matches_by_file(matches: &[GrepMatch]) -> Vec<Value> {
    let mut order: Vec<&str> = Vec::new();
    let mut grouped: std::collections::HashMap<&str, Vec<Value>> = std::collections::HashMap::new();
    for entry in matches {
        let group = grouped.entry(entry.path.as_str()).or_insert_with(|| {
            order.push(entry.path.as_str());
            Vec::new()
        });
        group.push(json!({ "lineNumber": entry.line_number, "line": entry.line }));
    }
    order
        .into_iter()
        .map(|path| json!({ "path": path, "matches": grouped[path] }))
        .collect()
}

/// Narrow opaque result metadata to a search card; malformed or absent data
/// declines to the generic fallback. A zero-result meta is a valid empty
/// card — "no matches" is a real result.
pub fn search_view_from_meta(meta: &Value) -> Option<SearchResultView> {
    let map = meta.as_object()?;
    let truncated = map.get("truncated")?.as_bool()?;
    let total = map.get("total")?.as_u64()?;
    match map.get("shape")?.as_str()? {
        "paths" => {
            let paths: Vec<String> = map
                .get("paths")?
                .as_array()?
                .iter()
                .map(|path| path.as_str().map(str::to_string))
                .collect::<Option<Vec<_>>>()?;
            Some(SearchResultView::Paths(SearchPathsResultView {
                title: None,
                paths,
                truncated,
                total,
            }))
        }
        "matches" => {
            let files: Vec<SearchFileMatches> = map
                .get("files")?
                .as_array()?
                .iter()
                .map(|file| {
                    let file = file.as_object()?;
                    let matches = file
                        .get("matches")?
                        .as_array()?
                        .iter()
                        .map(|entry| {
                            let entry = entry.as_object()?;
                            Some(SearchLineMatch {
                                line_number: entry.get("lineNumber")?.as_u64()?,
                                line: entry.get("line")?.as_str()?.to_string(),
                            })
                        })
                        .collect::<Option<Vec<_>>>()?;
                    Some(SearchFileMatches {
                        path: file.get("path")?.as_str()?.to_string(),
                        matches,
                    })
                })
                .collect::<Option<Vec<_>>>()?;
            Some(SearchResultView::Matches(SearchMatchesResultView {
                title: None,
                files,
                truncated,
                total,
            }))
        }
        _ => None,
    }
}

fn params(spec: Value) -> anyhow::Result<ParameterSchemaSpec> {
    Ok(ParameterSchemaSpec::from_author_value(&spec, "parameters")?)
}

fn value_spec(spec: Value) -> anyhow::Result<ValueSchemaSpec> {
    Ok(ValueSchemaSpec::from_author_value(&spec, "schema")?)
}

// --- glob -----------------------------------------------------------------

fn parse_glob_args(args: &Value) -> anyhow::Result<(String, Option<String>)> {
    let pattern = args["pattern"].as_str().unwrap_or_default();
    if pattern.trim().is_empty() {
        anyhow::bail!("pattern must be a non-empty string");
    }
    let path = match args.get("path") {
        None | Some(Value::Null) => None,
        Some(value) => {
            let path = value.as_str().unwrap_or_default();
            if path.trim().is_empty() {
                anyhow::bail!("path must be a non-empty string when given");
            }
            Some(path.to_string())
        }
    };
    Ok((pattern.to_string(), path))
}

/// Render one bounded path page: a result that fits is shown whole in
/// modification-time order; an over-cap result keeps the head and reports
/// the unsaved remainder.
fn render_glob_paths(paths: &[String], max_results: usize) -> String {
    if paths.is_empty() {
        return "No files found".to_string();
    }
    if paths.len() <= max_results {
        return paths.join("\n");
    }
    let page = &paths[..max_results];
    format!(
        "{}\n\n(Showing {} of {} paths. The complete result could not be saved; narrow pattern or path to see more.)",
        page.join("\n"),
        page.len(),
        paths.len()
    )
}

fn register_glob_tool(
    ctx: &Context,
    tools: &Rc<ToolRuntime>,
    config: SearchToolsConfig,
) -> anyhow::Result<EffectHandle> {
    let definition = define_tool(DefineToolOptions {
        name: "glob".into(),
        description: format!(
            "Find files whose paths match a glob pattern. Returns matching file paths — never directories — \
             including hidden files (VCS metadata and .gitignore-excluded files are skipped). \
             Up to {max} paths come back in modification-time order; a larger result returns the first {max} \
             paths in modification-time order and says so. This tool does not enumerate directory entries.",
            max = config.glob_max_results
        ),
        parameters: params(json!({
            "pattern": {
                "type": "string",
                "required": true,
                "description": "Glob pattern to match file paths against (e.g. \"**/*.ts\", \"src/**/*.test.js\"). A pattern with no \"/\" matches the basename at any depth, so \"*\" and \"*.ts\" both search the whole tree; include a separator to anchor the depth.",
            },
            "path": { "type": "string", "description": "Directory to search in. Defaults to the session workspace; a relative path resolves against it." },
        }))?,
        output: DefineToolOutput {
            schema: value_spec(json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "root": { "type": "string", "required": true },
                    "paths": { "type": "array", "required": true, "items": { "type": "string" } },
                },
            }))?,
            render: Rc::new(move |_args, value| {
                let paths: Vec<String> = value["paths"]
                    .as_array()
                    .map(|paths| {
                        paths
                            .iter()
                            .map(|path| path.as_str().unwrap_or_default().to_string())
                            .collect()
                    })
                    .unwrap_or_default();
                Ok(vec![ContentBlock::Text {
                    text: render_glob_paths(&paths, config.glob_max_results),
                }])
            }),
            presentation_meta: Some(Rc::new(move |_args, value| {
                let paths = value["paths"].as_array().cloned().unwrap_or_default();
                let total = paths.len();
                let truncated = total > config.glob_max_results;
                let page: Vec<Value> = paths.into_iter().take(config.glob_max_results).collect();
                Ok(cap_meta_bytes(
                    json!({ "shape": "paths", "paths": page, "truncated": truncated, "total": total }),
                    "paths",
                    config.search_meta_max_bytes,
                ))
            })),
        },
        timeout_ms: Some(config.timeout_ms),
        is_concurrency_safe: None,
        execute: Rc::new(move |args, exec| {
            async move {
                let (pattern, path) = parse_glob_args(&args)?;
                let signal = exec.signal();
                if signal.aborted() {
                    return Err(search_tool_error(abort_error("glob")));
                }
                let workdir = workdir_of(&exec)?;
                let root = resolve_root(&workdir, path.as_deref());
                let mut found: Vec<(PathBuf, std::time::SystemTime)> = Vec::new();
                walk_files(
                    "glob",
                    &WalkSpec {
                        root: &root,
                        include: Some(&pattern),
                        include_hidden: true,
                        prune_vcs: true,
                    },
                    &signal,
                    |file| {
                        let modified = std::fs::metadata(file)
                            .and_then(|meta| meta.modified())
                            .unwrap_or(std::time::UNIX_EPOCH);
                        found.push((file.to_path_buf(), modified));
                        Ok(())
                    },
                )
                .map_err(search_tool_error)?;
                found.sort_by_key(|(_, modified)| *modified);
                let paths: Vec<String> = found
                    .iter()
                    .map(|(file, _)| to_workdir_relative(file, &workdir))
                    .collect();
                let root_display = match &path {
                    None => ".".to_string(),
                    Some(_) => to_workdir_relative(&root, &workdir),
                };
                Ok(json!({ "root": root_display, "paths": paths }))
            }
            .boxed_local()
        }),
        finalize_content: None,
        present_call: Some(Rc::new(|args| {
            let pattern = args["pattern"].as_str().unwrap_or_default();
            let location = args["path"]
                .as_str()
                .map(|path| format!(" in {path}"))
                .unwrap_or_default();
            Some(ToolCallView::Generic(GenericCallView {
                title: format!("Glob {pattern}{location}"),
                kind: Some(ToolCallKind::Search),
                raw_input: Some(json!(pattern)),
                content: None,
                locations: None,
            }))
        })),
        present_result: Some(Rc::new(|_args, result| {
            if result.is_error {
                return None;
            }
            match search_view_from_meta(result.meta.as_ref()?)? {
                view @ SearchResultView::Paths(_) => Some(ToolResultView::Search(view)),
                SearchResultView::Matches(_) => None,
            }
        })),
    })?;
    tools.register(ctx, definition)
}

// --- grep -----------------------------------------------------------------

/// Reject an `include` that is not ONE positive glob filter: blanks, negated
/// patterns, and comma-separated lists (brace alternation is one glob).
fn validate_include(include: &str) -> anyhow::Result<()> {
    if include.trim().is_empty() {
        anyhow::bail!("include must be a non-empty glob when given");
    }
    if include.starts_with('!') {
        anyhow::bail!(
            "include must be a positive glob filter; negated patterns (\"!…\") are not supported"
        );
    }
    let mut brace_depth = 0usize;
    for character in include.chars() {
        match character {
            '{' => brace_depth += 1,
            '}' => brace_depth = brace_depth.saturating_sub(1),
            ',' if brace_depth == 0 => {
                anyhow::bail!(
                    "include must be one glob, not a comma-separated list (use {{a,b}} alternation instead)"
                );
            }
            _ => {}
        }
    }
    Ok(())
}

fn parse_grep_args(args: &Value) -> anyhow::Result<(String, Option<String>, Option<String>)> {
    // Whitespace is a legitimate regex, so only the empty pattern fails.
    let pattern = args["pattern"].as_str().unwrap_or_default();
    if pattern.is_empty() {
        anyhow::bail!("pattern must be a non-empty string");
    }
    let path = match args.get("path") {
        None | Some(Value::Null) => None,
        Some(value) => {
            let path = value.as_str().unwrap_or_default();
            if path.trim().is_empty() {
                anyhow::bail!("path must be a non-empty string when given");
            }
            Some(path.to_string())
        }
    };
    let include = match args.get("include") {
        None | Some(Value::Null) => None,
        Some(value) => {
            let include = value.as_str().unwrap_or_default();
            validate_include(include)?;
            Some(include.to_string())
        }
    };
    Ok((pattern.to_string(), path, include))
}

fn match_noun(count: usize) -> &'static str {
    if count == 1 { "match" } else { "matches" }
}

/// Group retained matches by file (first-seen order) into the model-facing
/// body: the display path, then one `Line N: <text>` row per match.
pub fn format_grep_matches(matches: &[GrepMatch]) -> String {
    let mut order: Vec<&str> = Vec::new();
    let mut grouped: std::collections::HashMap<&str, Vec<&GrepMatch>> =
        std::collections::HashMap::new();
    for entry in matches {
        grouped
            .entry(entry.path.as_str())
            .or_insert_with(|| {
                order.push(entry.path.as_str());
                Vec::new()
            })
            .push(entry);
    }
    order
        .into_iter()
        .map(|path| {
            let rows: Vec<String> = grouped[path]
                .iter()
                .map(|entry| format!("Line {}: {}", entry.line_number, entry.line))
                .collect();
            format!("{path}\n{}", rows.join("\n"))
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Apply the shared inline cap: preview every retained line, keep the head.
fn retain_grep_matches(
    matches: &[GrepMatch],
    max_matches: usize,
    max_line_bytes: usize,
) -> Vec<GrepMatch> {
    matches
        .iter()
        .take(max_matches)
        .map(|entry| GrepMatch {
            path: entry.path.clone(),
            line_number: entry.line_number,
            line: preview_line(&entry.line, max_line_bytes),
        })
        .collect()
}

fn render_grep_matches(matches: &[GrepMatch], max_matches: usize, max_line_bytes: usize) -> String {
    if matches.is_empty() {
        return "No matches found".to_string();
    }
    let retained = retain_grep_matches(matches, max_matches, max_line_bytes);
    let truncated = matches.len() > retained.len();
    let header = if truncated {
        format!("Found {} of {} matches", retained.len(), matches.len())
    } else {
        format!("Found {} {}", matches.len(), match_noun(matches.len()))
    };
    let body = format_grep_matches(&retained);
    if truncated {
        format!(
            "{header}\n\n{body}\n\n(The complete result could not be saved; narrow pattern, path, or include to see more.)"
        )
    } else {
        format!("{header}\n\n{body}")
    }
}

fn grep_file(
    file: &Path,
    display: String,
    regex: &regex::Regex,
    out: &mut Vec<GrepMatch>,
) -> Result<(), SearchError> {
    // An unreadable file fails the search (upstream: rg reports and exits
    // nonzero); a binary file is silently skipped like ripgrep skips it.
    let content = std::fs::read(file).map_err(|error| {
        SearchError::new(
            format!(
                "grep search failed: cannot read \"{}\": {error}",
                file.display()
            ),
            SearchErrorCode::Failed,
        )
    })?;
    if content.iter().take(8192).any(|&byte| byte == 0) {
        return Ok(());
    }
    let mut pieces: Vec<&[u8]> = content.split(|&byte| byte == b'\n').collect();
    if pieces.last().is_some_and(|piece| piece.is_empty()) {
        pieces.pop();
    }
    for (index, raw) in pieces.iter().enumerate() {
        let line_number = index as u64 + 1;
        match std::str::from_utf8(raw) {
            Ok(text) => {
                let text = text.strip_suffix('\r').unwrap_or(text);
                if regex.is_match(text) {
                    out.push(GrepMatch {
                        path: display.clone(),
                        line_number,
                        line: text.to_string(),
                    });
                }
            }
            Err(_) => {
                // A non-UTF-8 line yields a placeholder preview instead of
                // failing the search (upstream's rg `bytes` arm).
                let lossy = String::from_utf8_lossy(raw);
                if regex.is_match(&lossy) {
                    out.push(GrepMatch {
                        path: display.clone(),
                        line_number,
                        line: "(line is not valid UTF-8)".to_string(),
                    });
                }
            }
        }
    }
    Ok(())
}

fn register_grep_tool(
    ctx: &Context,
    tools: &Rc<ToolRuntime>,
    config: SearchToolsConfig,
) -> anyhow::Result<EffectHandle> {
    let definition = define_tool(DefineToolOptions {
        name: "grep".into(),
        description: format!(
            "Search file contents with a regular expression. Returns matching lines with line numbers, \
             grouped by file. Returns the first {} matches inline. Use read on a matched file for surrounding context.",
            config.grep_max_matches
        ),
        parameters: params(json!({
            "pattern": { "type": "string", "required": true, "description": "Regular expression to search for (Rust regex syntax, the ripgrep family)." },
            "path": { "type": "string", "description": "File or directory to search. Defaults to the session workspace; a relative path resolves against it." },
            "include": { "type": "string", "description": "One glob filter for which files to search (e.g. \"*.ts\", \"*.{js,jsx}\"). Not a list; negation is not supported." },
        }))?,
        output: DefineToolOutput {
            schema: value_spec(json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "matches": {
                        "type": "array",
                        "required": true,
                        "items": {
                            "type": "object",
                            "additionalProperties": false,
                            "properties": {
                                "path": { "type": "string", "required": true },
                                "lineNumber": { "type": "integer", "required": true },
                                "line": { "type": "string", "required": true },
                            },
                        },
                    },
                },
            }))?,
            render: Rc::new(move |_args, value| {
                let matches = grep_matches_of(value);
                Ok(vec![ContentBlock::Text {
                    text: render_grep_matches(
                        &matches,
                        config.grep_max_matches,
                        config.grep_max_line_bytes,
                    ),
                }])
            }),
            presentation_meta: Some(Rc::new(move |_args, value| {
                let matches = grep_matches_of(value);
                let retained = retain_grep_matches(
                    &matches,
                    config.grep_max_matches,
                    config.grep_max_line_bytes,
                );
                Ok(cap_meta_bytes(
                    json!({
                        "shape": "matches",
                        "files": group_matches_by_file(&retained),
                        "truncated": matches.len() > retained.len(),
                        "total": matches.len(),
                    }),
                    "files",
                    config.search_meta_max_bytes,
                ))
            })),
        },
        timeout_ms: Some(config.timeout_ms),
        is_concurrency_safe: None,
        execute: Rc::new(move |args, exec| {
            async move {
                let (pattern, path, include) = parse_grep_args(&args)?;
                let signal = exec.signal();
                if signal.aborted() {
                    return Err(search_tool_error(abort_error("grep")));
                }
                let regex = regex::Regex::new(&pattern).map_err(|error| {
                    search_tool_error(SearchError::new(
                        format!("grep pattern rejected: {error}"),
                        SearchErrorCode::InvalidPattern,
                    ))
                })?;
                let workdir = workdir_of(&exec)?;
                let root = resolve_root(&workdir, path.as_deref());
                let mut matches: Vec<GrepMatch> = Vec::new();
                walk_files(
                    "grep",
                    &WalkSpec {
                        root: &root,
                        include: include.as_deref(),
                        include_hidden: false,
                        prune_vcs: false,
                    },
                    &signal,
                    |file| grep_file(file, to_workdir_relative(file, &workdir), &regex, &mut matches),
                )
                .map_err(search_tool_error)?;
                let rows: Vec<Value> = matches
                    .iter()
                    .map(|entry| {
                        json!({ "path": entry.path, "lineNumber": entry.line_number, "line": entry.line })
                    })
                    .collect();
                Ok(json!({ "matches": rows }))
            }
            .boxed_local()
        }),
        finalize_content: None,
        present_call: Some(Rc::new(|args| {
            let pattern = args["pattern"].as_str().unwrap_or_default();
            let location = args["path"]
                .as_str()
                .map(|path| format!(" in {path}"))
                .unwrap_or_default();
            let filter = args["include"]
                .as_str()
                .map(|include| format!(" ({include})"))
                .unwrap_or_default();
            Some(ToolCallView::Generic(GenericCallView {
                title: format!("Grep {pattern}{location}{filter}"),
                kind: Some(ToolCallKind::Search),
                raw_input: Some(json!(pattern)),
                content: None,
                locations: None,
            }))
        })),
        present_result: Some(Rc::new(|_args, result| {
            if result.is_error {
                return None;
            }
            match search_view_from_meta(result.meta.as_ref()?)? {
                view @ SearchResultView::Matches(_) => Some(ToolResultView::Search(view)),
                SearchResultView::Paths(_) => None,
            }
        })),
    })?;
    tools.register(ctx, definition)
}

fn grep_matches_of(value: &Value) -> Vec<GrepMatch> {
    value["matches"]
        .as_array()
        .map(|rows| {
            rows.iter()
                .map(|row| GrepMatch {
                    path: row["path"].as_str().unwrap_or_default().to_string(),
                    line_number: row["lineNumber"].as_u64().unwrap_or(0),
                    line: row["line"].as_str().unwrap_or_default().to_string(),
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Register the `glob` / `grep` discovery suite; the returned handles are
/// the exact registration disposers.
pub fn register_search_tools(
    ctx: &Context,
    tools: &Rc<ToolRuntime>,
    config: SearchToolsConfig,
) -> anyhow::Result<Vec<EffectHandle>> {
    if config.glob_max_results < 1
        || config.grep_max_matches < 1
        || config.grep_max_line_bytes < 1
        || config.search_meta_max_bytes < 1
    {
        anyhow::bail!("tool-fs-search: every search cap must be a positive integer");
    }
    if !config.timeout_ms.is_finite() || config.timeout_ms <= 0.0 {
        anyhow::bail!("tool-fs-search: timeout_ms must be a positive finite number");
    }
    Ok(vec![
        register_glob_tool(ctx, tools, config)?,
        register_grep_tool(ctx, tools, config)?,
    ])
}
