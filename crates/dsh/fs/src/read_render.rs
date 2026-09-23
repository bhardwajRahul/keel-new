//! Pure read presentation (port of upstream
//! `packages/fs/tool-fs/src/read-render.ts`): decoded text becomes a
//! bounded, line-numbered window plus the model-facing envelope, and the
//! persisted read `meta` is defensively narrowed back into a UI card.

use crate::types::{FsError, FsErrorCode};
use dsh_tools::ReadFileLine;
use serde_json::Value;

/// Default line cap per `read` call.
pub const READ_LIMIT: u64 = 2000;
/// Default per-line character cap before truncation.
pub const READ_MAX_LINE_LENGTH: usize = 2000;
/// Default byte cap on one call's selected lines.
pub const READ_MAX_BYTES: usize = 50 * 1024;

/// A resolved read window; the caller applied its defaults and caps.
#[derive(Debug, Clone, Copy)]
pub struct ReadWindow {
    /// 1-based first line to return.
    pub offset: u64,
    /// Maximum number of lines to return.
    pub limit: u64,
    /// Per-line character cap; overflow is truncated with a suffix.
    pub max_line_length: usize,
    /// Byte cap on the selected output; overflow stops the scan.
    pub max_bytes: usize,
}

/// The windowed result: numbered lines, the exact total line count, and
/// whether the byte cap cut the window short.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowResult {
    pub lines: Vec<ReadFileLine>,
    pub total_lines: u64,
    pub truncated_by_bytes: bool,
}

fn truncate_line(line: &str, max_line_length: usize) -> String {
    if line.chars().count() > max_line_length {
        let cut: String = line.chars().take(max_line_length).collect();
        format!("{cut}... (line truncated to {max_line_length} chars)")
    } else {
        line.to_string()
    }
}

/// Build one bounded window over the whole decoded text, still scanning to
/// an exact total line count. A requested offset past EOF fails with
/// `FS_NOT_FOUND` (offset 1 over an empty file is the empty window).
pub fn build_window(
    text: &str,
    request: ReadWindow,
    display_path: &str,
) -> Result<WindowResult, FsError> {
    let mut lines: Vec<ReadFileLine> = Vec::new();
    let mut total_lines: u64 = 0;
    let mut output_bytes: usize = 0;
    let mut truncated_by_bytes = false;

    if !text.is_empty() {
        let mut parts: Vec<&str> = text.split('\n').collect();
        // A trailing newline terminates the final line; it is not an extra
        // empty line.
        if parts.last() == Some(&"") {
            parts.pop();
        }
        for raw in parts {
            total_lines += 1;
            if truncated_by_bytes
                || total_lines < request.offset
                || (lines.len() as u64) >= request.limit
            {
                continue;
            }
            let stripped = raw.strip_suffix('\r').unwrap_or(raw);
            let rendered = truncate_line(stripped, request.max_line_length);
            let bytes = rendered.len() + usize::from(!lines.is_empty());
            if output_bytes + bytes > request.max_bytes {
                truncated_by_bytes = true;
                continue;
            }
            output_bytes += bytes;
            lines.push(ReadFileLine {
                number: total_lines,
                text: rendered,
            });
        }
    }

    if !truncated_by_bytes
        && request.offset > total_lines
        && !(total_lines == 0 && request.offset == 1)
    {
        return Err(FsError::new(
            format!(
                "offset {} is out of range for \"{display_path}\" ({total_lines} lines)",
                request.offset
            ),
            FsErrorCode::NotFound,
        ));
    }
    Ok(WindowResult {
        lines,
        total_lines,
        truncated_by_bytes,
    })
}

/// Outcome fields [`format_read_output`] renders.
#[derive(Debug, Clone)]
pub struct FileReadOutcome<'a> {
    pub offset: u64,
    pub lines: &'a [ReadFileLine],
    pub total_lines: u64,
    pub truncated_by_bytes: bool,
}

/// Render one read as the model-facing envelope: numbered lines plus a
/// continuation or end-of-file footer.
pub fn format_read_output(display_path: &str, outcome: &FileReadOutcome) -> String {
    let end_line = outcome
        .lines
        .last()
        .map(|line| line.number)
        .unwrap_or_else(|| outcome.offset.saturating_sub(1));
    let footer = if outcome.truncated_by_bytes {
        format!(
            "(Output capped. Showing lines {}-{end_line}. Use offset={} to continue.)",
            outcome.offset,
            end_line + 1
        )
    } else if end_line < outcome.total_lines {
        format!(
            "(Showing lines {}-{end_line} of {}. Use offset={} to continue.)",
            outcome.offset,
            outcome.total_lines,
            end_line + 1
        )
    } else {
        format!("(End of file - total {} lines)", outcome.total_lines)
    };
    let body = if outcome.lines.is_empty() {
        footer
    } else {
        let numbered: Vec<String> = outcome
            .lines
            .iter()
            .map(|line| format!("{}: {}", line.number, line.text))
            .collect();
        format!("{}\n\n{footer}", numbered.join("\n"))
    };
    format!("<path>{display_path}</path>\n<type>file</type>\n<content>\n{body}\n</content>")
}

/// Lowercased extension to syntax-highlighting hint; a UI treats an absent
/// key as plain text. Deliberately small — common source, config, and markup
/// extensions, not a registry.
const LANG_BY_EXTENSION: &[(&str, &str)] = &[
    ("ts", "ts"),
    ("tsx", "tsx"),
    ("mts", "ts"),
    ("cts", "ts"),
    ("js", "js"),
    ("jsx", "jsx"),
    ("mjs", "js"),
    ("cjs", "js"),
    ("json", "json"),
    ("jsonc", "json"),
    ("py", "py"),
    ("rb", "rb"),
    ("go", "go"),
    ("rs", "rs"),
    ("java", "java"),
    ("c", "c"),
    ("h", "c"),
    ("cc", "cpp"),
    ("cpp", "cpp"),
    ("hpp", "cpp"),
    ("cxx", "cpp"),
    ("cs", "cs"),
    ("kt", "kotlin"),
    ("swift", "swift"),
    ("php", "php"),
    ("sh", "sh"),
    ("bash", "sh"),
    ("zsh", "sh"),
    ("yaml", "yaml"),
    ("yml", "yaml"),
    ("toml", "toml"),
    ("ini", "ini"),
    ("md", "md"),
    ("markdown", "md"),
    ("mdx", "mdx"),
    ("html", "html"),
    ("htm", "html"),
    ("css", "css"),
    ("scss", "scss"),
    ("less", "less"),
    ("sql", "sql"),
    ("xml", "xml"),
    ("lua", "lua"),
];

/// Highlighting hint derived from the path's extension. Case-insensitive on
/// the extension; a dotfile (`.gitignore`) and an unknown extension both
/// yield `None`.
pub fn lang_from_path(path: &str) -> Option<&'static str> {
    let base = path.rsplit(['/', '\\']).next().unwrap_or(path);
    let dot = base.rfind('.')?;
    if dot == 0 {
        return None;
    }
    let ext = base[dot + 1..].to_lowercase();
    LANG_BY_EXTENSION
        .iter()
        .find(|(key, _)| *key == ext)
        .map(|(_, lang)| *lang)
}

/// The `read` tool's persisted `meta` payload, narrowed back from opaque
/// JSON on replay.
#[derive(Debug, Clone, PartialEq)]
pub struct FsReadMeta {
    pub path: String,
    pub offset: u64,
    pub lines: Vec<ReadFileLine>,
    pub total_lines: u64,
    pub lang: Option<String>,
}

fn file_text_line(value: &Value) -> Option<ReadFileLine> {
    let map = value.as_object()?;
    let number = map.get("number")?.as_u64()?;
    if number < 1 {
        return None;
    }
    let text = map.get("text")?.as_str()?;
    Some(ReadFileLine {
        number,
        text: text.to_string(),
    })
}

/// Narrow opaque result metadata to a structured read window. Malformed or
/// semantically invalid data (non-1-based offsets, non-increasing or
/// overcounting line numbers) declines to `None` so presentation falls back
/// to the generic card instead of failing a replay.
pub fn read_meta_from_meta(meta: &Value) -> Option<FsReadMeta> {
    let map = meta.as_object()?;
    let path = map.get("path")?.as_str()?.to_string();
    let offset = map.get("offset")?.as_u64()?;
    if offset < 1 {
        return None;
    }
    let total_lines = map.get("totalLines")?.as_u64()?;
    let lines: Vec<ReadFileLine> = map
        .get("lines")?
        .as_array()?
        .iter()
        .map(file_text_line)
        .collect::<Option<Vec<_>>>()?;
    let lang = match map.get("lang") {
        None | Some(Value::Null) => None,
        Some(Value::String(lang)) => Some(lang.clone()),
        Some(_) => return None,
    };
    let mut previous = offset - 1;
    for line in &lines {
        if line.number <= previous || line.number > total_lines {
            return None;
        }
        previous = line.number;
    }
    Some(FsReadMeta {
        path,
        offset,
        lines,
        total_lines,
        lang,
    })
}
