//! Result-time contextual diff presentation for write and edit (port of
//! upstream `packages/fs/tool-fs/src/diff.ts`). Storage returns before/after
//! text; this layer derives one three-line-context card per applied hunk.
//!
//! Divergence: hunks come from the workspace `similar` crate
//! (`TextDiff::grouped_ops`) instead of the npm `diff` package's
//! `structuredPatch`; the produced per-hunk old/new blocks are equivalent.

use dsh_tools::FileDiff;
use serde_json::Value;
use similar::{ChangeTag, TextDiff};

/// Context lines shown on each side of an applied hunk.
pub const DIFF_CONTEXT: usize = 3;

/// One [`FileDiff`] per hunk between `before` and `after`, each carrying the
/// change plus [`DIFF_CONTEXT`] context lines. Pure insertions report
/// `old_text: None`; identical texts produce no hunks; scattered
/// replacements stay separate hunks.
pub fn compute_hunk_diffs(path: &str, before: &str, after: &str) -> Vec<FileDiff> {
    let diff = TextDiff::from_lines(before, after);
    let mut diffs = Vec::new();
    for group in diff.grouped_ops(DIFF_CONTEXT) {
        let mut old_lines: Vec<String> = Vec::new();
        let mut new_lines: Vec<String> = Vec::new();
        for op in &group {
            for change in diff.iter_changes(op) {
                let text = change
                    .value()
                    .strip_suffix('\n')
                    .unwrap_or(change.value())
                    .to_string();
                match change.tag() {
                    ChangeTag::Delete => old_lines.push(text),
                    ChangeTag::Insert => new_lines.push(text),
                    ChangeTag::Equal => {
                        old_lines.push(text.clone());
                        new_lines.push(text);
                    }
                }
            }
        }
        diffs.push(FileDiff {
            path: path.to_string(),
            old_text: (!old_lines.is_empty()).then(|| old_lines.join("\n")),
            new_text: new_lines.join("\n"),
        });
    }
    diffs
}

fn file_diff(value: &Value) -> Option<FileDiff> {
    let map = value.as_object()?;
    let path = map.get("path")?.as_str()?.to_string();
    let old_text = match map.get("oldText")? {
        Value::Null => None,
        Value::String(text) => Some(text.clone()),
        _ => return None,
    };
    let new_text = map.get("newText")?.as_str()?.to_string();
    Some(FileDiff {
        path,
        old_text,
        new_text,
    })
}

/// Narrow opaque result metadata to non-empty file diffs; malformed or empty
/// payloads decline to `None` so replay presentation can fall back.
pub fn diffs_from_meta(meta: &Value) -> Option<Vec<FileDiff>> {
    let diffs = meta.as_object()?.get("diffs")?.as_array()?;
    if diffs.is_empty() {
        return None;
    }
    diffs.iter().map(file_diff).collect()
}
