//! Render-intent vocabulary (port of upstream `src/presentation.ts`): the
//! provider-neutral view types a tool returns from `present_call` /
//! `present_result` to tell a UI how one call renders, without the UI
//! special-casing tool names. Pure data; every type serializes with the
//! upstream wire field names.

use dsh_llm::ContentBlock;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Category of a tool call, used by a UI to pick an icon or treatment;
/// `Other` is the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolCallKind {
    Read,
    Edit,
    Delete,
    Move,
    Search,
    Execute,
    Fetch,
    Other,
}

/// A file location a tool reads or modifies, for editor follow-along.
/// `line` is an optional 1-based line to focus.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileLocation {
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u64>,
}

/// A single-file change for inline-diff rendering. `old_text` is `None` for
/// a new-file create or an overwrite (no prior content available at call
/// time).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileDiff {
    pub path: String,
    /// Prior content, or `None` when there is nothing to diff against.
    pub old_text: Option<String>,
    /// Content after the change.
    pub new_text: String,
}

/// One numbered line of a file carried by a [`ReadResultView`]; `number` is
/// the 1-based line number in the file, `text` the line without its
/// trailing newline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReadFileLine {
    pub number: u64,
    pub text: String,
}

/// The default pending card: a titled row with an optional category icon,
/// salient raw input, extra content, and follow-along locations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenericCallView {
    /// Short always-visible label for what THIS call does.
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<ToolCallKind>,
    /// The salient input for a detail view — not the raw args object unless
    /// that is genuinely what a reader wants.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_input: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<ContentBlock>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub locations: Option<Vec<FileLocation>>,
}

/// A call that IS a shell command in a working directory; a capable UI
/// renders a terminal card.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalCallView {
    /// The command, shown as the card's header line.
    pub title: String,
    /// One-line summary rendered above the card.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Working directory; relative paths resolve against the session
    /// workspace, omission defers to it entirely.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
}

/// A call that creates or modifies files, rendered as an inline diff card;
/// diffs derive from the call arguments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiffCallView {
    pub title: String,
    /// One entry per file the call changes.
    pub diffs: Vec<FileDiff>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub locations: Option<Vec<FileLocation>>,
}

/// Provider-neutral pending-call presentation, tagged by `card`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "card", rename_all = "lowercase")]
pub enum ToolCallView {
    Generic(GenericCallView),
    Terminal(TerminalCallView),
    Diff(DiffCallView),
}

/// The default completed card: optional replacement title and reformatted
/// content; omission keeps the pending title / raw result content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenericResultView {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<ContentBlock>>,
}

/// The completed state of a [`TerminalCallView`]: captured output and exit
/// status. `exit_code` and `signal` are mutually exclusive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalResultView {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Captured command output (stdout+stderr as the tool combined them).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    /// Process exit code when the run ended by exiting (not a signal).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    /// Signal name that killed the process (e.g. `SIGTERM`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signal: Option<String>,
}

/// A completed file mutation rendered as an inline diff card — returned even
/// when it repeats the call-time diff, because the completed card replaces
/// the pending one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiffResultView {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// The change to show, in file order.
    pub diffs: Vec<FileDiff>,
}

/// One matched line inside a [`SearchFileMatches`] group.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchLineMatch {
    /// 1-based line number of the match within its file.
    pub line_number: u64,
    /// The matched line text as the tool surfaced it.
    pub line: String,
}

/// One file's grouped content matches, in first-seen file order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchFileMatches {
    /// The file the matches belong to (model-facing display path).
    pub path: String,
    /// The file's matched lines, in output order.
    pub matches: Vec<SearchLineMatch>,
}

/// A completed content search (`grep`) grouped by file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchMatchesResultView {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Matched lines grouped by file, in first-seen file order.
    pub files: Vec<SearchFileMatches>,
    /// Whether the inline result was capped, so a UI never presents a
    /// partial group as complete.
    pub truncated: bool,
    /// Total matches found before capping.
    pub total: u64,
}

/// A completed path search (`glob`) as a flat path list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchPathsResultView {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Discovered paths in result order (the retained page when truncated).
    pub paths: Vec<String>,
    /// Whether the inline result was capped.
    pub truncated: bool,
    /// Total paths found before capping.
    pub total: u64,
}

/// A completed search card, `shape`-discriminated so the discriminant never
/// collides with the [`ToolCallKind`] a bridge reads off a call view. There
/// is no call-time analogue — the pending state has nothing to show yet.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "lowercase")]
pub enum SearchResultView {
    Matches(SearchMatchesResultView),
    Paths(SearchPathsResultView),
}

/// A completed file read rendered as a line-numbered code view; the
/// structured fields ride the tool's `presentation_meta` so replay can
/// rebuild this view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadResultView {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// The read file's path (model-facing; the bridge relativizes it).
    pub path: String,
    /// 1-based first line the window requested, kept even for an empty
    /// window so a continuation knows where to resume.
    pub offset: u64,
    /// The returned window's lines, keeping the file's own numbering.
    pub lines: Vec<ReadFileLine>,
    /// Exact total line count in the file.
    pub total_lines: u64,
    /// Syntax-highlighting hint derived from the extension; omitted when
    /// unknown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lang: Option<String>,
    /// Envelope-stripped model-facing content for a UI without the read
    /// capability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<ContentBlock>>,
}

/// One citeable source in a [`WebSearchResultView`]; must evolve together
/// with the web service's source type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebSource {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
    /// Provider-supplied ISO-8601 timestamp, when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub published_at: Option<String>,
}

/// The completed state of a `web_search` call: structured sources, an
/// optional provider answer, and the truncation signal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebSearchResultView {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// The faithful structured sources the render text cannot losslessly
    /// carry.
    pub sources: Vec<WebSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,
    /// True when the source list was cut to the result cap.
    pub truncated: bool,
}

/// The completed state of a `web_fetch` call: the fetched URL, HTTP status,
/// and whether the content was cut. The body itself is in the raw result
/// content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebFetchResultView {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// The final URL after allowed redirects.
    pub url: String,
    /// HTTP status code of the fetched response.
    pub status_code: u16,
    /// True when any cap trimmed the rendered text.
    pub truncated: bool,
}

/// A completed web retrieval card. `kind` is this union's own discriminant;
/// the values deliberately match the tools' pending [`ToolCallKind`]s so a
/// call and its result read as one category.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum WebResultView {
    Search(WebSearchResultView),
    Fetch(WebFetchResultView),
}

/// How a tool wants the COMPLETED call shown, tagged by `card` and mirroring
/// [`ToolCallView`]. Lets a tool reformat its result for a UI distinctly
/// from the model-facing text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "card", rename_all = "lowercase")]
pub enum ToolResultView {
    Generic(GenericResultView),
    Terminal(TerminalResultView),
    Diff(DiffResultView),
    Search(SearchResultView),
    Read(ReadResultView),
    Web(WebResultView),
}
