//! Filesystem capability for the harness, ported at the contract level from
//! the upstream TypeScript packages and collapsed into ONE crate (upstream
//! splits them across five):
//!
//! - `packages/fs/fs` — the `ctx.fs` vocabulary, error taxonomy, and the
//!   `fs/*` events ([`types`], [`events`]).
//! - `packages/fs/fs-local` — the local provider ([`local`]); it registers
//!   directly as the `fs` service because it is the only backend ported
//!   (sandbox and e2b providers are out of scope), so its methods are the
//!   seam contract.
//! - `packages/fs/fs-observation-policy` — read-before-edit policy
//!   ([`observation`]): reads record observations, an edit to a file the
//!   session has not read is rejected, and a stale observation (the file
//!   changed underneath) fails the guarded mutation.
//! - `packages/fs/tool-fs` — the `read` / `write` / `edit` tools ([`tools`],
//!   [`read_render`], [`diff`]).
//! - `packages/fs/tool-fs-search` — the `glob` / `grep` tools ([`search`]),
//!   reimplemented in-process on the `ignore` + `regex` crates instead of
//!   spawning the packaged ripgrep binary.
//!
//! Cross-cutting divergences (per-module docs carry the details): local I/O
//! is synchronous `std::fs` under the single-threaded runtime; observation
//! owners are session ids instead of weakly-held session objects;
//! observations are dispatched awaited (`ctx.parallel`) because the Rust
//! bus's `emit` defers listener futures; system-prompt sections, streaming
//! reads, `read_image`, `read_bytes`, file URLs, the spill store, sandbox
//! escalation fields, and over-cap glob sampling are not ported.

mod diff;
mod events;
mod local;
mod observation;
mod read_render;
mod search;
mod tools;
mod types;

pub use crate::diff::{DIFF_CONTEXT, compute_hunk_diffs, diffs_from_meta};
pub use crate::events::{
    FsEditIntentSlot, FsObservedEvent, FsOwnerKey, FsWriteIntentSlot, owner_of,
};
pub use crate::local::{
    DEFAULT_DIFF_BASIS_MAX_BYTES, LineEndings, LocalFileSystem, LocalFileSystemConfig,
    apply_literal_edit, detect_line_endings, normalize_line_endings, restore_line_endings,
};
pub use crate::observation::install_observation_policy;
pub use crate::read_render::{
    FileReadOutcome, FsReadMeta, READ_LIMIT, READ_MAX_BYTES, READ_MAX_LINE_LENGTH, ReadWindow,
    WindowResult, build_window, format_read_output, lang_from_path, read_meta_from_meta,
};
pub use crate::search::{
    GLOB_MAX_RESULTS, GLOB_VCS_EXCLUDES, GREP_MAX_LINE_BYTES, GREP_MAX_MATCHES, GrepMatch,
    SEARCH_META_MAX_BYTES, SEARCH_TIMEOUT_MS, SearchError, SearchErrorCode, SearchToolsConfig,
    format_grep_matches, preview_line, register_search_tools, search_view_from_meta,
    to_workdir_relative,
};
pub use crate::tools::{FsToolsConfig, format_edit_output, format_write_output, register_fs_tools};
pub use crate::types::{
    FsDirEntry, FsEditOutcome, FsEditRequest, FsEntryType, FsError, FsErrorCode, FsInfo,
    FsObservation, FsPathEntryType, FsPathInfo, FsTarget, FsTargetKey, FsVersion, FsWriteIntent,
    FsWriteOperation, FsWriteOutcome,
};
