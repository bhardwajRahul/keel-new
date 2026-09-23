//! Vocabulary for the filesystem capability (port of upstream
//! `packages/fs/fs/src/types.ts`): opaque target/version identities, `stat`
//! metadata, guarded write intents and mutation outcomes, the literal-edit
//! request, and the typed error taxonomy every layer raises from.

use serde::{Deserialize, Serialize};

/// Opaque key identifying one target across path aliases (the local backend
/// derives it from the realpath). Consumers must not parse it or assume it is
/// an openable path — [`crate::LocalFileSystem::process_path`] exists for
/// that.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FsTargetKey(String);

impl FsTargetKey {
    /// Brand a backend's raw key string; no validation happens here.
    pub fn new(key: impl Into<String>) -> Self {
        FsTargetKey(key.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Opaque freshness token guarded by conditional writes and edits. The local
/// backend derives it from high-resolution stat identity; consumers record
/// and replay it but never interpret it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FsVersion(String);

impl FsVersion {
    /// Brand a backend's raw version string; no validation happens here.
    pub fn new(version: impl Into<String>) -> Self {
        FsVersion(version.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// One authoritative observation of a target: present at a version (the
/// basis for a guarded replacement) or confirmed absent (authorizes only a
/// guarded create).
#[derive(Debug, Clone, PartialEq)]
pub enum FsObservation {
    Present { version: FsVersion },
    Absent,
}

/// A resolved path: the stable identity plus the path shown to the model and
/// the UI.
#[derive(Debug, Clone, PartialEq)]
pub struct FsTarget {
    /// Opaque identity for stale guards and lookups.
    pub target_key: FsTargetKey,
    /// Model/UI-facing path (absolute for the local backend).
    pub display_path: String,
}

/// Coarse target classification from `stat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsEntryType {
    File,
    Directory,
    Other,
}

/// Path-entry classification from `lstat`, which can additionally report a
/// symlink because the final component is not followed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsPathEntryType {
    File,
    Directory,
    Symlink,
    Other,
}

/// Metadata for a resolved target; `stat` returns `None` for an absent one.
#[derive(Debug, Clone, PartialEq)]
pub struct FsInfo {
    /// Freshness token of the target right now.
    pub version: FsVersion,
    pub entry_type: FsEntryType,
    /// Byte size of a regular file.
    pub size: Option<u64>,
}

/// Metadata for a path probed without following the final symlink component.
#[derive(Debug, Clone, PartialEq)]
pub struct FsPathInfo {
    pub version: FsVersion,
    pub entry_type: FsPathEntryType,
    pub size: Option<u64>,
}

/// One direct child from a directory listing: metadata and a resolved target
/// only, never file content.
#[derive(Debug, Clone, PartialEq)]
pub struct FsDirEntry {
    /// Basename inside the listed directory.
    pub name: String,
    pub entry_type: FsEntryType,
    /// Resolved child target for follow-up operations.
    pub target: FsTarget,
    pub version: Option<FsVersion>,
    /// Byte size, reported for regular files only.
    pub size: Option<u64>,
}

/// Guarded write intent. `CreateIfAbsent` rejects an existing target with
/// `FS_NOT_OBSERVED`; `ReplaceIfVersion` rejects absence or a version
/// mismatch with `FS_STALE_VERSION`. Passing no intent means an
/// unconditional (but still atomic) create-or-overwrite.
#[derive(Debug, Clone, PartialEq)]
pub enum FsWriteIntent {
    CreateIfAbsent,
    ReplaceIfVersion(FsVersion),
}

/// Whether a write created a new file or replaced an existing one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FsWriteOperation {
    Create,
    Update,
}

/// Outcome of a full-file write.
#[derive(Debug, Clone, PartialEq)]
pub struct FsWriteOutcome {
    pub operation: FsWriteOperation,
    /// Version of the file after the write.
    pub version: FsVersion,
    /// LF-normalized content BEFORE the write — the contextual-diff basis.
    /// `None` for a create or when the prior content was undiffable (binary,
    /// invalid UTF-8, or either side at/over the diff-basis byte limit);
    /// consumers then fall back to a whole-file diff.
    pub before: Option<String>,
    /// LF-normalized content AFTER the write (shares `before`'s diff basis).
    pub after: String,
}

/// A literal-replacement edit request.
#[derive(Debug, Clone, PartialEq)]
pub struct FsEditRequest {
    /// Literal non-empty text to replace; must match exactly after
    /// line-ending normalization.
    pub old_string: String,
    /// Literal replacement; empty deletes the match.
    pub new_string: String,
    /// Replace every match instead of requiring exactly one.
    pub replace_all: bool,
}

/// Outcome of a literal edit; `before`/`after` are the LF-normalized diff
/// basis, never a rendered diff.
#[derive(Debug, Clone, PartialEq)]
pub struct FsEditOutcome {
    pub version: FsVersion,
    pub before: String,
    pub after: String,
}

/// Stable machine-routable codes for filesystem failures; retry, policy, and
/// UI layers branch on these instead of parsing messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsErrorCode {
    NotFound,
    NotDirectory,
    NotText,
    NotRegularFile,
    TooLarge,
    PermissionDenied,
    IoError,
    StaleVersion,
    NotObserved,
    AmbiguousEdit,
    EditNotFound,
    Aborted,
}

impl FsErrorCode {
    /// The upstream wire spelling of the code.
    pub fn as_str(self) -> &'static str {
        match self {
            FsErrorCode::NotFound => "FS_NOT_FOUND",
            FsErrorCode::NotDirectory => "FS_NOT_DIRECTORY",
            FsErrorCode::NotText => "FS_NOT_TEXT",
            FsErrorCode::NotRegularFile => "FS_NOT_REGULAR_FILE",
            FsErrorCode::TooLarge => "FS_TOO_LARGE",
            FsErrorCode::PermissionDenied => "FS_PERMISSION_DENIED",
            FsErrorCode::IoError => "FS_IO_ERROR",
            FsErrorCode::StaleVersion => "FS_STALE_VERSION",
            FsErrorCode::NotObserved => "FS_NOT_OBSERVED",
            FsErrorCode::AmbiguousEdit => "FS_AMBIGUOUS_EDIT",
            FsErrorCode::EditNotFound => "FS_EDIT_NOT_FOUND",
            FsErrorCode::Aborted => "FS_ABORTED",
        }
    }
}

/// Typed filesystem failure: a human-readable message plus a stable
/// [`FsErrorCode`]. The provider, the observation policy, and the tools all
/// raise this one vocabulary.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{message}")]
pub struct FsError {
    pub message: String,
    pub code: FsErrorCode,
}

impl FsError {
    pub fn new(message: impl Into<String>, code: FsErrorCode) -> Self {
        FsError {
            message: message.into(),
            code,
        }
    }
}
