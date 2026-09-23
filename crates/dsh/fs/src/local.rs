//! Local filesystem provider registered as the `fs` service (port of
//! upstream `packages/fs/fs-local`, `fsio.ts` folded in). Target identity is
//! realpath-derived so path aliases share one stale guard; mutations stage a
//! private temp file and publish atomically; guarded creates use a
//! no-replace hard link so a concurrent creator's file survives.
//!
//! Divergences: I/O is synchronous `std::fs` under the single-threaded
//! runtime, so the upstream per-target async lock is unnecessary — a
//! mutation's probe→guard→publish window cannot interleave with another
//! in-process mutation. `AbortSignal`s are still honored at operation entry.
//! Unix-only version identity (`dev:ino:size:mtimeNs:ctimeNs`); the Win32
//! DACL paths are not ported.

use crate::types::{
    FsDirEntry, FsEditOutcome, FsEditRequest, FsEntryType, FsError, FsErrorCode, FsInfo,
    FsPathEntryType, FsPathInfo, FsTarget, FsTargetKey, FsVersion, FsWriteIntent, FsWriteOperation,
    FsWriteOutcome,
};
use dsh_cordis::{Context, Service};
use dsh_timeout::AbortSignal;
use std::io::Write;
use std::os::unix::fs::MetadataExt;
use std::os::unix::fs::PermissionsExt;
use std::path::{Component, Path, PathBuf};
use std::rc::Rc;

/// Bytes sampled from the head of a file when rejecting binary content.
const BINARY_SAMPLE_BYTES: usize = 8192;

/// Default exclusive byte limit for each overwrite-diff side (10 MiB).
pub const DEFAULT_DIFF_BASIS_MAX_BYTES: usize = 10 * 1024 * 1024;

/// Configuration for the local backend.
#[derive(Debug, Clone, Default)]
pub struct LocalFileSystemConfig {
    /// Base directory for relative paths; defaults to the process cwd. A
    /// resolution default, NOT a containment boundary.
    pub cwd: Option<PathBuf>,
    /// Exclusive byte limit on each overwrite-diff side; defaults to
    /// [`DEFAULT_DIFF_BASIS_MAX_BYTES`].
    pub diff_basis_max_bytes: Option<usize>,
}

/// The host-filesystem backend, registered as the `fs` service. The upstream
/// seam/provider package split is collapsed here: this is the only backend
/// ported (sandbox and e2b providers are out of scope), so its public
/// methods ARE the `ctx.fs` contract.
pub struct LocalFileSystem {
    cwd: PathBuf,
    diff_basis_max_bytes: usize,
}

impl Service for LocalFileSystem {
    const NAME: &'static str = "fs";
}

fn aborted(signal: Option<&AbortSignal>, verb: &str) -> Result<(), FsError> {
    if signal.is_some_and(AbortSignal::aborted) {
        return Err(FsError::new(
            format!("{verb} aborted"),
            FsErrorCode::Aborted,
        ));
    }
    Ok(())
}

/// High-resolution identity + freshness version token.
fn version_of(meta: &std::fs::Metadata) -> FsVersion {
    let mtime_ns = meta.mtime() as i128 * 1_000_000_000 + meta.mtime_nsec() as i128;
    let ctime_ns = meta.ctime() as i128 * 1_000_000_000 + meta.ctime_nsec() as i128;
    FsVersion::new(format!(
        "{}:{}:{}:{}:{}",
        meta.dev(),
        meta.ino(),
        meta.len(),
        mtime_ns,
        ctime_ns
    ))
}

fn entry_type_of(meta: &std::fs::Metadata) -> FsEntryType {
    if meta.is_file() {
        FsEntryType::File
    } else if meta.is_dir() {
        FsEntryType::Directory
    } else {
        FsEntryType::Other
    }
}

/// A missing path and a path whose parent segment is a regular file both
/// read as "absent" — neither can hold a target.
fn is_absent_error(error: &std::io::Error) -> bool {
    matches!(
        error.kind(),
        std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
    )
}

fn is_permission_error(error: &std::io::Error) -> bool {
    error.kind() == std::io::ErrorKind::PermissionDenied
}

/// Lexically absolutize + normalize a path against a base (`.` dropped,
/// `..` folded), without touching the filesystem — the display-path form.
fn absolutize(base: &Path, path: &str) -> PathBuf {
    let joined = if Path::new(path).is_absolute() {
        PathBuf::from(path)
    } else {
        base.join(path)
    };
    let mut out = PathBuf::new();
    for component in joined.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Collapse CRLF pairs to LF — the canonical in-memory form every edit and
/// diff basis uses. Lone `\r` bytes are left alone.
pub fn normalize_line_endings(content: &str) -> String {
    content.replace("\r\n", "\n")
}

/// Line-ending style detected before normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineEndings {
    Lf,
    Crlf,
}

/// Detect the dominant style in the first 4 KiB of raw text.
pub fn detect_line_endings(raw: &str) -> LineEndings {
    let sample: String = raw.chars().take(4096).collect();
    let crlf = sample.matches("\r\n").count();
    let lf = sample.matches('\n').count() - crlf;
    if crlf > lf {
        LineEndings::Crlf
    } else {
        LineEndings::Lf
    }
}

/// Convert LF-normalized content back to the detected storage style. The
/// CRLF arm re-normalizes first so an already-CRLF run is never doubled.
pub fn restore_line_endings(content: &str, endings: LineEndings) -> String {
    match endings {
        LineEndings::Lf => content.to_string(),
        LineEndings::Crlf => normalize_line_endings(content).replace('\n', "\r\n"),
    }
}

fn count_occurrences(content: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    let mut count = 0;
    let mut index = 0;
    while let Some(found) = content[index..].find(needle) {
        count += 1;
        index += found + needle.len();
    }
    count
}

/// Apply a literal replacement to LF-normalized content. An empty or absent
/// `old_string` fails as `FS_EDIT_NOT_FOUND`; multiple matches fail as
/// `FS_AMBIGUOUS_EDIT` unless `replace_all` is set.
pub fn apply_literal_edit(
    content: &str,
    old_string: &str,
    new_string: &str,
    replace_all: bool,
    display_path: &str,
) -> Result<(String, usize), FsError> {
    let old = normalize_line_endings(old_string);
    if old.is_empty() {
        return Err(FsError::new(
            "old_string must be a non-empty string",
            FsErrorCode::EditNotFound,
        ));
    }
    let new = normalize_line_endings(new_string);
    let replacements = count_occurrences(content, &old);
    if replacements == 0 {
        return Err(FsError::new(
            format!("old_string was not found in \"{display_path}\""),
            FsErrorCode::EditNotFound,
        ));
    }
    if !replace_all && replacements > 1 {
        return Err(FsError::new(
            format!(
                "old_string matched {replacements} times in \"{display_path}\"; provide a more specific old_string or set replace_all to true"
            ),
            FsErrorCode::AmbiguousEdit,
        ));
    }
    Ok((content.replace(&old, &new), replacements))
}

impl LocalFileSystem {
    /// Build a backend without registering it (test/composition helper).
    pub fn new(config: LocalFileSystemConfig) -> anyhow::Result<LocalFileSystem> {
        let diff_basis_max_bytes = config
            .diff_basis_max_bytes
            .unwrap_or(DEFAULT_DIFF_BASIS_MAX_BYTES);
        if diff_basis_max_bytes == 0 {
            anyhow::bail!("fs-local: diff_basis_max_bytes must be a positive integer");
        }
        let cwd = match config.cwd {
            Some(cwd) => cwd,
            None => std::env::current_dir()?,
        };
        Ok(LocalFileSystem {
            cwd,
            diff_basis_max_bytes,
        })
    }

    /// Build and register the backend as the `fs` service.
    pub fn provide(
        ctx: &Context,
        config: LocalFileSystemConfig,
    ) -> anyhow::Result<Rc<LocalFileSystem>> {
        let fs = Rc::new(LocalFileSystem::new(config)?);
        ctx.provide_service(fs.clone())?;
        Ok(fs)
    }

    /// Resolve a path into a stable target. The identity is the realpath;
    /// for a missing target, the nearest existing ancestor's realpath plus
    /// the missing suffix, so the key survives creation of intermediate
    /// directories and stays shared across symlinked aliases.
    pub fn resolve(
        &self,
        path: &str,
        cwd: Option<&Path>,
        signal: Option<&AbortSignal>,
    ) -> Result<FsTarget, FsError> {
        aborted(signal, "resolve")?;
        if path.trim().is_empty() {
            return Err(FsError::new(
                "file_path must be a non-empty string",
                FsErrorCode::NotFound,
            ));
        }
        let display = absolutize(cwd.unwrap_or(&self.cwd), path);
        let display_path = display.to_string_lossy().into_owned();
        match std::fs::canonicalize(&display) {
            Ok(real) => {
                return Ok(FsTarget {
                    target_key: FsTargetKey::new(real.to_string_lossy().into_owned()),
                    display_path,
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotADirectory => {
                return Err(FsError::new(
                    format!(
                        "cannot resolve \"{display_path}\": a parent path segment is not a directory"
                    ),
                    FsErrorCode::NotFound,
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(FsError::new(
                    format!("cannot resolve \"{display_path}\": {error}"),
                    FsErrorCode::IoError,
                ));
            }
        }
        // Absent target: realpath the nearest existing ancestor and re-append
        // the missing suffix so the key is stable across creation.
        let mut missing: Vec<std::ffi::OsString> =
            vec![display.file_name().unwrap_or_default().to_os_string()];
        let mut ancestor = display.parent().map(Path::to_path_buf);
        while let Some(current) = ancestor {
            match std::fs::canonicalize(&current) {
                Ok(real) => {
                    let mut key = real;
                    for part in missing.iter().rev() {
                        key.push(part);
                    }
                    return Ok(FsTarget {
                        target_key: FsTargetKey::new(key.to_string_lossy().into_owned()),
                        display_path,
                    });
                }
                Err(error) if is_absent_error(&error) => {
                    missing.push(current.file_name().unwrap_or_default().to_os_string());
                    ancestor = current.parent().map(Path::to_path_buf);
                }
                Err(error) => {
                    return Err(FsError::new(
                        format!("cannot resolve \"{display_path}\": {error}"),
                        FsErrorCode::IoError,
                    ));
                }
            }
        }
        // Even the root failed to realpath: fall back to the display path.
        Ok(FsTarget {
            target_key: FsTargetKey::new(display_path.clone()),
            display_path,
        })
    }

    /// The absolute path a subprocess in this execution world can open.
    pub fn process_path(&self, target: &FsTarget) -> String {
        target.target_key.as_str().to_string()
    }

    /// Canonical containment: whether `child` is `parent` or one of its
    /// descendants. Both targets must come from this backend, so the check is
    /// lexical over canonical keys — the path-traversal guard consumers use
    /// instead of parsing keys themselves.
    pub fn contains(&self, parent: &FsTarget, child: &FsTarget) -> bool {
        Path::new(child.target_key.as_str())
            .strip_prefix(Path::new(parent.target_key.as_str()))
            .is_ok()
    }

    fn probe(&self, key: &FsTargetKey) -> Result<Option<std::fs::Metadata>, FsError> {
        match std::fs::metadata(key.as_str()) {
            Ok(meta) => Ok(Some(meta)),
            Err(error) if is_absent_error(&error) => Ok(None),
            Err(error) => Err(FsError::new(
                format!("cannot stat \"{}\": {error}", key.as_str()),
                if is_permission_error(&error) {
                    FsErrorCode::PermissionDenied
                } else {
                    FsErrorCode::IoError
                },
            )),
        }
    }

    /// Target metadata, or `None` when the target is absent.
    pub fn stat(
        &self,
        target: &FsTarget,
        signal: Option<&AbortSignal>,
    ) -> Result<Option<FsInfo>, FsError> {
        aborted(signal, "stat")?;
        let Some(meta) = self.probe(&target.target_key)? else {
            return Ok(None);
        };
        let entry_type = entry_type_of(&meta);
        Ok(Some(FsInfo {
            version: version_of(&meta),
            entry_type,
            size: (entry_type == FsEntryType::File).then(|| meta.len()),
        }))
    }

    /// Path metadata without following the final symlink component, so a
    /// consumer can reject a repository-owned link before resolution follows
    /// it. `None` when the path is absent.
    pub fn lstat(
        &self,
        path: &str,
        cwd: Option<&Path>,
        signal: Option<&AbortSignal>,
    ) -> Result<Option<FsPathInfo>, FsError> {
        aborted(signal, "lstat")?;
        if path.trim().is_empty() {
            return Err(FsError::new(
                "file_path must be a non-empty string",
                FsErrorCode::NotFound,
            ));
        }
        let full = absolutize(cwd.unwrap_or(&self.cwd), path);
        let meta = match std::fs::symlink_metadata(&full) {
            Ok(meta) => meta,
            Err(error) if is_absent_error(&error) => return Ok(None),
            Err(error) => {
                return Err(FsError::new(
                    format!("cannot lstat \"{}\": {error}", full.display()),
                    FsErrorCode::IoError,
                ));
            }
        };
        let entry_type = if meta.file_type().is_symlink() {
            FsPathEntryType::Symlink
        } else if meta.is_file() {
            FsPathEntryType::File
        } else if meta.is_dir() {
            FsPathEntryType::Directory
        } else {
            FsPathEntryType::Other
        };
        Ok(Some(FsPathInfo {
            version: version_of(&meta),
            entry_type,
            size: (entry_type == FsPathEntryType::File).then(|| meta.len()),
        }))
    }

    fn read_raw(&self, target: &FsTarget, verb: &str) -> Result<Vec<u8>, FsError> {
        let Some(meta) = self.probe(&target.target_key)? else {
            return Err(FsError::new(
                format!("cannot {verb} \"{}\": not found", target.display_path),
                FsErrorCode::NotFound,
            ));
        };
        if !meta.is_file() {
            return Err(FsError::new(
                format!(
                    "cannot {verb} \"{}\": not a regular file",
                    target.display_path
                ),
                FsErrorCode::NotRegularFile,
            ));
        }
        std::fs::read(target.target_key.as_str()).map_err(|error| {
            FsError::new(
                format!("cannot {verb} \"{}\": {error}", target.display_path),
                if is_permission_error(&error) {
                    FsErrorCode::PermissionDenied
                } else {
                    FsErrorCode::IoError
                },
            )
        })
    }

    /// Read a whole regular UTF-8 text file. Rejects non-regular files,
    /// NUL-byte binary samples, and invalid UTF-8; the content comes back
    /// byte-for-byte (no line-ending normalization).
    pub fn read_text(
        &self,
        target: &FsTarget,
        signal: Option<&AbortSignal>,
    ) -> Result<String, FsError> {
        aborted(signal, "read")?;
        let raw = self.read_raw(target, "read")?;
        if raw.iter().take(BINARY_SAMPLE_BYTES).any(|&byte| byte == 0) {
            return Err(FsError::new(
                format!("cannot read \"{}\": binary file", target.display_path),
                FsErrorCode::NotText,
            ));
        }
        String::from_utf8(raw).map_err(|_| {
            FsError::new(
                format!(
                    "cannot read \"{}\": invalid UTF-8 text",
                    target.display_path
                ),
                FsErrorCode::NotText,
            )
        })
    }

    /// List direct children in stable name order: resolved targets plus
    /// cheap metadata only, never file content.
    pub fn list_dir(
        &self,
        target: &FsTarget,
        signal: Option<&AbortSignal>,
    ) -> Result<Vec<FsDirEntry>, FsError> {
        aborted(signal, "list")?;
        let Some(meta) = self.probe(&target.target_key)? else {
            return Err(FsError::new(
                format!("cannot list \"{}\": not found", target.display_path),
                FsErrorCode::NotFound,
            ));
        };
        if !meta.is_dir() {
            return Err(FsError::new(
                format!("cannot list \"{}\": not a directory", target.display_path),
                FsErrorCode::NotDirectory,
            ));
        }
        let mut names: Vec<String> = Vec::new();
        let read_dir = std::fs::read_dir(target.target_key.as_str()).map_err(|error| {
            FsError::new(
                format!("cannot list \"{}\": {error}", target.display_path),
                if is_permission_error(&error) {
                    FsErrorCode::PermissionDenied
                } else {
                    FsErrorCode::IoError
                },
            )
        })?;
        for entry in read_dir {
            let entry = entry.map_err(|error| {
                FsError::new(
                    format!("cannot list \"{}\": {error}", target.display_path),
                    FsErrorCode::IoError,
                )
            })?;
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
        names.sort();
        let mut out = Vec::with_capacity(names.len());
        for name in names {
            aborted(signal, "list")?;
            let child = self.resolve(&name, Some(Path::new(target.target_key.as_str())), signal)?;
            let child = FsTarget {
                target_key: child.target_key,
                display_path: Path::new(&target.display_path)
                    .join(&name)
                    .to_string_lossy()
                    .into_owned(),
            };
            let info = self.probe(&child.target_key)?;
            let entry_type = info.as_ref().map_or(FsEntryType::Other, entry_type_of);
            out.push(FsDirEntry {
                name,
                entry_type,
                version: info.as_ref().map(version_of),
                size: info
                    .as_ref()
                    .filter(|meta| meta.is_file())
                    .map(std::fs::Metadata::len),
                target: child,
            });
        }
        Ok(out)
    }

    /// Best-effort overwrite-diff basis: LF-normalized text, or `None` for a
    /// binary, invalid-UTF-8, at/over-limit, or unreadable prior file — the
    /// write still succeeds and presentation falls back to a whole-file diff.
    fn read_text_for_diff(&self, key: &FsTargetKey) -> Option<String> {
        let meta = std::fs::metadata(key.as_str()).ok()?;
        if !meta.is_file() || meta.len() >= self.diff_basis_max_bytes as u64 {
            return None;
        }
        let raw = std::fs::read(key.as_str()).ok()?;
        if raw.len() >= self.diff_basis_max_bytes || raw.contains(&0) {
            return None;
        }
        let text = String::from_utf8(raw).ok()?;
        Some(normalize_line_endings(&text))
    }

    /// Stage the full content in a private sibling temp file (0600), sync
    /// it, then publish atomically: `create_if_absent` publishes with a
    /// no-replace hard link so a concurrent creator survives and this write
    /// reports `FS_NOT_OBSERVED`; otherwise rename replaces the target.
    fn write_file_atomic(
        &self,
        key: &FsTargetKey,
        content: &str,
        mode: Option<u32>,
        create_if_absent: Option<&str>,
    ) -> Result<(), FsError> {
        let target_path = Path::new(key.as_str());
        let directory = target_path.parent().unwrap_or(Path::new("/"));
        std::fs::create_dir_all(directory).map_err(|error| {
            FsError::new(
                format!("cannot write \"{}\": {error}", key.as_str()),
                FsErrorCode::IoError,
            )
        })?;
        let io_error = |error: std::io::Error| {
            FsError::new(
                format!("cannot write \"{}\": {error}", key.as_str()),
                if is_permission_error(&error) {
                    FsErrorCode::PermissionDenied
                } else {
                    FsErrorCode::IoError
                },
            )
        };
        let mut temp = tempfile::Builder::new()
            .prefix(&format!(
                ".{}.",
                target_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
            ))
            .suffix(".tmp")
            .tempfile_in(directory)
            .map_err(io_error)?;
        temp.write_all(content.as_bytes()).map_err(io_error)?;
        temp.as_file().sync_all().map_err(io_error)?;
        if let Some(mode) = mode {
            std::fs::set_permissions(temp.path(), std::fs::Permissions::from_mode(mode))
                .map_err(io_error)?;
        }
        match create_if_absent {
            Some(display_path) => {
                // Hard-link no-replace publication: an existing entry (a
                // racing creator, a dangling symlink) is preserved and this
                // guarded create fails instead of clobbering it.
                match std::fs::hard_link(temp.path(), target_path) {
                    Ok(()) => Ok(()),
                    Err(error) => {
                        let existing = std::fs::symlink_metadata(target_path).ok();
                        match existing {
                            Some(meta) if !meta.is_file() => Err(FsError::new(
                                format!("cannot write \"{display_path}\": not a regular file"),
                                FsErrorCode::NotRegularFile,
                            )),
                            Some(_) => Err(FsError::new(
                                format!(
                                    "cannot overwrite existing \"{display_path}\" without reading it first"
                                ),
                                FsErrorCode::NotObserved,
                            )),
                            None if error.kind() == std::io::ErrorKind::AlreadyExists => {
                                Err(FsError::new(
                                    format!(
                                        "cannot overwrite existing \"{display_path}\" without reading it first"
                                    ),
                                    FsErrorCode::NotObserved,
                                ))
                            }
                            None => Err(io_error(error)),
                        }
                    }
                }
            }
            None => temp
                .persist(target_path)
                .map(|_| ())
                .map_err(|error| io_error(error.error)),
        }
    }

    fn version_after_write(&self, key: &FsTargetKey) -> Result<FsVersion, FsError> {
        Ok(self
            .probe(key)?
            .map(|meta| version_of(&meta))
            // A concurrent unlink between publish and stat: sentinel version.
            .unwrap_or_else(|| FsVersion::new(format!("missing:{}", key.as_str()))))
    }

    /// Atomically create or replace UTF-8 text. `expected` guards intent and
    /// freshness (see [`FsWriteIntent`]); `None` is an unconditional but
    /// still atomic create-or-overwrite.
    pub fn write_text(
        &self,
        target: &FsTarget,
        content: &str,
        expected: Option<&FsWriteIntent>,
        signal: Option<&AbortSignal>,
    ) -> Result<FsWriteOutcome, FsError> {
        aborted(signal, "write")?;
        let existing = self.probe(&target.target_key)?;
        if let Some(meta) = &existing {
            if !meta.is_file() {
                return Err(FsError::new(
                    format!(
                        "cannot write \"{}\": not a regular file",
                        target.display_path
                    ),
                    FsErrorCode::NotRegularFile,
                ));
            }
        }
        match expected {
            Some(FsWriteIntent::ReplaceIfVersion(version)) => match &existing {
                None => {
                    return Err(FsError::new(
                        format!(
                            "cannot write \"{}\": file no longer exists",
                            target.display_path
                        ),
                        FsErrorCode::StaleVersion,
                    ));
                }
                Some(meta) if version_of(meta) != *version => {
                    return Err(FsError::new(
                        format!(
                            "cannot write \"{}\": file changed since it was read",
                            target.display_path
                        ),
                        FsErrorCode::StaleVersion,
                    ));
                }
                Some(_) => {}
            },
            Some(FsWriteIntent::CreateIfAbsent) if existing.is_some() => {
                return Err(FsError::new(
                    format!(
                        "cannot overwrite existing \"{}\" without reading it first",
                        target.display_path
                    ),
                    FsErrorCode::NotObserved,
                ));
            }
            _ => {}
        }
        let diffable = existing.is_some() && content.len() < self.diff_basis_max_bytes;
        let before = if diffable {
            self.read_text_for_diff(&target.target_key)
        } else {
            None
        };
        self.write_file_atomic(
            &target.target_key,
            content,
            existing
                .as_ref()
                .map(|meta| meta.permissions().mode() & 0o777),
            matches!(expected, Some(FsWriteIntent::CreateIfAbsent))
                .then_some(target.display_path.as_str()),
        )?;
        Ok(FsWriteOutcome {
            operation: if existing.is_some() {
                FsWriteOperation::Update
            } else {
                FsWriteOperation::Create
            },
            version: self.version_after_write(&target.target_key)?,
            before,
            after: normalize_line_endings(content),
        })
    }

    /// Atomically edit literal text. The version guard is checked BEFORE
    /// matching, so a stale basis reports `FS_STALE_VERSION` rather than a
    /// match failure against newer content; `None` edits unconditionally.
    pub fn edit_text(
        &self,
        target: &FsTarget,
        edit: &FsEditRequest,
        expected: Option<&FsVersion>,
        signal: Option<&AbortSignal>,
    ) -> Result<FsEditOutcome, FsError> {
        aborted(signal, "edit")?;
        let Some(meta) = self.probe(&target.target_key)? else {
            // Missing targets report stale on guarded and unconditional paths
            // alike: the file the caller based the edit on is gone.
            return Err(FsError::new(
                format!(
                    "cannot edit \"{}\": file changed since it was read",
                    target.display_path
                ),
                FsErrorCode::StaleVersion,
            ));
        };
        if !meta.is_file() {
            return Err(FsError::new(
                format!(
                    "cannot edit \"{}\": not a regular file",
                    target.display_path
                ),
                FsErrorCode::NotRegularFile,
            ));
        }
        if let Some(version) = expected {
            if version_of(&meta) != *version {
                return Err(FsError::new(
                    format!(
                        "cannot edit \"{}\": file changed since it was read",
                        target.display_path
                    ),
                    FsErrorCode::StaleVersion,
                ));
            }
        }
        let raw = self.read_raw(target, "edit")?;
        if raw.contains(&0) {
            return Err(FsError::new(
                format!("cannot edit \"{}\": binary file", target.display_path),
                FsErrorCode::NotText,
            ));
        }
        let raw = String::from_utf8(raw).map_err(|_| {
            FsError::new(
                format!(
                    "cannot edit \"{}\": invalid UTF-8 text",
                    target.display_path
                ),
                FsErrorCode::NotText,
            )
        })?;
        let endings = detect_line_endings(&raw);
        let content = normalize_line_endings(&raw);
        let (edited, _) = apply_literal_edit(
            &content,
            &edit.old_string,
            &edit.new_string,
            edit.replace_all,
            &target.display_path,
        )?;
        let stored = restore_line_endings(&edited, endings);
        self.write_file_atomic(
            &target.target_key,
            &stored,
            Some(meta.permissions().mode() & 0o777),
            None,
        )?;
        Ok(FsEditOutcome {
            version: self.version_after_write(&target.target_key)?,
            before: content,
            after: edited,
        })
    }
}
