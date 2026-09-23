//! Port of `packages/util/atomic-write` (`@deepseek-ai/dsh-atomic-write`):
//! atomic file replacement plus cross-process writer coordination.
//!
//! [`write_file_atomic`] writes a random-suffix sibling opened with exclusive
//! create and the caller's permission bits, then renames it over the target,
//! so a reader observes either the old or the new complete content and a
//! replaced file ends up with exactly the stated mode. [`with_file_lock`]
//! serializes cross-process writers of one file through an exclusively-created
//! `<file>.lock` sibling, so a read-modify-write cycle cannot resurrect state
//! another writer just replaced; readers never take the lock because the
//! rename commit is atomic.
//!
//! Divergences from the TS original:
//! - The API is blocking (`std::fs` + the `tempfile` crate) instead of
//!   promise-based; contention backoff sleeps the calling thread.
//! - The temp sibling name uses `tempfile`'s random alphanumeric characters
//!   rather than 12 hex characters; the shape (`<file>.<random>.tmp`, same
//!   directory, exclusive create) is unchanged.
//! - Permission bits apply on Unix only; on other platforms they are ignored,
//!   matching the upstream tests' Windows carve-out.
//! - The Cordis `./invariant` companion is not ported: the package declares no
//!   runtime invariant (its contract is enforced by these unit tests).

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Filesystem options for [`write_file_atomic`]; `mode` is required so the
/// permission decision stays visible at every call site.
#[derive(Debug, Clone, Copy)]
pub struct WriteFileAtomicOptions {
    /// Permission bits stamped on the fresh temp inode and carried through the
    /// rename (subject to the process umask, like every fresh inode).
    pub mode: u32,
    /// Permission bits for parent directories this call creates (subject to
    /// the umask; existing directories keep their mode). `None` uses the
    /// platform default — pass `Some(0o700)` when the tree holds user-private
    /// data.
    pub dir_mode: Option<u32>,
}

/// Append a suffix to a path's final component (`doc.yaml` -> `doc.yaml.lock`),
/// unlike `Path::with_extension`, which would replace the extension.
fn sibling(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

/// Replace `filename` with `content` in one atomic step, creating parent
/// directories.
///
/// The content first lands in a random-suffix sibling opened with exclusive
/// create: the open refuses a symlink planted at the temp path, and the fresh
/// inode carries `options.mode` through the rename, so replacing a
/// wider-permission file narrows it without a chmod race. The rename also
/// replaces a symlinked target itself instead of writing through to its
/// referent, and the same-directory sibling keeps the rename on one
/// filesystem. On any failure the temp file is removed and the error
/// returned. Crash durability (fsync) is out of scope, as upstream.
pub fn write_file_atomic(
    filename: &Path,
    content: &str,
    options: WriteFileAtomicOptions,
) -> io::Result<()> {
    let parent = filename.parent().unwrap_or(Path::new("."));
    let mut dirs = std::fs::DirBuilder::new();
    dirs.recursive(true);
    #[cfg(unix)]
    if let Some(dir_mode) = options.dir_mode {
        use std::os::unix::fs::DirBuilderExt;
        dirs.mode(dir_mode);
    }
    dirs.create(parent)?;

    let mut prefix = filename
        .file_name()
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "filename has no final component",
            )
        })?
        .to_owned();
    prefix.push(".");
    let mut builder = tempfile::Builder::new();
    builder.prefix(&prefix).suffix(".tmp").rand_bytes(12);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        builder.permissions(std::fs::Permissions::from_mode(options.mode));
    }
    // NamedTempFile removes the sibling on drop, so every failure path below
    // (write error, failed rename via PersistError) cleans up before returning.
    let mut temp = builder.tempfile_in(parent)?;
    temp.write_all(content.as_bytes())?;
    temp.persist(filename).map_err(|error| error.error)?;
    Ok(())
}

/// Writer-lock protocol constants. These are robustness invariants of the
/// cross-process write protocol, not deployment tunables: contention normally
/// resolves within the retry deadline, while expiry fails the contender
/// without guessing whether the existing lock still has an owner.
const LOCK_RETRY_INITIAL: Duration = Duration::from_millis(20);
const LOCK_RETRY_MAX: Duration = Duration::from_millis(200);
const LOCK_TIMEOUT: Duration = Duration::from_millis(2_000);

/// Hold the cross-process writer lock for `filename` around one operation.
///
/// The lock is an exclusively-created sibling (`<filename>.lock`); paired with
/// the rename commit of [`write_file_atomic`], readers stay lock-free and only
/// writers contend. Contention backs off exponentially and fails with a
/// timed-out error after the deadline. A contender never removes an existing
/// lock, because file age cannot prove its owner stopped; orphan recovery is
/// an operator action. The parent directory must exist.
///
/// Returns the operation's result; the lock releases whether or not the
/// operation's own result is an error (the operation reports failure through
/// its return value `T`, e.g. a `Result`).
pub fn with_file_lock<T>(filename: &Path, operation: impl FnOnce() -> T) -> io::Result<T> {
    let lock_path = sibling(filename, ".lock");
    let deadline = Instant::now() + LOCK_TIMEOUT;
    let mut delay = LOCK_RETRY_INITIAL;
    loop {
        let mut open = std::fs::OpenOptions::new();
        open.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            open.mode(0o600);
        }
        match open.open(&lock_path) {
            Ok(mut lock) => {
                let _ = writeln!(lock, "{}", std::process::id());
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!(
                    "atomic-write: timed out waiting for the writer lock at {}",
                    lock_path.display()
                ),
            ));
        }
        std::thread::sleep(delay);
        delay = (delay * 2).min(LOCK_RETRY_MAX);
    }
    let result = operation();
    match std::fs::remove_file(&lock_path) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    Ok(result)
}
