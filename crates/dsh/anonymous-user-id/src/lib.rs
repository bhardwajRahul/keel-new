//! Port of `packages/identity/anonymous-user-id`
//! (`@deepseek-ai/dsh-anonymous-user-id`): the per-harness-home anonymous
//! user id shared by telemetry and feedback.
//!
//! The id is a random UUID persisted as a bare line in `.anonymous-user-id`
//! inside the harness home (`$DSH_HOME` > `~/.dsh`), never derived from the
//! hostname, network address, git remote, or any other identifying source. It
//! is scoped to the harness home, not the machine: every process sharing one
//! `$DSH_HOME` reports the same id, and deleting the file mints a fresh
//! identity on the next launch. The API is synchronous so boot-time and
//! command consumers share one entry point, and the result is memoized per
//! resolved file path: one process touches the disk once, and a file deleted
//! mid-run keeps the process's id until the next launch.
//!
//! Divergences from the TS original:
//! - `options.env` (a whole environment record) becomes
//!   [`AnonymousUserIdOptions::dsh_home`], the one `$DSH_HOME` value the
//!   lookup consults; `None` reads the process environment.
//! - The memo is a process-wide `Mutex`-guarded map instead of a
//!   module-level `Map` (Rust tests run on multiple threads).
//! - The `./invariant` companion is not ported (`dsh-invariants` does not
//!   exist in this workspace).

use dsh_home_paths::{resolve_dsh_home, resolve_dsh_home_from};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// Compile-time marker for [`AnonymousUserId`].
pub enum AnonymousUserIdMark {}

/// A harness-home-scoped anonymous user id (random UUID v4).
pub type AnonymousUserId = dsh_brand::Branded<AnonymousUserIdMark>;

/// File inside the harness home storing the id: a bare UUID line, no wrapper
/// format.
pub const ANONYMOUS_USER_ID_FILE_NAME: &str = ".anonymous-user-id";

/// Ambient hooks for locating and generating the id; every field has a
/// default.
#[derive(Default)]
pub struct AnonymousUserIdOptions<'a> {
    /// `$DSH_HOME` value to resolve the harness home from; `None` consults
    /// the process environment.
    pub dsh_home: Option<&'a str>,
    /// UUID generator; `None` uses a random v4 UUID (test hook).
    pub random_uuid: Option<&'a dyn Fn() -> String>,
}

/// Process-lifetime memo keyed by resolved file path, so distinct test homes
/// never share an id.
fn memo() -> &'static Mutex<HashMap<PathBuf, AnonymousUserId>> {
    static MEMO: OnceLock<Mutex<HashMap<PathBuf, AnonymousUserId>>> = OnceLock::new();
    MEMO.get_or_init(Mutex::default)
}

/// Whether a trimmed line is an 8-4-4-4-12 hex UUID (either letter case).
fn is_uuid(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 36
        && bytes.iter().enumerate().all(|(i, byte)| match i {
            8 | 13 | 18 | 23 => *byte == b'-',
            _ => byte.is_ascii_hexdigit(),
        })
}

/// Read a valid persisted id from the file, or `None` when absent or corrupt
/// (the caller mints and persists a fresh one).
fn read_persisted_id(file: &Path) -> Option<AnonymousUserId> {
    let text = std::fs::read_to_string(file).ok()?;
    let value = text.trim();
    is_uuid(value).then(|| AnonymousUserId::new(value))
}

/// Write the id with exclusive create, failing when the file already exists.
fn write_exclusive(file: &Path, id: &str) -> std::io::Result<()> {
    if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut handle = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(file)?;
    handle.write_all(format!("{id}\n").as_bytes())
}

/// Return the harness home's anonymous user id, creating and persisting one
/// on first use.
///
/// A concurrent first launch is settled by an exclusive-create write: the
/// loser rereads the winner's id. (A reread landing inside the winner's
/// narrow create-to-write window can still yield two per-process ids for that
/// run; the next launch converges on the persisted one.) Persistence is
/// best-effort — a write failure (read-only home) still returns a usable id
/// for the current run so feedback and telemetry are never blocked.
pub fn get_or_create_anonymous_user_id(options: AnonymousUserIdOptions<'_>) -> AnonymousUserId {
    let home = match options.dsh_home {
        Some(value) => resolve_dsh_home_from(None, Some(value)),
        None => resolve_dsh_home(None),
    };
    let file = home.join(ANONYMOUS_USER_ID_FILE_NAME);
    if let Some(cached) = memo().lock().unwrap().get(&file) {
        return cached.clone();
    }

    let id = read_persisted_id(&file).unwrap_or_else(|| {
        let created = match options.random_uuid {
            Some(generate) => generate(),
            None => uuid::Uuid::new_v4().to_string(),
        };
        match write_exclusive(&file, &created) {
            Ok(()) => AnonymousUserId::new(created),
            // An exclusive-create refusal covers both a concurrent winner and
            // a pre-existing corrupt file: the reread adopts a valid winner,
            // and an invalid reread falls through to the overwrite path.
            // Other failures (read-only home) land there too, best-effort.
            Err(_) => read_persisted_id(&file).unwrap_or_else(|| {
                let _ = std::fs::write(&file, format!("{created}\n"));
                AnonymousUserId::new(created)
            }),
        }
    });
    memo().lock().unwrap().insert(file, id.clone());
    id
}
