//! Port of `packages/credentials/credentials-local`
//! (`@deepseek-ai/dsh-credentials-local`): the file-backed credentials
//! provider over `$DSH_HOME/.credentials.yaml`, layered against the
//! environment by trust:
//!
//! ```text
//! inherited process environment      (read-only, wins)
//! > $DSH_HOME/.credentials.yaml      (provider-managed, writable)
//! > <invocation cwd>/.env            (read-only fallback)
//! > $DSH_HOME/.env                   (read-only fallback)
//! ```
//!
//! The inherited environment wins because `DEEPSEEK_API_KEY=… dsh`, a CI
//! secret, or a container `-e` states this run's explicit intent; it cannot
//! be edited from inside the process, so it is visibly read-only instead of
//! silently shadowing writes. Everything below it loses to the managed store,
//! so a key written through a UI takes effect even when an older key sits in
//! a `.env`. The invoking project's `.env` ranks above the user's home file
//! (the more specific location wins) and below the store.
//!
//! Every write re-reads the document under the cross-process writer lock and
//! patches only its own entry, so comments and untouched entries survive;
//! external edits hot-publish through the seam, and each reload replaces the
//! snapshot wholesale so a deleted entry never lingers. The document is a
//! strict `CredentialRef`-to-non-empty-string mapping — never a dotenv file —
//! and anything unaddressable through the seam is rejected loud rather than
//! skipped.
//!
//! Divergences from the TS original:
//! - No YAML library is available in this workspace, so the document is a
//!   strict line-oriented subset of YAML: `#` comment lines, blank lines, and
//!   top-level `KEY: value` entries whose value is either a plain scalar or a
//!   JSON-style double-quoted scalar. The subset round-trips any string value
//!   and preserves comments on line edits; hand-written YAML outside the
//!   subset (block scalars, flow collections, anchors, single quotes, inline
//!   comments) is rejected as an invalid document. Unsetting the last entry
//!   leaves an empty file instead of `{}`.
//! - chokidar becomes a polling change detector on the tokio `LocalSet`:
//!   every `max(debounceMs, 20)` ms the file's (mtime, size) signature is
//!   compared and a change triggers the reload; the text cache still
//!   suppresses self-write echoes, and one reconcile runs at watcher start to
//!   close the boot-read race (upstream's `ready` reconcile).
//! - Storage operations are synchronous on the single thread, so the
//!   upstream promise queue (and its drain-on-dispose choreography) is
//!   unnecessary: a started operation completes before anything else runs,
//!   and the two-phase "queued after dispose" failure collapses into the one
//!   "is disposed" refusal. The cross-process writer lock blocks the thread
//!   under contention (bounded by its 2s deadline).
//! - The `INVARIANT` failure channel (rethrow-through-commit, reload-queue
//!   poisoning) is not ported: `dsh-invariants` does not exist in this
//!   workspace and dsh-cordis listeners are spawned futures whose failures
//!   cannot reach the emitter. The `./invariant` companion is likewise not
//!   ported.
//! - `launchEnvironmentOf` lives here (the Rust `dsh-launch-environment`
//!   crate is Cordis-free): the snapshot is read from the
//!   [`DSH_LAUNCH_ENVIRONMENT_KEY`] service slot, falling back to the live
//!   process environment per call.
//!
//! Diagnostic policy carried over exactly: a key name may appear in an error,
//! a value never does — parse failures report only a code with line/column,
//! because in this document the offending line is a secret.

use dsh_atomic_write::{WriteFileAtomicOptions, with_file_lock, write_file_atomic};
use dsh_cordis::{Context, Disposer, Effect, Plugin, validate_as};
use dsh_credentials::{
    CredentialInfo, CredentialProvider, CredentialRef, Credentials, ResolvedCredential,
    credential_ref, notify_updated,
};
use dsh_home_paths::{canonicalize_watch_path, resolve_dsh_home};
use dsh_launch_environment::{LaunchEnvironmentSnapshot, LaunchEnvironmentSource};
use futures::future::LocalBoxFuture;
use serde::{Deserialize, Serialize};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::io;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{Duration, SystemTime};

use futures::FutureExt;

/// Basename of the credentials document inside the harness home.
pub const CREDENTIALS_FILENAME: &str = ".credentials.yaml";

/// Service slot a launcher fills with this run's [`LaunchEnvironmentSnapshot`]
/// (as `Rc<LaunchEnvironmentSnapshot>`) before config entries mount.
pub const DSH_LAUNCH_ENVIRONMENT_KEY: &str = "launchEnvironment";

/// The launcher's snapshot, or the live inherited environment as the sole
/// layer when the host provided none.
pub fn launch_environment_of(ctx: &Context) -> Rc<LaunchEnvironmentSnapshot> {
    ctx.try_get_raw(DSH_LAUNCH_ENVIRONMENT_KEY, true)
        .and_then(|any| any.downcast::<LaunchEnvironmentSnapshot>().ok())
        .unwrap_or_else(|| Rc::new(LaunchEnvironmentSnapshot::from_process_env()))
}

/// Plugin config: file location and hot-reload behavior.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Config {
    /// Credentials document path; defaults to [`CREDENTIALS_FILENAME`] under
    /// the harness home.
    pub path: Option<String>,
    /// Harness home used when `path` is omitted; defaults to `$DSH_HOME` or
    /// `~/.dsh`.
    pub dsh_home: Option<String>,
    /// Watch the document and hot-publish external edits; defaults to true.
    pub watch: Option<bool>,
    /// Change-detection settle window in milliseconds; defaults to 100.
    pub debounce_ms: Option<u64>,
}

/// Fully resolved provider parameters; defaulting happens here, never inline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSpec {
    pub filename: PathBuf,
    pub watch: bool,
    pub debounce_ms: u64,
}

/// Resolve the runtime spec from plugin config: an explicit `path` wins,
/// otherwise the document lives at `<harness home>/.credentials.yaml`.
pub fn resolve_spec(config: &Config) -> ResolvedSpec {
    let filename = match &config.path {
        Some(path) => PathBuf::from(path),
        None => resolve_dsh_home(config.dsh_home.as_deref()).join(CREDENTIALS_FILENAME),
    };
    let filename = std::path::absolute(&filename).unwrap_or(filename);
    ResolvedSpec {
        filename,
        watch: config.watch.unwrap_or(true),
        debounce_ms: config.debounce_ms.unwrap_or(100),
    }
}

/// Permission bits outside the owner; a credentials document must have none.
#[cfg(unix)]
const GROUP_OTHER_BITS: u32 = 0o077;

/// Reject a credentials document other OS users can read, before its contents
/// are read at all. The provider creates and replaces the file at `0600`, but
/// a hand-written or externally generated one carries whatever umask produced
/// it, and silently serving secrets out of a world-readable file would make
/// the promised mode meaningless. Unix only: other platforms have no POSIX
/// mode to inspect, so the check is skipped rather than faked.
fn assert_owner_only(filename: &Path) -> anyhow::Result<()> {
    let metadata = match std::fs::metadata(filename) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            // Absence is fine, but the path hierarchy must still be valid: a
            // regular-file ancestor is a misconfiguration, not "no
            // credentials yet".
            canonicalize_watch_path(filename)?;
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let mode = metadata.mode();
        if mode & GROUP_OTHER_BITS != 0 {
            anyhow::bail!(
                "credentials-local: {} is readable beyond its owner (mode {:o}); \
                 run \"chmod 600 {}\" before starting again",
                filename.display(),
                mode & 0o777,
                filename.display(),
            );
        }
    }
    #[cfg(not(unix))]
    let _ = metadata;
    Ok(())
}

/// Whether a plain scalar would read back as a YAML non-string (number,
/// boolean, null); such values must be quoted when written and are type
/// errors when read.
fn is_non_string_scalar(value: &str) -> bool {
    matches!(
        value,
        "~" | "null" | "Null" | "NULL" | "true" | "True" | "TRUE" | "false" | "False" | "FALSE"
    ) || value.parse::<f64>().is_ok()
}

/// Whether a value can be written as a plain scalar and read back verbatim
/// within the document subset. Anything else is JSON-quoted.
fn is_safe_plain(value: &str) -> bool {
    if value.is_empty() || value != value.trim() {
        return false;
    }
    if value
        .chars()
        .any(|c| c == '"' || c == '#' || c == ':' || c.is_control())
    {
        return false;
    }
    let first = value.chars().next().expect("non-empty");
    if "-?[]{}&*!|>%@`'".contains(first) {
        return false;
    }
    !is_non_string_scalar(value)
}

/// Encode one value for a `KEY: value` line: plain when safe, otherwise a
/// JSON-style double-quoted scalar (also valid YAML).
fn encode_value(value: &str) -> String {
    if is_safe_plain(value) {
        value.to_string()
    } else {
        serde_json::to_string(value).expect("string serialization is infallible")
    }
}

fn is_comment_or_blank(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.is_empty() || trimmed.starts_with('#')
}

/// The entry key of a document line, `None` for comments and blanks. Only
/// called on text that already parsed, so entry lines are well-formed.
fn entry_key(line: &str) -> Option<&str> {
    if is_comment_or_blank(line) {
        return None;
    }
    line.split_once(':').map(|(key, _)| key)
}

/// Parse one credentials document into its entries. The document is a strict
/// mapping of credential reference to non-empty string: a non-mapping root, a
/// key that is not a POSIX identifier, a non-string value, and an empty
/// string are all rejected rather than skipped — this file holds nothing but
/// credentials, and a silently ignored entry would read as "the key I stored
/// has no effect". Duplicate keys are rejected. An empty document is an empty
/// store. Diagnostics carry the key name, a code, and a position — never the
/// value or the source line.
pub fn parse_credentials_document(
    text: &str,
    filename: &Path,
) -> anyhow::Result<HashMap<String, String>> {
    let file = filename.display();
    let mut entries: HashMap<String, String> = HashMap::new();
    for (index, line) in text.lines().enumerate() {
        let n = index + 1;
        if is_comment_or_blank(line) {
            continue;
        }
        if line.starts_with([' ', '\t']) {
            anyhow::bail!(
                "credentials-local: invalid document at {file}: \
                 unsupported indented line at line {n}, column 1"
            );
        }
        let Some((key, rest)) = line.split_once(':') else {
            anyhow::bail!(
                "credentials-local: {file} must be a mapping of credential reference to value"
            );
        };
        // The reference constraint is exactly what makes a stored key
        // addressable through the seam.
        credential_ref(key)?;
        if rest.is_empty() {
            anyhow::bail!("credentials-local: the value for \"{key}\" in {file} must be a string");
        }
        if !rest.starts_with(' ') {
            anyhow::bail!(
                "credentials-local: {file} must be a mapping of credential reference to value"
            );
        }
        let raw = rest.trim();
        let value = if raw.starts_with('"') {
            let column = line.len() - line.trim_start_matches(|c| c != '"').len() + 1;
            match serde_json::from_str::<String>(raw) {
                Ok(decoded) => decoded,
                // Never the parser's own message: it could quote the source.
                Err(_) => anyhow::bail!(
                    "credentials-local: invalid document at {file}: \
                     malformed quoted scalar at line {n}, column {column}"
                ),
            }
        } else {
            if raw.is_empty() {
                anyhow::bail!(
                    "credentials-local: the value for \"{key}\" in {file} must be a string"
                );
            }
            if raw.starts_with('\'') || raw.contains(" #") {
                anyhow::bail!(
                    "credentials-local: invalid document at {file}: \
                     unsupported scalar syntax at line {n}"
                );
            }
            if is_non_string_scalar(raw) {
                anyhow::bail!(
                    "credentials-local: the value for \"{key}\" in {file} must be a string"
                );
            }
            raw.to_string()
        };
        if value.is_empty() {
            anyhow::bail!(
                "credentials-local: the value for \"{key}\" in {file} is empty; \
                 remove the key instead"
            );
        }
        if entries.insert(key.to_string(), value).is_some() {
            anyhow::bail!(
                "credentials-local: invalid document at {file}: \
                 duplicate key \"{key}\" at line {n}"
            );
        }
    }
    Ok(entries)
}

/// Render the next document text with one reference set or deleted. Editing
/// lines rather than rebuilding keeps comments and the formatting of every
/// untouched entry; deletion also removes the contiguous comment/blank block
/// directly above the entry (its annotation). An absent document starts
/// fresh. Only called on text that parsed successfully.
fn render_document(text: Option<&str>, key: &str, value: Option<&str>) -> String {
    let mut lines: Vec<String> = match text {
        Some(text) => text.lines().map(String::from).collect(),
        None => Vec::new(),
    };
    let position = lines.iter().position(|line| entry_key(line) == Some(key));
    match (position, value) {
        (Some(index), Some(value)) => {
            lines[index] = format!("{key}: {}", encode_value(value));
        }
        (Some(index), None) => {
            let mut start = index;
            while start > 0 && is_comment_or_blank(&lines[start - 1]) {
                start -= 1;
            }
            lines.drain(start..=index);
        }
        (None, Some(value)) => lines.push(format!("{key}: {}", encode_value(value))),
        (None, None) => {}
    }
    if lines.is_empty() {
        String::new()
    } else {
        let mut rendered = lines.join("\n");
        rendered.push('\n');
        rendered
    }
}

/// File modification signature used by the polling change detector; `None`
/// while the file is absent or unreadable.
fn stat_sig(path: &Path) -> Option<(SystemTime, u64)> {
    let metadata = std::fs::metadata(path).ok()?;
    Some((metadata.modified().ok()?, metadata.len()))
}

/// File-backed credentials provider (`$DSH_HOME/.credentials.yaml`).
pub struct LocalCredentialProvider {
    ctx: Context,
    spec: ResolvedSpec,
    /// Raw text of the last read or persisted document; `None` while the
    /// file is absent. A reload whose content equals this cache is a no-op,
    /// which is also the self-write suppression.
    text: RefCell<Option<String>>,
    /// Parsed document snapshot; replaced wholesale on every reload.
    values: RefCell<HashMap<String, String>>,
    /// Set at dispose: refuse new writes and stop the change detector.
    closed: Cell<bool>,
}

impl LocalCredentialProvider {
    fn new(ctx: Context, spec: ResolvedSpec) -> LocalCredentialProvider {
        LocalCredentialProvider {
            ctx,
            spec,
            text: RefCell::new(None),
            values: RefCell::new(HashMap::new()),
            closed: Cell::new(false),
        }
    }

    /// The resolved file location and watch behavior.
    pub fn spec(&self) -> &ResolvedSpec {
        &self.spec
    }

    /// The inherited-environment value for a reference, or `None` when empty
    /// or unset.
    fn inherited(&self, r: &CredentialRef) -> Option<String> {
        let entry = launch_environment_of(&self.ctx)
            .get_from(r.as_str(), &[LaunchEnvironmentSource::Process])?;
        (!entry.value.is_empty()).then_some(entry.value)
    }

    /// The `.env` fallback for a reference — below the managed store, never
    /// above it. The invoking project ranks over the user's home file,
    /// matching the environment layering: the more specific location wins.
    fn dotenv_fallback(&self, r: &CredentialRef) -> Option<(String, &'static str)> {
        let entry = launch_environment_of(&self.ctx).get_from(
            r.as_str(),
            &[
                LaunchEnvironmentSource::ProjectEnv,
                LaunchEnvironmentSource::UserEnv,
            ],
        )?;
        if entry.value.is_empty() {
            return None;
        }
        let source = match entry.source {
            LaunchEnvironmentSource::ProjectEnv => "project-env",
            LaunchEnvironmentSource::UserEnv => "user-env",
            LaunchEnvironmentSource::Process => "env",
        };
        Some((entry.value, source))
    }

    /// Boot read: an absent file is an empty store; an invalid one fails the
    /// plugin's activation, because a credentials document that exists but
    /// cannot be trusted must never be treated as "no credentials stored".
    fn load_initial(&self) -> anyhow::Result<()> {
        assert_owner_only(&self.spec.filename)?;
        let text = match std::fs::read_to_string(&self.spec.filename) {
            Ok(text) => text,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error.into()),
        };
        *self.values.borrow_mut() = parse_credentials_document(&text, &self.spec.filename)?;
        *self.text.borrow_mut() = Some(text);
        Ok(())
    }

    /// Re-read after a change signal. Unchanged content (including this
    /// provider's own writes) is a no-op; an unreadable or invalid document
    /// keeps the last good snapshot and warns — a live hot-reload must never
    /// take the process down.
    fn refresh(&self) {
        if self.closed.get() {
            return;
        }
        if let Err(error) = self.reconcile_from_disk() {
            // The error never carries a credential value: parse diagnostics
            // are position-only by construction and io errors carry paths.
            tracing::warn!(
                "credentials-local: reload failed at {}; keeping the last good document: {error:#}",
                self.spec.filename.display(),
            );
        }
    }

    /// Compare the on-disk text against the cache and publish any difference
    /// into the seam. Absence publishes the empty store; an unreadable or
    /// invalid document errors, so each caller picks its policy — a reload
    /// warns and keeps the last good snapshot, a write fails loud rather than
    /// overwriting a document it could not understand.
    fn reconcile_from_disk(&self) -> anyhow::Result<()> {
        // Re-checked on every reload and before every write: an external
        // editor or a restored backup can loosen the mode after boot.
        assert_owner_only(&self.spec.filename)?;
        let text = match std::fs::read_to_string(&self.spec.filename) {
            Ok(text) => Some(text),
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        if text == *self.text.borrow() || self.closed.get() {
            return Ok(());
        }
        let next = match &text {
            Some(text) => parse_credentials_document(text, &self.spec.filename)?,
            None => HashMap::new(),
        };
        let changed = changed_refs(&self.values.borrow(), &next);
        *self.text.borrow_mut() = text;
        *self.values.borrow_mut() = next;
        // Publish only after the snapshot committed.
        for r in &changed {
            notify_updated(&self.ctx, r);
        }
        Ok(())
    }

    /// Reject a write the inherited environment would shadow into apparent
    /// no-effect. Only that layer can shadow a write: everything else this
    /// provider resolves ranks below the document being written.
    fn assert_unshadowed(&self, r: &CredentialRef, verb: &str) -> anyhow::Result<()> {
        if self.inherited(r).is_some() {
            anyhow::bail!(
                "credentials-local: \"{r}\" is supplied read-only by the launching environment, \
                 so {verb} would be shadowed; unset it in the shell you start dsh from instead"
            );
        }
        Ok(())
    }

    /// One line edit under the cross-process writer lock. The read-modify-
    /// write folds in any on-disk state this process has not observed yet —
    /// an external edit inside the settle window, a change the detector
    /// missed, another process's write — so the edit can never resurrect a
    /// stale document.
    fn write(&self, r: &CredentialRef, value: Option<&str>) -> anyhow::Result<()> {
        let verb = if value.is_some() { "set" } else { "unset" };
        if self.closed.get() {
            anyhow::bail!("credentials-local is disposed: cannot {verb} \"{r}\"");
        }
        self.assert_unshadowed(r, verb)?;
        // The writer lock's exclusive create needs the parent to exist; 0700
        // because the harness home holds user-private data.
        if let Some(parent) = self.spec.filename.parent() {
            let mut builder = std::fs::DirBuilder::new();
            builder.recursive(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt;
                builder.mode(0o700);
            }
            builder.create(parent)?;
        }
        // ponytail: the whole locked section is synchronous, so in-process
        // writers serialize by construction; cross-process contention blocks
        // the thread inside with_file_lock's bounded retry.
        with_file_lock(&self.spec.filename, || -> anyhow::Result<()> {
            self.reconcile_from_disk()?;
            let existing = self.values.borrow().contains_key(r.as_str());
            if value.is_none() && !existing {
                // Removing an absent reference is a no-op.
                return Ok(());
            }
            let next_text = render_document(self.text.borrow().as_deref(), r.as_str(), value);
            // 0600: a document holding secrets is never world-readable.
            write_file_atomic(
                &self.spec.filename,
                &next_text,
                WriteFileAtomicOptions {
                    mode: 0o600,
                    dir_mode: Some(0o700),
                },
            )?;
            *self.text.borrow_mut() = Some(next_text);
            match value {
                Some(value) => {
                    self.values
                        .borrow_mut()
                        .insert(r.as_str().to_string(), value.to_string());
                }
                None => {
                    self.values.borrow_mut().remove(r.as_str());
                }
            }
            // After the commit: a broken observer must never make the
            // durable write look failed.
            notify_updated(&self.ctx, r);
            Ok(())
        })??;
        Ok(())
    }

    /// Polling change detector standing in for the upstream file watcher.
    /// Runs one reconcile immediately (the `ready` reconcile closing the
    /// boot-read race), then re-checks the file signature each interval.
    async fn poll_loop(self: Rc<Self>) {
        let interval = Duration::from_millis(self.spec.debounce_ms.max(20));
        let mut last = stat_sig(&self.spec.filename);
        self.refresh();
        loop {
            tokio::time::sleep(interval).await;
            if self.closed.get() {
                break;
            }
            let now = stat_sig(&self.spec.filename);
            if now != last {
                last = now;
                self.refresh();
            }
        }
    }
}

/// Entries whose stored value changed between two snapshots; the parser has
/// already proven every key addressable.
fn changed_refs(
    prev: &HashMap<String, String>,
    next: &HashMap<String, String>,
) -> Vec<CredentialRef> {
    let keys: HashSet<&String> = prev.keys().chain(next.keys()).collect();
    let mut changed: Vec<CredentialRef> = keys
        .into_iter()
        .filter(|key| prev.get(*key) != next.get(*key))
        .map(|key| credential_ref(key).expect("parser-proven reference"))
        .collect();
    // Deterministic emission order for observers.
    changed.sort();
    changed
}

#[async_trait::async_trait(?Send)]
impl CredentialProvider for LocalCredentialProvider {
    async fn resolve(&self, r: &CredentialRef) -> anyhow::Result<Option<ResolvedCredential>> {
        if let Some(value) = self.inherited(r) {
            return Ok(Some(ResolvedCredential {
                value,
                source: "env".into(),
            }));
        }
        if let Some(value) = self.values.borrow().get(r.as_str()) {
            return Ok(Some(ResolvedCredential {
                value: value.clone(),
                source: "file".into(),
            }));
        }
        if let Some((value, source)) = self.dotenv_fallback(r) {
            return Ok(Some(ResolvedCredential {
                value,
                source: source.into(),
            }));
        }
        Ok(None)
    }

    async fn describe(&self, r: &CredentialRef) -> anyhow::Result<CredentialInfo> {
        // Only the inherited environment is unwritable: it is the one layer
        // this process cannot edit. A `.env` value is writable in the sense
        // that matters — storing a key replaces it as the effective one.
        if self.inherited(r).is_some() {
            return Ok(CredentialInfo {
                configured: true,
                source: Some("env".into()),
                writable: false,
            });
        }
        if self.values.borrow().contains_key(r.as_str()) {
            return Ok(CredentialInfo {
                configured: true,
                source: Some("file".into()),
                writable: true,
            });
        }
        if let Some((_, source)) = self.dotenv_fallback(r) {
            return Ok(CredentialInfo {
                configured: true,
                source: Some(source.to_string()),
                writable: true,
            });
        }
        Ok(CredentialInfo {
            configured: false,
            source: None,
            writable: true,
        })
    }

    async fn set(&self, r: &CredentialRef, value: &str) -> anyhow::Result<()> {
        if value.is_empty() {
            anyhow::bail!(
                "credentials-local: an empty value cannot be stored for \"{r}\"; use unset"
            );
        }
        self.write(r, Some(value))
    }

    async fn unset(&self, r: &CredentialRef) -> anyhow::Result<()> {
        self.write(r, None)
    }
}

/// The `credentials-local` plugin: mounts a [`LocalCredentialProvider`] as
/// the `ctx.credentials` service.
pub struct LocalCredentials;

impl Plugin for LocalCredentials {
    fn name(&self) -> Option<String> {
        Some("credentials-local".into())
    }

    fn validate_config(&self, config: serde_json::Value) -> dsh_cordis::Result<serde_json::Value> {
        validate_as::<Config>(config)
    }

    fn apply(
        &self,
        ctx: Context,
        config: serde_json::Value,
    ) -> LocalBoxFuture<'static, anyhow::Result<()>> {
        async move {
            let config: Config = serde_json::from_value(config)?;
            let spec = resolve_spec(&config);
            let provider = Rc::new(LocalCredentialProvider::new(ctx.clone(), spec));
            provider.load_initial()?;

            // Disposal refuses new writes and quiesces the change detector;
            // synchronous operations cannot be in flight across it.
            let closer = provider.clone();
            ctx.effect_labeled("credentials-local.close", move |_| {
                Ok(Effect::One(Disposer::sync(move || closer.closed.set(true))))
            })?;

            ctx.provide_service(Rc::new(Credentials(provider.clone())))?;

            if provider.spec.watch {
                tokio::task::spawn_local(provider.poll_loop());
            }
            Ok(())
        }
        .boxed_local()
    }
}
