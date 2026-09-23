//! Rust port of `packages/settings/settings-file`
//! (`@deepseek-ai/dsh-settings-file`): the file-backed settings provider. One
//! YAML or JSON document under the user's harness home carries every
//! namespace section; external edits hot-publish through the seam, and every
//! write re-reads the document under the cross-process writer lock before
//! rendering the next text.
//!
//! Divergences from the TS original:
//! - Comment preservation is NOT ported: upstream patches the document
//!   through the `yaml` package's comment-preserving `Document` tree
//!   (`setIn`/`deleteIn` leaf diffs); `serde_yaml` has no comment model, so a
//!   write re-serializes the parsed document — unregistered sections survive,
//!   comments do not. The `patchNode` diff machinery is therefore gone.
//! - chokidar is replaced by the `notify` crate watching the document's
//!   parent directory; events hop from notify's thread onto the local set
//!   through a channel, and the debounce is a settle-sleep before the queued
//!   refresh (upstream `awaitWriteFinish`).
//! - File IO is blocking (`std::fs`, `dsh_atomic_write`); the exclusive
//!   operation chain still serializes writes and reloads in queue order.
//! - The plugin config is a serde struct validated through
//!   `dsh_cordis::validate_as` instead of a schemastery schema.
//! - The `./invariant` companion is not ported (no `dsh-invariants` in this
//!   workspace; upstream's installer is empty anyway).

use anyhow::{anyhow, bail};
use dsh_atomic_write::{WriteFileAtomicOptions, with_file_lock, write_file_atomic};
use dsh_cordis::{Context, Disposer, Effect, Plugin, validate_as};
use dsh_home_paths::{canonicalize_watch_path, resolve_dsh_home};
use dsh_settings::{SettingsBackend, SettingsNamespace, SettingsService};
use futures::FutureExt;
use futures::future::{LocalBoxFuture, Shared};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::cell::{Cell, RefCell};
use std::io;
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};
use std::time::Duration;

/// Plugin config: file location and hot-reload behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Config {
    /// Settings document path; defaults to `settings.yaml` under the harness
    /// home.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// Harness home used when `path` is omitted; defaults to `$DSH_HOME` or
    /// `~/.dsh`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dsh_home: Option<String>,
    /// Watch the document and hot-publish external edits.
    #[serde(default = "default_watch")]
    pub watch: bool,
    /// Watcher write-settle window in milliseconds.
    #[serde(default = "default_debounce_ms")]
    pub debounce_ms: u64,
}

fn default_watch() -> bool {
    true
}

fn default_debounce_ms() -> u64 {
    100
}

impl Default for Config {
    fn default() -> Config {
        Config {
            path: None,
            dsh_home: None,
            watch: default_watch(),
            debounce_ms: default_debounce_ms(),
        }
    }
}

/// Document format derived from the configured file extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsFormat {
    Yaml,
    Json,
}

/// Fully resolved provider parameters; defaulting happens here, never inline.
#[derive(Debug, Clone)]
pub struct ResolvedSpec {
    pub filename: PathBuf,
    pub format: SettingsFormat,
    pub watch: bool,
    pub debounce_ms: u64,
}

/// Resolve the runtime spec from plugin config: an explicit `path` wins,
/// otherwise the document lives at `<harness home>/settings.yaml`.
pub fn resolve_spec(config: &Config) -> anyhow::Result<ResolvedSpec> {
    let filename = match &config.path {
        Some(path) => PathBuf::from(path),
        None => resolve_dsh_home(config.dsh_home.as_deref()).join("settings.yaml"),
    };
    let filename = std::path::absolute(&filename)?;
    let extension = filename.extension().and_then(|e| e.to_str()).unwrap_or("");
    let format = match extension {
        "yaml" | "yml" => SettingsFormat::Yaml,
        "json" => SettingsFormat::Json,
        _ => bail!(
            "settings-file: extension \".{extension}\" is not supported (use .yaml, .yml, or .json)"
        ),
    };
    Ok(ResolvedSpec {
        filename,
        format,
        watch: config.watch,
        debounce_ms: config.debounce_ms,
    })
}

fn is_not_found(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::NotFound
}

/// Create `dir` (recursively) with owner-only permissions on fresh segments:
/// the harness home holds user-private documents.
fn create_private_dir(dir: &Path) -> io::Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(dir)
}

type Tail = Shared<LocalBoxFuture<'static, ()>>;

fn ready_tail() -> Tail {
    futures::future::ready(()).boxed_local().shared()
}

/// File storage backend for the settings seam (`settings.yaml`/`.json`).
pub struct FileBackend {
    weak_self: Weak<FileBackend>,
    spec: ResolvedSpec,
    /// Raw text of the last successfully parsed or persisted document;
    /// `None` while the file is absent. Reads whose content equals this
    /// cache are no-ops, which is also the self-write suppression.
    text: RefCell<Option<String>>,
    /// Single exclusive operation chain: watcher reloads and document writes
    /// run one at a time in queue order (settled tail), so a write can never
    /// render from text a concurrent reload is busy replacing.
    operations: RefCell<Tail>,
    /// Set at dispose: refuse new reloads and let in-flight work no-op.
    closed: Cell<bool>,
    service: RefCell<Weak<SettingsService>>,
}

impl FileBackend {
    pub fn new(config: &Config) -> anyhow::Result<Rc<FileBackend>> {
        let spec = resolve_spec(config)?;
        Ok(Rc::new_cyclic(|weak| FileBackend {
            weak_self: weak.clone(),
            spec,
            text: RefCell::new(None),
            operations: RefCell::new(ready_tail()),
            closed: Cell::new(false),
            service: RefCell::new(Weak::new()),
        }))
    }

    /// The fully resolved file location and watch behavior.
    pub fn spec(&self) -> &ResolvedSpec {
        &self.spec
    }

    fn rc(&self) -> Rc<FileBackend> {
        self.weak_self.upgrade().expect("backend alive")
    }

    /// Queue one exclusive document operation behind every earlier one. The
    /// operation itself is synchronous file work; the chain only orders it
    /// against concurrent writes and reloads.
    fn enqueue<T: 'static>(
        &self,
        operation: impl FnOnce(Rc<FileBackend>) -> anyhow::Result<T> + 'static,
    ) -> LocalBoxFuture<'static, anyhow::Result<T>> {
        let previous = self.operations.borrow().clone();
        let (result_tx, result_rx) = futures::channel::oneshot::channel();
        let (done_tx, done_rx) = futures::channel::oneshot::channel::<()>();
        let tail: Tail = async move {
            let _ = done_rx.await;
        }
        .boxed_local()
        .shared();
        *self.operations.borrow_mut() = tail;
        let this = self.rc();
        tokio::task::spawn_local(async move {
            previous.await;
            let outcome = operation(this);
            let _ = result_tx.send(outcome);
            let _ = done_tx.send(());
        });
        Box::pin(async move {
            result_rx
                .await
                .map_err(|_| anyhow!("settings-file: operation task vanished"))?
        })
    }

    /// Queue a reload after a watcher event; failures are contained inside
    /// [`FileBackend::refresh`], so the chain stays alive.
    fn queue_refresh(&self) {
        let refresh = self.enqueue(|this| {
            this.refresh();
            Ok(())
        });
        tokio::task::spawn_local(async move {
            let _ = refresh.await;
        });
    }

    /// Re-read the document after a watcher event. Unchanged content
    /// (including this provider's own writes) is a no-op; an unreadable or
    /// unparsable document keeps the last good sections and warns — a live
    /// hot-reload must never take the process down.
    fn refresh(&self) {
        if self.closed.get() {
            return;
        }
        if let Err(error) = self.reconcile_from_disk() {
            tracing::warn!(
                "settings-file: reload failed at {}; keeping the last good document: {error:#}",
                self.spec.filename.display()
            );
        }
    }

    /// Compare the on-disk text against the cache and publish any difference
    /// into the seam. Absence publishes the empty document; an unreadable or
    /// unparsable file errors, so each caller picks its policy — a reload
    /// warns and keeps the last good document, a write fails loud.
    fn reconcile_from_disk(&self) -> anyhow::Result<()> {
        let text = match std::fs::read_to_string(&self.spec.filename) {
            Ok(text) => Some(text),
            Err(error) if is_not_found(&error) => None,
            Err(error) => return Err(error.into()),
        };
        if text == *self.text.borrow() || self.closed.get() {
            return Ok(());
        }
        match text {
            None => {
                *self.text.borrow_mut() = None;
                self.publish_doc(Map::new());
            }
            Some(text) => {
                let doc = self.parse(&text)?;
                *self.text.borrow_mut() = Some(text);
                self.publish_doc(doc);
            }
        }
        Ok(())
    }

    fn publish_doc(&self, doc: Map<String, Value>) {
        if let Some(service) = self.service.borrow().upgrade() {
            service.publish(doc);
        }
    }

    /// Parse one document text into raw sections, failing on a non-map root.
    fn parse(&self, text: &str) -> anyhow::Result<Map<String, Value>> {
        let root: Value = if text.trim().is_empty() {
            Value::Null
        } else {
            match self.spec.format {
                SettingsFormat::Yaml => serde_yaml::from_str(text).map_err(|error| {
                    anyhow!(
                        "settings-file: invalid document at {}: {error}",
                        self.spec.filename.display()
                    )
                })?,
                SettingsFormat::Json => serde_json::from_str(text).map_err(|error| {
                    anyhow!(
                        "settings-file: invalid document at {}: {error}",
                        self.spec.filename.display()
                    )
                })?,
            }
        };
        match root {
            // An empty or comment-only document holds no sections yet.
            Value::Null => Ok(Map::new()),
            Value::Object(map) => Ok(map),
            _ => bail!(
                "settings-file: {} must be a map of namespace sections",
                self.spec.filename.display()
            ),
        }
    }

    /// Render the next document text with one namespace section replaced,
    /// keeping every other (registered or not) section from the cached text.
    fn render(
        &self,
        ns: &SettingsNamespace,
        section: &Map<String, Value>,
    ) -> anyhow::Result<String> {
        let mut root = match &*self.text.borrow() {
            // The cache only ever holds content that parsed successfully.
            Some(text) => self.parse(text)?,
            None => Map::new(),
        };
        root.insert(ns.to_string(), Value::Object(section.clone()));
        match self.spec.format {
            SettingsFormat::Yaml => Ok(serde_yaml::to_string(&Value::Object(root))?),
            SettingsFormat::Json => Ok(format!(
                "{}\n",
                serde_json::to_string_pretty(&Value::Object(root))?
            )),
        }
    }

    fn persist_section(
        &self,
        ns: &SettingsNamespace,
        section: &Map<String, Value>,
    ) -> anyhow::Result<()> {
        // The writer lock's exclusive create needs the parent to exist before
        // write_file_atomic gets its own chance to create it.
        create_private_dir(self.spec.filename.parent().unwrap_or(Path::new(".")))?;
        with_file_lock(&self.spec.filename, || -> anyhow::Result<()> {
            // Read-modify-write: fold in any on-disk state this process has
            // not observed yet — an external edit still inside the watcher
            // debounce window, or another process's write — so the render
            // below can never resurrect a stale document. An unparsable
            // on-disk document fails the write loud instead of silently
            // overwriting a user's manual edit.
            self.reconcile_from_disk()?;
            let output = self.render(ns, section)?;
            // 0600: a document that may hold personal values is never
            // world-readable.
            write_file_atomic(
                &self.spec.filename,
                &output,
                WriteFileAtomicOptions {
                    mode: 0o600,
                    dir_mode: Some(0o700),
                },
            )?;
            *self.text.borrow_mut() = Some(output);
            Ok(())
        })??;
        Ok(())
    }

    fn load_sync(&self) -> anyhow::Result<Map<String, Value>> {
        let text = match std::fs::read_to_string(&self.spec.filename) {
            Ok(text) => text,
            Err(error) if is_not_found(&error) => {
                *self.text.borrow_mut() = None;
                return Ok(Map::new());
            }
            Err(error) => return Err(error.into()),
        };
        let doc = self.parse(&text)?;
        *self.text.borrow_mut() = Some(text);
        Ok(doc)
    }
}

impl SettingsBackend for FileBackend {
    /// The local document is always writable through the seam.
    fn writable(&self) -> bool {
        true
    }

    fn document_path(&self) -> Option<PathBuf> {
        Some(self.spec.filename.clone())
    }

    /// Materialize an absent owner-only document, then return its resolved
    /// path; an existing document is left untouched.
    fn prepare_document(&self) -> LocalBoxFuture<'static, anyhow::Result<Option<PathBuf>>> {
        self.enqueue(|this| {
            create_private_dir(this.spec.filename.parent().unwrap_or(Path::new(".")))?;
            with_file_lock(&this.spec.filename, || -> anyhow::Result<()> {
                let mut open = std::fs::OpenOptions::new();
                open.write(true).create_new(true);
                #[cfg(unix)]
                {
                    use std::os::unix::fs::OpenOptionsExt;
                    open.mode(0o600);
                }
                match open.open(&this.spec.filename) {
                    Ok(_) => {
                        *this.text.borrow_mut() = Some(String::new());
                        if !this.closed.get() {
                            this.publish_doc(Map::new());
                        }
                        Ok(())
                    }
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => Ok(()),
                    Err(error) => Err(error.into()),
                }
            })??;
            Ok(Some(this.spec.filename.clone()))
        })
    }

    fn attach(&self, service: Weak<SettingsService>) {
        *self.service.borrow_mut() = service;
    }

    /// Boot read: an existing-but-invalid document fails loud here, never
    /// silently ignored or overwritten.
    fn load(&self) -> LocalBoxFuture<'static, anyhow::Result<Map<String, Value>>> {
        let result = self.load_sync();
        Box::pin(async move { result })
    }

    fn persist(
        &self,
        ns: &SettingsNamespace,
        section: &Map<String, Value>,
    ) -> LocalBoxFuture<'static, anyhow::Result<()>> {
        // One document backs every namespace, so writes from different
        // namespace queues serialize with each other and with watcher reloads
        // on the one operation chain.
        let ns = ns.clone();
        let section = section.clone();
        self.enqueue(move |this| this.persist_section(&ns, &section))
    }
}

/// Watch the document's parent directory and queue a debounced reconcile for
/// every event that names the document. Returns the watcher handle; dropping
/// it stops events and ends the forwarding loop.
fn start_watcher(backend: &Rc<FileBackend>) -> anyhow::Result<notify::RecommendedWatcher> {
    use notify::Watcher as _;
    let canonical = canonicalize_watch_path(&backend.spec.filename)?;
    let parent = canonical
        .parent()
        .ok_or_else(|| anyhow!("settings-file: document path has no parent directory"))?
        .to_path_buf();
    let file_name = canonical.file_name().map(|name| name.to_owned());
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<()>();
    let watched = backend.spec.filename.display().to_string();
    // The event handler runs on notify's own thread; it only forwards a ping
    // onto the local set, where all state lives.
    let mut watcher =
        notify::recommended_watcher(move |result: notify::Result<notify::Event>| match result {
            Ok(event) => {
                let relevant = event.paths.is_empty()
                    || event
                        .paths
                        .iter()
                        .any(|path| path.file_name() == file_name.as_deref());
                if relevant {
                    let _ = tx.send(());
                }
            }
            Err(error) => {
                tracing::warn!("settings-file: watcher error on {watched}: {error}");
            }
        })?;
    watcher.watch(&parent, notify::RecursiveMode::NonRecursive)?;
    let debounce = Duration::from_millis(backend.spec.debounce_ms);
    let loop_backend = backend.clone();
    tokio::task::spawn_local(async move {
        while rx.recv().await.is_some() {
            // Write-settle window: coalesce an event burst into one refresh.
            // ponytail: sleep-then-drain, not a resetting timer — a refresh
            // on still-moving content is a no-op reconcile anyway.
            tokio::time::sleep(debounce).await;
            while rx.try_recv().is_ok() {}
            if loop_backend.closed.get() {
                break;
            }
            loop_backend.queue_refresh();
        }
    });
    // The boot load raced the watcher's setup: a change written between that
    // read and the watch becoming active never fires an event. One reconcile
    // now closes the gap.
    backend.queue_refresh();
    Ok(watcher)
}

/// File-backed settings provider plugin (`settings.yaml`/`.json`).
pub struct FileSettingsProvider;

impl Plugin for FileSettingsProvider {
    fn name(&self) -> Option<String> {
        Some("settings-file".into())
    }

    fn validate_config(&self, config: Value) -> dsh_cordis::Result<Value> {
        validate_as::<Config>(config)
    }

    fn apply(&self, ctx: Context, config: Value) -> LocalBoxFuture<'static, anyhow::Result<()>> {
        Box::pin(async move {
            let config: Config = serde_json::from_value(config)?;
            let backend = FileBackend::new(&config)?;
            dsh_settings::mount(&ctx, backend.clone()).await?;
            let watcher = if backend.spec.watch {
                Some(start_watcher(&backend)?)
            } else {
                None
            };
            let teardown = backend.clone();
            ctx.effect_labeled("settings-file:quiesce", move |_| {
                Ok(Effect::One(Disposer::asynchronous(move || async move {
                    // Quiesce the operation chain (and stop the watcher) even
                    // when no watcher is configured; this runs before the
                    // seam's own write drain and un-provide.
                    teardown.closed.set(true);
                    drop(watcher);
                    let tail = teardown.operations.borrow().clone();
                    tail.await;
                })))
            })?;
            Ok(())
        })
    }
}
