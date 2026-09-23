//! Rust port of `packages/settings/settings` (`@deepseek-ai/dsh-settings`):
//! the user-settings capability seam. A provider stores one raw document of
//! per-namespace sections; plugins register a namespace schema and read the
//! resolved value, which layers schema defaults, the registrant's composition
//! `base`, and the user document section, in that order.
//!
//! Divergences from the TS original:
//! - Schemastery is replaced by the crate-local [`Schema`] type (defaults,
//!   structural validation, `role("secret")` metadata, serializable
//!   description); `describe()` serializes that node tree instead of the
//!   schemastery `{uid, refs}` envelope.
//! - The abstract `SettingsProvider` base class becomes composition: a
//!   provider implements [`SettingsBackend`] (load/persist/writable plus the
//!   document-path metadata) and calls [`mount`], which owns registration,
//!   resolution, write serialization, publication, and teardown draining —
//!   the base-class half of the upstream contract.
//! - Values are `serde_json::Value`; `deepFreeze` has no counterpart (handed
//!   out values are detached clones), and the JSON-compatibility walk
//!   (`cloneJsonShaped`) is unnecessary — `Value` cannot hold functions,
//!   dates, cycles, or non-finite numbers, so those rejections are
//!   statically impossible.
//! - Upstream binds `this.ctx` to the accessing context through the cordis
//!   proxy; here [`SettingsService::register`] takes the registrant's
//!   [`Context`] explicitly so the registration effect rides the right fiber.
//! - Event listener failure containment (and the `INVARIANT` rethrow) is not
//!   ported: cordis-rs notification listeners have no error channel, and the
//!   `dsh-invariants` registry does not exist in this workspace. Watcher
//!   containment (the seam's own callbacks) is ported and tested. The
//!   `./invariant` companion module is likewise not ported.
//! - `settings/updated` and `settings/document-updated` are the typed events
//!   [`SettingsUpdated`] and [`SettingsDocumentUpdated`], dispatched through
//!   `ctx.emit` (fire-and-forget on the local set).

mod redact;
mod schema;

pub use redact::{RedactedSecret, RedactedValue, redact_secrets};
pub use schema::{Schema, SchemaKind};

use anyhow::bail;
use dsh_brand::Branded;
use dsh_cordis::{Context, Disposer, Effect, Event, Fiber, FiberState, Inject, plugin_fn};
use futures::FutureExt;
use futures::future::{LocalBoxFuture, Shared};
use serde_json::{Map, Value};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::{Rc, Weak};

/// Marker for [`SettingsNamespace`].
pub enum SettingsNamespaceMark {}

/// Nominal id of one registered settings namespace.
pub type SettingsNamespace = Branded<SettingsNamespaceMark>;

/// Brand a raw string as a [`SettingsNamespace`]. The name must be lowercase
/// kebab-case (`^[a-z][a-z0-9-]*$`), like plugin short names.
pub fn settings_namespace(value: &str) -> anyhow::Result<SettingsNamespace> {
    let mut chars = value.chars();
    let head_ok = chars.next().is_some_and(|c| c.is_ascii_lowercase());
    let rest_ok = chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
    if !head_ok || !rest_ok {
        bail!("settings namespace \"{value}\" must match /^[a-z][a-z0-9-]*$/");
    }
    Ok(SettingsNamespace::new(value))
}

/// Origin of one committed settings change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsUpdateSource {
    /// The change entered through an in-process write (`update`/`replace`/`mutate`).
    Update,
    /// The change was published by the provider (external storage edit).
    Provider,
}

/// When a namespace's changes take effect for its owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SettingsApplies {
    /// The owner reacts to committed changes immediately.
    #[default]
    Live,
    /// The owner reads the value at startup only.
    Restart,
}

/// Committed change to one registered namespace's resolved value. Emitted
/// after the provider persisted (`Update`) or published (`Provider`) the
/// change; never emitted when the resolved value is deep-equal.
pub struct SettingsUpdated;
impl Event for SettingsUpdated {
    const NAME: &'static str = "settings/updated";
    type Args = (SettingsNamespace, Value, Value, SettingsUpdateSource);
    type Ret = ();
}

/// One registered namespace's RAW user section changed, whether or not the
/// resolved value did. Configuration surfaces listen here to learn that a
/// field moved between inherited and overridden and that a held revision is
/// stale.
pub struct SettingsDocumentUpdated;
impl Event for SettingsDocumentUpdated {
    const NAME: &'static str = "settings/document-updated";
    type Args = (SettingsNamespace, u64);
    type Ret = ();
}

/// Deep equality over JSON data — the seam's single change-detection
/// predicate. `serde_json::Value` equality is already structural; the export
/// exists so companions check exactly the implementation's relation.
pub fn deep_equal_json(a: &Value, b: &Value) -> bool {
    a == b
}

/// A write refused because the namespace moved since the caller read it. The
/// serialized write queue orders writes; it cannot tell a fresh writer from
/// one holding a stale snapshot, which is what this reports.
#[derive(Debug, thiserror::Error)]
#[error(
    "settings namespace \"{ns}\" changed since it was read (expected revision {expected}, now {actual})"
)]
pub struct SettingsConflictError {
    /// The namespace whose write was refused.
    pub ns: SettingsNamespace,
    /// The revision the write expected.
    pub expected: u64,
    /// The revision the namespace actually stands at.
    pub actual: u64,
}

impl SettingsConflictError {
    /// Stable machine code for wire layers mapping this to their own taxonomy.
    pub const CODE: &'static str = "SETTINGS_CONFLICT";
}

/// One path-addressed edit to a namespace's user section. Path mutation
/// exists for a caller holding an incomplete (redacted) view: it can name the
/// field it means without restating the section, so it cannot delete secrets
/// the wire never returned.
#[derive(Debug, Clone)]
pub enum SettingsPathOp {
    /// Set the value at `path`, creating intermediate objects as needed. The
    /// empty path replaces the section itself (the value must be an object).
    Set { path: Vec<String>, value: Value },
    /// Remove the value at `path`; unsetting through an absent path is a
    /// no-op. The empty path clears the section.
    Unset { path: Vec<String> },
}

enum PathAction<'a> {
    Set(&'a Value),
    Unset,
}

/// Apply one path op to a detached section, returning the next section.
fn apply_path_op(
    section: Map<String, Value>,
    op: &SettingsPathOp,
) -> anyhow::Result<Map<String, Value>> {
    let (path, action) = match op {
        SettingsPathOp::Set { path, value } => (path.as_slice(), PathAction::Set(value)),
        SettingsPathOp::Unset { path } => (path.as_slice(), PathAction::Unset),
    };
    apply_path(section, path, action)
}

fn apply_path(
    mut section: Map<String, Value>,
    path: &[String],
    action: PathAction<'_>,
) -> anyhow::Result<Map<String, Value>> {
    let Some((head, rest)) = path.split_first() else {
        return match action {
            PathAction::Unset => Ok(Map::new()),
            PathAction::Set(Value::Object(map)) => Ok(map.clone()),
            PathAction::Set(_) => {
                bail!("settings mutate: setting the section root requires a plain object")
            }
        };
    };
    if rest.is_empty() {
        match action {
            PathAction::Set(value) => {
                section.insert(head.clone(), value.clone());
            }
            PathAction::Unset => {
                section.remove(head);
            }
        }
        return Ok(section);
    }
    let child = match section.get(head) {
        Some(Value::Object(map)) => map.clone(),
        _ => match action {
            // Unsetting through an absent path is already satisfied; setting
            // through one creates the intermediate objects it needs.
            PathAction::Unset => return Ok(section),
            PathAction::Set(_) => Map::new(),
        },
    };
    let replaced = apply_path(child, rest, action)?;
    section.insert(head.clone(), Value::Object(replaced));
    Ok(section)
}

/// Layer `over` onto `under`: objects merge recursively, every other value
/// (arrays included) replaces the lower layer wholesale.
fn merge_layers(under: Option<&Value>, over: Option<&Value>) -> Option<Value> {
    match (under, over) {
        (under, None) => under.cloned(),
        (Some(Value::Object(under)), Some(Value::Object(over))) => {
            let mut merged = under.clone();
            for (key, value) in over {
                let next = match merged.get(key) {
                    Some(existing) => merge_layers(Some(existing), Some(value))
                        .expect("merging two present values yields a value"),
                    None => value.clone(),
                };
                merged.insert(key.clone(), next);
            }
            Some(Value::Object(merged))
        }
        (_, Some(over)) => Some(over.clone()),
    }
}

/// Owner-supplied cross-field check run on the resolved (schema-valid) value.
pub type ValidateFn = Rc<dyn Fn(&Value) -> anyhow::Result<()>>;

/// Resolve one namespace value: schema defaults, then `base`, then the user
/// layer, then the owner's cross-field check.
fn resolve_value(
    schema: &Schema,
    base: Option<&Value>,
    section: Option<&Map<String, Value>>,
    validate: Option<&ValidateFn>,
) -> anyhow::Result<Value> {
    let section_value = section.map(|map| Value::Object(map.clone()));
    let merged = merge_layers(base, section_value.as_ref());
    let value = schema
        .resolve(merged.as_ref())
        .map_err(anyhow::Error::msg)?
        .unwrap_or(Value::Null);
    if let Some(validate) = validate {
        validate(&value)?;
    }
    Ok(value)
}

/// Registration options beyond the namespace schema.
#[derive(Default)]
pub struct SettingsRegisterOptions {
    /// Composition-layer values resolved below the user layer.
    pub base: Option<Value>,
    /// Owner's effect timing, surfaced to configuration UIs.
    pub applies: SettingsApplies,
    /// Cross-field check the schema cannot express. A failing check refuses
    /// the write that produced the value; once registered, a stored section
    /// that fails it keeps the last good value (like a schema failure), while
    /// at registration a failing stored section refuses the registration.
    pub validate: Option<ValidateFn>,
}

/// One registered namespace as surfaced to configuration UIs.
#[derive(Debug, Clone)]
pub struct SettingsDescriptor {
    /// The registered namespace.
    pub ns: SettingsNamespace,
    /// Serialized schema description ([`Schema::to_json`]).
    pub schema: Value,
    /// Current resolved value.
    pub value: Value,
    /// Monotonic revision of the raw user section this descriptor was read
    /// at; send it back as `expected_revision` on a write to refuse a stale
    /// one.
    pub revision: u64,
    /// Registrant's composition `base` layer (detached), when declared.
    pub base: Option<Value>,
    /// Raw user section from the stored document (detached), when present and
    /// well-formed; a field's presence here marks it user-overridden.
    pub user: Option<Value>,
    /// Owner's declared effect timing.
    pub applies: SettingsApplies,
    /// Schema-declared secret positions; present only under redaction.
    pub secrets: Option<Vec<RedactedSecret>>,
}

/// Options for [`SettingsService::describe`].
#[derive(Default, Clone, Copy)]
pub struct SettingsDescribeOptions {
    /// Strip secret fields from `value`/`base`/`user` and enumerate them in
    /// each descriptor's `secrets`. Every wire surface MUST pass this; the
    /// verbatim default exists for same-process configuration UIs only.
    pub redact_secrets: bool,
}

type Tail = Shared<LocalBoxFuture<'static, ()>>;

fn ready_tail() -> Tail {
    futures::future::ready(()).boxed_local().shared()
}

type WatchCallback = Rc<dyn Fn(Value, Value) -> LocalBoxFuture<'static, anyhow::Result<()>>>;

/// One registered watcher and its serialized invocation chain.
struct Watcher {
    callback: WatchCallback,
    /// Settled tail: invocations of this callback run one at a time, in
    /// commit order.
    tail: RefCell<Tail>,
    /// Cleared by the disposer: a queued invocation checks this before
    /// starting.
    active: Cell<bool>,
}

/// One live namespace registration owned by a registrant fiber.
struct Registration {
    ns: SettingsNamespace,
    schema: Schema,
    base: Option<Value>,
    applies: SettingsApplies,
    validate: Option<ValidateFn>,
    resolved: RefCell<Value>,
    /// Monotonic counter over this namespace's RAW user section — bumped by
    /// any stored change, including one whose resolved value is unchanged.
    revision: Cell<u64>,
    watchers: RefCell<Vec<Rc<Watcher>>>,
}

/// Storage half of a settings provider. Implementations own raw-document
/// storage; [`mount`] wires them into a [`SettingsService`], which owns
/// everything else (registration, resolution, serialization, publication).
pub trait SettingsBackend: 'static {
    /// Whether in-process writes may persist through this provider.
    fn writable(&self) -> bool;

    /// Absolute path of the user-editable document when storage is one local
    /// file; `None` for non-file storage.
    fn document_path(&self) -> Option<PathBuf> {
        None
    }

    /// Prepare the user-editable document for a native editor (file backends
    /// may materialize an absent document first).
    fn prepare_document(&self) -> LocalBoxFuture<'static, anyhow::Result<Option<PathBuf>>> {
        let path = self.document_path();
        Box::pin(async move { Ok(path) })
    }

    /// Receive the mounted service handle so external storage changes can be
    /// pushed through [`SettingsService::publish`].
    fn attach(&self, _service: Weak<SettingsService>) {}

    /// Read the provider's current raw document (namespace to raw section).
    fn load(&self) -> LocalBoxFuture<'static, anyhow::Result<Map<String, Value>>>;

    /// Durably store one namespace's complete merged user section.
    fn persist(
        &self,
        ns: &SettingsNamespace,
        section: &Map<String, Value>,
    ) -> LocalBoxFuture<'static, anyhow::Result<()>>;
}

enum WriteInput {
    Merge(Map<String, Value>),
    Replace(Map<String, Value>),
    Mutate(Vec<SettingsPathOp>),
}

/// The `ctx.settings` service: namespace registration, layered resolution,
/// validation, serialized writes, change detection, and the commit events.
pub struct SettingsService {
    ctx: Context,
    backend: Rc<dyn SettingsBackend>,
    weak_self: Weak<SettingsService>,
    registrations: RefCell<Vec<Rc<Registration>>>,
    /// Latest published raw document; empty until the provider's first
    /// publish.
    document: RefCell<Map<String, Value>>,
    /// Per-namespace settled write tails: a failed write never poisons the
    /// queue for later callers.
    write_queues: RefCell<HashMap<String, Tail>>,
    /// In-flight watcher invocation segments, drained by the dispose
    /// teardown.
    pending_tails: RefCell<HashMap<u64, Tail>>,
    tail_counter: Cell<u64>,
    /// Set at service dispose: refuse new writes while queued ones drain.
    stopped: Cell<bool>,
}

impl dsh_cordis::Service for SettingsService {
    const NAME: &'static str = "settings";
}

/// Mount a settings service over a backend on the provider plugin's context:
/// registers the write-drain teardown, loads and publishes the document
/// before the service becomes injectable, then provides `settings`.
pub async fn mount(
    ctx: &Context,
    backend: Rc<dyn SettingsBackend>,
) -> anyhow::Result<Rc<SettingsService>> {
    let service = Rc::new_cyclic(|weak| SettingsService {
        ctx: ctx.clone(),
        backend: backend.clone(),
        weak_self: weak.clone(),
        registrations: RefCell::new(Vec::new()),
        document: RefCell::new(Map::new()),
        write_queues: RefCell::new(HashMap::new()),
        pending_tails: RefCell::new(HashMap::new()),
        tail_counter: Cell::new(0),
        stopped: Cell::new(false),
    });
    backend.attach(Rc::downgrade(&service));
    let teardown = service.clone();
    ctx.effect_labeled("settings:drain", move |_| {
        Ok(Effect::One(Disposer::asynchronous(move || async move {
            // Refuse new writes and new watcher starts, then wait until every
            // queued write chain and every started watcher invocation
            // settles, so disposal completes only once storage and observers
            // are quiescent.
            teardown.stopped.set(true);
            let tails: Vec<Tail> = teardown
                .write_queues
                .borrow()
                .values()
                .cloned()
                .chain(teardown.pending_tails.borrow().values().cloned())
                .collect();
            for tail in tails {
                tail.await;
            }
        })))
    })?;
    let doc = backend.load().await?;
    service.publish(doc);
    ctx.provide_service(service.clone())?;
    Ok(service)
}

impl SettingsService {
    fn registration(&self, ns: &SettingsNamespace) -> Option<Rc<Registration>> {
        self.registrations
            .borrow()
            .iter()
            .find(|registration| registration.ns == *ns)
            .cloned()
    }

    /// Whether in-process writes may persist through this provider.
    pub fn writable(&self) -> bool {
        self.backend.writable()
    }

    /// Absolute path of the provider's user-editable document, when its
    /// storage is one local file.
    pub fn document_path(&self) -> Option<PathBuf> {
        self.backend.document_path()
    }

    /// Prepare the provider's user-editable document for a native editor.
    pub fn prepare_document(&self) -> LocalBoxFuture<'static, anyhow::Result<Option<PathBuf>>> {
        self.backend.prepare_document()
    }

    /// Read one namespace's raw user section, rejecting non-object sections.
    fn section(&self, ns: &SettingsNamespace) -> anyhow::Result<Option<Map<String, Value>>> {
        match self.document.borrow().get(ns.as_str()) {
            None => Ok(None),
            Some(Value::Object(map)) => Ok(Some(map.clone())),
            Some(_) => bail!("settings section \"{ns}\" must be an object of keys"),
        }
    }

    /// Register a namespace schema and receive its owner scope. The
    /// registration is an effect on `ctx`'s fiber: disposing that fiber
    /// removes the namespace and its observers. An invalid stored section
    /// fails the registration itself — the earliest point where the schema
    /// can judge it.
    pub fn register(
        &self,
        ctx: &Context,
        ns: SettingsNamespace,
        schema: Schema,
        options: SettingsRegisterOptions,
    ) -> anyhow::Result<SettingsScope> {
        if self.registration(&ns).is_some() {
            bail!("settings namespace \"{ns}\" is already registered");
        }
        let section = self.section(&ns)?;
        let resolved = resolve_value(
            &schema,
            options.base.as_ref(),
            section.as_ref(),
            options.validate.as_ref(),
        )?;
        let registration = Rc::new(Registration {
            ns: ns.clone(),
            schema,
            base: options.base,
            applies: options.applies,
            validate: options.validate,
            resolved: RefCell::new(resolved),
            revision: Cell::new(0),
            watchers: RefCell::new(Vec::new()),
        });
        let weak = self.weak_self.clone();
        let effect_registration = registration.clone();
        ctx.effect_labeled(
            &format!("settings.register({:?})", ns.as_str()),
            move |_| {
                let Some(service) = weak.upgrade() else {
                    return Ok(Effect::None);
                };
                service
                    .registrations
                    .borrow_mut()
                    .push(effect_registration.clone());
                Ok(Effect::One(Disposer::sync(move || {
                    if let Some(service) = weak.upgrade() {
                        service
                            .registrations
                            .borrow_mut()
                            .retain(|entry| !Rc::ptr_eq(entry, &effect_registration));
                    }
                })))
            },
        )?;
        Ok(SettingsScope {
            service: self
                .weak_self
                .upgrade()
                .expect("service alive during register"),
            registration,
        })
    }

    /// Describe every registered namespace for configuration surfaces, in
    /// registration order, including the composition `base` and raw user
    /// layers so a form can mark which fields the user overrode.
    pub fn describe(&self, options: SettingsDescribeOptions) -> Vec<SettingsDescriptor> {
        let registrations: Vec<Rc<Registration>> = self.registrations.borrow().clone();
        registrations
            .iter()
            .map(|registration| {
                // A malformed stored section already warned at publish and
                // kept the last good resolved value; describe it as "no user
                // layer" to keep this read total.
                let user = self
                    .section(&registration.ns)
                    .ok()
                    .flatten()
                    .map(Value::Object);
                let mut descriptor = SettingsDescriptor {
                    ns: registration.ns.clone(),
                    schema: registration.schema.to_json(),
                    value: registration.resolved.borrow().clone(),
                    revision: registration.revision.get(),
                    base: registration.base.clone(),
                    user,
                    applies: registration.applies,
                    secrets: None,
                };
                if options.redact_secrets {
                    let redacted = redact_secrets(&registration.schema, Some(&descriptor.value));
                    descriptor.value = redacted.value.unwrap_or(Value::Null);
                    descriptor.base = descriptor.base.take().map(|base| {
                        redact_secrets(&registration.schema, Some(&base))
                            .value
                            .unwrap_or(Value::Null)
                    });
                    descriptor.user = descriptor.user.take().map(|user| {
                        redact_secrets(&registration.schema, Some(&user))
                            .value
                            .unwrap_or(Value::Null)
                    });
                    descriptor.secrets = Some(redacted.secrets);
                }
                descriptor
            })
            .collect()
    }

    /// Read one registered namespace's resolved value; `None` while
    /// unregistered.
    pub fn get(&self, ns: &SettingsNamespace) -> Option<Value> {
        self.registration(ns)
            .map(|registration| registration.resolved.borrow().clone())
    }

    /// Merge a patch into one registered namespace's user layer, validate the
    /// resolved candidate, persist, then commit and emit. A validation
    /// failure rejects before anything is persisted; writes to one namespace
    /// apply in call order.
    pub fn update(
        &self,
        ns: &SettingsNamespace,
        patch: Value,
    ) -> LocalBoxFuture<'static, anyhow::Result<()>> {
        self.update_expecting(ns, patch, None)
    }

    /// [`SettingsService::update`] with a revision expectation: a namespace
    /// that moved past it rejects with [`SettingsConflictError`].
    pub fn update_expecting(
        &self,
        ns: &SettingsNamespace,
        patch: Value,
        expected_revision: Option<u64>,
    ) -> LocalBoxFuture<'static, anyhow::Result<()>> {
        self.write(ns, WriteMode::Merge, patch, expected_revision)
    }

    /// Replace one registered namespace's user section wholesale; absent keys
    /// re-inherit the composition `base` and schema defaults (`replace({})`
    /// resets all).
    pub fn replace(
        &self,
        ns: &SettingsNamespace,
        section: Value,
    ) -> LocalBoxFuture<'static, anyhow::Result<()>> {
        self.replace_expecting(ns, section, None)
    }

    /// [`SettingsService::replace`] with a revision expectation.
    pub fn replace_expecting(
        &self,
        ns: &SettingsNamespace,
        section: Value,
        expected_revision: Option<u64>,
    ) -> LocalBoxFuture<'static, anyhow::Result<()>> {
        self.write(ns, WriteMode::Replace, section, expected_revision)
    }

    /// Apply path-addressed edits to one registered namespace's user section.
    /// Ops read the section as it stands at the front of the write queue, so
    /// a caller holding a redacted view cannot delete fields it never saw.
    pub fn mutate(
        &self,
        ns: &SettingsNamespace,
        ops: Vec<SettingsPathOp>,
    ) -> LocalBoxFuture<'static, anyhow::Result<()>> {
        self.mutate_expecting(ns, ops, None)
    }

    /// [`SettingsService::mutate`] with a revision expectation.
    pub fn mutate_expecting(
        &self,
        ns: &SettingsNamespace,
        ops: Vec<SettingsPathOp>,
        expected_revision: Option<u64>,
    ) -> LocalBoxFuture<'static, anyhow::Result<()>> {
        match self.begin_write(ns, WriteInput::Mutate(ops), expected_revision, "mutate") {
            Ok(rx) => await_write(rx),
            Err(error) => Box::pin(futures::future::ready(Err(error))),
        }
    }

    fn write(
        &self,
        ns: &SettingsNamespace,
        mode: WriteMode,
        input: Value,
        expected_revision: Option<u64>,
    ) -> LocalBoxFuture<'static, anyhow::Result<()>> {
        let verb = match mode {
            WriteMode::Merge => "update",
            WriteMode::Replace => "replace",
        };
        let input = match input {
            Value::Object(map) => map,
            _ => {
                let error = anyhow::anyhow!("settings {verb} for \"{ns}\" must be a plain object");
                return Box::pin(futures::future::ready(Err(error)));
            }
        };
        let input = match mode {
            WriteMode::Merge => WriteInput::Merge(input),
            WriteMode::Replace => WriteInput::Replace(input),
        };
        match self.begin_write(ns, input, expected_revision, verb) {
            Ok(rx) => await_write(rx),
            Err(error) => Box::pin(futures::future::ready(Err(error))),
        }
    }

    /// Validate a write eagerly, then queue it on the namespace's serialized
    /// write chain. Queueing happens at call time (before the returned future
    /// is polled), so concurrent writes apply in call order.
    fn begin_write(
        &self,
        ns: &SettingsNamespace,
        input: WriteInput,
        expected_revision: Option<u64>,
        verb: &'static str,
    ) -> anyhow::Result<futures::channel::oneshot::Receiver<anyhow::Result<()>>> {
        let Some(registration) = self.registration(ns) else {
            bail!("settings namespace \"{ns}\" is not registered");
        };
        if self.stopped.get() {
            bail!("settings service is disposed: \"{ns}\" cannot be written");
        }
        if !self.backend.writable() {
            bail!("settings provider is read-only: \"{ns}\" cannot be updated in-process");
        }
        let previous = self.write_queues.borrow().get(ns.as_str()).cloned();
        let (result_tx, result_rx) = futures::channel::oneshot::channel();
        let (done_tx, done_rx) = futures::channel::oneshot::channel::<()>();
        let tail: Tail = async move {
            let _ = done_rx.await;
        }
        .boxed_local()
        .shared();
        let service = self
            .weak_self
            .upgrade()
            .expect("service alive during write");
        let ns = ns.clone();
        self.write_queues.borrow_mut().insert(ns.to_string(), tail);
        tokio::task::spawn_local(async move {
            // Chain past a failed predecessor: one rejected write must not
            // poison the namespace queue for every later caller.
            if let Some(previous) = previous {
                previous.await;
            }
            let outcome = service
                .run_write(&ns, registration, input, expected_revision, verb)
                .await;
            let _ = result_tx.send(outcome);
            let _ = done_tx.send(());
        });
        Ok(result_rx)
    }

    async fn run_write(
        self: &Rc<Self>,
        ns: &SettingsNamespace,
        registration: Rc<Registration>,
        input: WriteInput,
        expected_revision: Option<u64>,
        verb: &'static str,
    ) -> anyhow::Result<()> {
        if self.stopped.get() {
            bail!("settings service was disposed before the queued \"{ns}\" {verb} ran");
        }
        let still_owner = self
            .registration(ns)
            .is_some_and(|entry| Rc::ptr_eq(&entry, &registration));
        if !still_owner {
            bail!(
                "settings namespace \"{ns}\" registration was disposed before the queued {verb} ran"
            );
        }
        // Every mode derives from the section as it stands NOW, at the front
        // of the queue — never from whatever the caller last saw.
        let current = self.section(ns)?.unwrap_or_default();
        // The revision check belongs here, not at call time: the queue orders
        // writes but cannot tell a fresh writer from one holding a snapshot a
        // predecessor already superseded.
        if let Some(expected) = expected_revision {
            let actual = registration.revision.get();
            if expected != actual {
                return Err(SettingsConflictError {
                    ns: ns.clone(),
                    expected,
                    actual,
                }
                .into());
            }
        }
        let section = match input {
            WriteInput::Merge(patch) => {
                match merge_layers(
                    Some(&Value::Object(current.clone())),
                    Some(&Value::Object(patch)),
                ) {
                    Some(Value::Object(map)) => map,
                    _ => unreachable!("merging two objects yields an object"),
                }
            }
            WriteInput::Replace(section) => section,
            WriteInput::Mutate(ops) => {
                let mut section = current.clone();
                for op in &ops {
                    section = apply_path_op(section, op)?;
                }
                section
            }
        };
        let next = resolve_value(
            &registration.schema,
            registration.base.as_ref(),
            Some(&section),
            registration.validate.as_ref(),
        )?;
        self.backend.persist(ns, &section).await?;
        // The write reached storage either way; the cache must say so. Commit
        // only when this registration is still the namespace owner — a fiber
        // disposed (or replaced) mid-persist must not receive the
        // notification.
        self.document
            .borrow_mut()
            .insert(ns.to_string(), Value::Object(section.clone()));
        let still_owner = self
            .registration(ns)
            .is_some_and(|entry| Rc::ptr_eq(&entry, &registration));
        if still_owner && !self.stopped.get() {
            self.bump_revision(&registration, Some(&current), Some(&section));
            self.commit(&registration, next, SettingsUpdateSource::Update);
        }
        Ok(())
    }

    /// Commit a complete raw document observed in storage. Each registered
    /// namespace re-resolves; an invalid section keeps that namespace's last
    /// good value and warns, other namespaces still commit.
    pub fn publish(&self, doc: Map<String, Value>) {
        self.publish_with_source(doc, SettingsUpdateSource::Provider);
    }

    fn publish_with_source(&self, doc: Map<String, Value>, source: SettingsUpdateSource) {
        let registrations: Vec<Rc<Registration>> = self.registrations.borrow().clone();
        // Read every raw section BEFORE swapping the document, so the
        // revision bump compares what was stored with what now is — an
        // external edit moves the revision exactly like an in-process write.
        let before: Vec<Option<Map<String, Value>>> = registrations
            .iter()
            .map(|registration| self.section(&registration.ns).ok().flatten())
            .collect();
        *self.document.borrow_mut() = doc;
        for (registration, before) in registrations.iter().zip(before) {
            let next = self.section(&registration.ns).and_then(|section| {
                resolve_value(
                    &registration.schema,
                    registration.base.as_ref(),
                    section.as_ref(),
                    registration.validate.as_ref(),
                )
            });
            let next = match next {
                Ok(next) => next,
                Err(error) => {
                    tracing::warn!(
                        "settings: keeping last good \"{}\" after invalid stored section: {error:#}",
                        registration.ns
                    );
                    continue;
                }
            };
            let after = self.section(&registration.ns).ok().flatten();
            self.bump_revision(registration, before.as_ref(), after.as_ref());
            self.commit(registration, next, source);
        }
    }

    /// Advance a namespace's revision when its RAW section changed, and
    /// announce it. Deliberately independent of `commit`'s resolved-value
    /// equality: storing an override equal to the composition base leaves the
    /// resolved value alone but changes what the document says.
    fn bump_revision(
        &self,
        registration: &Rc<Registration>,
        before: Option<&Map<String, Value>>,
        after: Option<&Map<String, Value>>,
    ) {
        if before == after {
            return;
        }
        registration.revision.set(registration.revision.get() + 1);
        self.ctx.emit::<SettingsDocumentUpdated>(&(
            registration.ns.clone(),
            registration.revision.get(),
        ));
    }

    /// Commit a resolved value when changed: swap, notify watchers, emit the
    /// event.
    fn commit(&self, registration: &Rc<Registration>, next: Value, source: SettingsUpdateSource) {
        let prev = registration.resolved.borrow().clone();
        if next == prev {
            return;
        }
        *registration.resolved.borrow_mut() = next.clone();
        let watchers: Vec<Rc<Watcher>> = registration.watchers.borrow().clone();
        for watcher in watchers {
            self.chain_watcher(registration.ns.clone(), watcher, next.clone(), prev.clone());
        }
        self.ctx
            .emit::<SettingsUpdated>(&(registration.ns.clone(), next, prev, source));
    }

    /// Queue one watcher invocation behind that watcher's previous one, so a
    /// slow stale invocation can never apply after a newer one. The activity
    /// check runs when the queued invocation would start, so a disposer (or
    /// service stop) that ran while it waited prevents the start entirely;
    /// started invocations drain at service dispose. A failing callback is
    /// contained and logged.
    fn chain_watcher(&self, ns: SettingsNamespace, watcher: Rc<Watcher>, next: Value, prev: Value) {
        let weak = self.weak_self.clone();
        let previous = watcher.tail.borrow().clone();
        let watcher_for_segment = watcher.clone();
        let segment: Tail = async move {
            previous.await;
            let stopped = weak
                .upgrade()
                .map(|service| service.stopped.get())
                .unwrap_or(true);
            if !watcher_for_segment.active.get() || stopped {
                return;
            }
            if let Err(error) = (watcher_for_segment.callback)(next, prev).await {
                tracing::warn!("settings: watcher for \"{ns}\" failed: {error:#}");
            }
        }
        .boxed_local()
        .shared();
        *watcher.tail.borrow_mut() = segment.clone();
        self.tail_counter.set(self.tail_counter.get() + 1);
        let id = self.tail_counter.get();
        self.pending_tails.borrow_mut().insert(id, segment.clone());
        let weak = self.weak_self.clone();
        tokio::task::spawn_local(async move {
            segment.await;
            if let Some(service) = weak.upgrade() {
                service.pending_tails.borrow_mut().remove(&id);
            }
        });
    }
}

enum WriteMode {
    Merge,
    Replace,
}

fn await_write(
    rx: futures::channel::oneshot::Receiver<anyhow::Result<()>>,
) -> LocalBoxFuture<'static, anyhow::Result<()>> {
    Box::pin(async move {
        rx.await
            .map_err(|_| anyhow::anyhow!("settings: write task vanished"))?
    })
}

/// Owner-facing handle for one registered namespace.
pub struct SettingsScope {
    service: Rc<SettingsService>,
    registration: Rc<Registration>,
}

impl SettingsScope {
    /// The registered namespace.
    pub fn ns(&self) -> &SettingsNamespace {
        &self.registration.ns
    }

    /// Current resolved value: schema defaults, then `base`, then the user
    /// layer (a detached clone).
    pub fn get(&self) -> Value {
        self.registration.resolved.borrow().clone()
    }

    /// Observe committed changes to this namespace's resolved value.
    /// Invocations of one callback run asynchronously, one at a time, in
    /// commit order; a failure is contained and logged. After the disposer
    /// runs, no further invocation starts — one already queued is skipped;
    /// one already started still settles, and service disposal waits for it.
    pub fn watch<F, Fut>(&self, callback: F) -> SettingsWatchDisposer
    where
        F: Fn(Value, Value) -> Fut + 'static,
        Fut: std::future::Future<Output = anyhow::Result<()>> + 'static,
    {
        let watcher = Rc::new(Watcher {
            callback: Rc::new(move |next, prev| callback(next, prev).boxed_local()),
            tail: RefCell::new(ready_tail()),
            active: Cell::new(true),
        });
        self.registration
            .watchers
            .borrow_mut()
            .push(watcher.clone());
        SettingsWatchDisposer {
            registration: self.registration.clone(),
            watcher,
        }
    }

    /// Merge a partial patch into this namespace's user layer and persist it.
    pub fn update(&self, patch: Value) -> LocalBoxFuture<'static, anyhow::Result<()>> {
        self.service.update(&self.registration.ns, patch)
    }

    /// Replace this namespace's user section wholesale; absent keys
    /// re-inherit the composition `base` and schema defaults.
    pub fn replace(&self, section: Value) -> LocalBoxFuture<'static, anyhow::Result<()>> {
        self.service.replace(&self.registration.ns, section)
    }
}

/// Disposer for one [`SettingsScope::watch`] observer; idempotent.
pub struct SettingsWatchDisposer {
    registration: Rc<Registration>,
    watcher: Rc<Watcher>,
}

impl SettingsWatchDisposer {
    /// Remove the observer; a queued invocation that has not started is
    /// skipped, one already started still settles.
    pub fn dispose(&self) {
        self.watcher.active.set(false);
        self.registration
            .watchers
            .borrow_mut()
            .retain(|entry| !Rc::ptr_eq(entry, &self.watcher));
    }
}

/// Whether the consumer's own fiber is tearing down (not just losing the
/// settings service).
fn is_unloading(ctx: &Context) -> bool {
    matches!(
        ctx.fiber().state(),
        FiberState::Unloading | FiberState::Disposed
    )
}

/// Hooks a consumer hands to [`install_settings_section`].
pub struct SettingsSectionHooks {
    /// Receive the active configuration source: the resolved settings scope
    /// while one is attached, the composition entry otherwise. Called before
    /// the matching `on_change` at attach and at detach.
    pub set_source: Rc<dyn Fn(Rc<dyn Fn() -> Value>)>,
    /// Re-judge anything derived from the source after an attach, a detach,
    /// or a committed change.
    pub on_change: Rc<dyn Fn()>,
    /// Cross-field check forwarded to the registration.
    pub validate: Option<ValidateFn>,
}

/// Install the canonical optional-settings consumer wiring: while a settings
/// service exists, register `ns` with the consumer's composition entry as the
/// `base` layer and point the source thunk at the resolved scope; when the
/// service goes away, fall back to the entry so the consumer keeps working
/// exactly as composed. The registration rides a scoped child fiber that
/// injects `settings`, so no settings service ever mounted means none of this
/// runs.
pub fn install_settings_section(
    ctx: &Context,
    ns: SettingsNamespace,
    schema: Schema,
    entry: Value,
    hooks: SettingsSectionHooks,
) -> dsh_cordis::Result<Fiber> {
    let consumer_ctx = ctx.clone();
    let hooks = Rc::new(hooks);
    let plugin = plugin_fn(
        "settings-section",
        Inject::names(["settings"]),
        move |sctx, _config| {
            let consumer_ctx = consumer_ctx.clone();
            let hooks = hooks.clone();
            let ns = ns.clone();
            let schema = schema.clone();
            let entry = entry.clone();
            async move {
                let service = sctx
                    .service::<SettingsService>()
                    .map_err(anyhow::Error::from)?;
                let scope = Rc::new(service.register(
                    &sctx,
                    ns,
                    schema,
                    SettingsRegisterOptions {
                        base: Some(entry.clone()),
                        applies: SettingsApplies::default(),
                        validate: hooks.validate.clone(),
                    },
                )?);
                let source_scope = scope.clone();
                (hooks.set_source)(Rc::new(move || source_scope.get()));
                let fallback_ctx = consumer_ctx.clone();
                let fallback_hooks = hooks.clone();
                let fallback_entry = entry.clone();
                sctx.effect(move |_| {
                    Ok(Effect::One(Disposer::sync(move || {
                        // This disposer runs for two different reasons. A
                        // settings provider detaching leaves the consumer
                        // running, so it falls back to its composition entry
                        // and re-judges what it derived. The consumer's own
                        // unload runs it too — and there `on_change` would
                        // re-register against resources the teardown is
                        // releasing, so it must stay silent.
                        if is_unloading(&fallback_ctx) {
                            return;
                        }
                        let entry = fallback_entry.clone();
                        (fallback_hooks.set_source)(Rc::new(move || entry.clone()));
                        (fallback_hooks.on_change)();
                    })))
                })
                .map_err(anyhow::Error::from)?;
                (hooks.on_change)();
                let watch_ctx = consumer_ctx.clone();
                let watch_hooks = hooks.clone();
                scope.watch(move |_next, _prev| {
                    // A stored change landing while the consumer unloads
                    // reaches the watcher before the registration is
                    // released; notifying then is exactly as harmful as
                    // notifying from the disposer above.
                    if !is_unloading(&watch_ctx) {
                        (watch_hooks.on_change)();
                    }
                    futures::future::ready(Ok(()))
                });
                Ok(())
            }
        },
    );
    ctx.plugin(Rc::new(plugin), Value::Null)
}
