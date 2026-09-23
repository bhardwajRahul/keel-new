//! The Cordis core state machine: contexts, fibers, service store, and the
//! plugin registry — ported from `vendor/cordis/src/{context,fiber,reflect,
//! registry}.ts`.
//!
//! Divergences from the TS implementation, all forced by the language:
//! - No `Proxy`: services are read with [`Context::get_raw`] /
//!   `Context::service::<S>()` instead of property access.
//! - Single-threaded: the whole framework runs inside one tokio `LocalSet`
//!   (matching JS run-to-completion); nothing here is `Send`.
//! - Plugins are identified by `TypeId` instead of function identity; two
//!   instances of the same plugin type share one runtime record, matching
//!   upstream class-plugin identity.
//!
//! Everything else follows upstream semantics: fibers move through
//! PENDING → LOADING → ACTIVE → UNLOADING (or FAILED/DISPOSED), driven by an
//! epoch string of provider uids; registrations are effects whose disposers
//! run in reverse order on unload; providing or removing a service re-checks
//! every fiber that injects it.

use crate::disposer::{DisposableList, Disposer, Effect};
use crate::error::{CordisError, Result};
use crate::events::Event;
use crate::plugin::{Plugin, PluginKey};
use futures::FutureExt;
use futures::future::{LocalBoxFuture, Shared};
use serde_json::Value;
use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

pub type FiberId = u64;
pub(crate) type Label = u64;
type Inertia = Shared<LocalBoxFuture<'static, ()>>;

/// Lifecycle state for one plugin fiber (upstream `FiberState`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiberState {
    /// Waiting for required services.
    Pending,
    /// The plugin body is running.
    Loading,
    /// Loaded and providing.
    Active,
    /// The body or its config threw.
    Failed,
    /// Removed; cannot restart.
    Disposed,
    /// Disposers are running.
    Unloading,
}

/// Epoch of one activation: `Active` carries the concatenated provider uids
/// so a dependency swap forces a reload even when all names stay satisfied.
#[derive(Clone, PartialEq, Eq)]
pub(crate) enum Epoch {
    Inactive,
    Active(String),
}

/// Concrete service implementation record (upstream `Impl`).
pub(crate) struct ImplRecord {
    pub name: String,
    pub fiber: FiberId,
    pub value: RefCell<Rc<dyn Any>>,
    pub check: Option<Rc<dyn Fn(&Context) -> bool>>,
}

/// Mutable registry record shared by all fibers of one plugin
/// (upstream `Plugin.Runtime`).
pub(crate) struct RuntimeRecord {
    pub name: Option<String>,
    pub plugin: Rc<dyn Plugin>,
    pub fibers: Vec<FiberId>,
}

pub(crate) struct FiberData {
    pub uid: Option<u64>,
    pub parent: Option<FiberId>,
    /// The context the plugin was loaded from (isolate chain of the caller).
    pub parent_ctx: Option<Context>,
    /// The fiber's own plugin context.
    pub ctx: Context,
    pub state: FiberState,
    pub inject: HashMap<String, Option<Value>>,
    pub runtime: Option<PluginKey>,
    pub config_raw: Value,
    pub config: Value,
    pub error: Option<CordisError>,
    pub epoch: Epoch,
    /// Snapshot of satisfying impls while loaded (upstream `fiber.store`).
    pub store_snapshot: Option<HashMap<String, Rc<ImplRecord>>>,
    /// Live dependency-check results (upstream `fiber._store`).
    pub live_store: HashMap<String, Rc<ImplRecord>>,
    pub disposables: DisposableList<EffectHandle>,
    pub inertia: Option<Inertia>,
}

pub(crate) struct AppState {
    pub fibers: RefCell<HashMap<FiberId, FiberData>>,
    pub runtimes: RefCell<HashMap<PluginKey, RuntimeRecord>>,
    pub hooks: RefCell<HashMap<String, Vec<crate::events::Hook>>>,
    pub store: RefCell<HashMap<Label, Rc<ImplRecord>>>,
    pub root_isolate: RefCell<HashMap<String, Label>>,
    counter: Cell<u64>,
    label_counter: Cell<Label>,
    hook_counter: Cell<u64>,
}

impl AppState {
    pub(crate) fn next_uid(&self) -> u64 {
        self.counter.set(self.counter.get() + 1);
        self.counter.get()
    }

    pub(crate) fn next_label(&self) -> Label {
        self.label_counter.set(self.label_counter.get() + 1);
        self.label_counter.get()
    }

    pub(crate) fn next_hook_id(&self) -> u64 {
        self.hook_counter.set(self.hook_counter.get() + 1);
        self.hook_counter.get()
    }
}

/// Isolation chain node: service name → scope label, prototypally inherited
/// (upstream `Context[symbols.isolate]`). Root labels live in
/// `AppState::root_isolate`.
pub(crate) struct IsolateNode {
    map: HashMap<String, Label>,
    parent: Option<Rc<IsolateNode>>,
}

/// Intercept chain node: service name → config merged into that service's
/// per-plugin config (upstream `Context[symbols.intercept]`).
pub(crate) struct InterceptNode {
    map: HashMap<String, Value>,
    parent: Option<Rc<InterceptNode>>,
}

/// Dependency container handle (upstream `Context`). Cheap to clone; child
/// contexts share the app state and differ only in fiber, isolate, and
/// intercept chains.
#[derive(Clone)]
pub struct Context {
    app: Rc<AppState>,
    fiber: FiberId,
    isolate: Option<Rc<IsolateNode>>,
    intercept: Option<Rc<InterceptNode>>,
}

/// Runtime instance handle of one plugin application (upstream `Fiber`).
#[derive(Clone)]
pub struct Fiber {
    app: Rc<AppState>,
    id: FiberId,
}

/// Disposal handle returned by `Context::effect` and every registration.
/// Idempotent: the first `dispose()` runs collected disposers in reverse
/// order; later calls join the same task.
#[derive(Clone)]
pub struct EffectHandle(Rc<EffectState>);

impl std::fmt::Debug for EffectHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EffectHandle")
            .field("label", &self.0.label)
            .field("disposed", &self.0.disposed.get())
            .finish()
    }
}

struct EffectState {
    label: String,
    disposed: Cell<bool>,
    disposers: RefCell<Vec<Disposer>>,
    task: RefCell<Option<Inertia>>,
}

impl EffectHandle {
    /// Tear the effect down; resolves once every disposer has settled.
    pub fn dispose(&self) -> LocalBoxFuture<'static, ()> {
        if let Some(task) = self.0.task.borrow().clone() {
            return task.boxed_local();
        }
        if self.0.disposed.get() {
            return futures::future::ready(()).boxed_local();
        }
        self.0.disposed.set(true);
        let disposers = {
            let mut list = self.0.disposers.borrow_mut();
            let mut taken: Vec<Disposer> = list.drain(..).collect();
            taken.reverse();
            taken
        };
        let fut = async move {
            for disposer in disposers {
                disposer.run().await;
            }
        }
        .boxed_local()
        .shared();
        *self.0.task.borrow_mut() = Some(fut.clone());
        fut.boxed_local()
    }

    /// Effect label shown in diagnostics (upstream `EffectMeta.label`).
    pub fn label(&self) -> &str {
        &self.0.label
    }
}

/// The application root: owns the state and the root context.
pub struct App {
    root: Context,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    /// Create the root context with an active root fiber (uid 0).
    pub fn new() -> App {
        let app = Rc::new(AppState {
            fibers: RefCell::new(HashMap::new()),
            runtimes: RefCell::new(HashMap::new()),
            hooks: RefCell::new(HashMap::new()),
            store: RefCell::new(HashMap::new()),
            root_isolate: RefCell::new(HashMap::new()),
            counter: Cell::new(0),
            label_counter: Cell::new(0),
            hook_counter: Cell::new(0),
        });
        let root = Context {
            app: app.clone(),
            fiber: 0,
            isolate: None,
            intercept: None,
        };
        let data = FiberData {
            uid: Some(0),
            parent: None,
            parent_ctx: None,
            ctx: root.clone(),
            state: FiberState::Active,
            inject: HashMap::new(),
            runtime: None,
            config_raw: Value::Null,
            config: Value::Null,
            error: None,
            epoch: Epoch::Active(String::new()),
            store_snapshot: Some(HashMap::new()),
            live_store: HashMap::new(),
            disposables: DisposableList::default(),
            inertia: None,
        };
        app.fibers.borrow_mut().insert(0, data);
        App { root }
    }

    /// The root context.
    pub fn root(&self) -> Context {
        self.root.clone()
    }

    /// Dispose every root-owned effect (reverse order) — application
    /// shutdown. The upstream root fiber's `dispose` restarts instead; a Rust
    /// process wants a real teardown.
    pub async fn shutdown(&self) {
        let handles = {
            let mut fibers = self.root.app.fibers.borrow_mut();
            let root = fibers.get_mut(&0).expect("root fiber");
            root.disposables.clear()
        };
        for handle in handles {
            handle.dispose().await;
        }
    }
}

impl Context {
    pub(crate) fn app(&self) -> &AppState {
        &self.app
    }

    pub(crate) fn app_rc(&self) -> Rc<AppState> {
        self.app.clone()
    }

    pub fn fiber_id(&self) -> FiberId {
        self.fiber
    }

    /// The fiber (plugin runtime instance) that owns this context.
    pub fn fiber(&self) -> Fiber {
        Fiber {
            app: self.app.clone(),
            id: self.fiber,
        }
    }

    fn with_fiber(&self, fiber: FiberId) -> Context {
        Context {
            fiber,
            ..self.clone()
        }
    }

    /// Resolve the isolation label for a service name: the local chain first,
    /// then the root map (upstream prototype-chain lookup).
    pub(crate) fn isolate_lookup(&self, name: &str) -> Option<Label> {
        let mut node = self.isolate.as_ref();
        while let Some(current) = node {
            if let Some(label) = current.map.get(name) {
                return Some(*label);
            }
            node = current.parent.as_ref();
        }
        self.app.root_isolate.borrow().get(name).copied()
    }

    /// Create a child context with an independent service scope for `name`
    /// (upstream `ctx.isolate`). Passing the same `label` to two calls joins
    /// their scopes.
    pub fn isolate(&self, name: &str, label: Option<Label>) -> Context {
        let label = label.unwrap_or_else(|| self.app.next_label());
        let mut map = HashMap::new();
        map.insert(name.to_string(), label);
        Context {
            isolate: Some(Rc::new(IsolateNode {
                map,
                parent: self.isolate.clone(),
            })),
            ..self.clone()
        }
    }

    /// Add service-specific intercept config for plugins started below this
    /// context (upstream `ctx.intercept`).
    pub fn intercept(&self, name: &str, config: Value) -> Context {
        let mut map = HashMap::new();
        map.insert(name.to_string(), config);
        Context {
            intercept: Some(Rc::new(InterceptNode {
                map,
                parent: self.intercept.clone(),
            })),
            ..self.clone()
        }
    }

    /// Collect intercept configs for a service, root-most first (upstream
    /// `Service[symbols.resolveConfig]` walk).
    pub fn intercept_configs(&self, name: &str) -> Vec<Value> {
        let mut configs = Vec::new();
        let mut node = self.intercept.as_ref();
        while let Some(current) = node {
            if let Some(config) = current.map.get(name) {
                configs.push(config.clone());
            }
            node = current.parent.as_ref();
        }
        configs.reverse();
        configs
    }

    /// Register a cleanup-aware effect on the owning fiber (upstream
    /// `ctx.effect`). The body runs immediately; its disposers run in reverse
    /// order when the handle is disposed or the fiber unloads.
    pub fn effect(&self, body: impl FnOnce(&Context) -> Result<Effect>) -> Result<EffectHandle> {
        self.effect_labeled("anonymous", body)
    }

    /// Same as [`Context::effect`] with a diagnostics label.
    pub fn effect_labeled(
        &self,
        label: &str,
        body: impl FnOnce(&Context) -> Result<Effect>,
    ) -> Result<EffectHandle> {
        {
            let fibers = self.app.fibers.borrow();
            let data = fibers.get(&self.fiber).ok_or(CordisError::InactiveEffect)?;
            if data.uid.is_none() || data.state == FiberState::Unloading {
                return Err(CordisError::InactiveEffect);
            }
        }
        let effect = body(self)?;
        let handle = EffectHandle(Rc::new(EffectState {
            label: label.to_string(),
            disposed: Cell::new(false),
            disposers: RefCell::new(effect.into_disposers()),
            task: RefCell::new(None),
        }));
        let mut fibers = self.app.fibers.borrow_mut();
        if let Some(data) = fibers.get_mut(&self.fiber) {
            data.disposables.push(handle.clone());
        }
        Ok(handle)
    }

    /// Register a service implementation owned by the current fiber
    /// (upstream `ctx.provide` via `ReflectService.provide`). The service is
    /// visible to dependents in the same isolation scope while the fiber is
    /// active; disposing unregisters it and re-settles dependents.
    pub fn provide(
        &self,
        name: &str,
        value: Rc<dyn Any>,
        check: Option<Rc<dyn Fn(&Context) -> bool>>,
    ) -> Result<EffectHandle> {
        let name = name.to_string();
        let label = {
            // Root label is created on first provide (upstream `??= Symbol(name)`),
            // but resolution still honors the local isolate chain below. The
            // root borrow must end before `isolate_lookup` re-reads the map.
            {
                let mut root = self.app.root_isolate.borrow_mut();
                if !root.contains_key(&name) {
                    let fresh = self.app.next_label();
                    root.insert(name.clone(), fresh);
                }
            }
            self.isolate_lookup(&name).expect("label just ensured")
        };
        let fiber = self.fiber;
        let app = self.app.clone();
        let ctx = self.clone();
        self.effect_labeled(&format!("ctx.provide({name:?})"), move |_| {
            {
                let store = app.store.borrow();
                if let Some(existing) = store.get(&label) {
                    let provider = Fiber {
                        app: app.clone(),
                        id: existing.fiber,
                    }
                    .name();
                    return Err(CordisError::ServiceConflict { name, provider });
                }
            }
            let record = Rc::new(ImplRecord {
                name: name.clone(),
                fiber,
                value: RefCell::new(value),
                check,
            });
            app.store.borrow_mut().insert(label, record.clone());
            {
                let mut fibers = app.fibers.borrow_mut();
                if let Some(data) = fibers.get_mut(&fiber) {
                    // Self access: the provider sees its own service while
                    // loaded (upstream `this.ctx.fiber.store![name] = impl`).
                    if let Some(snapshot) = data.store_snapshot.as_mut() {
                        snapshot.insert(name.clone(), record.clone());
                    }
                    data.live_store.insert(name.clone(), record.clone());
                }
            }
            let active = {
                let fibers = app.fibers.borrow();
                fibers.get(&fiber).map(|d| d.state) == Some(FiberState::Active)
            };
            if active {
                notify(&ctx, &[name.clone()]);
            }
            let app2 = app.clone();
            let ctx2 = ctx.clone();
            let name2 = name.clone();
            Ok(Effect::One(Disposer::asynchronous(move || async move {
                app2.store.borrow_mut().remove(&label);
                let woken = notify(&ctx2, &[name2.clone()]);
                for id in woken {
                    let _ = (Fiber {
                        app: app2.clone(),
                        id,
                    })
                    .await_ready()
                    .await;
                }
                // Ensure self access lasted through dependency cleanup
                // (upstream deletes the snapshot entry only afterwards).
                let mut fibers = app2.fibers.borrow_mut();
                if let Some(data) = fibers.get_mut(&fiber) {
                    if let Some(snapshot) = data.store_snapshot.as_mut() {
                        snapshot.remove(&name2);
                    }
                    data.live_store.remove(&name2);
                }
            })))
        })
    }

    /// Overwrite a provided service's value; only the providing fiber may set
    /// (upstream `ctx.set`).
    pub fn set(&self, name: &str, value: Rc<dyn Any>) -> Result<()> {
        let label = self
            .isolate_lookup(name)
            .ok_or_else(|| CordisError::ServiceMissing(name.to_string()))?;
        let store = self.app.store.borrow();
        let record = store
            .get(&label)
            .ok_or_else(|| CordisError::ServiceMissing(name.to_string()))?;
        if record.fiber != self.fiber {
            return Err(CordisError::ServiceConflict {
                name: name.to_string(),
                provider: Fiber {
                    app: self.app.clone(),
                    id: record.fiber,
                }
                .name(),
            });
        }
        *record.value.borrow_mut() = value;
        Ok(())
    }

    /// Read a service through the inject walk (upstream context-proxy `get`):
    /// the fiber chain's store snapshots are consulted upward; an injected
    /// but unsatisfied name errors; crossing an isolate boundary errors.
    pub fn get_raw(&self, name: &str) -> Result<Rc<dyn Any>> {
        let label = self.isolate_lookup(name);
        let fibers = self.app.fibers.borrow();
        let mut current = self.fiber;
        loop {
            let data = fibers
                .get(&current)
                .ok_or_else(|| CordisError::ServiceMissing(name.to_string()))?;
            if let Some(record) = data.store_snapshot.as_ref().and_then(|s| s.get(name)) {
                return Ok(record.value.borrow().clone());
            }
            if data.inject.contains_key(name) {
                return Err(CordisError::ServiceInactive(name.to_string()));
            }
            if data.runtime.is_none() {
                // Root context: fall back to a non-strict store read
                // (upstream `if (!ctx.fiber.runtime) return ctx.reflect.get(prop, false)`).
                drop(fibers);
                return self
                    .try_get_raw(name, false)
                    .ok_or_else(|| CordisError::ServiceMissing(name.to_string()));
            }
            let parent_label = data
                .parent_ctx
                .as_ref()
                .and_then(|ctx| ctx.isolate_lookup(name));
            if parent_label != label {
                return Err(CordisError::ServiceMissing(name.to_string()));
            }
            current = data.parent.expect("non-root fiber has a parent");
        }
    }

    /// Read a service from the store without the inject requirement
    /// (upstream `ctx.reflect.get`). With `strict`, only implementations
    /// whose providing fiber is active are returned.
    pub fn try_get_raw(&self, name: &str, strict: bool) -> Option<Rc<dyn Any>> {
        let label = self.isolate_lookup(name)?;
        let store = self.app.store.borrow();
        let record = store.get(&label)?;
        if strict {
            let fibers = self.app.fibers.borrow();
            if fibers.get(&record.fiber).map(|d| d.state) != Some(FiberState::Active) {
                return None;
            }
        }
        Some(record.value.borrow().clone())
    }

    /// Load a plugin in the current context (upstream `ctx.plugin`).
    pub fn plugin(&self, plugin: Rc<dyn Plugin>, config: Value) -> Result<Fiber> {
        // Assert the current fiber is active (upstream `assertActive`).
        {
            let fibers = self.app.fibers.borrow();
            let data = fibers.get(&self.fiber).ok_or(CordisError::InactiveEffect)?;
            if data.uid.is_none() {
                return Err(CordisError::InactiveEffect);
            }
        }
        let key = plugin.key();
        let inject = plugin.inject();
        {
            let mut runtimes = self.app.runtimes.borrow_mut();
            runtimes.entry(key).or_insert_with(|| RuntimeRecord {
                name: plugin.name(),
                plugin: plugin.clone(),
                fibers: Vec::new(),
            });
        }

        let uid = self.app.next_uid();
        let id: FiberId = uid;
        // Intercept entries from the inject map (upstream fiber constructor).
        let mut child_ctx = self.with_fiber(id);
        for (name, config) in inject.0.iter() {
            if let Some(config) = config {
                child_ctx = child_ctx.intercept(name, config.clone());
            }
        }

        let data = FiberData {
            uid: Some(uid),
            parent: Some(self.fiber),
            parent_ctx: Some(self.clone()),
            ctx: child_ctx.clone(),
            state: FiberState::Pending,
            inject: inject.0,
            runtime: Some(key),
            config_raw: config,
            config: Value::Null,
            error: None,
            epoch: Epoch::Inactive,
            store_snapshot: None,
            live_store: HashMap::new(),
            disposables: DisposableList::default(),
            inertia: None,
        };
        self.app.fibers.borrow_mut().insert(id, data);
        self.app
            .runtimes
            .borrow_mut()
            .get_mut(&key)
            .expect("runtime just ensured")
            .fibers
            .push(id);

        let fiber = Fiber {
            app: self.app.clone(),
            id,
        };

        // Parent-owned disposer: unloading the parent disposes the child
        // (upstream wraps `dispose` in `parent.fiber.effect`).
        let child = fiber.clone();
        self.effect_labeled("ctx.plugin()", move |_| {
            Ok(Effect::One(Disposer::asynchronous(move || async move {
                child.dispose().await;
            })))
        })?;

        self.emit::<InternalPlugin>(&fiber);

        // Resolve dependencies only after publication (upstream keeps the
        // initial notification's PENDING view).
        let parent_unloading = {
            let fibers = self.app.fibers.borrow();
            fibers.get(&self.fiber).map(|d| d.state) == Some(FiberState::Unloading)
        };
        let still_alive = {
            let fibers = self.app.fibers.borrow();
            fibers.get(&id).and_then(|d| d.uid).is_some()
        };
        if still_alive && !parent_unloading {
            let names: Vec<String> = {
                let fibers = self.app.fibers.borrow();
                fibers
                    .get(&id)
                    .map(|d| d.inject.keys().cloned().collect())
                    .unwrap_or_default()
            };
            for name in &names {
                check_impl(&self.app, id, name);
            }
            refresh(&self.app, id);
        }
        Ok(fiber)
    }
}

/// A plugin fiber was created, or its uid was cleared on disposal
/// (upstream `internal/plugin`).
pub struct InternalPlugin;
impl Event for InternalPlugin {
    const NAME: &'static str = "internal/plugin";
    type Args = Fiber;
    type Ret = ();
}

/// A fiber changed lifecycle state (upstream `internal/status`).
pub struct InternalStatus;
impl Event for InternalStatus {
    const NAME: &'static str = "internal/status";
    type Args = (Fiber, FiberState);
    type Ret = ();
}

/// A service binding changed (upstream `internal/service`).
pub struct InternalService;
impl Event for InternalService {
    const NAME: &'static str = "internal/service";
    type Args = String;
    type Ret = ();
}

/// Waterfall: resolve raw plugin config after injections become active
/// (upstream `internal/config`).
pub struct InternalConfig;
impl Event for InternalConfig {
    const NAME: &'static str = "internal/config";
    type Args = Value;
    type Ret = Value;
}

/// Waterfall: a fiber config update is being applied; skipping `next` vetoes
/// the restart (upstream `internal/update`).
pub struct InternalUpdate;
impl Event for InternalUpdate {
    const NAME: &'static str = "internal/update";
    type Args = (Value, bool);
    type Ret = ();
}

impl Fiber {
    pub fn id(&self) -> FiberId {
        self.id
    }

    /// Unique id within the registry; `None` once disposed (upstream `uid`).
    pub fn uid(&self) -> Option<u64> {
        self.app.fibers.borrow().get(&self.id).and_then(|d| d.uid)
    }

    /// Current lifecycle state.
    pub fn state(&self) -> FiberState {
        self.app
            .fibers
            .borrow()
            .get(&self.id)
            .map(|d| d.state)
            .unwrap_or(FiberState::Disposed)
    }

    /// The fiber's plugin context.
    pub fn ctx(&self) -> Context {
        self.app
            .fibers
            .borrow()
            .get(&self.id)
            .map(|d| d.ctx.clone())
            .expect("fiber data")
    }

    /// The plugin's display name, inherited from the nearest named ancestor,
    /// else `"root"` (upstream `Fiber.name`).
    pub fn name(&self) -> String {
        let fibers = self.app.fibers.borrow();
        let runtimes = self.app.runtimes.borrow();
        let mut current = self.id;
        loop {
            let Some(data) = fibers.get(&current) else {
                return "root".into();
            };
            if let Some(key) = data.runtime {
                if let Some(name) = runtimes.get(&key).and_then(|r| r.name.clone()) {
                    return name;
                }
            }
            match data.parent {
                Some(parent) => current = parent,
                None => return "root".into(),
            }
        }
    }

    /// Startup error, if the last activation failed.
    pub fn error(&self) -> Option<CordisError> {
        self.app
            .fibers
            .borrow()
            .get(&self.id)
            .and_then(|d| d.error.clone())
    }

    /// Wait for current lifecycle work and rethrow startup errors
    /// (upstream `Fiber.await`).
    pub async fn await_ready(&self) -> Result<()> {
        loop {
            let inertia = {
                let fibers = self.app.fibers.borrow();
                fibers.get(&self.id).and_then(|d| d.inertia.clone())
            };
            match inertia {
                Some(task) => task.await,
                None => break,
            }
        }
        match self.error() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// Dispose: unload the plugin, then settle once cleanup finished
    /// (upstream `Fiber.dispose`).
    pub async fn dispose(&self) {
        let already = {
            let fibers = self.app.fibers.borrow();
            fibers
                .get(&self.id)
                .map(|d| d.uid.is_none())
                .unwrap_or(true)
        };
        if !already {
            {
                let mut fibers = self.app.fibers.borrow_mut();
                if let Some(data) = fibers.get_mut(&self.id) {
                    data.uid = None;
                }
            }
            let ctx = self.ctx();
            ctx.emit::<InternalPlugin>(self);
            // Remove from the runtime record; drop the record when empty.
            let key = {
                let fibers = self.app.fibers.borrow();
                fibers.get(&self.id).and_then(|d| d.runtime)
            };
            if let Some(key) = key {
                let mut runtimes = self.app.runtimes.borrow_mut();
                if let Some(runtime) = runtimes.get_mut(&key) {
                    runtime.fibers.retain(|id| *id != self.id);
                    if runtime.fibers.is_empty() {
                        runtimes.remove(&key);
                    }
                }
            }
            set_epoch(&self.app, self.id, Epoch::Inactive);
            // A PENDING fiber can already own effects registered by an
            // internal/plugin observer; drain them before reporting disposal.
            let needs_drain = {
                let fibers = self.app.fibers.borrow();
                let data = fibers.get(&self.id).expect("fiber data");
                data.inertia.is_none() && !data.disposables.is_empty()
            };
            if needs_drain {
                start_transition(&self.app, self.id, FiberState::Unloading);
            }
        }
        // Await every in-flight transition.
        loop {
            let inertia = {
                let fibers = self.app.fibers.borrow();
                fibers.get(&self.id).and_then(|d| d.inertia.clone())
            };
            match inertia {
                Some(task) => task.await,
                None => break,
            }
        }
        let mut fibers = self.app.fibers.borrow_mut();
        if let Some(data) = fibers.get_mut(&self.id) {
            data.state = FiberState::Disposed;
        }
    }

    /// Dispose and immediately reload with the current config
    /// (upstream `Fiber.restart`).
    pub async fn restart(&self) -> Result<()> {
        {
            let fibers = self.app.fibers.borrow();
            let data = fibers.get(&self.id).ok_or(CordisError::InactiveEffect)?;
            if data.uid.is_none() {
                return Err(CordisError::InactiveEffect);
            }
        }
        set_epoch(&self.app, self.id, Epoch::Inactive);
        refresh(&self.app, self.id);
        self.await_ready().await
    }

    /// Validate and apply new config, then restart through the
    /// `internal/update` waterfall (upstream `Fiber.update`).
    pub async fn update(&self, config: Value, no_save: bool) -> Result<()> {
        {
            let mut fibers = self.app.fibers.borrow_mut();
            let data = fibers
                .get_mut(&self.id)
                .ok_or(CordisError::InactiveEffect)?;
            if data.uid.is_none() {
                return Err(CordisError::InactiveEffect);
            }
            data.config_raw = config.clone();
            if data.state != FiberState::Active {
                // Config resolution may access injected services; defer it
                // until the fiber can activate.
                data.error = None;
                drop(fibers);
                set_epoch(&self.app, self.id, Epoch::Inactive);
                refresh(&self.app, self.id);
                return Ok(());
            }
        }
        let ctx = self.ctx();
        let this = self.clone();
        ctx.waterfall::<InternalUpdate, _, _>((config, no_save), move |(config, _)| async move {
            {
                let mut fibers = this.app.fibers.borrow_mut();
                if let Some(data) = fibers.get_mut(&this.id) {
                    data.config = config;
                    data.error = None;
                }
            }
            this.restart().await?;
            Ok(())
        })
        .await
        .map_err(|e| CordisError::plugin(e))
    }
}

/// Re-run the availability check for one injected service on one fiber
/// (upstream `Fiber._checkImpl`).
pub(crate) fn check_impl(app: &Rc<AppState>, id: FiberId, name: &str) {
    let record = {
        let fibers = app.fibers.borrow();
        let Some(data) = fibers.get(&id) else { return };
        let label = data.ctx.isolate_lookup(name);
        label.and_then(|label| app.store.borrow().get(&label).cloned())
    };
    let Some(record) = record else {
        let mut fibers = app.fibers.borrow_mut();
        if let Some(data) = fibers.get_mut(&id) {
            data.live_store.remove(name);
        }
        return;
    };
    if let Some(check) = record.check.clone() {
        let provider_ctx = {
            let fibers = app.fibers.borrow();
            fibers.get(&record.fiber).map(|d| d.ctx.clone())
        };
        let ok = provider_ctx.map(|ctx| check(&ctx)).unwrap_or(false);
        if !ok {
            let mut fibers = app.fibers.borrow_mut();
            if let Some(data) = fibers.get_mut(&id) {
                data.live_store.remove(name);
            }
            return;
        }
    }
    let mut fibers = app.fibers.borrow_mut();
    if let Some(data) = fibers.get_mut(&id) {
        data.live_store.insert(name.to_string(), record);
    }
}

/// Recompute the epoch from the live dependency store and drive transitions
/// (upstream `Fiber._refresh`).
pub(crate) fn refresh(app: &Rc<AppState>, id: FiberId) {
    let epoch = {
        let fibers = app.fibers.borrow();
        let Some(data) = fibers.get(&id) else { return };
        let mut epoch = String::new();
        let mut inactive = false;
        for name in data.inject.keys() {
            match data.live_store.get(name) {
                Some(record) => {
                    let provider_uid = fibers
                        .get(&record.fiber)
                        .and_then(|d| d.uid)
                        .unwrap_or(u64::MAX);
                    epoch.push(':');
                    epoch.push_str(&provider_uid.to_string());
                }
                None => {
                    inactive = true;
                    break;
                }
            }
        }
        if inactive {
            Epoch::Inactive
        } else {
            Epoch::Active(epoch)
        }
    };
    set_epoch(app, id, epoch);
}

/// Apply a new epoch; start a load/unload transition when the activation
/// boundary is crossed and no transition is in flight
/// (upstream `Fiber._setEpoch`).
pub(crate) fn set_epoch(app: &Rc<AppState>, id: FiberId, epoch: Epoch) {
    let start = {
        let mut fibers = app.fibers.borrow_mut();
        let Some(data) = fibers.get_mut(&id) else {
            return;
        };
        if data.epoch == epoch {
            return;
        }
        let old = std::mem::replace(&mut data.epoch, epoch.clone());
        if data.inertia.is_some() {
            None
        } else if epoch != Epoch::Inactive && old == Epoch::Inactive {
            Some(FiberState::Loading)
        } else {
            Some(FiberState::Unloading)
        }
    };
    if let Some(state) = start {
        start_transition(app, id, state);
    }
}

fn update_state(app: &Rc<AppState>, id: FiberId, new_state: FiberState) {
    let (old, fiber_ctx) = {
        let mut fibers = app.fibers.borrow_mut();
        let Some(data) = fibers.get_mut(&id) else {
            return;
        };
        let old = std::mem::replace(&mut data.state, new_state);
        (old, data.ctx.clone())
    };
    if old == new_state {
        return;
    }
    let fiber = Fiber {
        app: app.clone(),
        id,
    };
    fiber_ctx.emit::<InternalStatus>(&(fiber, old));

    // Only notify on changes crossing the ACTIVE boundary (upstream).
    if (old == FiberState::Active) == (new_state == FiberState::Active) {
        return;
    }
    let names: Vec<String> = {
        let store = app.store.borrow();
        store
            .values()
            .filter(|record| record.fiber == id)
            .map(|record| record.name.clone())
            .collect()
    };
    if !names.is_empty() {
        notify(&fiber_ctx, &names);
    }
}

/// Compute the settled state from current data (upstream `Fiber._getState`).
fn settled_state(app: &Rc<AppState>, id: FiberId) -> FiberState {
    let fibers = app.fibers.borrow();
    let Some(data) = fibers.get(&id) else {
        return FiberState::Disposed;
    };
    if data.uid.is_none() {
        return FiberState::Disposed;
    }
    if data.error.is_some() {
        return FiberState::Failed;
    }
    if data.epoch != Epoch::Inactive {
        return FiberState::Active;
    }
    FiberState::Pending
}

/// Spawn the load/unload driver for a fiber and store its shared completion
/// as the fiber's inertia (upstream `_reload`/`_unload` recursion).
fn start_transition(app: &Rc<AppState>, id: FiberId, initial: FiberState) {
    let (tx, rx) = futures::channel::oneshot::channel::<()>();
    let inertia: Inertia = async move {
        let _ = rx.await;
    }
    .boxed_local()
    .shared();
    {
        let mut fibers = app.fibers.borrow_mut();
        let Some(data) = fibers.get_mut(&id) else {
            return;
        };
        data.inertia = Some(inertia);
    }
    update_state(app, id, initial);
    let app = app.clone();
    tokio::task::spawn_local(async move {
        let mut phase = initial;
        loop {
            match phase {
                FiberState::Loading => do_load(&app, id).await,
                FiberState::Unloading => do_unload(&app, id).await,
                _ => break,
            }
            // Decide the next phase: the epoch may have flipped mid-flight.
            let next = {
                let fibers = app.fibers.borrow();
                let Some(data) = fibers.get(&id) else { break };
                match (&phase, &data.epoch) {
                    (FiberState::Loading, Epoch::Inactive) => Some(FiberState::Unloading),
                    (FiberState::Loading, Epoch::Active(_)) => None,
                    (FiberState::Unloading, Epoch::Inactive) => None,
                    (FiberState::Unloading, Epoch::Active(_)) => Some(FiberState::Loading),
                    _ => None,
                }
            };
            match next {
                Some(next_phase) => {
                    phase = next_phase;
                    update_state(&app, id, next_phase);
                }
                None => break,
            }
        }
        {
            let mut fibers = app.fibers.borrow_mut();
            if let Some(data) = fibers.get_mut(&id) {
                data.inertia = None;
            }
        }
        let settled = settled_state(&app, id);
        update_state(&app, id, settled);
        let _ = tx.send(());
    });
}

async fn do_load(app: &Rc<AppState>, id: FiberId) {
    // Snapshot the live store and capture the expected epoch.
    let (expected, ctx, key, config_raw) = {
        let mut fibers = app.fibers.borrow_mut();
        let Some(data) = fibers.get_mut(&id) else {
            return;
        };
        data.store_snapshot = Some(data.live_store.clone());
        (
            data.epoch.clone(),
            data.ctx.clone(),
            data.runtime,
            data.config_raw.clone(),
        )
    };
    // Yield once so a disposer queued before this checkpoint can invalidate
    // the load (upstream `await Promise.resolve()`).
    tokio::task::yield_now().await;
    let still_expected = {
        let fibers = app.fibers.borrow();
        fibers
            .get(&id)
            .map(|d| d.epoch == expected)
            .unwrap_or(false)
    };
    if still_expected {
        let plugin = key.and_then(|key| {
            let runtimes = app.runtimes.borrow();
            runtimes.get(&key).map(|r| r.plugin.clone())
        });
        let Some(plugin) = plugin else { return };
        let result: anyhow::Result<()> = async {
            let config = ctx
                .waterfall::<InternalConfig, _, _>(config_raw, |config| async move { Ok(config) })
                .await?;
            let config = plugin
                .validate_config(config)
                .map_err(anyhow::Error::from)?;
            {
                let mut fibers = app.fibers.borrow_mut();
                if let Some(data) = fibers.get_mut(&id) {
                    data.config = config.clone();
                }
            }
            plugin.apply(ctx.clone(), config).await
        }
        .await;
        match result {
            Ok(()) => {
                let mut fibers = app.fibers.borrow_mut();
                if let Some(data) = fibers.get_mut(&id) {
                    data.error = None;
                }
            }
            Err(error) => {
                tracing::error!(fiber = %(Fiber { app: app.clone(), id }).name(), "plugin failed: {error:#}");
                let mut fibers = app.fibers.borrow_mut();
                if let Some(data) = fibers.get_mut(&id) {
                    data.error = Some(CordisError::plugin(error));
                    data.epoch = Epoch::Inactive;
                }
            }
        }
    }
}

async fn do_unload(app: &Rc<AppState>, id: FiberId) {
    let handles = {
        let mut fibers = app.fibers.borrow_mut();
        let Some(data) = fibers.get_mut(&id) else {
            return;
        };
        data.disposables.clear()
    };
    for handle in handles {
        handle.dispose().await;
    }
    let mut fibers = app.fibers.borrow_mut();
    if let Some(data) = fibers.get_mut(&id) {
        data.store_snapshot = None;
    }
}

/// Re-evaluate every fiber that injects one of the given services
/// (upstream `ReflectService.notify`). Returns the refreshed fiber ids.
pub(crate) fn notify(ctx: &Context, names: &[String]) -> Vec<FiberId> {
    let app = ctx.app_rc();
    let candidates: Vec<FiberId> = {
        let runtimes = app.runtimes.borrow();
        runtimes
            .values()
            .flat_map(|r| r.fibers.iter().copied())
            .collect()
    };
    let mut refreshed = Vec::new();
    for id in candidates {
        let matching: Vec<String> = {
            let fibers = app.fibers.borrow();
            let Some(data) = fibers.get(&id) else {
                continue;
            };
            names
                .iter()
                .filter(|name| data.inject.contains_key(*name))
                .filter(|name| {
                    // Scope filter: the dependent resolves the name to the
                    // same label as the notifying context.
                    data.ctx.isolate_lookup(name) == ctx.isolate_lookup(name)
                })
                .cloned()
                .collect()
        };
        if matching.is_empty() {
            continue;
        }
        for name in &matching {
            check_impl(&app, id, name);
        }
        refresh(&app, id);
        refreshed.push(id);
    }
    for name in names {
        ctx.emit::<InternalService>(name);
    }
    refreshed
}
