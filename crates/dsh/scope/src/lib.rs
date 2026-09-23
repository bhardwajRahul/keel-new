//! Rust port of `packages/core/scope`: mint a Cordis context tagged with an
//! opaque scope identity, walk scope parent chains, and route events so that
//! a listener tagged with a scope (or any of its ancestors) receives only
//! that scope's dispatches while untagged listeners stay global.
//!
//! Divergences from the TypeScript package (contract-level; forced by the
//! Rust host or the dsh-cordis port):
//! - [`ScopeKey`] is a minted handle compared by identity, not "any object";
//!   its parent link lives inside the key instead of a module-level WeakMap,
//!   so bind/rebind/cycle checks need no global state.
//! - dsh-cordis contexts have no dynamic properties (`ctx.extend`), so the
//!   scope tag rides the context's intercept chain under a reserved name and
//!   is inherited by every fiber loaded below the scoped context. A small
//!   thread-local table maps tag ids back to keys (entries live for the
//!   thread; scopes are long-lived agents).
//! - The Rust `emit` has no receiver parameter (`Context.filter` is not
//!   ported), so scope-filtered dispatch is library-owned: dispatch with
//!   [`ScopedEvents::emit_scoped`] and listen with [`ScopedEvents::on_scoped`].
//!   A plain `ctx.on::<E>` listener does not observe scoped dispatches of `E`;
//!   an `on_scoped` listener on an untagged context has global semantics, and
//!   `EventOptions::global` skips filtering entirely, as upstream.
//! - [`scope_target`] cannot read a filter off an arbitrary base object;
//!   [`scope_target_filtered`] takes the base filter as a closure. The
//!   `Scoped<T>` carrier keeps the subject brand as a type parameter; a
//!   concrete type replaces upstream's WeakMap-backed `isScopeCarrier` /
//!   `carrierKeyOf` (use [`Scoped::key`]).
//! - `bindScopeParent` throws become `Result`s.
//! - `raw_dispose` returns the backing fiber's disposal future (cordis-rs
//!   disposers are values, not identity-compared functions); `dispose`
//!   memoizes one shared quiescent teardown, as upstream.
//! - The generated `scoped-events` subject-resolver table is not ported (the
//!   harness event catalog does not exist in this workspace yet); see
//!   [`invariant`] for the companion consequences.

pub mod invariant;
mod store;

pub use store::{AnonymousEntries, NamedEntries, ScopeLayer, ScopedLayers, Undo};

use dsh_cordis::{Context, EffectHandle, Event, EventOptions, Fiber, Inject, plugin_fn};
use futures::FutureExt;
use futures::future::{LocalBoxFuture, Shared};
use serde_json::Value;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::rc::Rc;

/// Reserved intercept name carrying the scope tag down the fiber tree.
const SCOPE_TAG: &str = "dsh.scope";

thread_local! {
    static NEXT_KEY: Cell<u64> = const { Cell::new(0) };
    /// Tag id → key, for contexts minted by [`create_scope`].
    /// ponytail: entries live for the thread; add reclamation if scope churn
    /// ever makes this map matter.
    static MINTED: RefCell<HashMap<u64, ScopeKey>> = RefCell::new(HashMap::new());
}

struct KeyInner {
    id: u64,
    parent: RefCell<Option<ScopeKey>>,
}

/// An opaque, identity-compared scope key.
#[derive(Clone)]
pub struct ScopeKey(Rc<KeyInner>);

impl ScopeKey {
    /// Mint a fresh identity.
    pub fn new() -> ScopeKey {
        let id = NEXT_KEY.with(|next| {
            let id = next.get() + 1;
            next.set(id);
            id
        });
        ScopeKey(Rc::new(KeyInner {
            id,
            parent: RefCell::new(None),
        }))
    }
}

impl Default for ScopeKey {
    fn default() -> Self {
        ScopeKey::new()
    }
}

impl PartialEq for ScopeKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.id == other.0.id
    }
}
impl Eq for ScopeKey {}

impl std::hash::Hash for ScopeKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.id.hash(state);
    }
}

impl std::fmt::Debug for ScopeKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ScopeKey({})", self.0.id)
    }
}

/// Cycle-checked write shared by the bind and every rebind.
fn link_scope_parent(key: &ScopeKey, parent: &ScopeKey) -> anyhow::Result<()> {
    let mut cursor = Some(parent.clone());
    while let Some(current) = cursor {
        if current == *key {
            anyhow::bail!("dsh-scope: scope parent link would form a cycle");
        }
        cursor = current.0.parent.borrow().clone();
    }
    *key.0.parent.borrow_mut() = Some(parent.clone());
    Ok(())
}

/// The privileged handle to move one scope key's parent link. Only the
/// original binder receives it, so a scope's ancestry cannot be moved from
/// the outside.
pub struct ScopeParentBinding {
    key: ScopeKey,
}

impl ScopeParentBinding {
    /// Re-link the bound key to a different parent, with the same cycle check
    /// as the bind. Valid only while nothing produced under the old parent is
    /// retained (the blank-session recompose contract, upheld by the holder).
    pub fn rebind(&self, parent: &ScopeKey) -> anyhow::Result<()> {
        link_scope_parent(&self.key, parent)
    }
}

/// Bind `parent` as `key`'s enclosing scope, once. A key that already has a
/// parent is rejected — re-linking requires the returned binding — and a
/// link that would close a cycle is rejected because every chain consumer
/// walks parents to the root.
pub fn bind_scope_parent(key: &ScopeKey, parent: &ScopeKey) -> anyhow::Result<ScopeParentBinding> {
    if key.0.parent.borrow().is_some() {
        anyhow::bail!(
            "dsh-scope: scope key is already bound to a parent; re-linking requires the binding returned by the original bind"
        );
    }
    link_scope_parent(key, parent)?;
    Ok(ScopeParentBinding { key: key.clone() })
}

/// Read one key's enclosing scope (`None` for a root scope).
pub fn scope_parent_of(key: &ScopeKey) -> Option<ScopeKey> {
    key.0.parent.borrow().clone()
}

/// The chain from a key to its root ancestor, nearest first:
/// `[key, parent, grandparent, …]`; `None` yields the empty chain.
pub fn scope_chain_of(key: Option<&ScopeKey>) -> Vec<ScopeKey> {
    let mut chain = Vec::new();
    let mut cursor = key.cloned();
    while let Some(current) = cursor {
        cursor = scope_parent_of(&current);
        chain.push(current);
    }
    chain
}

/// A minted registration scope and its quiescent disposal boundaries.
pub struct Scope {
    ctx: Context,
    fiber: Fiber,
    disposing: RefCell<Option<Shared<LocalBoxFuture<'static, ()>>>>,
}

impl Scope {
    /// Context through which scope-owned registrations are made.
    pub fn ctx(&self) -> Context {
        self.ctx.clone()
    }

    /// The backing no-op fiber.
    pub fn fiber(&self) -> Fiber {
        self.fiber.clone()
    }

    /// The raw disposal future, for nesting this scope in an ordered
    /// composite effect (upstream's exact Cordis disposer).
    pub fn raw_dispose(&self) -> LocalBoxFuture<'static, ()> {
        let fiber = self.fiber.clone();
        async move { fiber.dispose().await }.boxed_local()
    }

    /// Dispose every scope-owned registration; racing calls await the same
    /// completion.
    pub fn dispose(&self) -> impl std::future::Future<Output = ()> + 'static {
        let shared = self
            .disposing
            .borrow_mut()
            .get_or_insert_with(|| self.raw_dispose().shared())
            .clone();
        async move { shared.await }
    }
}

/// Mint a scope under `ctx`. The scoped context inherits the minting
/// context's dependency API and owns every registration made through it;
/// `parent` (upstream `options.parent`) binds the enclosing scope before the
/// scope is usable, keeping the binding internal.
pub fn create_scope(
    ctx: &Context,
    key: ScopeKey,
    parent: Option<&ScopeKey>,
) -> anyhow::Result<Scope> {
    if let Some(parent) = parent {
        bind_scope_parent(&key, parent)?;
    }
    let fiber = ctx.plugin(
        Rc::new(plugin_fn(
            "scope",
            Inject::default(),
            |_ctx, _config| async { Ok(()) },
        )),
        Value::Null,
    )?;
    MINTED.with(|minted| minted.borrow_mut().insert(key.0.id, key.clone()));
    let scoped = fiber
        .ctx()
        .intercept(SCOPE_TAG, serde_json::json!(key.0.id));
    Ok(Scope {
        ctx: scoped,
        fiber,
        disposing: RefCell::new(None),
    })
}

/// Read the nearest scope tag inherited by a context (`None` when unscoped).
pub fn scope_of(ctx: &Context) -> Option<ScopeKey> {
    let configs = ctx.intercept_configs(SCOPE_TAG);
    let id = configs.last()?.as_u64()?;
    MINTED.with(|minted| minted.borrow().get(&id).cloned())
}

#[derive(Clone)]
struct CarrierData {
    key: Option<ScopeKey>,
    base_filter: Option<Rc<dyn Fn(&Context) -> bool>>,
}

impl CarrierData {
    /// Preserve the base filter, admit untagged listeners globally, and admit
    /// tagged listeners for a matching key or any of its ancestors. A tag
    /// BELOW the dispatch key stays excluded — events flow up the chain,
    /// never down.
    fn admits(&self, listener_ctx: &Context) -> bool {
        if let Some(filter) = &self.base_filter {
            if !filter(listener_ctx) {
                return false;
            }
        }
        let Some(tag) = scope_of(listener_ctx) else {
            return true;
        };
        let mut cursor = self.key.clone();
        while let Some(current) = cursor {
            if current == tag {
                return true;
            }
            cursor = scope_parent_of(&current);
        }
        false
    }
}

/// A routing-only event receiver built by [`scope_target`]. The type
/// parameter records the subject type for dispatch checking; the carrier
/// never exposes the subject — event payloads carry the real subject.
pub struct Scoped<T: 'static = ()> {
    data: CarrierData,
    _subject: PhantomData<fn() -> T>,
}

impl<T: 'static> Clone for Scoped<T> {
    fn clone(&self) -> Self {
        Scoped {
            data: self.data.clone(),
            _subject: PhantomData,
        }
    }
}

impl<T: 'static> Scoped<T> {
    /// The carrier's routing key (`None` for an unscoped subject).
    pub fn key(&self) -> Option<&ScopeKey> {
        self.data.key.as_ref()
    }

    /// Whether this carrier admits a listener registered on `listener_ctx`.
    pub fn admits(&self, listener_ctx: &Context) -> bool {
        self.data.admits(listener_ctx)
    }

    fn erased(&self) -> Scoped<()> {
        Scoped {
            data: self.data.clone(),
            _subject: PhantomData,
        }
    }
}

/// Build an opaque receiver routing to `key` (or globally for `None`).
pub fn scope_target<T: 'static>(key: Option<&ScopeKey>) -> Scoped<T> {
    Scoped {
        data: CarrierData {
            key: key.cloned(),
            base_filter: None,
        },
        _subject: PhantomData,
    }
}

/// [`scope_target`] with a preserved base filter, consulted with each
/// listener's registration context before scope admission (upstream reads
/// the filter off the base object; Rust takes it as a closure).
pub fn scope_target_filtered<T: 'static>(
    filter: impl Fn(&Context) -> bool + 'static,
    key: Option<&ScopeKey>,
) -> Scoped<T> {
    Scoped {
        data: CarrierData {
            key: key.cloned(),
            base_filter: Some(Rc::new(filter)),
        },
        _subject: PhantomData,
    }
}

/// Carrier-tagged twin of an event `E`, dispatched by
/// [`ScopedEvents::emit_scoped`]. Shares `E`'s name; the distinct type keeps
/// its hooks separate from plain `E` listeners.
pub struct ScopedEvent<E>(PhantomData<E>);

impl<E: Event> Event for ScopedEvent<E> {
    const NAME: &'static str = E::NAME;
    type Args = (Scoped<()>, E::Args);
    type Ret = E::Ret;
}

/// Scope-filtered dispatch over the cordis event bus.
pub trait ScopedEvents {
    /// Dispatch `E` with a scope carrier; only listeners the carrier admits
    /// run (fire-and-forget, like `ctx.emit`).
    fn emit_scoped<E: Event, T: 'static>(&self, carrier: &Scoped<T>, args: E::Args);

    /// Register a listener for scoped dispatches of `E`. The listener's
    /// admission tag is this context's scope (untagged ⇒ global);
    /// `options.global` receives every dispatch regardless of filters.
    fn on_scoped<E, F, Fut>(
        &self,
        options: EventOptions,
        listener: F,
    ) -> dsh_cordis::Result<EffectHandle>
    where
        E: Event,
        F: Fn(&Context, &E::Args) -> Fut + 'static,
        Fut: std::future::Future<Output = Option<E::Ret>> + 'static;
}

impl ScopedEvents for Context {
    fn emit_scoped<E: Event, T: 'static>(&self, carrier: &Scoped<T>, args: E::Args) {
        self.emit::<ScopedEvent<E>>(&(carrier.erased(), args));
    }

    fn on_scoped<E, F, Fut>(
        &self,
        options: EventOptions,
        listener: F,
    ) -> dsh_cordis::Result<EffectHandle>
    where
        E: Event,
        F: Fn(&Context, &E::Args) -> Fut + 'static,
        Fut: std::future::Future<Output = Option<E::Ret>> + 'static,
    {
        let registered_ctx = self.clone();
        self.on::<ScopedEvent<E>, _, _>(options, move |ctx, payload| {
            let (carrier, args) = payload;
            let admitted = options.global || carrier.admits(&registered_ctx);
            let inner = if admitted {
                Some(listener(ctx, args))
            } else {
                None
            };
            async move {
                match inner {
                    Some(future) => future.await,
                    None => None,
                }
            }
        })
    }
}
