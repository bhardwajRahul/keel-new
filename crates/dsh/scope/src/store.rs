//! Shared insertion-ordered storage and effect ownership for scope-aware
//! registries (port of upstream `src/store.ts`).
//!
//! Divergences: entry tables use interior mutability and hand back boxed
//! idempotent undos; iteration returns snapshots instead of upstream's live
//! generation-scoped JS Map iterators; the merged view is an ordered
//! `Vec<(String, V)>` (no ordered map in the workspace); layer factories and
//! change notifications are fallible `Result`s instead of throwing.

use crate::{ScopeKey, scope_chain_of, scope_of};
use dsh_cordis::{Context, CordisError, Disposer, Effect, EffectHandle};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

/// Idempotent undo removing exactly one insertion.
pub type Undo = Box<dyn Fn()>;

/// One scope's aggregate contribution to a registry.
pub trait ScopeLayer {
    /// Whether every table in this layer is empty.
    fn is_empty(&self) -> bool;
}

/// Insertion-ordered named entries with caller-owned duplicate diagnostics.
///
/// Each successful insertion returns an idempotent undo for that exact
/// entry: a spent undo never removes a later insertion under the same name.
pub struct NamedEntries<V> {
    // ponytail: Vec-backed linear scan; registries hold tens of entries.
    data: Rc<RefCell<Vec<(String, V)>>>,
    duplicate_error: Rc<dyn Fn(&str) -> anyhow::Error>,
}

impl<V: 'static> NamedEntries<V> {
    /// Build a table whose duplicate-name error the caller owns.
    pub fn new(duplicate_error: impl Fn(&str) -> anyhow::Error + 'static) -> Self {
        NamedEntries {
            data: Rc::default(),
            duplicate_error: Rc::new(duplicate_error),
        }
    }

    /// Insert one unique name, returning its idempotent undo.
    pub fn insert(&self, name: impl Into<String>, value: V) -> anyhow::Result<Undo> {
        let name = name.into();
        {
            let mut data = self.data.borrow_mut();
            if data.iter().any(|(existing, _)| *existing == name) {
                return Err((self.duplicate_error)(&name));
            }
            data.push((name.clone(), value));
        }
        let data = self.data.clone();
        let active = Cell::new(true);
        Ok(Box::new(move || {
            if !active.get() {
                return;
            }
            active.set(false);
            data.borrow_mut().retain(|(existing, _)| *existing != name);
        }))
    }

    /// Read one named value.
    pub fn get(&self, name: &str) -> Option<V>
    where
        V: Clone,
    {
        self.data
            .borrow()
            .iter()
            .find(|(existing, _)| existing == name)
            .map(|(_, value)| value.clone())
    }

    /// Test one name for membership.
    pub fn has(&self, name: &str) -> bool {
        self.data
            .borrow()
            .iter()
            .any(|(existing, _)| existing == name)
    }

    /// Names in insertion order (snapshot).
    pub fn keys(&self) -> Vec<String> {
        self.data
            .borrow()
            .iter()
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Entries in insertion order (snapshot).
    pub fn entries(&self) -> Vec<(String, V)>
    where
        V: Clone,
    {
        self.data.borrow().clone()
    }

    /// Values in insertion order (snapshot).
    pub fn values(&self) -> Vec<V>
    where
        V: Clone,
    {
        self.data
            .borrow()
            .iter()
            .map(|(_, value)| value.clone())
            .collect()
    }

    /// Whether this table has no entries.
    pub fn is_empty(&self) -> bool {
        self.data.borrow().is_empty()
    }
}

struct AnonInner<V> {
    next: u64,
    items: Vec<(u64, V)>,
}

/// Insertion-ordered anonymous entries with independent registration
/// identity: equal values remain separate registrations, and each append's
/// undo removes only that append.
pub struct AnonymousEntries<V> {
    data: Rc<RefCell<AnonInner<V>>>,
}

impl<V: 'static> Default for AnonymousEntries<V> {
    fn default() -> Self {
        AnonymousEntries::new()
    }
}

impl<V: 'static> AnonymousEntries<V> {
    pub fn new() -> Self {
        AnonymousEntries {
            data: Rc::new(RefCell::new(AnonInner {
                next: 0,
                items: Vec::new(),
            })),
        }
    }

    /// Append one independently owned value, returning its idempotent undo.
    pub fn append(&self, value: V) -> Undo {
        let id = {
            let mut data = self.data.borrow_mut();
            data.next += 1;
            let id = data.next;
            data.items.push((id, value));
            id
        };
        let data = self.data.clone();
        let active = Cell::new(true);
        Box::new(move || {
            if !active.get() {
                return;
            }
            active.set(false);
            data.borrow_mut()
                .items
                .retain(|(existing, _)| *existing != id);
        })
    }

    /// Values in insertion order (snapshot).
    pub fn values(&self) -> Vec<V>
    where
        V: Clone,
    {
        self.data
            .borrow()
            .items
            .iter()
            .map(|(_, value)| value.clone())
            .collect()
    }

    /// Whether this table has no entries.
    pub fn is_empty(&self) -> bool {
        self.data.borrow().items.is_empty()
    }
}

/// Owns the global and exact-scope layers for one registry.
///
/// Reads never create scoped layers. Registrations derive both visibility
/// and effect ownership from the supplied Cordis context, collect undo
/// before notification, and reclaim only a completely empty aggregate layer.
pub struct ScopedLayers<L: ScopeLayer + 'static> {
    global: Rc<L>,
    scoped: Rc<RefCell<HashMap<ScopeKey, Rc<L>>>>,
    create_layer: Box<dyn Fn(Option<&ScopeKey>) -> anyhow::Result<L>>,
    on_change: Rc<dyn Fn() -> anyhow::Result<()>>,
}

impl<L: ScopeLayer + 'static> ScopedLayers<L> {
    /// Build the registry storage; the context-global layer is constructed
    /// eagerly.
    pub fn new(
        create_layer: impl Fn(Option<&ScopeKey>) -> anyhow::Result<L> + 'static,
        on_change: impl Fn() -> anyhow::Result<()> + 'static,
    ) -> anyhow::Result<Self> {
        let global = Rc::new(create_layer(None)?);
        Ok(ScopedLayers {
            global,
            scoped: Rc::default(),
            create_layer: Box::new(create_layer),
            on_change: Rc::new(on_change),
        })
    }

    /// The eagerly constructed context-global layer.
    pub fn global(&self) -> &L {
        &self.global
    }

    /// Read an existing exact-scope overlay without creating one.
    /// Deliberately chain-blind: callers addressing one scope's OWN
    /// contributions must not silently pick up an ancestor's — use
    /// [`ScopedLayers::chain_layers`] where inheritance is the point.
    pub fn peek(&self, scope: Option<&ScopeKey>) -> Option<Rc<L>> {
        self.scoped.borrow().get(scope?).cloned()
    }

    /// Existing overlays along the scope's parent chain, farthest ancestor
    /// first and the exact scope last, so layering in order gives the
    /// nearest scope the final word. Absent overlays are skipped.
    pub fn chain_layers(&self, scope: Option<&ScopeKey>) -> Vec<Rc<L>> {
        let map = self.scoped.borrow();
        let mut chain = scope_chain_of(scope);
        chain.reverse();
        chain
            .iter()
            .filter_map(|key| map.get(key).cloned())
            .collect()
    }

    /// Materialize global named entries followed by scope-chain shadows,
    /// farthest ancestor first, so the nearest scope's entry wins a name;
    /// a shadow keeps the original insertion position.
    pub fn merge<V: Clone + 'static>(
        &self,
        scope: Option<&ScopeKey>,
        pick: impl Fn(&L) -> &NamedEntries<V>,
    ) -> Vec<(String, V)> {
        let mut merged = pick(&self.global).entries();
        for layer in self.chain_layers(scope) {
            for (name, value) in pick(&layer).entries() {
                match merged.iter_mut().find(|(existing, _)| *existing == name) {
                    Some(slot) => slot.1 = value,
                    None => merged.push((name, value)),
                }
            }
        }
        merged
    }

    /// Attach one synchronous layer mutation to its registration context:
    /// the context determines both scope visibility and effect ownership,
    /// `action` returns its undo, and the returned handle is the exact
    /// cordis effect disposer. A failed action or notification rolls the
    /// mutation back and reclaims a just-created empty layer.
    pub fn effect(
        &self,
        ctx: &Context,
        action: impl FnOnce(&L) -> anyhow::Result<Undo>,
        label: &str,
        notify: bool,
    ) -> anyhow::Result<EffectHandle> {
        let scope = scope_of(ctx);
        let handle = ctx.effect_labeled(label, |_| {
            let (layer, created) = match &scope {
                None => (self.global.clone(), false),
                Some(key) => {
                    let existing = self.scoped.borrow().get(key).cloned();
                    match existing {
                        Some(layer) => (layer, false),
                        None => {
                            let layer = Rc::new(
                                (self.create_layer)(Some(key)).map_err(CordisError::plugin)?,
                            );
                            self.scoped.borrow_mut().insert(key.clone(), layer.clone());
                            (layer, true)
                        }
                    }
                }
            };
            let undo = match action(&layer) {
                Ok(undo) => undo,
                Err(error) => {
                    if let Some(key) = &scope {
                        if created && layer.is_empty() {
                            self.scoped.borrow_mut().remove(key);
                        }
                    }
                    return Err(CordisError::plugin(error));
                }
            };
            let rollback: Rc<dyn Fn()> = {
                let scope = scope.clone();
                let scoped = self.scoped.clone();
                let on_change = self.on_change.clone();
                Rc::new(move || {
                    undo();
                    if let Some(key) = &scope {
                        if layer.is_empty() {
                            scoped.borrow_mut().remove(key);
                        }
                    }
                    if notify {
                        let _ = on_change();
                    }
                })
            };
            if notify {
                if let Err(error) = (self.on_change)() {
                    rollback();
                    return Err(CordisError::plugin(error));
                }
            }
            Ok(Effect::One(Disposer::sync(move || rollback())))
        })?;
        Ok(handle)
    }
}
