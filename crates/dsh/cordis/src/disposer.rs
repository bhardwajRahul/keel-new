//! Disposers, effects, and the ordered disposable list, ported from
//! `vendor/cordis/src/utils.ts` (`DisposableList`) and `fiber.ts` (`Effect`).
//!
//! Registrations are effects: every contribution returns a disposer, and
//! disposers run in reverse registration order when the owning fiber unloads.

use futures::future::LocalBoxFuture;
use std::collections::BTreeMap;

/// Cleanup function released when an effect or fiber is disposed.
///
/// Sync disposers run inline; async disposers are awaited during unload,
/// matching upstream's promise-returning disposers.
pub enum Disposer {
    Sync(Box<dyn FnOnce()>),
    Async(Box<dyn FnOnce() -> LocalBoxFuture<'static, ()>>),
}

impl Disposer {
    /// Wrap a synchronous cleanup closure.
    pub fn sync(f: impl FnOnce() + 'static) -> Self {
        Disposer::Sync(Box::new(f))
    }

    /// Wrap an asynchronous cleanup closure.
    pub fn asynchronous<Fut>(f: impl FnOnce() -> Fut + 'static) -> Self
    where
        Fut: std::future::Future<Output = ()> + 'static,
    {
        Disposer::Async(Box::new(move || Box::pin(f())))
    }

    /// Run the disposer to completion.
    pub async fn run(self) {
        match self {
            Disposer::Sync(f) => f(),
            Disposer::Async(f) => f().await,
        }
    }
}

/// Effect body result accepted by `Fiber::effect` and plugin startup: zero or
/// more disposers registered together. Upstream generator effects collapse to
/// the `Many` case.
pub enum Effect {
    None,
    One(Disposer),
    Many(Vec<Disposer>),
}

impl Effect {
    /// Flatten into a plain disposer list.
    pub fn into_disposers(self) -> Vec<Disposer> {
        match self {
            Effect::None => Vec::new(),
            Effect::One(d) => vec![d],
            Effect::Many(ds) => ds,
        }
    }
}

impl From<Disposer> for Effect {
    fn from(d: Disposer) -> Self {
        Effect::One(d)
    }
}

impl From<Vec<Disposer>> for Effect {
    fn from(ds: Vec<Disposer>) -> Self {
        Effect::Many(ds)
    }
}

impl From<()> for Effect {
    fn from(_: ()) -> Self {
        Effect::None
    }
}

/// Ordered collection of disposables with O(log n) deletion by serial,
/// ported from upstream `DisposableList` (insertion order preserved,
/// `clear()` returns values newest-first for reverse-order teardown).
pub struct DisposableList<T> {
    sn: u64,
    map: BTreeMap<u64, T>,
}

impl<T> Default for DisposableList<T> {
    fn default() -> Self {
        DisposableList {
            sn: 0,
            map: BTreeMap::new(),
        }
    }
}

impl<T> DisposableList<T> {
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Insert a value, returning its serial for later `remove`.
    pub fn push(&mut self, value: T) -> u64 {
        self.sn += 1;
        self.map.insert(self.sn, value);
        self.sn
    }

    /// Remove by serial; `true` when the entry was still present.
    pub fn remove(&mut self, sn: u64) -> Option<T> {
        self.map.remove(&sn)
    }

    /// Drain every value, newest first (reverse registration order).
    pub fn clear(&mut self) -> Vec<T> {
        let mut values: Vec<T> = std::mem::take(&mut self.map).into_values().collect();
        values.reverse();
        values
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.map.values()
    }
}
