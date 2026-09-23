//! Typed event bus with the five upstream dispatch modes, ported from
//! `vendor/cordis/src/events.ts`.
//!
//! Upstream events are string-named with TypeScript declaration merging; here
//! an event is a marker type implementing [`Event`], carrying its name, its
//! argument tuple, and its return type. Listeners are owned by the fiber that
//! registered them and are removed automatically when that fiber unloads.
//!
//! Dispatch modes (`@mode` in upstream JSDoc):
//! - `emit` — fire listeners without awaiting them (returned futures are
//!   spawned on the local set).
//! - `parallel` — await all listeners together.
//! - `serial` — await listeners in registration order until one returns
//!   `Some` (the bail value).
//! - `bail` — upstream's synchronous variant of `serial`; folded into
//!   [`Context::serial`] here because Rust listeners are uniformly async.
//! - `waterfall` — around-middleware: each listener receives the args and a
//!   `next` continuation; not calling `next` vetoes the rest of the chain.

use crate::core::{Context, FiberId};
use crate::error::Result;
use futures::FutureExt;
use futures::future::LocalBoxFuture;
use std::any::{Any, TypeId};
use std::rc::Rc;

/// A typed event: name, argument payload, and return value.
///
/// Notification modes (`emit`/`parallel`/`serial`) use listeners of shape
/// `Fn(&Context, &Args) -> Future<Option<Ret>>`; the waterfall mode uses
/// `Fn(&Context, Args, Next) -> Future<Result<Ret>>`.
pub trait Event: 'static {
    const NAME: &'static str;
    type Args: 'static;
    type Ret: 'static;
}

/// Continuation passed to waterfall listeners; calling it delegates to the
/// rest of the chain (finally the built-in behavior).
pub type Next<E> = Box<
    dyn FnOnce(<E as Event>::Args) -> LocalBoxFuture<'static, anyhow::Result<<E as Event>::Ret>>,
>;

pub(crate) type NotifyFn<E> =
    dyn Fn(&Context, &<E as Event>::Args) -> LocalBoxFuture<'static, Option<<E as Event>::Ret>>;

pub(crate) type WaterfallFn<E> =
    dyn Fn(
        &Context,
        <E as Event>::Args,
        Next<E>,
    ) -> LocalBoxFuture<'static, anyhow::Result<<E as Event>::Ret>>;

/// Type-erased holder so hooks for different events share one list.
pub(crate) struct NotifyHolder<E: Event>(pub Rc<NotifyFn<E>>);
pub(crate) struct WaterfallHolder<E: Event>(pub Rc<WaterfallFn<E>>);

/// One registered listener record. The callback's concrete holder type
/// (`NotifyHolder` vs `WaterfallHolder`) discriminates the listener kind.
pub(crate) struct Hook {
    pub id: u64,
    #[allow(dead_code)] // reserved for scoped dispatch filtering (upstream Context.filter)
    pub fiber: FiberId,
    pub event_type: TypeId,
    pub callback: Rc<dyn Any>,
    /// Receive the event regardless of context filter checks.
    #[allow(dead_code)] // reserved for scoped dispatch filtering
    pub global: bool,
}

/// Options accepted by `Context::on` (upstream `EventOptions`).
#[derive(Default, Clone, Copy)]
pub struct EventOptions {
    /// Add the listener before existing listeners for the same event.
    pub prepend: bool,
    /// Receive the event regardless of context filter checks.
    pub global: bool,
}

impl Context {
    fn collect_notify<E: Event>(&self) -> Vec<(Rc<NotifyFn<E>>, FiberId)> {
        let hooks = self.app().hooks.borrow();
        let Some(list) = hooks.get(E::NAME) else {
            return Vec::new();
        };
        list.iter()
            .filter(|hook| hook.event_type == TypeId::of::<E>())
            .filter_map(|hook| {
                hook.callback
                    .downcast_ref::<NotifyHolder<E>>()
                    .map(|holder| (holder.0.clone(), hook.fiber))
            })
            .collect()
    }

    fn collect_waterfall<E: Event>(&self) -> Vec<Rc<WaterfallFn<E>>> {
        let hooks = self.app().hooks.borrow();
        let Some(list) = hooks.get(E::NAME) else {
            return Vec::new();
        };
        list.iter()
            .filter(|hook| hook.event_type == TypeId::of::<E>())
            .filter_map(|hook| {
                hook.callback
                    .downcast_ref::<WaterfallHolder<E>>()
                    .map(|holder| holder.0.clone())
            })
            .collect()
    }

    /// Dispatch an event without awaiting listeners; returned futures are
    /// spawned on the current local set (upstream `emit`).
    pub fn emit<E: Event>(&self, args: &E::Args) {
        for (callback, _) in self.collect_notify::<E>() {
            let fut = callback(self, args);
            tokio::task::spawn_local(async move {
                let _ = fut.await;
            });
        }
    }

    /// Dispatch an event, awaiting all listeners together (upstream
    /// `parallel`).
    pub async fn parallel<E: Event>(&self, args: &E::Args) {
        let futures: Vec<_> = self
            .collect_notify::<E>()
            .into_iter()
            .map(|(callback, _)| callback(self, args))
            .collect();
        futures::future::join_all(futures).await;
    }

    /// Dispatch an event, awaiting listeners in order until one bails with
    /// `Some` (upstream `serial`; also covers upstream's sync `bail`).
    pub async fn serial<E: Event>(&self, args: &E::Args) -> Option<E::Ret> {
        for (callback, _) in self.collect_notify::<E>() {
            if let Some(value) = callback(self, args).await {
                return Some(value);
            }
        }
        None
    }

    /// Dispatch a waterfall event: listeners compose around `inner`,
    /// outermost-first; a listener that does not call `next` vetoes the rest
    /// of the chain including `inner` (upstream `waterfall`).
    pub fn waterfall<E, F, Fut>(
        &self,
        args: E::Args,
        inner: F,
    ) -> LocalBoxFuture<'static, anyhow::Result<E::Ret>>
    where
        E: Event,
        F: FnOnce(E::Args) -> Fut + 'static,
        Fut: std::future::Future<Output = anyhow::Result<E::Ret>> + 'static,
    {
        let chain = Rc::new(self.collect_waterfall::<E>());
        let ctx = self.clone();
        fn step<E: Event>(
            ctx: Context,
            chain: Rc<Vec<Rc<WaterfallFn<E>>>>,
            index: usize,
            args: E::Args,
            inner: Box<dyn FnOnce(E::Args) -> LocalBoxFuture<'static, anyhow::Result<E::Ret>>>,
        ) -> LocalBoxFuture<'static, anyhow::Result<E::Ret>> {
            let callback = chain.get(index).cloned();
            match callback {
                None => inner(args),
                Some(callback) => {
                    let next_ctx = ctx.clone();
                    let next: Next<E> =
                        Box::new(move |args| step::<E>(next_ctx, chain, index + 1, args, inner));
                    callback(&ctx, args, next)
                }
            }
        }
        let inner: Box<dyn FnOnce(E::Args) -> LocalBoxFuture<'static, anyhow::Result<E::Ret>>> =
            Box::new(move |args| inner(args).boxed_local());
        step::<E>(ctx, chain, 0, args, inner)
    }

    /// Register a notification listener owned by the current fiber
    /// (upstream `ctx.on`). The listener is removed when the fiber unloads.
    pub fn on<E, F, Fut>(
        &self,
        options: EventOptions,
        listener: F,
    ) -> Result<crate::core::EffectHandle>
    where
        E: Event,
        F: Fn(&Context, &E::Args) -> Fut + 'static,
        Fut: std::future::Future<Output = Option<E::Ret>> + 'static,
    {
        let callback: Rc<NotifyFn<E>> = Rc::new(move |ctx, args| listener(ctx, args).boxed_local());
        self.register_hook::<E>(Rc::new(NotifyHolder::<E>(callback)), options)
    }

    /// Register a waterfall listener owned by the current fiber. The listener
    /// MUST call `next(args)` to delegate; returning without it vetoes the
    /// rest of the chain (upstream waterfall contract).
    pub fn on_waterfall<E, F, Fut>(
        &self,
        options: EventOptions,
        listener: F,
    ) -> Result<crate::core::EffectHandle>
    where
        E: Event,
        F: Fn(&Context, E::Args, Next<E>) -> Fut + 'static,
        Fut: std::future::Future<Output = anyhow::Result<E::Ret>> + 'static,
    {
        let callback: Rc<WaterfallFn<E>> =
            Rc::new(move |ctx, args, next| listener(ctx, args, next).boxed_local());
        self.register_hook::<E>(Rc::new(WaterfallHolder::<E>(callback)), options)
    }

    fn register_hook<E: Event>(
        &self,
        callback: Rc<dyn Any>,
        options: EventOptions,
    ) -> Result<crate::core::EffectHandle> {
        let app = self.app_rc();
        let fiber = self.fiber_id();
        let hook_id = app.next_hook_id();
        let label = format!("ctx.on({:?})", E::NAME);
        self.effect_labeled(&label, move |_ctx| {
            let hook = Hook {
                id: hook_id,
                fiber,
                event_type: TypeId::of::<E>(),
                callback,
                global: options.global,
            };
            let mut hooks = app.hooks.borrow_mut();
            let list = hooks.entry(E::NAME.to_string()).or_default();
            if options.prepend {
                list.insert(0, hook);
            } else {
                list.push(hook);
            }
            drop(hooks);
            let app = app.clone();
            Ok(crate::disposer::Effect::One(
                crate::disposer::Disposer::sync(move || {
                    let mut hooks = app.hooks.borrow_mut();
                    if let Some(list) = hooks.get_mut(E::NAME) {
                        list.retain(|hook| hook.id != hook_id);
                    }
                }),
            ))
        })
    }
}
