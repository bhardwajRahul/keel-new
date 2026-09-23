//! Rust port of the Cordis plugin framework as vendored by deepseek-harness
//! (`vendor/cordis`, `vendor/cosmokit`): contexts, fiber lifecycle, typed
//! events, services, and the plugin registry.
//!
//! Everything is a plugin: behavior mounts on a [`Context`], declares the
//! services it injects, provides services of its own, and registers every
//! contribution as a disposable effect. Fibers load when all injected
//! services are available and unload (reverse-order disposers) when any
//! leaves.
//!
//! The framework is single-threaded by design, mirroring the JS event loop:
//! run it inside a tokio `LocalSet` (see [`run`]). Nothing here is `Send`.
//!
//! Port notes (contract-level 1:1, Rust idiom where the language forces it):
//! - Property-style service access (`ctx.foo`) becomes
//!   `ctx.service::<Foo>()` / `ctx.get_raw("foo")`.
//! - Upstream's `bail` dispatch is folded into [`Context::serial`]; the other
//!   modes (`emit`, `parallel`, `serial`, `waterfall`) keep their semantics.
//! - cosmokit is absorbed: its JS-object utilities have no Rust counterpart.
//! - HMR and dynamic `import()` plugin loading are replaced by a
//!   compile-time registry (`ctx.plugin(...)` with plugin values).

mod core;
mod disposer;
mod error;
mod events;
mod plugin;
mod service;

pub use crate::core::{
    App, Context, EffectHandle, Fiber, FiberId, FiberState, InternalConfig, InternalPlugin,
    InternalService, InternalStatus, InternalUpdate,
};
pub use crate::disposer::{DisposableList, Disposer, Effect};
pub use crate::error::{CordisError, Result, ValidationIssues};
pub use crate::events::{Event, EventOptions, Next};
pub use crate::plugin::{FnPlugin, Inject, Plugin, PluginKey, plugin_fn, validate_as};
pub use crate::service::Service;

/// Run a future on a current-thread tokio runtime inside a `LocalSet` — the
/// execution environment the framework requires.
pub fn run<F, T>(future: F) -> T
where
    F: std::future::Future<Output = T>,
{
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build current-thread runtime");
    let local = tokio::task::LocalSet::new();
    local.block_on(&runtime, future)
}
