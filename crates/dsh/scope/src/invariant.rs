//! Package-owned scoped-dispatch invariant companion (port of upstream
//! `src/invariant.ts`).
//!
//! Divergence: upstream hooks `internal/dispatch` and consults the generated
//! subject-resolver table to reject scoped events dispatched without a
//! carrier, or with a carrier keyed to a different subject than the payload
//! names. In the Rust port scoped dispatch is library-owned —
//! `ScopedEvents::emit_scoped` requires a `Scoped` carrier by construction,
//! making the carrier-presence invariant a compile-time property — and the
//! generated resolver table has no counterpart until the harness event
//! catalog is ported. The companion therefore reserves this package's
//! ownership with an empty installer, mirroring the dsh-invariants
//! companion's own convention.

use dsh_cordis::{Context, Disposer, Effect, FnPlugin, Inject, plugin_fn};
use dsh_invariants::{InvariantInstaller, InvariantRegistry};
use futures::FutureExt;
use futures::future::LocalBoxFuture;
use serde_json::Value;

/// Package name registered by the companion.
pub const PACKAGE_NAME: &str = "dsh-scope";
/// Cordis companion plugin name.
pub const NAME: &str = "scope-invariant";

/// Build the companion plugin: requires the `invariants` service and
/// registers this package's contribution.
pub fn plugin()
-> FnPlugin<impl Fn(Context, Value) -> LocalBoxFuture<'static, anyhow::Result<()>> + 'static> {
    plugin_fn(NAME, Inject::names(["invariants"]), |ctx, _config| {
        async move {
            let registry = ctx.service::<InvariantRegistry>()?;
            let registration = registry
                .register(
                    PACKAGE_NAME,
                    InvariantInstaller::new(|_ctx, _fail| async { Ok(()) }),
                )
                .await?;
            ctx.effect_labeled(NAME, move |_| {
                Ok(Effect::One(Disposer::asynchronous(move || {
                    registration.dispose()
                })))
            })?;
            Ok(())
        }
        .boxed_local()
    })
}
