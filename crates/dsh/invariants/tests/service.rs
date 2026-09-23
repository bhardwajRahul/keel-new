//! Behavioral contract tests ported from upstream `tests/service.spec.ts`:
//! selection filters, validation, and registration lifecycle. The upstream
//! "publication failure" case (a throwing `internal/plugin` observer) has no
//! Rust counterpart — `emit` cannot throw into `ctx.plugin` — and event-time
//! `fail` throws become installer-returned errors (see the crate docs).

use dsh_cordis::{
    App, Context, Disposer, Effect, Event, EventOptions, Fiber, Inject, Service, plugin_fn,
};
use dsh_invariants::{
    Config, InvariantError, InvariantInstaller, InvariantRegistry, InvariantsPlugin,
};
use futures::FutureExt;
use serde_json::{Value, json};
use std::cell::Cell;
use std::rc::Rc;

struct Ping;
impl Event for Ping {
    const NAME: &'static str = "invariants-test/ping";
    type Args = ();
    type Ret = ();
}

/// Let `emit`-spawned listener tasks run.
async fn settle() {
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
}

async fn setup(ctx: &Context, config: Value) -> Fiber {
    let fiber = ctx.plugin(Rc::new(InvariantsPlugin), config).unwrap();
    fiber.await_ready().await.unwrap();
    fiber
}

fn registry(ctx: &Context) -> Rc<InvariantRegistry> {
    ctx.service::<InvariantRegistry>().unwrap()
}

/// Installer registering a Ping listener that bumps `count`.
fn probe(count: Rc<Cell<u32>>) -> InvariantInstaller {
    InvariantInstaller::new(move |ctx, _fail| {
        let count = count.clone();
        async move {
            ctx.on::<Ping, _, _>(EventOptions::default(), move |_ctx, _args| {
                count.set(count.get() + 1);
                std::future::ready(None)
            })?;
            Ok(())
        }
    })
}

fn trivial() -> InvariantInstaller {
    InvariantInstaller::new(|_ctx, _fail| async { Ok(()) })
}

/// `unwrap_err` needs `T: Debug`; `EffectHandle` has none.
fn expect_err<T>(result: anyhow::Result<T>) -> anyhow::Error {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(error) => error,
    }
}

#[test]
fn applies_defaults_when_constructed_directly() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let registry = InvariantRegistry::new(ctx.clone(), Config::default()).unwrap();
        let count: Rc<Cell<u32>> = Rc::default();
        let registration = registry
            .register("dsh-session", probe(count.clone()))
            .await
            .unwrap();
        ctx.emit::<Ping>(&());
        settle().await;
        assert_eq!(count.get(), 1);
        registration.dispose().await;
    });
}

#[test]
fn enables_by_default_and_treats_empty_lists_as_admit_all() {
    dsh_cordis::run(async {
        for config in [
            json!({}),
            json!({ "package_allowlist": [], "package_blocklist": [] }),
        ] {
            let app = App::new();
            let ctx = app.root();
            setup(&ctx, config).await;
            let count: Rc<Cell<u32>> = Rc::default();
            registry(&ctx)
                .register("dsh-session", probe(count.clone()))
                .await
                .unwrap();
            ctx.emit::<Ping>(&());
            settle().await;
            assert_eq!(count.get(), 1);
        }
    });
}

#[test]
fn disabled_still_reserves_package_ownership() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        setup(&ctx, json!({ "enabled": false })).await;
        let count: Rc<Cell<u32>> = Rc::default();
        let registration = registry(&ctx)
            .register("dsh-session", probe(count.clone()))
            .await
            .unwrap();
        let err = expect_err(registry(&ctx).register("dsh-session", trivial()).await);
        assert!(err.to_string().contains("already registered"), "{err:#}");
        ctx.emit::<Ping>(&());
        settle().await;
        assert_eq!(count.get(), 0);
        registration.dispose().await;
    });
}

#[test]
fn patterns_are_unanchored_and_case_sensitive() {
    dsh_cordis::run(async {
        // Unanchored source matches as a substring.
        let app = App::new();
        let ctx = app.root();
        setup(&ctx, json!({ "package_allowlist": ["session"] })).await;
        let count: Rc<Cell<u32>> = Rc::default();
        registry(&ctx)
            .register("dsh-session-extra", probe(count.clone()))
            .await
            .unwrap();
        ctx.emit::<Ping>(&());
        settle().await;
        assert_eq!(count.get(), 1);

        // Anchors bound the match.
        let app = App::new();
        let ctx = app.root();
        setup(&ctx, json!({ "package_allowlist": ["^dsh-session$"] })).await;
        let count: Rc<Cell<u32>> = Rc::default();
        registry(&ctx)
            .register("dsh-session-extra", probe(count.clone()))
            .await
            .unwrap();
        ctx.emit::<Ping>(&());
        settle().await;
        assert_eq!(count.get(), 0);

        // Matching is case-sensitive.
        let app = App::new();
        let ctx = app.root();
        setup(&ctx, json!({ "package_allowlist": ["Session"] })).await;
        let count: Rc<Cell<u32>> = Rc::default();
        registry(&ctx)
            .register("dsh-session", probe(count.clone()))
            .await
            .unwrap();
        ctx.emit::<Ping>(&());
        settle().await;
        assert_eq!(count.get(), 0);
    });
}

#[test]
fn blocklist_overrides_allowlist_match() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        setup(
            &ctx,
            json!({ "package_allowlist": ["^dsh-"], "package_blocklist": ["session"] }),
        )
        .await;
        let session: Rc<Cell<u32>> = Rc::default();
        let agent: Rc<Cell<u32>> = Rc::default();
        registry(&ctx)
            .register("dsh-session", probe(session.clone()))
            .await
            .unwrap();
        registry(&ctx)
            .register("dsh-agent", probe(agent.clone()))
            .await
            .unwrap();
        ctx.emit::<Ping>(&());
        settle().await;
        assert_eq!(session.get(), 0);
        assert_eq!(agent.get(), 1);
    });
}

#[test]
fn accepts_zero_match_patterns_for_later_packages() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        setup(&ctx, json!({ "package_allowlist": ["^later-invariants$"] })).await;
        let now: Rc<Cell<u32>> = Rc::default();
        let later: Rc<Cell<u32>> = Rc::default();
        registry(&ctx)
            .register("dsh-session", probe(now.clone()))
            .await
            .unwrap();
        registry(&ctx)
            .register("later-invariants", probe(later.clone()))
            .await
            .unwrap();
        ctx.emit::<Ping>(&());
        settle().await;
        assert_eq!(now.get(), 0);
        assert_eq!(later.get(), 1);
    });
}

#[test]
fn same_source_in_both_lists_blocks() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        setup(
            &ctx,
            json!({ "package_allowlist": ["agent"], "package_blocklist": ["agent"] }),
        )
        .await;
        let count: Rc<Cell<u32>> = Rc::default();
        registry(&ctx)
            .register("dsh-agent", probe(count.clone()))
            .await
            .unwrap();
        ctx.emit::<Ping>(&());
        settle().await;
        assert_eq!(count.get(), 0);
    });
}

#[test]
fn rejects_malformed_filter_config() {
    dsh_cordis::run(async {
        let cases = [
            (json!({ "package_allowlist": [""] }), "non-blank"),
            (json!({ "package_allowlist": [" "] }), "non-blank"),
            (
                json!({ "package_allowlist": [" session"] }),
                "surrounding whitespace",
            ),
            (
                json!({ "package_blocklist": ["session "] }),
                "surrounding whitespace",
            ),
            (
                json!({ "package_allowlist": ["session", "session"] }),
                "duplicate regex",
            ),
            (
                json!({ "package_blocklist": ["agent", "agent"] }),
                "duplicate regex",
            ),
            (json!({ "package_allowlist": ["["] }), "invalid regex"),
            (json!({ "package_blocklist": ["("] }), "invalid regex"),
        ];
        for (config, needle) in cases {
            let app = App::new();
            let ctx = app.root();
            let fiber = ctx
                .plugin(Rc::new(InvariantsPlugin), config.clone())
                .unwrap();
            let err = fiber.await_ready().await.unwrap_err();
            assert!(err.to_string().contains(needle), "{config}: {err}");
        }
    });
}

#[test]
fn rejects_malformed_package_names() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        setup(&ctx, json!({})).await;
        for name in ["", " ", " package", "pack age", "package\n"] {
            let err = expect_err(registry(&ctx).register(name, trivial()).await);
            assert!(err.to_string().contains("packageName"), "{name:?}: {err}");
        }
    });
}

#[test]
fn honors_installer_inject_in_child_fiber() {
    struct ProbeService;
    impl Service for ProbeService {
        const NAME: &'static str = "invariantProbe";
    }

    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        setup(&ctx, json!({})).await;
        let provider = plugin_fn("probe-provider", Inject::default(), |ctx, _| async move {
            ctx.provide_service(Rc::new(ProbeService))?;
            Ok(())
        });
        ctx.plugin(Rc::new(provider), Value::Null)
            .unwrap()
            .await_ready()
            .await
            .unwrap();

        let seen: Rc<Cell<bool>> = Rc::default();
        let seen2 = seen.clone();
        let installer = InvariantInstaller::new(move |ctx, _fail| {
            let seen = seen2.clone();
            async move {
                // Resolves only because the child fiber injected the service.
                ctx.service::<ProbeService>()?;
                seen.set(true);
                Ok(())
            }
        })
        .with_inject(Inject::names(["invariantProbe"]));
        let registration = registry(&ctx)
            .register("dsh-probe", installer)
            .await
            .unwrap();
        assert!(seen.get());
        registration.dispose().await;
    });
}

#[test]
fn attributes_failures_to_the_registering_package() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        setup(&ctx, json!({})).await;
        let err = registry(&ctx)
            .register(
                "dsh-session",
                InvariantInstaller::new(|_ctx, fail| async move {
                    Err(fail("seq must strictly increase").into())
                }),
            )
            .await
            .unwrap_err();
        let invariant = err.downcast_ref::<InvariantError>().unwrap();
        assert_eq!(invariant.package_name, "dsh-session");
        assert_eq!(InvariantError::CODE, "INVARIANT");
        assert_eq!(
            invariant.to_string(),
            "invariant violated by \"dsh-session\": seq must strictly increase"
        );
    });
}

#[test]
fn disposal_is_complete_and_permits_re_registration() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        setup(&ctx, json!({})).await;
        let first: Rc<Cell<u32>> = Rc::default();
        let registration = registry(&ctx)
            .register("dsh-session", probe(first.clone()))
            .await
            .unwrap();
        ctx.emit::<Ping>(&());
        settle().await;
        registration.dispose().await;
        ctx.emit::<Ping>(&());
        settle().await;
        assert_eq!(first.get(), 1);

        let second: Rc<Cell<u32>> = Rc::default();
        registry(&ctx)
            .register("dsh-session", probe(second.clone()))
            .await
            .unwrap();
        ctx.emit::<Ping>(&());
        settle().await;
        assert_eq!(first.get(), 1);
        assert_eq!(second.get(), 1);
    });
}

#[test]
fn reserves_ownership_until_async_child_disposal_completes() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        setup(&ctx, json!({})).await;
        let (tx, rx) = futures::channel::oneshot::channel::<()>();
        let gate = rx.shared();
        let installer = InvariantInstaller::new(move |ctx, _fail| {
            let gate = gate.clone();
            async move {
                ctx.effect(move |_| {
                    Ok(Effect::One(Disposer::asynchronous(move || async move {
                        let _ = gate.await;
                    })))
                })?;
                Ok(())
            }
        });
        let registration = registry(&ctx)
            .register("dsh-session", installer)
            .await
            .unwrap();

        let disposing = tokio::task::spawn_local({
            let registration = registration.clone();
            async move { registration.dispose().await }
        });
        settle().await;
        let err = expect_err(registry(&ctx).register("dsh-session", trivial()).await);
        assert!(err.to_string().contains("already registered"), "{err:#}");
        tx.send(()).unwrap();
        disposing.await.unwrap();

        let replacement = registry(&ctx)
            .register("dsh-session", trivial())
            .await
            .unwrap();
        replacement.dispose().await;
    });
}

#[test]
fn rolls_back_listeners_and_ownership_when_installer_fails() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        setup(&ctx, json!({})).await;
        let leaked: Rc<Cell<u32>> = Rc::default();
        let leaked2 = leaked.clone();
        let err = registry(&ctx)
            .register(
                "dsh-session",
                InvariantInstaller::new(move |ctx, _fail| {
                    let leaked = leaked2.clone();
                    async move {
                        ctx.on::<Ping, _, _>(EventOptions::default(), move |_ctx, _args| {
                            leaked.set(leaked.get() + 1);
                            std::future::ready(None)
                        })?;
                        anyhow::bail!("installer failed")
                    }
                }),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("installer failed"), "{err:#}");
        ctx.emit::<Ping>(&());
        settle().await;
        assert_eq!(leaked.get(), 0);

        let retry: Rc<Cell<u32>> = Rc::default();
        registry(&ctx)
            .register("dsh-session", probe(retry.clone()))
            .await
            .unwrap();
        ctx.emit::<Ping>(&());
        settle().await;
        assert_eq!(retry.get(), 1);
    });
}

#[test]
fn joins_async_checks_and_rolls_back_their_effects_on_failure() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        setup(&ctx, json!({})).await;
        let leaked: Rc<Cell<u32>> = Rc::default();
        let leaked2 = leaked.clone();
        let err = registry(&ctx)
            .register(
                "dsh-async-probe",
                InvariantInstaller::new(move |ctx, fail| {
                    let leaked = leaked2.clone();
                    async move {
                        ctx.on::<Ping, _, _>(EventOptions::default(), move |_ctx, _args| {
                            leaked.set(leaked.get() + 1);
                            std::future::ready(None)
                        })?;
                        tokio::task::yield_now().await;
                        Err(fail("asynchronous check failed").into())
                    }
                }),
            )
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("asynchronous check failed"),
            "{err:#}"
        );
        ctx.emit::<Ping>(&());
        settle().await;
        assert_eq!(leaked.get(), 0);

        let retry = registry(&ctx)
            .register(
                "dsh-async-probe",
                InvariantInstaller::new(|_ctx, _fail| async {
                    tokio::task::yield_now().await;
                    Ok(())
                }),
            )
            .await
            .unwrap();
        retry.dispose().await;
    });
}

#[test]
fn releases_reservation_when_service_fiber_is_inactive() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let fiber = setup(&ctx, json!({})).await;
        let service = registry(&ctx);
        fiber.dispose().await;
        let err = expect_err(service.register("dsh-session", trivial()).await);
        assert!(
            err.to_string().to_lowercase().contains("inactive"),
            "{err:#}"
        );
    });
}

#[test]
fn companion_reserves_its_own_package() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        setup(&ctx, json!({})).await;
        let companion = ctx
            .plugin(Rc::new(dsh_invariants::invariant::plugin()), Value::Null)
            .unwrap();
        companion.await_ready().await.unwrap();
        let err = registry(&ctx)
            .register(dsh_invariants::invariant::PACKAGE_NAME, trivial())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("already registered"), "{err:#}");
        // Unloading the companion releases the reservation.
        companion.dispose().await;
        let handle = registry(&ctx)
            .register(dsh_invariants::invariant::PACKAGE_NAME, trivial())
            .await
            .unwrap();
        handle.dispose().await;
    });
}
