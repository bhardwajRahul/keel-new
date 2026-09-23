//! Lifecycle behavior ported from upstream cordis test expectations:
//! dependency-gated loading, teardown order, service disposal waking
//! dependents, waterfall veto, and isolate scoping.

use dsh_cordis::{App, Effect, Event, FiberState, Inject, plugin_fn};
use serde_json::{Value, json};
use std::cell::RefCell;
use std::rc::Rc;

#[test]
fn plugin_loads_when_service_appears_and_unloads_when_it_leaves() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let log: Rc<RefCell<Vec<&'static str>>> = Rc::default();

        let log2 = log.clone();
        let consumer = plugin_fn("consumer", Inject::names(["counter"]), move |ctx, _| {
            let log = log2.clone();
            async move {
                log.borrow_mut().push("load");
                let log = log.clone();
                ctx.effect(move |_| {
                    Ok(Effect::One(dsh_cordis::Disposer::sync(move || {
                        log.borrow_mut().push("unload");
                    })))
                })?;
                Ok(())
            }
        });
        let consumer_fiber = ctx.plugin(Rc::new(consumer), Value::Null).unwrap();
        consumer_fiber.await_ready().await.unwrap();
        assert_eq!(consumer_fiber.state(), FiberState::Pending);
        assert!(log.borrow().is_empty());

        let provider = plugin_fn("provider", Inject::default(), move |ctx, _| async move {
            ctx.provide("counter", Rc::new(41u32), None)?;
            Ok(())
        });
        let provider_fiber = ctx.plugin(Rc::new(provider), Value::Null).unwrap();
        provider_fiber.await_ready().await.unwrap();
        consumer_fiber.await_ready().await.unwrap();
        assert_eq!(consumer_fiber.state(), FiberState::Active);
        assert_eq!(*log.borrow(), vec!["load"]);

        // Read through the inject walk from the consumer's context.
        let value = consumer_fiber.ctx().get_raw("counter").unwrap();
        assert_eq!(*value.downcast::<u32>().unwrap(), 41);

        // Disposing the provider unloads the consumer.
        provider_fiber.dispose().await;
        consumer_fiber.await_ready().await.unwrap();
        assert_eq!(consumer_fiber.state(), FiberState::Pending);
        assert_eq!(*log.borrow(), vec!["load", "unload"]);
    });
}

#[test]
fn disposers_run_in_reverse_registration_order() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let order: Rc<RefCell<Vec<u32>>> = Rc::default();

        let order2 = order.clone();
        let plugin = plugin_fn("ordered", Inject::default(), move |ctx, _| {
            let order = order2.clone();
            async move {
                for i in 0..3u32 {
                    let order = order.clone();
                    ctx.effect(move |_| {
                        Ok(Effect::One(dsh_cordis::Disposer::sync(move || {
                            order.borrow_mut().push(i);
                        })))
                    })?;
                }
                Ok(())
            }
        });
        let fiber = ctx.plugin(Rc::new(plugin), Value::Null).unwrap();
        fiber.await_ready().await.unwrap();
        fiber.dispose().await;
        assert_eq!(*order.borrow(), vec![2, 1, 0]);
    });
}

struct Greeting;
impl Event for Greeting {
    const NAME: &'static str = "test/greeting";
    type Args = String;
    type Ret = String;
}

#[test]
fn waterfall_composes_and_vetoes() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();

        // Outermost listener wraps; delegate via next.
        ctx.on_waterfall::<Greeting, _, _>(Default::default(), |_ctx, args, next| async move {
            let inner = next(format!("{args}!")).await?;
            Ok(format!("<{inner}>"))
        })
        .unwrap();

        let result = ctx
            .waterfall::<Greeting, _, _>("hi".to_string(), |args| async move { Ok(args) })
            .await
            .unwrap();
        assert_eq!(result, "<hi!>");

        // A vetoing listener (never calls next) short-circuits the chain.
        ctx.on_waterfall::<Greeting, _, _>(
            dsh_cordis::EventOptions {
                prepend: true,
                global: false,
            },
            |_ctx, _args, _next| async move { Ok("vetoed".to_string()) },
        )
        .unwrap();
        let result = ctx
            .waterfall::<Greeting, _, _>("hi".to_string(), |args| async move { Ok(args) })
            .await
            .unwrap();
        assert_eq!(result, "vetoed");
    });
}

#[test]
fn config_validation_failure_marks_fiber_failed() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();

        struct Strict;
        impl dsh_cordis::Plugin for Strict {
            fn validate_config(&self, config: Value) -> dsh_cordis::Result<Value> {
                if config.get("port").is_some() {
                    Ok(config)
                } else {
                    Err(dsh_cordis::CordisError::InvalidConfig(
                        "  - port required".into(),
                    ))
                }
            }
            fn apply(
                &self,
                _ctx: dsh_cordis::Context,
                _config: Value,
            ) -> futures::future::LocalBoxFuture<'static, anyhow::Result<()>> {
                Box::pin(async { Ok(()) })
            }
        }

        let fiber = ctx.plugin(Rc::new(Strict), json!({})).unwrap();
        assert!(fiber.await_ready().await.is_err());
        assert_eq!(fiber.state(), FiberState::Failed);

        // update() with valid config recovers.
        fiber.update(json!({"port": 8080}), false).await.unwrap();
        fiber.await_ready().await.unwrap();
        assert_eq!(fiber.state(), FiberState::Active);
    });
}

#[test]
fn isolate_scopes_hide_services() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();

        let provider = plugin_fn("provider", Inject::default(), |ctx, _| async move {
            ctx.provide("db", Rc::new("root-db".to_string()), None)?;
            Ok(())
        });
        ctx.plugin(Rc::new(provider), Value::Null)
            .unwrap()
            .await_ready()
            .await
            .unwrap();

        // A consumer below an isolate() for "db" must not see the root impl.
        let isolated = ctx.isolate("db", None);
        let seen: Rc<RefCell<Option<bool>>> = Rc::default();
        let seen2 = seen.clone();
        let consumer = plugin_fn("consumer", Inject::names(["db"]), move |_ctx, _| {
            let seen = seen2.clone();
            async move {
                *seen.borrow_mut() = Some(true);
                Ok(())
            }
        });
        let fiber = isolated.plugin(Rc::new(consumer), Value::Null).unwrap();
        fiber.await_ready().await.unwrap();
        assert_eq!(fiber.state(), FiberState::Pending);
        assert!(seen.borrow().is_none());
    });
}
