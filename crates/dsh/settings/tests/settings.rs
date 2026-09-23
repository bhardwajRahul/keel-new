//! Behavior suite for the settings Service Definition, ported from upstream
//! `tests/settings.spec.ts`. Not ported (impossible or meaningless in Rust):
//! frozen-value checks, `undefined`/null-prototype/non-JSON input rejections
//! (`serde_json::Value` cannot hold them), call-time snapshot mutation, and
//! the `settings/updated` listener-containment + INVARIANT cases (cordis-rs
//! notification listeners have no error channel).

mod common;

use common::{boot, boot_with, record_updates, settle, wait_until};
use dsh_cordis::{Context, Inject, plugin_fn};
use dsh_settings::{
    Schema, SettingsApplies, SettingsConflictError, SettingsDescribeOptions, SettingsNamespace,
    SettingsPathOp, SettingsRegisterOptions, SettingsScope, SettingsSectionHooks, SettingsService,
    SettingsUpdateSource, install_settings_section, settings_namespace,
};
use serde_json::{Value, json};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::Duration;

fn ns(value: &str) -> SettingsNamespace {
    settings_namespace(value).unwrap()
}

fn theme_schema() -> Schema {
    Schema::object([
        (
            "theme",
            Schema::union([json!("dark"), json!("light")]).default(json!("dark")),
        ),
        ("fontSize", Schema::number().default(json!(14))),
    ])
}

fn nested_schema() -> Schema {
    Schema::object([
        (
            "retry",
            Schema::object([
                ("attempts", Schema::number().default(json!(2))),
                ("delayMs", Schema::number().default(json!(100))),
            ]),
        ),
        (
            "tags",
            Schema::array(Schema::string()).default(json!(["default"])),
        ),
    ])
}

fn options() -> SettingsRegisterOptions {
    SettingsRegisterOptions::default()
}

fn base(value: Value) -> SettingsRegisterOptions {
    SettingsRegisterOptions {
        base: Some(value),
        ..Default::default()
    }
}

fn err_text(error: anyhow::Error) -> String {
    format!("{error:#}")
}

// ---------------------------------------------------------------- metadata

#[test]
fn does_not_advertise_a_local_document_unless_the_backend_overrides_it() {
    dsh_cordis::run(async {
        let booted = boot(Value::Null).await;
        assert!(booted.service.document_path().is_none());
        assert!(booted.service.prepare_document().await.unwrap().is_none());
    });
}

// ------------------------------------------------------- settings_namespace

#[test]
fn brands_lowercase_kebab_case_names() {
    assert_eq!(settings_namespace("ui-theme").unwrap().as_str(), "ui-theme");
}

#[test]
fn rejects_malformed_namespaces() {
    for value in ["", "UI", "9lives", "a_b", "-lead"] {
        assert!(
            settings_namespace(value).is_err(),
            "{value:?} should be rejected"
        );
    }
}

// ------------------------------------------------------------ registration

#[test]
fn resolves_schema_defaults_then_composition_base_then_the_user_layer() {
    dsh_cordis::run(async {
        let booted = boot(json!({"ui-theme": {"theme": "light"}})).await;
        let scope = booted
            .service
            .register(
                &booted.ctx,
                ns("ui-theme"),
                theme_schema(),
                base(json!({"fontSize": 16})),
            )
            .unwrap();
        // theme: user layer wins; fontSize: base wins over the schema default.
        assert_eq!(scope.get(), json!({"theme": "light", "fontSize": 16}));
    });
}

#[test]
fn refuses_a_write_its_owner_could_not_act_on_and_keeps_the_last_good_stored_value() {
    dsh_cordis::run(async {
        let booted = boot(Value::Null).await;
        // A constraint the schema cannot express: this owner cannot serve a
        // size it considers unreadable, whatever the schema admits.
        let scope = booted
            .service
            .register(
                &booted.ctx,
                ns("ui-theme"),
                theme_schema(),
                SettingsRegisterOptions {
                    validate: Some(Rc::new(|value| {
                        let size = value["fontSize"].as_i64().unwrap_or(0);
                        if size < 10 {
                            anyhow::bail!("font size {size} is unreadable");
                        }
                        Ok(())
                    })),
                    ..Default::default()
                },
            )
            .unwrap();
        let before = scope.get();

        let error = booted
            .service
            .update(&ns("ui-theme"), json!({"fontSize": 4}))
            .await
            .unwrap_err();
        assert!(err_text(error).contains("unreadable"));
        assert_eq!(scope.get(), before);

        // An externally edited document must not strand the owner: the
        // namespace keeps its last good value, exactly as a schema failure
        // would.
        booted.service.publish(
            json!({"ui-theme": {"fontSize": 4}})
                .as_object()
                .unwrap()
                .clone(),
        );
        assert_eq!(scope.get(), before);

        booted
            .service
            .update(&ns("ui-theme"), json!({"fontSize": 18}))
            .await
            .unwrap();
        assert_eq!(scope.get()["fontSize"], json!(18));
    });
}

#[test]
fn fails_the_registration_itself_when_the_already_stored_section_is_unserviceable() {
    dsh_cordis::run(async {
        // At cold start there is no last good value to keep, so a stored
        // section the owner cannot serve refuses the registration.
        let booted = boot(json!({"ui-theme": {"fontSize": 4}})).await;
        let result = booted.service.register(
            &booted.ctx,
            ns("ui-theme"),
            theme_schema(),
            SettingsRegisterOptions {
                validate: Some(Rc::new(|value| {
                    let size = value["fontSize"].as_i64().unwrap_or(0);
                    if size < 10 {
                        anyhow::bail!("font size {size} is unreadable");
                    }
                    Ok(())
                })),
                ..Default::default()
            },
        );
        assert!(err_text(result.err().unwrap()).contains("unreadable"));
    });
}

#[test]
fn rejects_a_duplicate_namespace_loud() {
    dsh_cordis::run(async {
        let booted = boot(Value::Null).await;
        booted
            .service
            .register(&booted.ctx, ns("ui-theme"), theme_schema(), options())
            .unwrap();
        let error = booted
            .service
            .register(&booted.ctx, ns("ui-theme"), theme_schema(), options())
            .err()
            .unwrap();
        assert!(err_text(error).contains("already registered"));
    });
}

#[test]
fn fails_registration_when_the_stored_section_is_invalid_for_the_schema() {
    dsh_cordis::run(async {
        let booted = boot(json!({"ui-theme": {"fontSize": "big"}})).await;
        assert!(
            booted
                .service
                .register(&booted.ctx, ns("ui-theme"), theme_schema(), options())
                .is_err()
        );
    });
}

#[test]
fn fails_registration_when_the_stored_section_is_not_an_object() {
    dsh_cordis::run(async {
        let booted = boot(json!({"ui-theme": "dark"})).await;
        let error = booted
            .service
            .register(&booted.ctx, ns("ui-theme"), theme_schema(), options())
            .err()
            .unwrap();
        assert!(err_text(error).contains("must be an object"));
    });
}

#[test]
fn describes_registered_namespaces_with_schema_json_value_and_applies() {
    dsh_cordis::run(async {
        let booted = boot(Value::Null).await;
        booted
            .service
            .register(&booted.ctx, ns("ui-theme"), theme_schema(), options())
            .unwrap();
        booted
            .service
            .register(
                &booted.ctx,
                ns("workspace"),
                nested_schema(),
                SettingsRegisterOptions {
                    applies: SettingsApplies::Restart,
                    ..Default::default()
                },
            )
            .unwrap();
        let descriptors = booted.service.describe(SettingsDescribeOptions::default());
        let listed: Vec<(String, SettingsApplies)> = descriptors
            .iter()
            .map(|entry| (entry.ns.to_string(), entry.applies))
            .collect();
        assert_eq!(
            listed,
            vec![
                ("ui-theme".to_string(), SettingsApplies::Live),
                ("workspace".to_string(), SettingsApplies::Restart),
            ]
        );
        assert_eq!(
            descriptors[0].value,
            json!({"theme": "dark", "fontSize": 14})
        );
        // Divergence: the serialized schema is this crate's node tree, not
        // schemastery's {uid, refs} envelope; the root still says "object".
        assert_eq!(descriptors[0].schema["type"], json!("object"));
    });
}

#[test]
fn reads_none_for_an_unregistered_namespace() {
    dsh_cordis::run(async {
        let booted = boot(Value::Null).await;
        assert!(booted.service.get(&ns("missing")).is_none());
    });
}

#[test]
fn removes_the_namespace_and_its_observers_when_the_registrant_fiber_disposes() {
    dsh_cordis::run(async {
        let booted = boot(Value::Null).await;
        let seen: Rc<RefCell<Vec<Value>>> = Rc::default();
        let scope_cell: Rc<RefCell<Option<Rc<SettingsScope>>>> = Rc::default();
        let seen_plugin = seen.clone();
        let scope_plugin = scope_cell.clone();
        let plugin = plugin_fn("registrant", Inject::names(["settings"]), move |ctx, _| {
            let seen = seen_plugin.clone();
            let scope_cell = scope_plugin.clone();
            async move {
                let service = ctx
                    .service::<SettingsService>()
                    .map_err(anyhow::Error::from)?;
                let scope =
                    Rc::new(service.register(&ctx, ns("ui-theme"), theme_schema(), options())?);
                let seen = seen.clone();
                scope.watch(move |next, _prev| {
                    seen.borrow_mut().push(next);
                    std::future::ready(Ok(()))
                });
                *scope_cell.borrow_mut() = Some(scope);
                Ok(())
            }
        });
        let fiber = booted.ctx.plugin(Rc::new(plugin), Value::Null).unwrap();
        fiber.await_ready().await.unwrap();
        assert_eq!(
            booted.service.get(&ns("ui-theme")),
            Some(json!({"theme": "dark", "fontSize": 14}))
        );

        fiber.dispose().await;
        assert!(booted.service.get(&ns("ui-theme")).is_none());
        assert!(
            booted
                .service
                .describe(SettingsDescribeOptions::default())
                .is_empty()
        );
        booted
            .backend
            .push_external(json!({"ui-theme": {"theme": "light"}}));
        settle().await;
        assert!(seen.borrow().is_empty());

        // The namespace is free again, and re-registration resolves the user
        // layer that kept living in storage while nobody owned it.
        let again = booted
            .service
            .register(&booted.ctx, ns("ui-theme"), theme_schema(), options())
            .unwrap();
        assert_eq!(again.get(), json!({"theme": "light", "fontSize": 14}));
    });
}

// ------------------------------------------------------------------ update

#[test]
fn persists_the_merged_user_section_without_baking_in_the_base_layer() {
    dsh_cordis::run(async {
        let booted = boot(json!({"ui-theme": {"theme": "light"}})).await;
        let scope = booted
            .service
            .register(
                &booted.ctx,
                ns("ui-theme"),
                theme_schema(),
                base(json!({"fontSize": 16})),
            )
            .unwrap();
        scope.update(json!({"theme": "dark"})).await.unwrap();
        let persisted = booted.backend.persisted.borrow();
        assert_eq!(persisted.len(), 1);
        assert_eq!(persisted[0].0, "ui-theme");
        assert_eq!(
            Value::Object(persisted[0].1.clone()),
            json!({"theme": "dark"})
        );
        assert_eq!(scope.get(), json!({"theme": "dark", "fontSize": 16}));
    });
}

#[test]
fn deep_merges_nested_objects_and_replaces_arrays_wholesale() {
    dsh_cordis::run(async {
        let booted = boot(json!({
            "workspace": {"retry": {"attempts": 5, "delayMs": 300}, "tags": ["a", "b"]}
        }))
        .await;
        let scope = booted
            .service
            .register(&booted.ctx, ns("workspace"), nested_schema(), options())
            .unwrap();
        scope
            .update(json!({"retry": {"attempts": 7}, "tags": ["c"]}))
            .await
            .unwrap();
        assert_eq!(
            Value::Object(booted.backend.persisted.borrow()[0].1.clone()),
            json!({"retry": {"attempts": 7, "delayMs": 300}, "tags": ["c"]})
        );
        assert_eq!(
            scope.get(),
            json!({"retry": {"attempts": 7, "delayMs": 300}, "tags": ["c"]})
        );
    });
}

#[test]
fn commits_notifies_watchers_and_emits_with_source_update() {
    dsh_cordis::run(async {
        let booted = boot(Value::Null).await;
        let events = record_updates(&booted.ctx);
        let scope = booted
            .service
            .register(&booted.ctx, ns("ui-theme"), theme_schema(), options())
            .unwrap();
        let calls: Rc<RefCell<Vec<(Value, Value)>>> = Rc::default();
        let calls_watch = calls.clone();
        scope.watch(move |next, prev| {
            calls_watch.borrow_mut().push((next, prev));
            std::future::ready(Ok(()))
        });
        scope.update(json!({"theme": "light"})).await.unwrap();
        settle().await;
        assert_eq!(
            *calls.borrow(),
            vec![(
                json!({"theme": "light", "fontSize": 14}),
                json!({"theme": "dark", "fontSize": 14})
            )]
        );
        let events = events.borrow();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, "ui-theme");
        assert_eq!(events[0].1, json!({"theme": "light", "fontSize": 14}));
        assert_eq!(events[0].2, json!({"theme": "dark", "fontSize": 14}));
        assert_eq!(events[0].3, SettingsUpdateSource::Update);
    });
}

#[test]
fn rejects_an_invalid_patch_before_persisting_anything() {
    dsh_cordis::run(async {
        let booted = boot(Value::Null).await;
        let events = record_updates(&booted.ctx);
        let scope = booted
            .service
            .register(&booted.ctx, ns("ui-theme"), theme_schema(), options())
            .unwrap();
        assert!(scope.update(json!({"fontSize": "big"})).await.is_err());
        settle().await;
        assert!(booted.backend.persisted.borrow().is_empty());
        assert!(events.borrow().is_empty());
        assert_eq!(scope.get(), json!({"theme": "dark", "fontSize": 14}));
        // The failed write must not poison the namespace queue.
        scope.update(json!({"fontSize": 18})).await.unwrap();
        assert_eq!(scope.get(), json!({"theme": "dark", "fontSize": 18}));
    });
}

#[test]
fn rejects_a_non_object_patch() {
    dsh_cordis::run(async {
        let booted = boot(Value::Null).await;
        let scope = booted
            .service
            .register(&booted.ctx, ns("ui-theme"), theme_schema(), options())
            .unwrap();
        let error = scope.update(json!([1])).await.unwrap_err();
        assert!(err_text(error).contains("must be a plain object"));
        let error = scope.replace(json!([1])).await.unwrap_err();
        assert!(err_text(error).contains("replace for \"ui-theme\""));
    });
}

#[test]
fn rejects_an_unregistered_namespace() {
    dsh_cordis::run(async {
        let booted = boot(Value::Null).await;
        let error = booted
            .service
            .update(&ns("missing"), json!({}))
            .await
            .unwrap_err();
        assert!(err_text(error).contains("not registered"));
    });
}

#[test]
fn rejects_on_a_read_only_provider_before_reaching_persist() {
    dsh_cordis::run(async {
        let booted = boot_with(Value::Null, false, 0).await;
        let scope = booted
            .service
            .register(&booted.ctx, ns("ui-theme"), theme_schema(), options())
            .unwrap();
        let error = scope.update(json!({"theme": "light"})).await.unwrap_err();
        assert!(err_text(error).contains("read-only"));
        assert!(booted.backend.persisted.borrow().is_empty());
    });
}

// ---------------------------------------------------------- deep_equal_json

#[test]
fn deep_equal_json_compares_structurally() {
    use dsh_settings::deep_equal_json;
    let cases: Vec<(Value, Value, bool)> = vec![
        (json!({"a": [1, 2]}), json!({"a": [1, 2]}), true),
        (json!({"a": [1, 2]}), json!({"a": [1]}), false),
        (json!({"a": [1]}), json!({"a": {"0": 1}}), false),
        (json!({"a": 1}), json!({"b": 1}), false),
        (json!({"a": 1}), json!({}), false),
        (json!({"a": null}), json!({"a": null}), true),
        (json!({"a": null}), json!({"a": {}}), false),
    ];
    for (a, b, equal) in cases {
        assert_eq!(deep_equal_json(&a, &b), equal, "{a} vs {b}");
    }
}

// ------------------------------------------------------- review regressions

#[test]
fn serializes_concurrent_updates_so_neither_patch_is_lost() {
    dsh_cordis::run(async {
        let booted = boot_with(Value::Null, true, 10).await;
        let scope = booted
            .service
            .register(&booted.ctx, ns("ui-theme"), theme_schema(), options())
            .unwrap();
        let first = scope.update(json!({"theme": "light"}));
        let second = scope.update(json!({"fontSize": 20}));
        let (first, second) = futures::join!(first, second);
        first.unwrap();
        second.unwrap();
        assert_eq!(
            booted.backend.doc.borrow().get("ui-theme").cloned(),
            Some(json!({"theme": "light", "fontSize": 20}))
        );
        assert_eq!(scope.get(), json!({"theme": "light", "fontSize": 20}));
    });
}

#[test]
fn contains_an_async_watcher_rejection() {
    dsh_cordis::run(async {
        let booted = boot(Value::Null).await;
        let scope = booted
            .service
            .register(&booted.ctx, ns("ui-theme"), theme_schema(), options())
            .unwrap();
        scope.watch(|_next, _prev| async { anyhow::bail!("async watcher boom") });
        booted
            .backend
            .push_external(json!({"ui-theme": {"theme": "light"}}));
        assert_eq!(scope.get()["theme"], json!("light"));
        // Containment: the failing watcher never wedges anything.
        settle().await;
    });
}

#[test]
fn replaces_the_user_section_wholesale_so_overrides_can_be_removed() {
    dsh_cordis::run(async {
        let booted = boot(json!({"ui-theme": {"theme": "light", "fontSize": 20}})).await;
        let scope = booted
            .service
            .register(
                &booted.ctx,
                ns("ui-theme"),
                theme_schema(),
                base(json!({"fontSize": 16})),
            )
            .unwrap();
        scope.replace(json!({"theme": "light"})).await.unwrap();
        // fontSize override is gone: resolution falls back to the base layer.
        assert_eq!(scope.get(), json!({"theme": "light", "fontSize": 16}));
        assert_eq!(
            booted.backend.doc.borrow().get("ui-theme").cloned(),
            Some(json!({"theme": "light"}))
        );
        scope.replace(json!({})).await.unwrap();
        assert_eq!(scope.get(), json!({"theme": "dark", "fontSize": 16}));
        assert_eq!(
            booted.backend.doc.borrow().get("ui-theme").cloned(),
            Some(json!({}))
        );
    });
}

#[test]
fn rejects_an_update_queued_after_the_registrant_fiber_disposed() {
    dsh_cordis::run(async {
        let booted = boot(Value::Null).await;
        let scope_cell: Rc<RefCell<Option<Rc<SettingsScope>>>> = Rc::default();
        let scope_plugin = scope_cell.clone();
        let plugin = plugin_fn("registrant", Inject::names(["settings"]), move |ctx, _| {
            let scope_cell = scope_plugin.clone();
            async move {
                let service = ctx
                    .service::<SettingsService>()
                    .map_err(anyhow::Error::from)?;
                *scope_cell.borrow_mut() = Some(Rc::new(service.register(
                    &ctx,
                    ns("ui-theme"),
                    theme_schema(),
                    options(),
                )?));
                Ok(())
            }
        });
        let fiber = booted.ctx.plugin(Rc::new(plugin), Value::Null).unwrap();
        fiber.await_ready().await.unwrap();
        fiber.dispose().await;
        let scope = scope_cell.borrow().clone().unwrap();
        let error = scope.update(json!({"theme": "light"})).await.unwrap_err();
        let text = err_text(error);
        assert!(
            text.contains("disposed") || text.contains("not registered"),
            "{text}"
        );
    });
}

#[test]
fn does_not_notify_a_registrant_disposed_while_its_update_was_in_flight() {
    dsh_cordis::run(async {
        let booted = boot_with(Value::Null, true, 30).await;
        let events = record_updates(&booted.ctx);
        let scope_cell: Rc<RefCell<Option<Rc<SettingsScope>>>> = Rc::default();
        let calls: Rc<RefCell<Vec<Value>>> = Rc::default();
        let scope_plugin = scope_cell.clone();
        let calls_plugin = calls.clone();
        let plugin = plugin_fn("registrant", Inject::names(["settings"]), move |ctx, _| {
            let scope_cell = scope_plugin.clone();
            let calls = calls_plugin.clone();
            async move {
                let service = ctx
                    .service::<SettingsService>()
                    .map_err(anyhow::Error::from)?;
                let scope =
                    Rc::new(service.register(&ctx, ns("ui-theme"), theme_schema(), options())?);
                let calls = calls.clone();
                scope.watch(move |next, _prev| {
                    calls.borrow_mut().push(next);
                    std::future::ready(Ok(()))
                });
                *scope_cell.borrow_mut() = Some(scope);
                Ok(())
            }
        });
        let fiber = booted.ctx.plugin(Rc::new(plugin), Value::Null).unwrap();
        fiber.await_ready().await.unwrap();
        let scope = scope_cell.borrow().clone().unwrap();
        let pending = scope.update(json!({"theme": "light"}));
        tokio::time::sleep(Duration::from_millis(5)).await;
        fiber.dispose().await;
        let _ = pending.await;
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(calls.borrow().is_empty());
        assert!(events.borrow().is_empty());
        // The persist was already in flight, so storage keeps the write —
        // but no commit reached the disposed registration.
        assert_eq!(
            booted.backend.doc.borrow().get("ui-theme").cloned(),
            Some(json!({"theme": "light"}))
        );
    });
}

#[test]
fn drains_in_flight_writes_at_service_dispose_and_rejects_later_ones() {
    dsh_cordis::run(async {
        let booted = boot_with(Value::Null, true, 20).await;
        let scope = booted
            .service
            .register(&booted.ctx, ns("ui-theme"), theme_schema(), options())
            .unwrap();
        let pending = scope.update(json!({"theme": "light"}));
        tokio::time::sleep(Duration::from_millis(5)).await;
        booted.fiber.dispose().await;
        // The teardown drained the in-flight write before completing…
        let _ = pending.await;
        let persisted_at_dispose = booted.backend.persisted.borrow().len();
        assert_eq!(persisted_at_dispose, 1);
        // …and afterwards nothing writes and new writes reject.
        let error = booted
            .service
            .update(&ns("ui-theme"), json!({"theme": "dark"}))
            .await
            .unwrap_err();
        let text = err_text(error);
        assert!(
            text.contains("disposed") || text.contains("not registered"),
            "{text}"
        );
        tokio::time::sleep(Duration::from_millis(40)).await;
        assert_eq!(
            booted.backend.persisted.borrow().len(),
            persisted_at_dispose
        );
    });
}

#[test]
fn serializes_invocations_of_one_async_watcher_in_commit_order() {
    dsh_cordis::run(async {
        let booted = boot(Value::Null).await;
        let scope = booted
            .service
            .register(&booted.ctx, ns("ui-theme"), theme_schema(), options())
            .unwrap();
        let applied: Rc<RefCell<Vec<i64>>> = Rc::default();
        let first_call = Rc::new(Cell::new(true));
        let applied_watch = applied.clone();
        scope.watch(move |next, _prev| {
            let applied = applied_watch.clone();
            let first_call = first_call.clone();
            async move {
                // The first (stale) invocation is slow; unserialized it would
                // finish last and clobber the newer applied state.
                let delay = if first_call.replace(false) { 30 } else { 0 };
                tokio::time::sleep(Duration::from_millis(delay)).await;
                applied
                    .borrow_mut()
                    .push(next["fontSize"].as_i64().unwrap());
                Ok(())
            }
        });
        booted
            .backend
            .push_external(json!({"ui-theme": {"fontSize": 1}}));
        booted
            .backend
            .push_external(json!({"ui-theme": {"fontSize": 2}}));
        wait_until(|| applied.borrow().len() == 2).await;
        assert_eq!(*applied.borrow(), vec![1, 2]);
    });
}

#[test]
fn rejects_a_write_still_queued_when_the_service_disposes() {
    dsh_cordis::run(async {
        let booted = boot_with(Value::Null, true, 20).await;
        let scope = booted
            .service
            .register(&booted.ctx, ns("ui-theme"), theme_schema(), options())
            .unwrap();
        let first = scope.update(json!({"theme": "light"}));
        let second = scope.update(json!({"fontSize": 20}));
        tokio::time::sleep(Duration::from_millis(5)).await;
        booted.fiber.dispose().await;
        first.await.unwrap();
        let error = second.await.unwrap_err();
        assert!(err_text(error).contains("disposed before the queued"));
    });
}

#[test]
fn rejects_a_write_still_queued_when_the_registrant_disposes() {
    dsh_cordis::run(async {
        let booted = boot_with(Value::Null, true, 20).await;
        let scope_cell: Rc<RefCell<Option<Rc<SettingsScope>>>> = Rc::default();
        let scope_plugin = scope_cell.clone();
        let plugin = plugin_fn("registrant", Inject::names(["settings"]), move |ctx, _| {
            let scope_cell = scope_plugin.clone();
            async move {
                let service = ctx
                    .service::<SettingsService>()
                    .map_err(anyhow::Error::from)?;
                *scope_cell.borrow_mut() = Some(Rc::new(service.register(
                    &ctx,
                    ns("ui-theme"),
                    theme_schema(),
                    options(),
                )?));
                Ok(())
            }
        });
        let fiber = booted.ctx.plugin(Rc::new(plugin), Value::Null).unwrap();
        fiber.await_ready().await.unwrap();
        let scope = scope_cell.borrow().clone().unwrap();
        let first = scope.update(json!({"theme": "light"}));
        let second = scope.update(json!({"fontSize": 20}));
        tokio::time::sleep(Duration::from_millis(5)).await;
        fiber.dispose().await;
        first.await.unwrap();
        let error = second.await.unwrap_err();
        assert!(err_text(error).contains("registration was disposed before the queued"));
    });
}

// ----------------------------------------------------------------- publish

#[test]
fn notifies_watchers_of_an_external_change_with_source_provider() {
    dsh_cordis::run(async {
        let booted = boot(Value::Null).await;
        let events = record_updates(&booted.ctx);
        let scope = booted
            .service
            .register(&booted.ctx, ns("ui-theme"), theme_schema(), options())
            .unwrap();
        let calls: Rc<RefCell<Vec<(Value, Value)>>> = Rc::default();
        let calls_watch = calls.clone();
        scope.watch(move |next, prev| {
            calls_watch.borrow_mut().push((next, prev));
            std::future::ready(Ok(()))
        });
        booted
            .backend
            .push_external(json!({"ui-theme": {"theme": "light"}}));
        wait_until(|| !calls.borrow().is_empty()).await;
        assert_eq!(
            calls.borrow()[0],
            (
                json!({"theme": "light", "fontSize": 14}),
                json!({"theme": "dark", "fontSize": 14})
            )
        );
        assert_eq!(events.borrow()[0].3, SettingsUpdateSource::Provider);
    });
}

#[test]
fn stays_silent_when_the_resolved_value_is_deep_equal() {
    dsh_cordis::run(async {
        let booted = boot(json!({"ui-theme": {"theme": "light"}})).await;
        let events = record_updates(&booted.ctx);
        let scope = booted
            .service
            .register(&booted.ctx, ns("ui-theme"), theme_schema(), options())
            .unwrap();
        let calls: Rc<RefCell<Vec<Value>>> = Rc::default();
        let calls_watch = calls.clone();
        scope.watch(move |next, _prev| {
            calls_watch.borrow_mut().push(next);
            std::future::ready(Ok(()))
        });
        booted
            .backend
            .push_external(json!({"ui-theme": {"theme": "light"}}));
        settle().await;
        assert!(calls.borrow().is_empty());
        assert!(events.borrow().is_empty());
    });
}

#[test]
fn keeps_the_last_good_value_for_an_invalid_section_while_other_namespaces_commit() {
    dsh_cordis::run(async {
        let booted = boot(Value::Null).await;
        let events = record_updates(&booted.ctx);
        let theme = booted
            .service
            .register(&booted.ctx, ns("ui-theme"), theme_schema(), options())
            .unwrap();
        let workspace = booted
            .service
            .register(&booted.ctx, ns("workspace"), nested_schema(), options())
            .unwrap();
        booted.backend.push_external(json!({
            "ui-theme": {"fontSize": "broken"},
            "workspace": {"retry": {"attempts": 9}},
        }));
        settle().await;
        assert_eq!(theme.get(), json!({"theme": "dark", "fontSize": 14}));
        assert_eq!(
            workspace.get(),
            json!({"retry": {"attempts": 9, "delayMs": 100}, "tags": ["default"]})
        );
        let namespaces: Vec<String> = events.borrow().iter().map(|e| e.0.clone()).collect();
        assert_eq!(namespaces, vec!["workspace".to_string()]);
    });
}

#[test]
fn recovers_from_a_bad_section_once_storage_turns_valid_again() {
    dsh_cordis::run(async {
        let booted = boot(Value::Null).await;
        let scope = booted
            .service
            .register(&booted.ctx, ns("ui-theme"), theme_schema(), options())
            .unwrap();
        booted
            .backend
            .push_external(json!({"ui-theme": {"fontSize": "broken"}}));
        assert_eq!(scope.get(), json!({"theme": "dark", "fontSize": 14}));
        booted
            .backend
            .push_external(json!({"ui-theme": {"fontSize": 18}}));
        assert_eq!(scope.get(), json!({"theme": "dark", "fontSize": 18}));
    });
}

// ---------------------------------------------------- third-review watchers

#[test]
fn skips_a_queued_watch_invocation_whose_disposer_ran_before_it_started() {
    dsh_cordis::run(async {
        let booted = boot(Value::Null).await;
        let scope = booted
            .service
            .register(&booted.ctx, ns("ui-theme"), theme_schema(), options())
            .unwrap();
        let calls: Rc<RefCell<Vec<Value>>> = Rc::default();
        let calls_watch = calls.clone();
        let disposer = scope.watch(move |next, _prev| {
            calls_watch.borrow_mut().push(next);
            std::future::ready(Ok(()))
        });
        // The commit queues the invocation as a task; the disposer runs in
        // the same synchronous frame, before that invocation could start.
        booted
            .backend
            .push_external(json!({"ui-theme": {"theme": "light"}}));
        disposer.dispose();
        settle().await;
        assert!(calls.borrow().is_empty());
    });
}

#[test]
fn waits_for_an_in_flight_watch_invocation_at_service_dispose() {
    dsh_cordis::run(async {
        let booted = boot(Value::Null).await;
        let scope = booted
            .service
            .register(&booted.ctx, ns("ui-theme"), theme_schema(), options())
            .unwrap();
        type Release = Rc<RefCell<Option<futures::channel::oneshot::Sender<()>>>>;
        let release: Release = Rc::default();
        let finished = Rc::new(Cell::new(false));
        let release_watch = release.clone();
        let finished_watch = finished.clone();
        scope.watch(move |_next, _prev| {
            let release = release_watch.clone();
            let finished = finished_watch.clone();
            async move {
                let (tx, rx) = futures::channel::oneshot::channel::<()>();
                *release.borrow_mut() = Some(tx);
                let _ = rx.await;
                finished.set(true);
                Ok(())
            }
        });
        booted
            .backend
            .push_external(json!({"ui-theme": {"theme": "light"}}));
        wait_until(|| release.borrow().is_some()).await;
        let disposed = Rc::new(Cell::new(false));
        let disposed_task = disposed.clone();
        let fiber = booted.fiber.clone();
        tokio::task::spawn_local(async move {
            fiber.dispose().await;
            disposed_task.set(true);
        });
        tokio::time::sleep(Duration::from_millis(15)).await;
        assert!(!disposed.get());
        release.borrow_mut().take().unwrap().send(()).unwrap();
        wait_until(|| disposed.get()).await;
        assert!(finished.get());
    });
}

// ------------------------------------------------------------------- watch

#[test]
fn stops_after_its_disposer_runs() {
    dsh_cordis::run(async {
        let booted = boot(Value::Null).await;
        let scope = booted
            .service
            .register(&booted.ctx, ns("ui-theme"), theme_schema(), options())
            .unwrap();
        let calls: Rc<RefCell<Vec<Value>>> = Rc::default();
        let calls_watch = calls.clone();
        let disposer = scope.watch(move |next, _prev| {
            calls_watch.borrow_mut().push(next);
            std::future::ready(Ok(()))
        });
        disposer.dispose();
        booted
            .backend
            .push_external(json!({"ui-theme": {"theme": "light"}}));
        settle().await;
        assert!(calls.borrow().is_empty());
    });
}

#[test]
fn contains_a_failing_watcher_without_blocking_the_commit_or_other_watchers() {
    dsh_cordis::run(async {
        let booted = boot(Value::Null).await;
        let events = record_updates(&booted.ctx);
        let scope = booted
            .service
            .register(&booted.ctx, ns("ui-theme"), theme_schema(), options())
            .unwrap();
        scope.watch(|_next, _prev| async { anyhow::bail!("watcher boom") });
        let calls: Rc<RefCell<Vec<Value>>> = Rc::default();
        let calls_watch = calls.clone();
        scope.watch(move |next, _prev| {
            calls_watch.borrow_mut().push(next);
            std::future::ready(Ok(()))
        });
        booted
            .backend
            .push_external(json!({"ui-theme": {"theme": "light"}}));
        wait_until(|| calls.borrow().len() == 1).await;
        assert_eq!(events.borrow().len(), 1);
        assert_eq!(scope.get(), json!({"theme": "light", "fontSize": 14}));
    });
}

// ------------------------------------------------- install_settings_section

type SourceCell = Rc<RefCell<Rc<dyn Fn() -> Value>>>;

fn source_cell(entry: Value) -> SourceCell {
    Rc::new(RefCell::new(
        Rc::new(move || entry.clone()) as Rc<dyn Fn() -> Value>
    ))
}

fn helper_schema() -> Schema {
    Schema::object([("theme", Schema::string().default(json!("default")))])
}

fn hooks_for(current: &SourceCell, on_change: impl Fn() + 'static) -> SettingsSectionHooks {
    let current = current.clone();
    SettingsSectionHooks {
        set_source: Rc::new(move |source| {
            *current.borrow_mut() = source;
        }),
        on_change: Rc::new(on_change),
        validate: None,
    }
}

#[test]
fn drives_the_source_through_attach_live_commits_and_detach() {
    dsh_cordis::run(async {
        let app = dsh_cordis::App::new();
        let ctx = app.root();
        let entry = json!({"theme": "entry"});
        let current = source_cell(entry.clone());
        let changes = Rc::new(Cell::new(0u32));
        let changes_hook = changes.clone();
        install_settings_section(
            &ctx,
            ns("helper-ns"),
            helper_schema(),
            entry.clone(),
            hooks_for(&current, move || changes_hook.set(changes_hook.get() + 1)),
        )
        .unwrap();
        settle().await;
        // No settings service mounted: nothing ran, the entry stays
        // authoritative.
        assert_eq!((current.borrow().clone())(), json!({"theme": "entry"}));
        assert_eq!(changes.get(), 0);

        let backend = common::MemoryBackend::new(json!({"helper-ns": {"theme": "user"}}));
        let plugin_backend = backend.clone();
        let plugin = plugin_fn("memory-settings", Inject::default(), move |ctx, _| {
            let backend = plugin_backend.clone();
            async move { dsh_settings::mount(&ctx, backend).await.map(|_| ()) }
        });
        let fiber = ctx.plugin(Rc::new(plugin), Value::Null).unwrap();
        fiber.await_ready().await.unwrap();
        wait_until({
            let current = current.clone();
            move || (current.borrow().clone())() == json!({"theme": "user"})
        })
        .await;
        assert_eq!(changes.get(), 1);

        let service = ctx.try_service::<SettingsService>().unwrap();
        service
            .update(&ns("helper-ns"), json!({"theme": "live"}))
            .await
            .unwrap();
        wait_until({
            let changes = changes.clone();
            move || changes.get() == 2
        })
        .await;
        assert_eq!((current.borrow().clone())(), json!({"theme": "live"}));

        fiber.dispose().await;
        wait_until({
            let changes = changes.clone();
            move || changes.get() == 3
        })
        .await;
        assert_eq!((current.borrow().clone())(), json!({"theme": "entry"}));
    });
}

#[test]
fn stays_silent_when_the_consumer_itself_unloads() {
    dsh_cordis::run(async {
        let booted = boot(json!({"helper-ns": {"theme": "user"}})).await;
        let entry = json!({"theme": "entry"});
        let current = source_cell(entry.clone());
        let changes: Rc<RefCell<Vec<String>>> = Rc::default();
        let changes_hook = changes.clone();
        let current_hook = current.clone();
        let entry_plugin = entry.clone();
        let consumer = plugin_fn("consumer", Inject::default(), move |ctx, _| {
            let current = current_hook.clone();
            let changes = changes_hook.clone();
            let entry = entry_plugin.clone();
            async move {
                let record = {
                    let current = current.clone();
                    let changes = changes.clone();
                    move || {
                        let theme = (current.borrow().clone())()["theme"]
                            .as_str()
                            .unwrap()
                            .to_string();
                        changes.borrow_mut().push(theme);
                    }
                };
                install_settings_section(
                    &ctx,
                    ns("helper-ns"),
                    helper_schema(),
                    entry.clone(),
                    hooks_for(&current, record),
                )
                .map_err(anyhow::Error::from)?;
                Ok(())
            }
        });
        let consumer = booted.ctx.plugin(Rc::new(consumer), Value::Null).unwrap();
        consumer.await_ready().await.unwrap();
        wait_until({
            let changes = changes.clone();
            move || *changes.borrow() == vec!["user".to_string()]
        })
        .await;

        // The consumer's own teardown must not re-derive anything.
        consumer.dispose().await;
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(*changes.borrow(), vec!["user".to_string()]);
    });
}

#[test]
fn stays_silent_for_a_stored_change_that_lands_while_the_consumer_unloads() {
    dsh_cordis::run(async {
        let booted = boot(json!({"helper-ns": {"theme": "user"}})).await;
        let entry = json!({"theme": "entry"});
        let current = source_cell(entry.clone());
        let changes: Rc<RefCell<Vec<String>>> = Rc::default();
        let changes_hook = changes.clone();
        let current_hook = current.clone();
        let entry_plugin = entry.clone();
        let consumer = plugin_fn("consumer", Inject::default(), move |ctx, _| {
            let current = current_hook.clone();
            let changes = changes_hook.clone();
            let entry = entry_plugin.clone();
            async move {
                let record = {
                    let current = current.clone();
                    let changes = changes.clone();
                    move || {
                        let theme = (current.borrow().clone())()["theme"]
                            .as_str()
                            .unwrap()
                            .to_string();
                        changes.borrow_mut().push(theme);
                    }
                };
                install_settings_section(
                    &ctx,
                    ns("helper-ns"),
                    helper_schema(),
                    entry.clone(),
                    hooks_for(&current, record),
                )
                .map_err(anyhow::Error::from)?;
                Ok(())
            }
        });
        let consumer = booted.ctx.plugin(Rc::new(consumer), Value::Null).unwrap();
        consumer.await_ready().await.unwrap();
        wait_until({
            let changes = changes.clone();
            move || *changes.borrow() == vec!["user".to_string()]
        })
        .await;

        // A change racing the unload must not notify either.
        let disposal = {
            let consumer = consumer.clone();
            tokio::task::spawn_local(async move { consumer.dispose().await })
        };
        booted
            .backend
            .push_external(json!({"helper-ns": {"theme": "racing"}}));
        disposal.await.unwrap();
        settle().await;
        assert_eq!(*changes.borrow(), vec!["user".to_string()]);
    });
}

// ------------------------------------------------------------------ mutate

fn keyed_schema() -> Schema {
    Schema::object([
        ("apiKey", Schema::string().role("secret")),
        ("baseURL", Schema::string()),
        ("reasoning", Schema::string()),
    ])
}

fn user_of(service: &SettingsService, target: &SettingsNamespace) -> Option<Value> {
    service
        .describe(SettingsDescribeOptions::default())
        .into_iter()
        .find(|d| d.ns == *target)
        .unwrap()
        .user
}

#[test]
fn removes_one_field_without_touching_a_secret_the_caller_never_saw() {
    dsh_cordis::run(async {
        // The data-loss shape this exists to prevent: a configuration UI
        // reads the REDACTED descriptor (no apiKey), the user resets baseURL,
        // and the client rebuilds the section from what it holds. A wholesale
        // replace of that rebuild deletes the stored literal key; a path
        // unset cannot.
        let booted = boot(json!({
            "keyed": {"apiKey": "sk-stored", "baseURL": "https://user", "reasoning": "high"}
        }))
        .await;
        booted
            .service
            .register(&booted.ctx, ns("keyed"), keyed_schema(), options())
            .unwrap();
        let redacted = booted
            .service
            .describe(SettingsDescribeOptions {
                redact_secrets: true,
            })
            .into_iter()
            .find(|d| d.ns == ns("keyed"))
            .unwrap();
        assert_eq!(
            redacted.user,
            Some(json!({"baseURL": "https://user", "reasoning": "high"}))
        );

        booted
            .service
            .mutate(
                &ns("keyed"),
                vec![SettingsPathOp::Unset {
                    path: vec!["baseURL".into()],
                }],
            )
            .await
            .unwrap();

        assert_eq!(
            user_of(&booted.service, &ns("keyed")),
            Some(json!({"apiKey": "sk-stored", "reasoning": "high"}))
        );
    });
}

#[test]
fn applies_set_and_unset_in_one_write_in_order() {
    dsh_cordis::run(async {
        let booted =
            boot(json!({"keyed": {"apiKey": "sk-stored", "baseURL": "https://old"}})).await;
        booted
            .service
            .register(&booted.ctx, ns("keyed"), keyed_schema(), options())
            .unwrap();
        booted
            .service
            .mutate(
                &ns("keyed"),
                vec![
                    SettingsPathOp::Set {
                        path: vec!["baseURL".into()],
                        value: json!("https://new"),
                    },
                    SettingsPathOp::Set {
                        path: vec!["reasoning".into()],
                        value: json!("low"),
                    },
                    SettingsPathOp::Unset {
                        path: vec!["reasoning".into()],
                    },
                ],
            )
            .await
            .unwrap();
        assert_eq!(
            user_of(&booted.service, &ns("keyed")),
            Some(json!({"apiKey": "sk-stored", "baseURL": "https://new"}))
        );
    });
}

#[test]
fn reads_the_section_as_it_stands_at_the_front_of_the_queue_not_at_call_time() {
    dsh_cordis::run(async {
        // Two concurrent writers: the mutate is issued against the pre-update
        // section but must observe the update that ran before it.
        let booted = boot(json!({"keyed": {"apiKey": "sk-stored"}})).await;
        booted
            .service
            .register(&booted.ctx, ns("keyed"), keyed_schema(), options())
            .unwrap();
        let first = booted.service.update(
            &ns("keyed"),
            json!({"baseURL": "https://first", "reasoning": "high"}),
        );
        let second = booted.service.mutate(
            &ns("keyed"),
            vec![SettingsPathOp::Unset {
                path: vec!["reasoning".into()],
            }],
        );
        let (first, second) = futures::join!(first, second);
        first.unwrap();
        second.unwrap();
        assert_eq!(
            user_of(&booted.service, &ns("keyed")),
            Some(json!({"apiKey": "sk-stored", "baseURL": "https://first"}))
        );
    });
}

#[test]
fn creates_intermediate_objects_for_a_nested_set_and_leaves_an_absent_unset_alone() {
    dsh_cordis::run(async {
        let booted = boot(Value::Null).await;
        booted
            .service
            .register(&booted.ctx, ns("workspace"), nested_schema(), options())
            .unwrap();
        booted
            .service
            .mutate(
                &ns("workspace"),
                vec![SettingsPathOp::Set {
                    path: vec!["retry".into(), "attempts".into()],
                    value: json!(5),
                }],
            )
            .await
            .unwrap();
        assert_eq!(
            user_of(&booted.service, &ns("workspace")),
            Some(json!({"retry": {"attempts": 5}}))
        );
        booted
            .service
            .mutate(
                &ns("workspace"),
                vec![SettingsPathOp::Unset {
                    path: vec!["missing".into(), "deep".into()],
                }],
            )
            .await
            .unwrap();
        assert_eq!(
            user_of(&booted.service, &ns("workspace")),
            Some(json!({"retry": {"attempts": 5}}))
        );
    });
}

#[test]
fn edits_one_leaf_of_an_existing_nested_object_without_replacing_its_siblings() {
    dsh_cordis::run(async {
        let booted = boot(json!({"workspace": {"retry": {"attempts": 5, "delayMs": 250}}})).await;
        booted
            .service
            .register(&booted.ctx, ns("workspace"), nested_schema(), options())
            .unwrap();
        booted
            .service
            .mutate(
                &ns("workspace"),
                vec![SettingsPathOp::Set {
                    path: vec!["retry".into(), "delayMs".into()],
                    value: json!(900),
                }],
            )
            .await
            .unwrap();
        assert_eq!(
            user_of(&booted.service, &ns("workspace")),
            Some(json!({"retry": {"attempts": 5, "delayMs": 900}}))
        );
    });
}

#[test]
fn addresses_the_section_itself_through_the_empty_path() {
    dsh_cordis::run(async {
        let booted =
            boot(json!({"keyed": {"apiKey": "sk-stored", "baseURL": "https://user"}})).await;
        booted
            .service
            .register(&booted.ctx, ns("keyed"), keyed_schema(), options())
            .unwrap();
        booted
            .service
            .mutate(
                &ns("keyed"),
                vec![SettingsPathOp::Set {
                    path: vec![],
                    value: json!({"reasoning": "low"}),
                }],
            )
            .await
            .unwrap();
        assert_eq!(
            user_of(&booted.service, &ns("keyed")),
            Some(json!({"reasoning": "low"}))
        );
        booted
            .service
            .mutate(&ns("keyed"), vec![SettingsPathOp::Unset { path: vec![] }])
            .await
            .unwrap();
        assert_eq!(user_of(&booted.service, &ns("keyed")), Some(json!({})));
    });
}

#[test]
fn refuses_a_non_object_at_the_section_root_leaving_the_stored_section_alone() {
    dsh_cordis::run(async {
        let booted = boot(json!({"keyed": {"apiKey": "sk-stored"}})).await;
        booted
            .service
            .register(&booted.ctx, ns("keyed"), keyed_schema(), options())
            .unwrap();
        let error = booted
            .service
            .mutate(
                &ns("keyed"),
                vec![SettingsPathOp::Set {
                    path: vec![],
                    value: json!("a whole section"),
                }],
            )
            .await
            .unwrap_err();
        assert!(err_text(error).contains("setting the section root requires a plain object"));
        assert_eq!(
            user_of(&booted.service, &ns("keyed")),
            Some(json!({"apiKey": "sk-stored"}))
        );
    });
}

// -------------------------------------------- revision and conflict detection

fn rev_schema() -> Schema {
    Schema::object([
        ("a", Schema::string().default(json!("base-a"))),
        ("b", Schema::string()),
    ])
}

fn revision_of(service: &SettingsService, target: &SettingsNamespace) -> u64 {
    service
        .describe(SettingsDescribeOptions::default())
        .into_iter()
        .find(|d| d.ns == *target)
        .unwrap()
        .revision
}

fn record_documents(ctx: &Context) -> Rc<RefCell<Vec<(String, u64)>>> {
    let documents: Rc<RefCell<Vec<(String, u64)>>> = Rc::default();
    let sink = documents.clone();
    ctx.on::<dsh_settings::SettingsDocumentUpdated, _, _>(Default::default(), move |_ctx, args| {
        sink.borrow_mut().push((args.0.to_string(), args.1));
        std::future::ready(None::<()>)
    })
    .unwrap();
    documents
}

#[test]
fn refuses_a_write_whose_expected_revision_is_stale_leaving_the_winner_in_place() {
    dsh_cordis::run(async {
        // Two editors open the same namespace, both holding revision 0. The
        // first to land wins; the second must be told rather than overwrite.
        let booted = boot(Value::Null).await;
        booted
            .service
            .register(&booted.ctx, ns("rev"), rev_schema(), options())
            .unwrap();
        let opened = revision_of(&booted.service, &ns("rev"));

        booted
            .service
            .update_expecting(&ns("rev"), json!({"b": "from-tab-B"}), Some(opened))
            .await
            .unwrap();
        let error = booted
            .service
            .update_expecting(&ns("rev"), json!({"a": "from-tab-A"}), Some(opened))
            .await
            .unwrap_err();
        assert!(err_text(error).contains("changed since it was read (expected revision 0, now 1)"));
        assert_eq!(
            user_of(&booted.service, &ns("rev")),
            Some(json!({"b": "from-tab-B"}))
        );
    });
}

#[test]
fn carries_the_machine_code_and_both_revisions_on_the_refusal() {
    dsh_cordis::run(async {
        let booted = boot(Value::Null).await;
        booted
            .service
            .register(&booted.ctx, ns("rev"), rev_schema(), options())
            .unwrap();
        booted
            .service
            .update(&ns("rev"), json!({"b": "first"}))
            .await
            .unwrap();
        let error = booted
            .service
            .update_expecting(&ns("rev"), json!({"b": "second"}), Some(0))
            .await
            .unwrap_err();
        let conflict = error.downcast_ref::<SettingsConflictError>().unwrap();
        assert_eq!(SettingsConflictError::CODE, "SETTINGS_CONFLICT");
        assert_eq!(conflict.expected, 0);
        assert_eq!(conflict.actual, 1);
    });
}

#[test]
fn accepts_a_write_that_carries_no_expectation_at_all() {
    dsh_cordis::run(async {
        let booted = boot(Value::Null).await;
        booted
            .service
            .register(&booted.ctx, ns("rev"), rev_schema(), options())
            .unwrap();
        booted
            .service
            .update(&ns("rev"), json!({"b": "one"}))
            .await
            .unwrap();
        booted
            .service
            .update(&ns("rev"), json!({"b": "two"}))
            .await
            .unwrap();
        assert_eq!(revision_of(&booted.service, &ns("rev")), 2);
    });
}

#[test]
fn announces_a_raw_change_whose_resolved_value_is_unchanged() {
    dsh_cordis::run(async {
        // Storing an override equal to the schema default leaves `value`
        // alone but changes what the document says: the field is now
        // overridden, not inherited, and another tab has to learn that.
        let booted = boot(Value::Null).await;
        booted
            .service
            .register(&booted.ctx, ns("rev"), rev_schema(), options())
            .unwrap();
        let documents = record_documents(&booted.ctx);
        let resolved = record_updates(&booted.ctx);

        booted
            .service
            .update(&ns("rev"), json!({"a": "base-a"}))
            .await
            .unwrap();
        settle().await;

        assert_eq!(*documents.borrow(), vec![("rev".to_string(), 1)]);
        assert!(resolved.borrow().is_empty());
        assert_eq!(
            user_of(&booted.service, &ns("rev")),
            Some(json!({"a": "base-a"}))
        );
    });
}

#[test]
fn does_not_move_the_revision_when_a_write_stores_an_identical_section() {
    dsh_cordis::run(async {
        let booted = boot(json!({"rev": {"b": "same"}})).await;
        booted
            .service
            .register(&booted.ctx, ns("rev"), rev_schema(), options())
            .unwrap();
        let documents = record_documents(&booted.ctx);
        booted
            .service
            .update(&ns("rev"), json!({"b": "same"}))
            .await
            .unwrap();
        settle().await;
        assert!(documents.borrow().is_empty());
        assert_eq!(revision_of(&booted.service, &ns("rev")), 0);
    });
}

#[test]
fn moves_the_revision_for_an_external_edit_the_provider_publishes() {
    dsh_cordis::run(async {
        let booted = boot(Value::Null).await;
        booted
            .service
            .register(&booted.ctx, ns("rev"), rev_schema(), options())
            .unwrap();
        let documents = record_documents(&booted.ctx);
        booted.service.publish(
            json!({"rev": {"b": "edited on disk"}})
                .as_object()
                .unwrap()
                .clone(),
        );
        settle().await;
        assert_eq!(*documents.borrow(), vec![("rev".to_string(), 1)]);
        // An editor that opened before the external edit is now refused.
        let error = booted
            .service
            .update_expecting(&ns("rev"), json!({"b": "stale"}), Some(0))
            .await
            .unwrap_err();
        assert!(error.downcast_ref::<SettingsConflictError>().is_some());
    });
}

#[test]
fn moves_the_revision_past_a_stored_section_that_was_not_an_object() {
    dsh_cordis::run(async {
        // A hand-edited file can leave a namespace holding a scalar. The
        // resolved value keeps its last good reading, and the repair that
        // follows still has to announce itself.
        let booted = boot(Value::Null).await;
        booted
            .service
            .register(&booted.ctx, ns("rev"), rev_schema(), options())
            .unwrap();
        booted
            .service
            .publish(json!({"rev": "not a section"}).as_object().unwrap().clone());
        settle().await;
        let documents = record_documents(&booted.ctx);
        booted.service.publish(
            json!({"rev": {"b": "repaired by hand"}})
                .as_object()
                .unwrap()
                .clone(),
        );
        settle().await;
        assert_eq!(*documents.borrow(), vec![("rev".to_string(), 1)]);
    });
}
