//! Behavioral contract tests ported from upstream `tests/store.spec.ts`.
//! Live-iterator "generation" cases are not ported: Rust iteration is
//! snapshot-based (see the store module docs). The "exact disposer" case is
//! covered through the effect handle's label — the handle *is* the cordis
//! disposer here.

use dsh_cordis::{App, Context};
use dsh_scope::{
    AnonymousEntries, NamedEntries, Scope, ScopeKey, ScopeLayer, ScopedLayers, Undo, create_scope,
};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

struct TestLayer {
    named: NamedEntries<i32>,
    anonymous: AnonymousEntries<&'static str>,
}

impl TestLayer {
    fn new(scope: Option<&ScopeKey>) -> Self {
        let scoped = scope.is_some();
        TestLayer {
            named: NamedEntries::new(move |name| {
                anyhow::anyhow!(
                    "{} duplicate: {name}",
                    if scoped { "scoped" } else { "global" }
                )
            }),
            anonymous: AnonymousEntries::new(),
        }
    }
}

impl ScopeLayer for TestLayer {
    fn is_empty(&self) -> bool {
        self.named.is_empty() && self.anonymous.is_empty()
    }
}

async fn mint_scope(ctx: &Context, key: ScopeKey) -> Scope {
    let scope = create_scope(ctx, key, None).unwrap();
    scope.fiber().await_ready().await.unwrap();
    scope
}

fn expect_err<T>(result: anyhow::Result<T>) -> anyhow::Error {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(error) => error,
    }
}

#[test]
fn named_entries_own_duplicates_lookup_order_and_exact_idempotent_undo() {
    let entries = NamedEntries::<i32>::new(|name| anyhow::anyhow!("caller duplicate: {name}"));
    let undo_a = entries.insert("a", 1).unwrap();
    let undo_b = entries.insert("b", 2).unwrap();

    assert_eq!(entries.keys(), vec!["a", "b"]);
    assert_eq!(entries.entries(), vec![("a".into(), 1), ("b".into(), 2)]);
    assert_eq!(entries.values(), vec![1, 2]);
    assert_eq!(entries.get("a"), Some(1));
    assert_eq!(entries.get("missing"), None);
    assert!(entries.has("b"));
    assert!(!entries.has("missing"));
    assert!(!entries.is_empty());
    let err = expect_err(entries.insert("a", 3));
    assert_eq!(err.to_string(), "caller duplicate: a");

    // A spent undo never removes a later insertion under the same name.
    undo_a();
    let _ = entries.insert("a", 3).unwrap();
    undo_a();
    assert_eq!(entries.get("a"), Some(3));
    undo_b();
    assert_eq!(entries.entries(), vec![("a".into(), 3)]);
}

#[test]
fn anonymous_entries_own_equal_values_independently_with_idempotent_undo() {
    let entries = AnonymousEntries::<i32>::new();
    let undo_first = entries.append(7);
    let undo_second = entries.append(7);

    assert_eq!(entries.values(), vec![7, 7]);
    undo_first();
    undo_first();
    assert_eq!(entries.values(), vec![7]);
    undo_second();
    assert!(entries.is_empty());
}

#[test]
fn constructs_global_eagerly_while_reads_stay_non_creating_and_merge_in_order() {
    let created: Rc<RefCell<Vec<Option<ScopeKey>>>> = Rc::default();
    let created2 = created.clone();
    let changed: Rc<Cell<u32>> = Rc::default();
    let changed2 = changed.clone();
    let layers = ScopedLayers::new(
        move |scope| {
            created2.borrow_mut().push(scope.cloned());
            Ok(TestLayer::new(scope))
        },
        move || {
            changed2.set(changed2.get() + 1);
            Ok(())
        },
    )
    .unwrap();
    let key = ScopeKey::new();
    let _ = layers.global().named.insert("a", 1).unwrap();
    let _ = layers.global().named.insert("shared", 2).unwrap();

    assert_eq!(*created.borrow(), vec![None]);
    assert!(layers.peek(None).is_none());
    assert!(layers.peek(Some(&key)).is_none());
    assert_eq!(
        layers.merge(Some(&key), |layer| &layer.named),
        vec![("a".into(), 1), ("shared".into(), 2)]
    );
    assert_eq!(*created.borrow(), vec![None]);
    assert_eq!(changed.get(), 0);
}

#[test]
fn uses_the_scoped_context_for_lazy_visibility_and_ownership_and_reclaims_only_empty_aggregates() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let key = ScopeKey::new();
        let scope = mint_scope(&ctx, key.clone()).await;
        let changed: Rc<Cell<u32>> = Rc::default();
        let changed2 = changed.clone();
        let created: Rc<RefCell<Vec<Option<ScopeKey>>>> = Rc::default();
        let created2 = created.clone();
        let layers = ScopedLayers::new(
            move |selected| {
                created2.borrow_mut().push(selected.cloned());
                Ok(TestLayer::new(selected))
            },
            move || {
                changed2.set(changed2.get() + 1);
                Ok(())
            },
        )
        .unwrap();
        let _ = layers.global().named.insert("a", 1).unwrap();
        let _ = layers.global().named.insert("shared", 1).unwrap();
        let remove_named = layers
            .effect(
                &scope.ctx(),
                |layer| layer.named.insert("shared", 2),
                "test.named",
                false,
            )
            .unwrap();
        let remove_tail = layers
            .effect(
                &scope.ctx(),
                |layer| layer.named.insert("c", 3),
                "test.tail",
                false,
            )
            .unwrap();
        let remove_anonymous = layers
            .effect(
                &scope.ctx(),
                |layer| Ok(layer.anonymous.append("kept")),
                "test.anonymous",
                false,
            )
            .unwrap();

        assert_eq!(*created.borrow(), vec![None, Some(key.clone())]);
        assert_eq!(
            layers.merge(Some(&key), |layer| &layer.named),
            vec![("a".into(), 1), ("shared".into(), 2), ("c".into(), 3)]
        );
        assert_eq!(changed.get(), 0);
        remove_named.dispose().await;
        assert!(layers.peek(Some(&key)).is_some());
        assert_eq!(
            layers.merge(Some(&key), |layer| &layer.named),
            vec![("a".into(), 1), ("shared".into(), 1), ("c".into(), 3)]
        );
        remove_tail.dispose().await;
        assert!(layers.peek(Some(&key)).is_some());
        remove_anonymous.dispose().await;
        assert!(layers.peek(Some(&key)).is_none());
        scope.dispose().await;
    });
}

#[test]
fn runs_action_notification_undo_and_disposal_notification_in_order_with_labels() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let events: Rc<RefCell<Vec<&'static str>>> = Rc::default();
        let notify_log = events.clone();
        let layers = ScopedLayers::new(
            |scope| Ok(TestLayer::new(scope)),
            move || {
                notify_log.borrow_mut().push("notify");
                Ok(())
            },
        )
        .unwrap();
        let action_log = events.clone();
        let handle = layers
            .effect(
                &ctx,
                |layer| {
                    action_log.borrow_mut().push("action");
                    let undo = layer.named.insert("x", 1)?;
                    let undo_log = action_log.clone();
                    Ok(Box::new(move || {
                        undo_log.borrow_mut().push("undo");
                        undo();
                    }) as Undo)
                },
                "store.order",
                true,
            )
            .unwrap();

        assert_eq!(*events.borrow(), vec!["action", "notify"]);
        assert_eq!(handle.label(), "store.order");
        handle.dispose().await;
        handle.dispose().await;
        assert_eq!(*events.borrow(), vec!["action", "notify", "undo", "notify"]);
        assert!(layers.global().is_empty());
    });
}

#[test]
fn cleans_up_failed_factories_and_empty_failed_actions_without_discarding_existing_layers() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let key = ScopeKey::new();
        let scope = mint_scope(&ctx, key.clone()).await;
        let fail_factory: Rc<Cell<bool>> = Rc::new(Cell::new(true));
        let fail_factory2 = fail_factory.clone();
        let layers = ScopedLayers::new(
            move |selected| {
                if selected.is_some() && fail_factory2.get() {
                    anyhow::bail!("factory failed");
                }
                Ok(TestLayer::new(selected))
            },
            || Ok(()),
        )
        .unwrap();

        let err = expect_err(layers.effect(
            &scope.ctx(),
            |layer| layer.named.insert("never", 1),
            "store.factory",
            false,
        ));
        assert!(err.to_string().contains("factory failed"), "{err:#}");
        assert!(layers.peek(Some(&key)).is_none());

        fail_factory.set(false);
        let err = expect_err(layers.effect(
            &scope.ctx(),
            |_layer| -> anyhow::Result<Undo> { anyhow::bail!("action failed") },
            "store.action",
            false,
        ));
        assert!(err.to_string().contains("action failed"), "{err:#}");
        assert!(layers.peek(Some(&key)).is_none());

        let dispose = layers
            .effect(
                &scope.ctx(),
                |layer| layer.named.insert("kept", 1),
                "store.kept",
                false,
            )
            .unwrap();
        let err = expect_err(layers.effect(
            &scope.ctx(),
            |_layer| -> anyhow::Result<Undo> { anyhow::bail!("second action failed") },
            "store.existing-action",
            false,
        ));
        assert!(err.to_string().contains("second action failed"), "{err:#}");
        assert_eq!(layers.peek(Some(&key)).unwrap().named.get("kept"), Some(1));
        dispose.dispose().await;
        scope.dispose().await;
    });
}

#[test]
fn rolls_back_a_scoped_insertion_when_notification_fails() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let key = ScopeKey::new();
        let scope = mint_scope(&ctx, key.clone()).await;
        let events: Rc<RefCell<Vec<&'static str>>> = Rc::default();
        let notifications: Rc<Cell<u32>> = Rc::default();
        let notify_log = events.clone();
        let layers = ScopedLayers::new(
            |selected| Ok(TestLayer::new(selected)),
            move || {
                notify_log.borrow_mut().push("notify");
                notifications.set(notifications.get() + 1);
                if notifications.get() == 1 {
                    anyhow::bail!("change failed");
                }
                Ok(())
            },
        )
        .unwrap();

        let action_log = events.clone();
        let err = expect_err(layers.effect(
            &scope.ctx(),
            |layer| {
                let undo = layer.named.insert("rollback", 1)?;
                let undo_log = action_log.clone();
                Ok(Box::new(move || {
                    undo_log.borrow_mut().push("undo");
                    undo();
                }) as Undo)
            },
            "store.rollback",
            true,
        ));
        assert!(err.to_string().contains("change failed"), "{err:#}");
        assert_eq!(*events.borrow(), vec!["notify", "undo", "notify"]);
        assert!(layers.peek(Some(&key)).is_none());
        scope.dispose().await;
    });
}
