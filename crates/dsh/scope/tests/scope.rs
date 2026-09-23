//! Behavioral contract tests ported from upstream `tests/scope.spec.ts`:
//! context tagging, scope lifecycle/disposal boundaries, carrier routing,
//! and the scope parent chain. Upstream's `ctx.extend({})` inheritance case
//! is exercised through a child plugin (the Rust tag rides the intercept
//! chain), and carrier opacity is a type-level property here.

use dsh_cordis::{App, Context, Disposer, Effect, Event, EventOptions, Inject, plugin_fn};
use dsh_scope::{
    Scope, ScopeKey, ScopedEvents, bind_scope_parent, create_scope, scope_chain_of, scope_of,
    scope_parent_of, scope_target, scope_target_filtered,
};
use serde_json::Value;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

struct Ping;
impl Event for Ping {
    const NAME: &'static str = "scope-test/ping";
    type Args = String;
    type Ret = ();
}

/// Let `emit`-spawned listener tasks run.
async fn settle() {
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
}

fn hear(
    ctx: &Context,
    label: &'static str,
    heard: Rc<RefCell<Vec<String>>>,
    options: EventOptions,
) {
    ctx.on_scoped::<Ping, _, _>(options, move |_ctx, value| {
        heard.borrow_mut().push(format!("{label}:{value}"));
        std::future::ready(None)
    })
    .unwrap();
}

#[test]
fn tags_contexts_and_derived_contexts_with_the_nearest_tag_winning() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let outer_key = ScopeKey::new();
        let inner_key = ScopeKey::new();
        let outer = create_scope(&ctx, outer_key.clone(), None).unwrap();
        let inner = create_scope(&outer.ctx(), inner_key.clone(), None).unwrap();

        assert_eq!(scope_of(&ctx), None);
        assert_eq!(scope_of(&outer.ctx()), Some(outer_key.clone()));
        assert_eq!(scope_of(&inner.ctx()), Some(inner_key.clone()));

        // A plugin loaded below the scoped context inherits the tag
        // (upstream's derived-context case).
        let seen: Rc<RefCell<Option<Option<ScopeKey>>>> = Rc::default();
        let seen2 = seen.clone();
        let probe = plugin_fn("derived-probe", Inject::default(), move |ctx, _| {
            let seen = seen2.clone();
            async move {
                *seen.borrow_mut() = Some(scope_of(&ctx));
                Ok(())
            }
        });
        let fiber = outer.ctx().plugin(Rc::new(probe), Value::Null).unwrap();
        fiber.await_ready().await.unwrap();
        assert_eq!(*seen.borrow(), Some(Some(outer_key)));

        inner.dispose().await;
        outer.dispose().await;
    });
}

#[test]
fn is_usable_synchronously_before_the_backing_fiber_activates() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let events: Rc<RefCell<Vec<&'static str>>> = Rc::default();
        let slot: Rc<RefCell<Option<Scope>>> = Rc::default();
        let events2 = events.clone();
        let slot2 = slot.clone();
        let host = plugin_fn("sync-host", Inject::default(), move |ctx, _| {
            let events = events2.clone();
            let slot = slot2.clone();
            async move {
                let scope = create_scope(&ctx, ScopeKey::new(), None)?;
                let disposed = events.clone();
                scope.ctx().effect(move |_| {
                    Ok(Effect::One(Disposer::sync(move || {
                        disposed.borrow_mut().push("disposed")
                    })))
                })?;
                events.borrow_mut().push("registered");
                *slot.borrow_mut() = Some(scope);
                Ok(())
            }
        });
        ctx.plugin(Rc::new(host), Value::Null)
            .unwrap()
            .await_ready()
            .await
            .unwrap();
        assert_eq!(*events.borrow(), vec!["registered"]);
        let scope = slot.borrow_mut().take().unwrap();
        scope.dispose().await;
        assert_eq!(*events.borrow(), vec!["registered", "disposed"]);
    });
}

#[test]
fn shares_quiescence_across_repeat_and_raw_disposer_first_calls() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let scope = create_scope(&ctx, ScopeKey::new(), None).unwrap();
        scope.fiber().await_ready().await.unwrap();
        let (tx, rx) = futures::channel::oneshot::channel::<()>();
        let gate = futures::FutureExt::shared(rx);
        let finished: Rc<Cell<bool>> = Rc::default();
        let gate2 = gate.clone();
        let finished2 = finished.clone();
        scope
            .ctx()
            .effect(move |_| {
                Ok(Effect::One(Disposer::asynchronous(move || async move {
                    let _ = gate2.await;
                    finished2.set(true);
                })))
            })
            .unwrap();

        let raw = tokio::task::spawn_local(scope.raw_dispose());
        let public_dispose = tokio::task::spawn_local(scope.dispose());
        settle().await;
        assert!(!finished.get());
        tx.send(()).unwrap();
        raw.await.unwrap();
        public_dispose.await.unwrap();
        scope.dispose().await;
        assert!(finished.get());
    });
}

#[test]
fn exposes_the_raw_disposer_for_ordered_composite_teardown() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let order: Rc<RefCell<Vec<&'static str>>> = Rc::default();
        let scope = create_scope(&ctx, ScopeKey::new(), None).unwrap();
        scope.fiber().await_ready().await.unwrap();
        let in_scope = order.clone();
        scope
            .ctx()
            .effect(move |_| {
                Ok(Effect::One(Disposer::sync(move || {
                    in_scope.borrow_mut().push("scope")
                })))
            })
            .unwrap();

        let raw = scope.raw_dispose();
        let outer = order.clone();
        let inner = order.clone();
        let handle = ctx
            .effect(move |_| {
                Ok(Effect::Many(vec![
                    Disposer::sync(move || outer.borrow_mut().push("outer")),
                    Disposer::asynchronous(move || raw),
                    Disposer::sync(move || inner.borrow_mut().push("inner")),
                ]))
            })
            .unwrap();
        handle.dispose().await;
        assert_eq!(*order.borrow(), vec!["inner", "scope", "outer"]);
    });
}

#[test]
fn routes_scoped_listeners_by_key_while_untagged_listeners_remain_global() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let key_a = ScopeKey::new();
        let key_b = ScopeKey::new();
        let scope_a = create_scope(&ctx, key_a.clone(), None).unwrap();
        let scope_b = create_scope(&ctx, key_b.clone(), None).unwrap();
        let heard: Rc<RefCell<Vec<String>>> = Rc::default();
        hear(&ctx, "global", heard.clone(), EventOptions::default());
        hear(&scope_a.ctx(), "A", heard.clone(), EventOptions::default());
        hear(&scope_b.ctx(), "B", heard.clone(), EventOptions::default());

        ctx.emit_scoped::<Ping, ()>(&scope_target(Some(&key_a)), "a".into());
        ctx.emit_scoped::<Ping, ()>(&scope_target(Some(&key_b)), "b".into());
        ctx.emit_scoped::<Ping, ()>(&scope_target(None), "none".into());
        settle().await;

        assert_eq!(
            *heard.borrow(),
            vec!["global:a", "A:a", "global:b", "B:b", "global:none"]
        );
        scope_a.dispose().await;
        scope_b.dispose().await;
    });
}

#[test]
fn preserves_a_base_filter() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let key = ScopeKey::new();
        let scope = create_scope(&ctx, key.clone(), None).unwrap();
        let heard: Rc<RefCell<Vec<String>>> = Rc::default();
        hear(&ctx, "global", heard.clone(), EventOptions::default());
        hear(&scope.ctx(), "A", heard.clone(), EventOptions::default());

        let called: Rc<Cell<bool>> = Rc::default();
        let called2 = called.clone();
        let carrier = scope_target_filtered::<()>(
            move |_ctx| {
                called2.set(true);
                false
            },
            Some(&key),
        );
        ctx.emit_scoped::<Ping, ()>(&carrier, "vetoed".into());
        settle().await;
        assert!(heard.borrow().is_empty());
        assert!(called.get());
        scope.dispose().await;
    });
}

#[test]
fn global_listeners_retain_global_semantics() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let scope = create_scope(&ctx, ScopeKey::new(), None).unwrap();
        let heard: Rc<RefCell<Vec<String>>> = Rc::default();
        hear(
            &scope.ctx(),
            "heard",
            heard.clone(),
            EventOptions {
                prepend: false,
                global: true,
            },
        );

        let foreign = ScopeKey::new();
        ctx.emit_scoped::<Ping, ()>(&scope_target(Some(&foreign)), "foreign".into());
        ctx.emit_scoped::<Ping, ()>(&scope_target(None), "none".into());
        settle().await;
        assert_eq!(*heard.borrow(), vec!["heard:foreign", "heard:none"]);
        scope.dispose().await;
    });
}

#[test]
fn carrier_tracks_its_key_separately_from_the_subject_brand() {
    struct Subject;
    let key = ScopeKey::new();
    let carrier = scope_target::<Subject>(Some(&key));
    assert_eq!(carrier.key(), Some(&key));
    assert_eq!(carrier.clone().key(), Some(&key));
    let unkeyed = scope_target::<Subject>(None);
    assert!(unkeyed.key().is_none());
}

#[test]
fn links_at_mint_walks_to_the_root_and_rejects_cycles() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let preset = ScopeKey::new();
        let agent = ScopeKey::new();
        let _preset_scope = create_scope(&ctx, preset.clone(), None).unwrap();
        let _agent_scope = create_scope(&ctx, agent.clone(), Some(&preset)).unwrap();

        assert_eq!(scope_parent_of(&agent), Some(preset.clone()));
        assert_eq!(scope_parent_of(&preset), None);
        assert_eq!(
            scope_chain_of(Some(&agent)),
            vec![agent.clone(), preset.clone()]
        );
        assert!(scope_chain_of(None).is_empty());
        let err = bind_scope_parent(&preset, &agent).map(|_| ()).unwrap_err();
        assert!(err.to_string().contains("cycle"), "{err:#}");
        let err = bind_scope_parent(&preset, &preset).map(|_| ()).unwrap_err();
        assert!(err.to_string().contains("cycle"), "{err:#}");
    });
}

#[test]
fn relinks_only_through_the_binding_held_by_the_original_binder() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let preset_a = ScopeKey::new();
        let preset_b = ScopeKey::new();
        let agent = ScopeKey::new();
        let _a = create_scope(&ctx, preset_a.clone(), None).unwrap();
        let _b = create_scope(&ctx, preset_b.clone(), None).unwrap();
        let binding = bind_scope_parent(&agent, &preset_a).unwrap();
        let _agent_scope = create_scope(&ctx, agent.clone(), None).unwrap();

        // A bound key cannot be re-bound from the outside; only the binding
        // moves it.
        let err = bind_scope_parent(&agent, &preset_b)
            .map(|_| ())
            .unwrap_err();
        assert!(err.to_string().contains("already bound"), "{err:#}");
        binding.rebind(&preset_b).unwrap();
        assert_eq!(
            scope_chain_of(Some(&agent)),
            vec![agent.clone(), preset_b.clone()]
        );

        // The rebind keeps the cycle check: a parent may not adopt its
        // ancestor.
        let child = ScopeKey::new();
        let _child_binding = bind_scope_parent(&child, &agent).unwrap();
        let err = binding.rebind(&child).unwrap_err();
        assert!(err.to_string().contains("cycle"), "{err:#}");
    });
}

#[test]
fn admits_an_ancestor_tagged_listener_for_a_descendant_dispatch_never_the_reverse() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let preset = ScopeKey::new();
        let agent = ScopeKey::new();
        let other = ScopeKey::new();
        let preset_scope = create_scope(&ctx, preset.clone(), None).unwrap();
        let agent_scope = create_scope(&ctx, agent.clone(), Some(&preset)).unwrap();
        let other_scope = create_scope(&ctx, other.clone(), None).unwrap();

        let heard: Rc<RefCell<Vec<String>>> = Rc::default();
        hear(&ctx, "untagged", heard.clone(), EventOptions::default());
        hear(
            &preset_scope.ctx(),
            "preset",
            heard.clone(),
            EventOptions::default(),
        );
        hear(
            &agent_scope.ctx(),
            "agent",
            heard.clone(),
            EventOptions::default(),
        );
        hear(
            &other_scope.ctx(),
            "other",
            heard.clone(),
            EventOptions::default(),
        );

        // Dispatch at the AGENT key: its own tag and its ancestor's admit; a
        // sibling root does not.
        ctx.emit_scoped::<Ping, ()>(&scope_target(Some(&agent)), "x".into());
        settle().await;
        let mut seen = heard.borrow().clone();
        seen.sort();
        assert_eq!(seen, vec!["agent:x", "preset:x", "untagged:x"]);

        // Dispatch at the PRESET key: the agent tag sits BELOW the dispatch
        // key and stays excluded — events flow up the chain, not down.
        heard.borrow_mut().clear();
        ctx.emit_scoped::<Ping, ()>(&scope_target(Some(&preset)), "y".into());
        settle().await;
        let mut seen = heard.borrow().clone();
        seen.sort();
        assert_eq!(seen, vec!["preset:y", "untagged:y"]);
    });
}
