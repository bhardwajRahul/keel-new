//! Behavior tests for scoped registration, shadowing, restrictions, guards,
//! and scope-filtered dispatch (mirroring the upstream scoped suite).

mod common;

use common::*;
use dsh_cordis::EventOptions;
use dsh_scope::ScopedEvents;
use dsh_tools::{
    PreToolDecision, ScopedWaterfall, ToolRestriction, ToolsChange, ToolsPreExecute, ToolsResult,
};
use serde_json::json;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

fn names(schemas: &[dsh_llm::ToolSchema]) -> Vec<String> {
    let mut names: Vec<String> = schemas.iter().map(|schema| schema.name.clone()).collect();
    names.sort();
    names
}

#[test]
fn a_scoped_tool_is_visible_and_executable_for_that_scope_only() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let (scope, agent, key) = mint_agent(&ctx, "a", None);
        let (_other_scope, other, _other_key) = mint_agent(&ctx, "other", None);
        tools.register(&ctx, tool("shared", "ran:shared")).unwrap();
        tools
            .register(&scope.ctx(), tool("mine", "ran:mine"))
            .unwrap();

        assert_eq!(names(&tools.schemas(Some(&key))), vec!["mine", "shared"]);
        assert_eq!(names(&tools.schemas(None)), vec!["shared"]);

        assert_eq!(run(&tools, "mine", Some(agent)).await, "ran:mine");
        // Out-of-view execution reads exactly like a nonexistent tool.
        assert_eq!(
            run(&tools, "mine", Some(other)).await,
            "Error: unknown tool \"mine\""
        );
        assert_eq!(
            run(&tools, "mine", None).await,
            "Error: unknown tool \"mine\""
        );
    });
}

#[test]
fn a_scoped_tool_shadows_a_global_in_either_registration_order() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let (scope, agent, key) = mint_agent(&ctx, "a", None);
        // Scoped first, global second.
        tools
            .register(&scope.ctx(), tool("bash", "restricted-bash"))
            .unwrap();
        tools.register(&ctx, tool("bash", "global-bash")).unwrap();

        assert_eq!(run(&tools, "bash", Some(agent)).await, "restricted-bash");
        assert_eq!(run(&tools, "bash", None).await, "global-bash");
        // Exactly one bash entry in the scope's schema view (the shadow).
        let bash_entries = tools
            .schemas(Some(&key))
            .iter()
            .filter(|schema| schema.name == "bash")
            .count();
        assert_eq!(bash_entries, 1);
    });
}

#[test]
fn duplicate_names_fail_with_layer_specific_messages() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let (scope, _agent, _key) = mint_agent(&ctx, "a", None);
        tools.register(&ctx, tool("x", "ok")).unwrap();
        let global = tools.register(&ctx, tool("x", "ok")).unwrap_err();
        assert!(global.to_string().contains("agent.ctx"), "{global}");
        tools.register(&scope.ctx(), tool("y", "ok")).unwrap();
        let scoped = tools.register(&scope.ctx(), tool("y", "ok")).unwrap_err();
        assert!(
            scoped
                .to_string()
                .contains("already registered in this scope"),
            "{scoped}"
        );
    });
}

#[test]
fn disposing_the_scope_unwinds_its_registrations() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let (scope, _agent, key) = mint_agent(&ctx, "a", None);
        tools.register(&scope.ctx(), tool("mine", "ok")).unwrap();
        assert!(tools.get("mine", Some(&key)).is_some());
        scope.dispose().await;
        assert!(tools.get("mine", Some(&key)).is_none());
        assert!(tools.schemas(Some(&key)).is_empty());
    });
}

#[test]
fn register_returns_a_working_disposer_and_notifies_change() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let changes = Rc::new(Cell::new(0u32));
        let counter = changes.clone();
        ctx.on::<ToolsChange, _, _>(EventOptions::default(), move |_ctx, _args| {
            counter.set(counter.get() + 1);
            async { None }
        })
        .unwrap();

        let handle = tools.register(&ctx, tool("disposable", "ok")).unwrap();
        assert_eq!(names(&tools.schemas(None)), vec!["disposable"]);
        assert_eq!(changes.get(), 1);
        handle.dispose().await;
        assert!(tools.schemas(None).is_empty());
        assert_eq!(changes.get(), 2);
    });
}

#[test]
fn restrict_masks_globals_but_keeps_scope_local_tools() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let (scope, agent, key) = mint_agent(&ctx, "a", None);
        tools.register(&ctx, tool("read", "ran:read")).unwrap();
        tools.register(&ctx, tool("bash", "ran:bash")).unwrap();
        tools
            .register(&scope.ctx(), tool("capture", "ran:capture"))
            .unwrap();
        tools
            .restrict(
                &scope.ctx(),
                ToolRestriction {
                    allow: Some(vec!["read".into()]),
                    deny: None,
                },
            )
            .unwrap();

        assert_eq!(names(&tools.schemas(Some(&key))), vec!["capture", "read"]);
        assert_eq!(
            run(&tools, "bash", Some(agent.clone())).await,
            "Error: unknown tool \"bash\""
        );
        assert_eq!(run(&tools, "read", Some(agent.clone())).await, "ran:read");
        assert_eq!(run(&tools, "capture", Some(agent)).await, "ran:capture");
        // The global view is untouched.
        assert_eq!(names(&tools.schemas(None)), vec!["bash", "read"]);
    });
}

#[test]
fn restrictions_intersect_and_lift_independently() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let (scope, _agent, key) = mint_agent(&ctx, "a", None);
        for name in ["a", "b", "c"] {
            tools.register(&ctx, tool(name, "ok")).unwrap();
        }
        let lift_allow = tools
            .restrict(
                &scope.ctx(),
                ToolRestriction {
                    allow: Some(vec!["a".into(), "b".into()]),
                    deny: None,
                },
            )
            .unwrap();
        tools
            .restrict(
                &scope.ctx(),
                ToolRestriction {
                    allow: None,
                    deny: Some(vec!["b".into()]),
                },
            )
            .unwrap();
        assert_eq!(names(&tools.schemas(Some(&key))), vec!["a"]);
        lift_allow.dispose().await;
        // The deny remains after the allow-list is lifted.
        assert_eq!(names(&tools.schemas(Some(&key))), vec!["a", "c"]);
    });
}

#[test]
fn restrict_fails_loud_on_misuse() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let (scope, _agent, _key) = mint_agent(&ctx, "a", None);
        tools.register(&ctx, tool("real", "ok")).unwrap();
        tools.register(&scope.ctx(), tool("local", "ok")).unwrap();

        let unscoped = tools
            .restrict(
                &ctx,
                ToolRestriction {
                    allow: None,
                    deny: Some(vec!["real".into()]),
                },
            )
            .unwrap_err();
        assert!(
            unscoped.to_string().contains("requires a scoped context"),
            "{unscoped}"
        );

        let empty = tools
            .restrict(&scope.ctx(), ToolRestriction::default())
            .unwrap_err();
        assert!(empty.to_string().contains("no-op"), "{empty}");

        // A scope's own registration is exempt from its own filter, so naming
        // it is a caller error rather than a silent no-op.
        let own = tools
            .restrict(
                &scope.ctx(),
                ToolRestriction {
                    allow: Some(vec!["local".into()]),
                    deny: None,
                },
            )
            .unwrap_err();
        assert!(
            own.to_string().contains("unknown global tool \"local\""),
            "{own}"
        );

        let typo = tools
            .restrict(
                &scope.ctx(),
                ToolRestriction {
                    allow: Some(vec!["reall".into()]),
                    deny: None,
                },
            )
            .unwrap_err();
        assert!(
            typo.to_string().contains("unknown global tool \"reall\""),
            "{typo}"
        );
        assert!(
            typo.to_string().contains("known global tools: real"),
            "{typo}"
        );

        let plural = tools
            .restrict(
                &scope.ctx(),
                ToolRestriction {
                    allow: None,
                    deny: Some(vec!["ghost".into(), "wraith".into()]),
                },
            )
            .unwrap_err();
        assert!(
            plural
                .to_string()
                .contains("unknown global tools \"ghost\", \"wraith\""),
            "{plural}"
        );

        let (_empty_app, empty_ctx, empty_tools) = setup();
        let (empty_scope, _a, _k) = mint_agent(&empty_ctx, "empty", None);
        let none = empty_tools
            .restrict(
                &empty_scope.ctx(),
                ToolRestriction {
                    allow: None,
                    deny: Some(vec!["ghost".into()]),
                },
            )
            .unwrap_err();
        assert!(
            none.to_string().contains("known global tools: (none)"),
            "{none}"
        );
    });
}

#[test]
fn restrictions_filter_tools_inherited_from_ancestor_scopes() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        // The preset shape: no global rows, everything contributed by an
        // ancestor scope the child joined.
        let (parent_scope, parent_agent, parent_key) = mint_agent(&ctx, "parent", None);
        tools
            .register(&parent_scope.ctx(), tool("bash", "ran:bash"))
            .unwrap();
        tools
            .register(&parent_scope.ctx(), tool("read", "ran:read"))
            .unwrap();
        let (child_scope, child_agent, child_key) = mint_agent(&ctx, "child", Some(&parent_key));

        assert_eq!(
            names(&tools.schemas(Some(&child_key))),
            vec!["bash", "read"]
        );
        tools
            .restrict(
                &child_scope.ctx(),
                ToolRestriction {
                    allow: None,
                    deny: Some(vec!["bash".into()]),
                },
            )
            .unwrap();

        assert_eq!(names(&tools.schemas(Some(&child_key))), vec!["read"]);
        assert_eq!(
            run(&tools, "bash", Some(child_agent.clone())).await,
            "Error: unknown tool \"bash\""
        );
        // The ancestor keeps its whole surface: a child's filter is its own.
        assert_eq!(
            names(&tools.schemas(Some(&parent_key))),
            vec!["bash", "read"]
        );
        assert_eq!(run(&tools, "bash", Some(parent_agent)).await, "ran:bash");

        // The child's own registrations sit outside its own filter.
        tools
            .register(&child_scope.ctx(), tool("report", "ran:report"))
            .unwrap();
        tools
            .restrict(
                &child_scope.ctx(),
                ToolRestriction {
                    allow: Some(vec!["read".into()]),
                    deny: None,
                },
            )
            .unwrap();
        assert_eq!(
            names(&tools.schemas(Some(&child_key))),
            vec!["read", "report"]
        );
        assert_eq!(run(&tools, "report", Some(child_agent)).await, "ran:report");
    });
}

#[test]
fn an_ancestor_restriction_reaches_every_nested_scope() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        tools.register(&ctx, tool("web", "ran:web")).unwrap();
        let (parent_scope, _parent_agent, parent_key) = mint_agent(&ctx, "parent", None);
        tools
            .register(&parent_scope.ctx(), tool("bash", "ran:bash"))
            .unwrap();
        let (_child_scope, _child_agent, child_key) = mint_agent(&ctx, "child", Some(&parent_key));
        tools
            .restrict(
                &parent_scope.ctx(),
                ToolRestriction {
                    allow: None,
                    deny: Some(vec!["web".into()]),
                },
            )
            .unwrap();

        assert_eq!(names(&tools.schemas(Some(&child_key))), vec!["bash"]);
        assert_eq!(names(&tools.schemas(Some(&parent_key))), vec!["bash"]);
    });
}

#[test]
fn a_scoped_pre_execute_listener_gates_only_its_own_agent() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let (scope, agent, _key) = mint_agent(&ctx, "a", None);
        let (_other_scope, other, _other_key) = mint_agent(&ctx, "other", None);
        tools.register(&ctx, tool("t", "ran:t")).unwrap();

        let seen: Rc<RefCell<Vec<bool>>> = Rc::default();
        let sink = seen.clone();
        scope
            .ctx()
            .on_waterfall_scoped::<ToolsPreExecute, _, _>(
                EventOptions::default(),
                move |_ctx, exec, _next| {
                    sink.borrow_mut().push(exec.agent().is_some());
                    async {
                        Ok(PreToolDecision::Deny {
                            reason: "scoped veto".into(),
                        })
                    }
                },
            )
            .unwrap();

        assert_eq!(run(&tools, "t", Some(agent)).await, "Error: scoped veto");
        assert_eq!(run(&tools, "t", Some(other)).await, "ran:t");
        assert_eq!(run(&tools, "t", None).await, "ran:t");
        assert_eq!(seen.borrow().len(), 1);
    });
}

#[test]
fn scoped_guards_run_after_pre_execute_and_unwind_independently() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let (scope, agent, _key) = mint_agent(&ctx, "a", None);
        let (_other_scope, other, _other_key) = mint_agent(&ctx, "other", None);
        let body_calls = Rc::new(Cell::new(0u32));
        let body = body_calls.clone();
        tools
            .register(
                &ctx,
                tool_with_execute("t", move |_args, _exec| {
                    body.set(body.get() + 1);
                    futures::FutureExt::boxed_local(async { Ok(json!("ran:t")) })
                }),
            )
            .unwrap();
        let guard = |exec: &dsh_tools::ToolExecution| {
            assert_eq!(exec.name(), "t");
            Some("terminal policy".to_string())
        };
        let lift_first = tools.register_guard(&scope.ctx(), guard).unwrap();
        tools.register_guard(&scope.ctx(), guard).unwrap();
        // A prepended listener that answers Allow without delegating can veto
        // the extensible waterfall, but not the owner-level monotonic guard.
        scope
            .ctx()
            .on_waterfall_scoped::<ToolsPreExecute, _, _>(
                EventOptions {
                    prepend: true,
                    global: false,
                },
                |_ctx, _exec, _next| async { Ok(PreToolDecision::Allow) },
            )
            .unwrap();

        assert_eq!(
            run(&tools, "t", Some(agent.clone())).await,
            "Error: terminal policy"
        );
        assert_eq!(run(&tools, "t", Some(other)).await, "ran:t");
        assert_eq!(body_calls.get(), 1);

        lift_first.dispose().await;
        assert_eq!(
            run(&tools, "t", Some(agent.clone())).await,
            "Error: terminal policy"
        );
        scope.dispose().await;
        assert_eq!(run(&tools, "t", Some(agent)).await, "ran:t");
        assert_eq!(body_calls.get(), 2);
    });
}

#[test]
fn global_guards_compose_monotonically() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let body_calls = Rc::new(Cell::new(0u32));
        let body = body_calls.clone();
        tools
            .register(
                &ctx,
                tool_with_execute("t", move |_args, _exec| {
                    body.set(body.get() + 1);
                    futures::FutureExt::boxed_local(async { Ok(json!("ran:t")) })
                }),
            )
            .unwrap();
        tools.register_guard(&ctx, |_exec| None).unwrap();
        tools
            .register_guard(&ctx, |_exec| Some("global denial".into()))
            .unwrap();

        assert_eq!(run(&tools, "t", None).await, "Error: global denial");
        assert_eq!(body_calls.get(), 0);
    });
}

#[test]
fn a_guard_replaced_during_its_own_run_applies_from_the_next_call() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let (scope, agent, _key) = mint_agent(&ctx, "a", None);
        tools.register(&ctx, tool("t", "ran:t")).unwrap();
        let calls: Rc<RefCell<Vec<&'static str>>> = Rc::default();
        let runtime = tools.clone();
        let scope_ctx = scope.ctx();
        let calls_probe = calls.clone();
        let added = Rc::new(Cell::new(false));
        let first = tools
            .register_guard(&scope.ctx(), move |_exec| {
                calls_probe.borrow_mut().push("first");
                // Register the replacement mid-run: the store iterates a
                // snapshot, so the replacement first applies on the NEXT call
                // (documented divergence from upstream's live iterators).
                if !added.get() {
                    added.set(true);
                    let calls = calls_probe.clone();
                    runtime
                        .register_guard(&scope_ctx, move |_exec| {
                            calls.borrow_mut().push("replacement");
                            Some("replacement denial".into())
                        })
                        .unwrap();
                }
                None
            })
            .unwrap();

        assert_eq!(run(&tools, "t", Some(agent.clone())).await, "ran:t");
        assert_eq!(*calls.borrow(), vec!["first"]);
        // Lift the registering guard; only the replacement remains.
        first.dispose().await;
        let second = run(&tools, "t", Some(agent)).await;
        assert_eq!(second, "Error: replacement denial");
        assert_eq!(*calls.borrow(), vec!["first", "replacement"]);
    });
}

#[test]
fn scoped_result_observers_see_only_their_agents_outcomes() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let (scope, agent, _key) = mint_agent(&ctx, "a", None);
        let (_other_scope, other, _other_key) = mint_agent(&ctx, "other", None);
        tools.register(&ctx, tool("t", "ran:t")).unwrap();
        let scoped_seen = Rc::new(Cell::new(0u32));
        let global_seen = Rc::new(Cell::new(0u32));
        let scoped_counter = scoped_seen.clone();
        scope
            .ctx()
            .on_scoped::<ToolsResult, _, _>(
                EventOptions::default(),
                move |_ctx, (_exec, _result)| {
                    scoped_counter.set(scoped_counter.get() + 1);
                    async { None }
                },
            )
            .unwrap();
        let global_counter = global_seen.clone();
        ctx.on_scoped::<ToolsResult, _, _>(
            EventOptions::default(),
            move |_ctx, (_exec, _result)| {
                global_counter.set(global_counter.get() + 1);
                async { None }
            },
        )
        .unwrap();

        tools.execute(input_for("t", Some(agent))).await;
        tools.execute(input_for("t", Some(other))).await;
        tools.execute(input_for("t", None)).await;

        assert_eq!(scoped_seen.get(), 1);
        assert_eq!(global_seen.get(), 3);
    });
}

#[test]
fn restrict_cannot_name_the_reserved_transport() {
    dsh_cordis::run(async {
        let (_app, ctx, tools) = setup();
        let (scope, _agent, _key) = mint_agent(&ctx, "a", None);
        let error = tools
            .restrict(
                &scope.ctx(),
                ToolRestriction {
                    allow: None,
                    deny: Some(vec!["run_code".into()]),
                },
            )
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("reserved Code Mode presentation transport")
        );
    });
}
