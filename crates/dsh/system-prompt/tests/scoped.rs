//! Scoped registration and dispatch tests ported from upstream
//! `tests/scoped.spec.ts`. Rust scopes are minted directly with
//! `dsh_scope::create_scope` (upstream mints inside a plugin injecting
//! `systemPrompt` for the property proxy; the Rust API takes the registering
//! context explicitly, so no minting plugin is needed).

use dsh_cordis::{App, Context, EffectHandle, EventOptions};
use dsh_llm::ToolSchema;
use dsh_scope::{ScopeKey, create_scope};
use dsh_system_prompt::{
    AssembleContext, AssembledSection, PromptContext, PromptSection, PromptText, SystemPrompt,
    SystemPromptPlugin, TOOL_ORDER_REST, ToolProviderResult, on_assemble, render_context_snapshot,
    render_prompt,
};
use serde_json::{Value, json};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

async fn mount(ctx: &Context, config: Value) -> Rc<SystemPrompt> {
    let fiber = ctx.plugin(Rc::new(SystemPromptPlugin), config).unwrap();
    fiber.await_ready().await.unwrap();
    ctx.service::<SystemPrompt>().unwrap()
}

fn section(name: &str, order: f64, text: &str) -> PromptSection {
    PromptSection {
        name: name.into(),
        order,
        text: PromptText::fixed(text),
        complete: false,
    }
}

fn context_entry(name: &str, order: f64, text: &str) -> PromptContext {
    PromptContext {
        name: name.into(),
        order,
        text: PromptText::fixed(text),
    }
}

fn schema(name: &str) -> ToolSchema {
    ToolSchema {
        name: name.into(),
        description: format!("tool {name}"),
        parameters: Default::default(),
    }
}

fn provided(tools: Vec<ToolSchema>) -> ToolProviderResult {
    ToolProviderResult {
        schemas: tools,
        known_names: None,
    }
}

fn scoped_context(key: &ScopeKey) -> AssembleContext {
    AssembleContext {
        scope: Some(key.clone()),
        payload: None,
    }
}

#[test]
fn a_scoped_persona_shadows_the_deployment_persona_for_that_scope_only() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, json!({"persona": "You are the deployment."})).await;
        let key = ScopeKey::new();
        let scope = create_scope(&ctx, key.clone(), None).unwrap();
        sp.section(
            &scope.ctx(),
            section("deployment:persona", 0.0, "You run tests."),
        )
        .unwrap();

        let scoped = render_prompt(&sp.assemble(scoped_context(&key)).await.unwrap()).unwrap();
        let global =
            render_prompt(&sp.assemble(AssembleContext::default()).await.unwrap()).unwrap();
        assert!(scoped.contains("You run tests."));
        assert!(!scoped.contains("You are the deployment."));
        assert!(global.contains("You are the deployment."));
        assert!(!global.contains("You run tests."));
    });
}

#[test]
fn scoped_only_sections_join_that_scope_alone_and_disposal_removes_them() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        let key = ScopeKey::new();
        let scope = create_scope(&ctx, key.clone(), None).unwrap();
        sp.section(
            &scope.ctx(),
            section("child:extra", 50.0, "Extra guidance."),
        )
        .unwrap();

        assert!(
            render_prompt(&sp.assemble(scoped_context(&key)).await.unwrap())
                .unwrap()
                .contains("Extra guidance.")
        );
        assert!(
            !render_prompt(&sp.assemble(AssembleContext::default()).await.unwrap())
                .unwrap()
                .contains("Extra guidance.")
        );
        scope.dispose().await;
        assert!(
            !render_prompt(&sp.assemble(scoped_context(&key)).await.unwrap())
                .unwrap()
                .contains("Extra guidance.")
        );
    });
}

#[test]
fn duplicate_names_throw_per_layer_naming_agent_ctx_for_the_global_case() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        let key = ScopeKey::new();
        let scope = create_scope(&ctx, key.clone(), None).unwrap();
        sp.section(&ctx, section("x", 1.0, "a")).unwrap();
        let global_dup = sp.section(&ctx, section("x", 1.0, "b")).unwrap_err();
        assert!(
            format!("{global_dup:#}").contains("agent.ctx"),
            "{global_dup:#}"
        );
        sp.section(&scope.ctx(), section("y", 1.0, "a")).unwrap();
        let scoped_dup = sp
            .section(&scope.ctx(), section("y", 1.0, "b"))
            .unwrap_err();
        assert!(
            format!("{scoped_dup:#}").contains("already registered in this scope"),
            "{scoped_dup:#}"
        );
    });
}

#[test]
fn shadows_a_global_section_before_evaluating_either_text_provider() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        let key = ScopeKey::new();
        let scope = create_scope(&ctx, key.clone(), None).unwrap();
        let global_calls = Rc::new(Cell::new(0));
        let scoped_calls = Rc::new(Cell::new(0));
        let global_calls2 = global_calls.clone();
        let scoped_calls2 = scoped_calls.clone();
        sp.section(
            &ctx,
            PromptSection {
                name: "shared".into(),
                order: 1.0,
                text: PromptText::provider(move |_| {
                    global_calls2.set(global_calls2.get() + 1);
                    "global text".into()
                }),
                complete: false,
            },
        )
        .unwrap();
        sp.section(
            &scope.ctx(),
            PromptSection {
                name: "shared".into(),
                order: 1.0,
                text: PromptText::provider(move |_| {
                    scoped_calls2.set(scoped_calls2.get() + 1);
                    "scoped text".into()
                }),
                complete: false,
            },
        )
        .unwrap();

        let assembly = sp.assemble(scoped_context(&key)).await.unwrap();
        let shared = assembly
            .sections
            .iter()
            .find(|s| s.name == "shared")
            .unwrap();
        assert_eq!(shared.text, "scoped text");
        assert_eq!(global_calls.get(), 0);
        assert_eq!(scoped_calls.get(), 1);
    });
}

#[test]
fn a_scoped_variable_shadows_its_global_name_twin_for_that_scope() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, json!({"persona": "Mode: {{mode}}."})).await;
        let key = ScopeKey::new();
        let scope = create_scope(&ctx, key.clone(), None).unwrap();
        sp.variable(&ctx, "mode", |_| Some("normal".into()))
            .unwrap();
        sp.variable(&scope.ctx(), "mode", |_| Some("strict".into()))
            .unwrap();

        assert!(
            render_prompt(&sp.assemble(scoped_context(&key)).await.unwrap())
                .unwrap()
                .contains("Mode: strict.")
        );
        assert!(
            render_prompt(&sp.assemble(AssembleContext::default()).await.unwrap())
                .unwrap()
                .contains("Mode: normal.")
        );
    });
}

#[test]
fn same_layer_duplicates_throw_and_the_scoped_layer_cleans_up_on_dispose() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        let scope = create_scope(&ctx, ScopeKey::new(), None).unwrap();
        sp.variable(&scope.ctx(), "v", |_| Some("1".into()))
            .unwrap();
        let dup = sp
            .variable(&scope.ctx(), "v", |_| Some("2".into()))
            .unwrap_err();
        assert!(
            format!("{dup:#}").contains("already registered in this scope"),
            "{dup:#}"
        );
        scope.dispose().await;
        // A freshly minted scope starts clean.
        let again = create_scope(&ctx, ScopeKey::new(), None).unwrap();
        sp.variable(&again.ctx(), "v", |_| Some("3".into()))
            .unwrap();
    });
}

#[test]
fn defers_a_scoped_variable_that_replaces_the_last_provider_in_its_generation() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, json!({"persona": "Mode: {{mode}}."})).await;
        let key = ScopeKey::new();
        let scope = create_scope(&ctx, key.clone(), None).unwrap();
        let calls: Rc<RefCell<Vec<&'static str>>> = Rc::default();
        sp.section(&scope.ctx(), section("scope:sibling", 1.0, "Scoped."))
            .unwrap();

        let handle_slot: Rc<RefCell<Option<EffectHandle>>> = Rc::default();
        let handle_slot2 = handle_slot.clone();
        let calls2 = calls.clone();
        let sp2 = sp.clone();
        let scope_ctx = scope.ctx();
        let handle = sp
            .variable(&scope.ctx(), "mode", move |_| {
                calls2.borrow_mut().push("first");
                if let Some(handle) = handle_slot2.borrow_mut().take() {
                    // The upstream disposer is synchronous; run the sync
                    // disposer chain to completion here.
                    futures::executor::block_on(handle.dispose());
                }
                let calls3 = calls2.clone();
                sp2.variable(&scope_ctx, "mode", move |_| {
                    calls3.borrow_mut().push("replacement");
                    Some("replacement".into())
                })
                .unwrap();
                Some("first".into())
            })
            .unwrap();
        *handle_slot.borrow_mut() = Some(handle);

        assert!(
            render_prompt(&sp.assemble(scoped_context(&key)).await.unwrap())
                .unwrap()
                .contains("Mode: first.")
        );
        assert_eq!(*calls.borrow(), ["first"]);
        assert!(
            render_prompt(&sp.assemble(scoped_context(&key)).await.unwrap())
                .unwrap()
                .contains("Mode: replacement.")
        );
        assert_eq!(*calls.borrow(), ["first", "replacement"]);
    });
}

#[test]
fn shadows_a_global_context_for_one_scope_and_cleans_up_with_that_scope() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        let key = ScopeKey::new();
        let scope = create_scope(&ctx, key.clone(), None).unwrap();
        sp.context(&ctx, context_entry("policy", 1.0, "global policy"))
            .unwrap();
        sp.context(&scope.ctx(), context_entry("policy", 1.0, "scoped policy"))
            .unwrap();
        let dup = sp
            .context(&scope.ctx(), context_entry("policy", 2.0, "duplicate"))
            .unwrap_err();
        assert!(
            format!("{dup:#}")
                .contains("prompt context \"policy\" is already registered in this scope"),
            "{dup:#}"
        );

        assert!(
            render_context_snapshot(&sp.assemble(scoped_context(&key)).await.unwrap())
                .unwrap()
                .contains("scoped policy")
        );
        assert!(
            render_context_snapshot(&sp.assemble(AssembleContext::default()).await.unwrap())
                .unwrap()
                .contains("global policy")
        );

        scope.dispose().await;
        assert!(
            render_context_snapshot(&sp.assemble(scoped_context(&key)).await.unwrap())
                .unwrap()
                .contains("global policy")
        );
    });
}

#[test]
fn suppresses_all_context_for_one_scope_and_restores_it_when_disposed() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        let key = ScopeKey::new();
        let scope = create_scope(&ctx, key.clone(), None).unwrap();
        sp.context(&ctx, context_entry("policy", 1.0, "global policy"))
            .unwrap();
        let handle = sp.suppress_runtime_context(&scope.ctx()).unwrap();

        let suppressed = sp.assemble(scoped_context(&key)).await.unwrap();
        assert!(suppressed.contexts.is_empty());
        let global = sp.assemble(AssembleContext::default()).await.unwrap();
        assert!(
            render_context_snapshot(&global)
                .unwrap()
                .contains("global policy")
        );

        handle.dispose().await;
        assert!(
            render_context_snapshot(&sp.assemble(scoped_context(&key)).await.unwrap())
                .unwrap()
                .contains("global policy")
        );
    });
}

#[test]
fn scoped_providers_are_consulted_only_for_their_scope() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        let key = ScopeKey::new();
        let scope = create_scope(&ctx, key.clone(), None).unwrap();
        sp.tools(&ctx, |_| provided(vec![schema("global_tool")]))
            .unwrap();
        sp.tools(&scope.ctx(), |_| provided(vec![schema("scoped_tool")]))
            .unwrap();

        let scoped = sp.assemble(scoped_context(&key)).await.unwrap();
        let global = sp.assemble(AssembleContext::default()).await.unwrap();
        assert_eq!(
            scoped
                .tools
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>(),
            ["global_tool", "scoped_tool"]
        );
        assert_eq!(
            global
                .tools
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>(),
            ["global_tool"]
        );
    });
}

#[test]
fn disposing_a_scoped_tool_provider_empties_its_layer_without_residue() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        let key = ScopeKey::new();
        let scope = create_scope(&ctx, key.clone(), None).unwrap();
        let handle = sp
            .tools(&scope.ctx(), |_| provided(vec![schema("scoped_tool")]))
            .unwrap();
        handle.dispose().await;
        let after = sp.assemble(scoped_context(&key)).await.unwrap();
        assert!(after.tools.is_empty());
        // Re-registering through the same scope starts a fresh layer.
        sp.tools(&scope.ctx(), |_| provided(vec![schema("again")]))
            .unwrap();
        let again = sp.assemble(scoped_context(&key)).await.unwrap();
        assert_eq!(
            again
                .tools
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>(),
            ["again"]
        );
    });
}

#[test]
fn a_restricted_tool_order_entry_is_a_normal_absence_while_a_typo_still_throws() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, json!({"toolOrder": ["bash", TOOL_ORDER_REST]})).await;
        // A provider mimicking the registry's restriction split: bash exists
        // (known_names) but is masked for this assembly (schemas).
        sp.tools(&ctx, |_| ToolProviderResult {
            schemas: vec![schema("read")],
            known_names: Some(vec!["read".into(), "bash".into()]),
        })
        .unwrap();
        let assembly = sp.assemble(AssembleContext::default()).await.unwrap();
        assert_eq!(
            assembly
                .tools
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>(),
            ["read"]
        );

        let bad_app = App::new();
        let bad_ctx = bad_app.root();
        let bad = mount(&bad_ctx, json!({"toolOrder": ["basj", TOOL_ORDER_REST]})).await;
        bad.tools(&bad_ctx, |_| ToolProviderResult {
            schemas: vec![schema("read")],
            known_names: Some(vec!["read".into(), "bash".into()]),
        })
        .unwrap();
        let error = bad.assemble(AssembleContext::default()).await.unwrap_err();
        assert!(
            format!("{error:#}")
                .contains("toolOrder lists unregistered tool \"basj\"; known tools: bash, read"),
            "{error:#}"
        );
    });
}

#[test]
fn a_scoped_assemble_listener_shapes_only_its_own_scopes_assemblies() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        let key = ScopeKey::new();
        let scope = create_scope(&ctx, key.clone(), None).unwrap();
        let shaped: Rc<RefCell<Vec<Option<ScopeKey>>>> = Rc::default();
        let shaped2 = shaped.clone();
        on_assemble(
            &scope.ctx(),
            EventOptions::default(),
            move |_ctx, assembly, context, next| {
                shaped2.borrow_mut().push(context.scope.clone());
                async move {
                    let mut result = next(assembly, context).await?;
                    result.sections.push(AssembledSection {
                        name: "listener:extra".into(),
                        text: "listener text".into(),
                    });
                    Ok(result)
                }
            },
        )
        .unwrap();

        let scoped = sp.assemble(scoped_context(&key)).await.unwrap();
        let global = sp.assemble(AssembleContext::default()).await.unwrap();
        assert!(scoped.sections.iter().any(|s| s.name == "listener:extra"));
        assert!(!global.sections.iter().any(|s| s.name == "listener:extra"));
        assert_eq!(shaped.borrow().len(), 1);
    });
}
