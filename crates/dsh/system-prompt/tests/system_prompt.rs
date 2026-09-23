//! Behavioral contract tests ported from upstream
//! `tests/system-prompt.spec.ts`. Not ported: the three
//! rollback-on-change-listener-throw specs (the Rust `emit` is
//! fire-and-forget, so a listener cannot fail a registration).

use dsh_cordis::{App, Context, EventOptions, Inject, plugin_fn};
use dsh_llm::ToolSchema;
use dsh_system_prompt::{
    AssembleContext, AssembledContext, AssembledSection, Change, Config, PromptAssembly,
    PromptContext, PromptSection, PromptText, SystemPrompt, SystemPromptPlugin, ToolProviderResult,
    on_assemble, render_context_snapshot, render_prompt,
};
use serde_json::{Value, json};
use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::rc::Rc;

const IDENTITY: &str = "You are an AI agent powered by DeepSeek Harness.";
const BUILT_IN: [&str; 2] = ["harness:identity", "deployment:persona"];

async fn mount(ctx: &Context, config: Value) -> Rc<SystemPrompt> {
    let fiber = ctx.plugin(Rc::new(SystemPromptPlugin), config).unwrap();
    fiber.await_ready().await.unwrap();
    ctx.service::<SystemPrompt>().unwrap()
}

/// Let `emit`-spawned listener tasks run.
async fn settle() {
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
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

fn tool(name: &str, description: &str) -> ToolSchema {
    ToolSchema {
        name: name.into(),
        description: description.into(),
        parameters: Default::default(),
    }
}

fn schemas(tools: Vec<ToolSchema>) -> ToolProviderResult {
    ToolProviderResult {
        schemas: tools,
        known_names: None,
    }
}

fn names(sections: &[AssembledSection]) -> Vec<&str> {
    sections.iter().map(|s| s.name.as_str()).collect()
}

fn contributed(assembly: &PromptAssembly) -> Vec<&AssembledSection> {
    assembly
        .sections
        .iter()
        .filter(|s| !BUILT_IN.contains(&s.name.as_str()))
        .collect()
}

fn bare(sections: Vec<(&str, &str)>) -> PromptAssembly {
    PromptAssembly {
        sections: sections
            .into_iter()
            .map(|(name, text)| AssembledSection {
                name: name.into(),
                text: text.into(),
            })
            .collect(),
        ..Default::default()
    }
}

#[test]
fn registers_the_harness_identity_and_the_configured_deployment_persona() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, json!({"persona": "You are DeepSeek Harness."})).await;

        let assembly = sp.assemble(AssembleContext::default()).await.unwrap();
        assert_eq!(
            names(&assembly.sections),
            ["harness:identity", "deployment:persona"]
        );
        assert_eq!(
            render_prompt(&assembly).unwrap(),
            format!("{IDENTITY}\n\nYou are DeepSeek Harness.")
        );
        // The names are reserved by the plugin — one owner per section.
        let error = sp
            .section(&ctx, section("deployment:persona", 0.0, "imposter"))
            .unwrap_err();
        assert!(
            format!("{error:#}")
                .contains("prompt section \"deployment:persona\" is already registered"),
            "{error:#}"
        );
    });
}

#[test]
fn renders_no_persona_section_for_a_persona_less_deployment() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        assert_eq!(
            render_prompt(&sp.assemble(AssembleContext::default()).await.unwrap()).unwrap(),
            IDENTITY
        );
    });
}

#[test]
fn can_omit_the_harness_identity() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(
            &ctx,
            json!({
                "includeHarnessIdentity": false,
                "persona": "You are a helpful software engineer assistant.",
            }),
        )
        .await;
        let assembly = sp.assemble(AssembleContext::default()).await.unwrap();
        assert_eq!(names(&assembly.sections), ["deployment:persona"]);
        assert_eq!(
            render_prompt(&assembly).unwrap(),
            "You are a helpful software engineer assistant."
        );
    });
}

#[test]
fn can_suppress_runtime_context_without_evaluating_providers_or_accepting_waterfall_additions() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, json!({"includeRuntimeContext": false})).await;
        let calls = Rc::new(Cell::new(0));
        let calls2 = calls.clone();
        sp.context(
            &ctx,
            PromptContext {
                name: "policy".into(),
                order: 0.0,
                text: PromptText::provider(move |_| {
                    calls2.set(calls2.get() + 1);
                    format!("policy {}", calls2.get())
                }),
            },
        )
        .unwrap();
        on_assemble(
            &ctx,
            EventOptions::default(),
            |_ctx, mut assembly, context, next| {
                assembly.contexts.push(AssembledContext {
                    name: "late".into(),
                    text: "late context".into(),
                });
                next(assembly, context)
            },
        )
        .unwrap();

        let assembly = sp.assemble(AssembleContext::default()).await.unwrap();
        assert!(assembly.contexts.is_empty());
        assert_eq!(calls.get(), 0);
    });
}

#[test]
fn tolerates_direct_construction_with_default_config() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = SystemPrompt::new(&ctx, Config::default()).unwrap();
        assert_eq!(
            render_prompt(&sp.assemble(AssembleContext::default()).await.unwrap()).unwrap(),
            IDENTITY
        );
    });
}

#[test]
fn assembles_sections_in_order_with_context_resolved_text_and_collected_tools() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, json!({"persona": "You are DeepSeek Harness."})).await;

        sp.section(
            &ctx,
            PromptSection {
                name: "cwd".into(),
                order: 20.0,
                text: PromptText::provider(|_| "cwd: /tmp".into()),
                complete: false,
            },
        )
        .unwrap();
        sp.section(&ctx, section("rules", 10.0, "Be precise."))
            .unwrap();
        sp.context(
            &ctx,
            PromptContext {
                name: "later".into(),
                order: 20.0,
                text: PromptText::provider(|_| "context 2".into()),
            },
        )
        .unwrap();
        sp.context(&ctx, context_entry("earlier", 10.0, "context 1"))
            .unwrap();
        sp.tools(&ctx, |_| schemas(vec![tool("echo", "echo back")]))
            .unwrap();

        let assembly = sp.assemble(AssembleContext::default()).await.unwrap();
        assert_eq!(
            names(&assembly.sections),
            ["harness:identity", "deployment:persona", "rules", "cwd"]
        );
        assert_eq!(
            assembly
                .sections
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>(),
            [
                IDENTITY,
                "You are DeepSeek Harness.",
                "Be precise.",
                "cwd: /tmp"
            ]
        );
        assert_eq!(
            assembly.contexts,
            vec![
                AssembledContext {
                    name: "earlier".into(),
                    text: "context 1".into()
                },
                AssembledContext {
                    name: "later".into(),
                    text: "context 2".into()
                },
            ]
        );
        assert_eq!(assembly.tools, vec![tool("echo", "echo back")]);
        assert!(assembly.variables.is_empty());
        assert_eq!(
            render_prompt(&assembly).unwrap(),
            format!("{IDENTITY}\n\nYou are DeepSeek Harness.\n\nBe precise.\n\ncwd: /tmp")
        );
        assert_eq!(
            render_context_snapshot(&assembly).unwrap(),
            "Current runtime context. This snapshot supersedes earlier runtime-context snapshots.\n\ncontext 1\n\ncontext 2"
        );
    });
}

#[test]
fn resolves_section_text_providers_against_the_assemble_context_at_each_call() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        let calls = Rc::new(Cell::new(0));
        let calls2 = calls.clone();
        sp.section(
            &ctx,
            PromptSection {
                name: "dynamic".into(),
                order: 0.0,
                text: PromptText::provider(move |context| {
                    calls2.set(calls2.get() + 1);
                    let who = context
                        .payload
                        .as_ref()
                        .and_then(|payload| payload.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "nobody".into());
                    format!("call {} for {who}", calls2.get())
                }),
                complete: false,
            },
        )
        .unwrap();

        let alice = AssembleContext {
            scope: None,
            payload: Some(Rc::new("alice".to_string())),
        };
        let first = sp.assemble(alice).await.unwrap();
        assert_eq!(contributed(&first)[0].text, "call 1 for alice");
        let second = sp.assemble(AssembleContext::default()).await.unwrap();
        assert_eq!(contributed(&second)[0].text, "call 2 for nobody");
    });
}

#[test]
fn removes_contributions_when_the_contributing_fiber_is_disposed() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;

        let fiber = ctx
            .plugin(
                Rc::new(plugin_fn(
                    "contributor",
                    Inject::names(["systemPrompt"]),
                    |ctx, _| async move {
                        let sp = ctx.service::<SystemPrompt>()?;
                        sp.section(&ctx, section("scoped", 0.0, "scoped section"))?;
                        sp.context(&ctx, context_entry("scoped-context", 0.0, "scoped context"))?;
                        sp.tools(&ctx, |_| schemas(vec![tool("scoped-tool", "")]))?;
                        sp.variable(&ctx, "scoped_var", |_| Some("v".into()))?;
                        Ok(())
                    },
                )),
                Value::Null,
            )
            .unwrap();
        fiber.await_ready().await.unwrap();

        let before = sp.assemble(AssembleContext::default()).await.unwrap();
        assert_eq!(contributed(&before).len(), 1);
        assert_eq!(before.contexts.len(), 1);
        assert_eq!(
            before.variables,
            BTreeMap::from([("scoped_var".to_string(), Some("v".to_string()))])
        );
        fiber.dispose().await;
        let assembly = sp.assemble(AssembleContext::default()).await.unwrap();
        assert!(contributed(&assembly).is_empty());
        assert!(assembly.contexts.is_empty());
        // The built-ins belong to the service fiber and survive this disposal.
        assert_eq!(names(&assembly.sections), BUILT_IN);
        assert!(assembly.tools.is_empty());
        assert!(assembly.variables.is_empty());
    });
}

#[test]
fn rejects_a_duplicate_section_name_without_leaking() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        sp.section(&ctx, section("dup", 0.0, "first")).unwrap();
        let error = sp.section(&ctx, section("dup", 1.0, "second")).unwrap_err();
        assert!(
            format!("{error:#}").contains("prompt section \"dup\" is already registered"),
            "{error:#}"
        );
        let assembly = sp.assemble(AssembleContext::default()).await.unwrap();
        assert_eq!(
            contributed(&assembly)
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>(),
            ["first"]
        );
    });
}

#[test]
fn rejects_a_non_finite_section_order() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        let error = sp
            .section(&ctx, section("bad-order", f64::NAN, "x"))
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("order must be a finite number"),
            "{error:#}"
        );
        assert!(contributed(&sp.assemble(AssembleContext::default()).await.unwrap()).is_empty());
    });
}

#[test]
fn rejects_duplicate_and_non_finite_context_registrations_without_leaking() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        sp.context(&ctx, context_entry("policy", 1.0, "first"))
            .unwrap();
        let dup = sp
            .context(&ctx, context_entry("policy", 2.0, "second"))
            .unwrap_err();
        assert!(
            format!("{dup:#}").contains("prompt context \"policy\" is already registered"),
            "{dup:#}"
        );
        let bad = sp
            .context(&ctx, context_entry("bad", f64::NAN, "x"))
            .unwrap_err();
        assert!(
            format!("{bad:#}").contains("prompt context \"bad\" order must be a finite number"),
            "{bad:#}"
        );
        assert_eq!(
            sp.assemble(AssembleContext::default())
                .await
                .unwrap()
                .contexts,
            vec![AssembledContext {
                name: "policy".into(),
                text: "first".into()
            }]
        );
    });
}

#[test]
fn snapshots_tool_provider_membership_before_evaluating_an_assembly() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        let added = Rc::new(Cell::new(false));
        let added2 = added.clone();
        let sp2 = sp.clone();
        let ctx2 = ctx.clone();
        sp.tools(&ctx, move |_| {
            if !added2.get() {
                added2.set(true);
                sp2.tools(&ctx2, |_| schemas(vec![tool("late", "")]))
                    .unwrap();
            }
            schemas(vec![tool("first", "")])
        })
        .unwrap();

        let tool_names = |assembly: &PromptAssembly| {
            assembly
                .tools
                .iter()
                .map(|t| t.name.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            tool_names(&sp.assemble(AssembleContext::default()).await.unwrap()),
            ["first"]
        );
        assert_eq!(
            tool_names(&sp.assemble(AssembleContext::default()).await.unwrap()),
            ["first", "late"]
        );
    });
}

#[test]
fn composes_multiple_assemble_waterfall_listeners_in_order_with_the_context() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        sp.section(&ctx, section("base", 0.0, "base")).unwrap();

        // Listener A records the context payload, appends, then delegates.
        let payloads: Rc<RefCell<Vec<Option<Rc<dyn Any>>>>> = Rc::default();
        let payloads2 = payloads.clone();
        on_assemble(
            &ctx,
            EventOptions::default(),
            move |_ctx, mut assembly, context, next| {
                payloads2.borrow_mut().push(context.payload.clone());
                assembly.sections.push(AssembledSection {
                    name: "from-a".into(),
                    text: "a".into(),
                });
                next(assembly, context)
            },
        )
        .unwrap();
        // Listener B (registered later, runs after A) sees A's contribution.
        let seen: Rc<RefCell<Vec<Vec<String>>>> = Rc::default();
        let seen2 = seen.clone();
        on_assemble(
            &ctx,
            EventOptions::default(),
            move |_ctx, assembly, context, next| {
                seen2
                    .borrow_mut()
                    .push(assembly.sections.iter().map(|s| s.name.clone()).collect());
                next(assembly, context)
            },
        )
        .unwrap();

        let marker: Rc<dyn Any> = Rc::new(42u32);
        let assembly = sp
            .assemble(AssembleContext {
                scope: None,
                payload: Some(marker.clone()),
            })
            .await
            .unwrap();
        assert_eq!(
            *seen.borrow(),
            vec![vec![
                "harness:identity".to_string(),
                "deployment:persona".to_string(),
                "base".to_string(),
                "from-a".to_string(),
            ]]
        );
        assert_eq!(
            names(&assembly.sections),
            ["harness:identity", "deployment:persona", "base", "from-a"]
        );
        // The caller's payload reaches listeners by identity.
        let recorded = payloads.borrow();
        assert!(Rc::ptr_eq(recorded[0].as_ref().unwrap(), &marker));
    });
}

#[test]
fn lets_a_waterfall_listener_short_circuit_by_not_calling_next() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        sp.section(&ctx, section("real", 0.0, "real")).unwrap();

        on_assemble(
            &ctx,
            EventOptions::default(),
            |_ctx, _assembly, _context, _next| async move { Ok(PromptAssembly::default()) },
        )
        .unwrap();

        let assembly = sp.assemble(AssembleContext::default()).await.unwrap();
        assert!(assembly.sections.is_empty());
    });
}

#[test]
fn restores_one_complete_section_after_the_assembly_waterfall() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        sp.section(
            &ctx,
            PromptSection {
                name: "complete".into(),
                order: 10.0,
                text: PromptText::fixed("Exact prompt."),
                complete: true,
            },
        )
        .unwrap();
        sp.section(&ctx, section("extra", 20.0, "extra")).unwrap();
        on_assemble(
            &ctx,
            EventOptions {
                prepend: true,
                ..Default::default()
            },
            |_ctx, mut assembly, context, next| {
                let position = assembly
                    .sections
                    .iter()
                    .position(|s| s.name == "complete")
                    .expect("complete section missing before waterfall");
                assembly.sections[position].text = "mutated".into();
                assembly.sections.push(AssembledSection {
                    name: "late".into(),
                    text: "late".into(),
                });
                next(assembly, context)
            },
        )
        .unwrap();

        assert_eq!(
            sp.assemble(AssembleContext::default())
                .await
                .unwrap()
                .sections,
            vec![AssembledSection {
                name: "complete".into(),
                text: "Exact prompt.".into()
            }]
        );
    });
}

#[test]
fn rejects_multiple_effective_complete_sections() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        sp.section(
            &ctx,
            PromptSection {
                name: "first".into(),
                order: 10.0,
                text: PromptText::fixed("first"),
                complete: true,
            },
        )
        .unwrap();
        sp.section(
            &ctx,
            PromptSection {
                name: "second".into(),
                order: 20.0,
                text: PromptText::fixed("second"),
                complete: true,
            },
        )
        .unwrap();

        let error = sp.assemble(AssembleContext::default()).await.unwrap_err();
        assert!(
            format!("{error:#}")
                .contains("multiple complete prompt sections are active: \"first\", \"second\""),
            "{error:#}"
        );
    });
}

#[test]
fn assembles_snapshots_so_one_step_mutations_do_not_leak_into_future_assemblies() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        sp.section(&ctx, section("base", 0.0, "base")).unwrap();
        let parameters = json!({"type": "object", "properties": {}})
            .as_object()
            .unwrap()
            .clone();
        let shared = ToolSchema {
            name: "t".into(),
            description: "tool".into(),
            parameters,
        };
        let shared2 = shared.clone();
        sp.tools(&ctx, move |_| ToolProviderResult {
            schemas: vec![shared2.clone()],
            known_names: None,
        })
        .unwrap();

        let mut first = sp.assemble(AssembleContext::default()).await.unwrap();
        first.sections[0].name = "mutated".into();
        first.sections[0].text = "mutated".into();
        first.contexts.push(AssembledContext {
            name: "mutated".into(),
            text: "mutated".into(),
        });
        first.tools[0].description = "mutated".into();
        first.tools[0]
            .parameters
            .insert("leak".into(), json!({"type": "string"}));

        let second = sp.assemble(AssembleContext::default()).await.unwrap();
        assert_eq!(
            names(&second.sections),
            ["harness:identity", "deployment:persona", "base"]
        );
        assert_eq!(second.sections[0].text, IDENTITY);
        assert!(second.contexts.is_empty());
        assert_eq!(second.tools, vec![shared]);
    });
}

#[test]
fn filters_out_empty_section_text_from_render_prompt() {
    let assembly = bare(vec![("empty", ""), ("real", "content")]);
    assert_eq!(render_prompt(&assembly).unwrap(), "content");
}

#[test]
fn filters_empty_context_interpolates_variables_and_returns_empty_without_active_context() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        sp.context(&ctx, context_entry("empty", 0.0, "")).unwrap();
        assert_eq!(
            render_context_snapshot(&sp.assemble(AssembleContext::default()).await.unwrap())
                .unwrap(),
            ""
        );
        sp.variable(&ctx, "mode", |_| Some("read-only".into()))
            .unwrap();
        sp.context(&ctx, context_entry("policy", 1.0, "Mode: {{mode}}."))
            .unwrap();
        assert_eq!(
            render_context_snapshot(&sp.assemble(AssembleContext::default()).await.unwrap())
                .unwrap(),
            "Current runtime context. This snapshot supersedes earlier runtime-context snapshots.\n\nMode: read-only."
        );
    });
}

#[test]
fn attributes_context_interpolation_failures_to_the_contributing_context() {
    let assembly = PromptAssembly {
        contexts: vec![AssembledContext {
            name: "policy".into(),
            text: "Mode: {{missing}}.".into(),
        }],
        ..Default::default()
    };
    let error = render_context_snapshot(&assembly).unwrap_err();
    assert!(
        format!("{error:#}").contains(
            "unknown prompt variable \"{{missing}}\" in context \"policy\"; registered variables: (none)"
        ),
        "{error:#}"
    );
}

#[test]
fn emits_change_when_a_tool_provider_is_registered_and_disposed() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;

        let count = Rc::new(Cell::new(0));
        let count2 = count.clone();
        ctx.on::<Change, _, _>(EventOptions::default(), move |_ctx, _| {
            count2.set(count2.get() + 1);
            std::future::ready(None)
        })
        .unwrap();

        let handle = sp.tools(&ctx, |_| schemas(vec![])).unwrap();
        settle().await;
        assert_eq!(count.get(), 1);

        handle.dispose().await;
        settle().await;
        assert_eq!(count.get(), 2);
    });
}

#[test]
fn emits_change_when_a_context_is_registered_and_disposed() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        let count = Rc::new(Cell::new(0));
        let count2 = count.clone();
        ctx.on::<Change, _, _>(EventOptions::default(), move |_ctx, _| {
            count2.set(count2.get() + 1);
            std::future::ready(None)
        })
        .unwrap();
        let handle = sp
            .context(&ctx, context_entry("policy", 0.0, "current"))
            .unwrap();
        settle().await;
        assert_eq!(count.get(), 1);
        handle.dispose().await;
        settle().await;
        assert_eq!(count.get(), 2);
    });
}

#[test]
fn cleans_up_tool_providers_on_fiber_dispose() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;

        let fiber = ctx
            .plugin(
                Rc::new(plugin_fn(
                    "tool-owner",
                    Inject::names(["systemPrompt"]),
                    |ctx, _| async move {
                        let sp = ctx.service::<SystemPrompt>()?;
                        sp.tools(&ctx, |_| schemas(vec![tool("fiber-tool", "")]))?;
                        Ok(())
                    },
                )),
                Value::Null,
            )
            .unwrap();
        fiber.await_ready().await.unwrap();

        assert_eq!(
            sp.assemble(AssembleContext::default())
                .await
                .unwrap()
                .tools
                .len(),
            1
        );
        fiber.dispose().await;
        assert!(
            sp.assemble(AssembleContext::default())
                .await
                .unwrap()
                .tools
                .is_empty()
        );
    });
}

#[test]
fn removes_section_when_returned_disposer_is_called_directly() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;

        let handle = sp
            .section(&ctx, section("direct", 0.0, "direct section"))
            .unwrap();
        assert_eq!(
            contributed(&sp.assemble(AssembleContext::default()).await.unwrap()).len(),
            1
        );

        handle.dispose().await;
        assert!(contributed(&sp.assemble(AssembleContext::default()).await.unwrap()).is_empty());
    });
}

#[test]
fn removes_tool_provider_when_returned_disposer_is_called_directly() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;

        let handle = sp
            .tools(&ctx, |_| schemas(vec![tool("direct-tool", "")]))
            .unwrap();
        assert_eq!(
            sp.assemble(AssembleContext::default())
                .await
                .unwrap()
                .tools
                .len(),
            1
        );

        handle.dispose().await;
        assert!(
            sp.assemble(AssembleContext::default())
                .await
                .unwrap()
                .tools
                .is_empty()
        );
    });
}

// ---- prompt variables ----

#[test]
fn resolves_each_variable_against_the_assemble_context_and_emits_change() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        let count = Rc::new(Cell::new(0));
        let count2 = count.clone();
        ctx.on::<Change, _, _>(EventOptions::default(), move |_ctx, _| {
            count2.set(count2.get() + 1);
            std::future::ready(None)
        })
        .unwrap();

        let handle = sp
            .variable(&ctx, "who", |context| {
                context
                    .payload
                    .as_ref()
                    .and_then(|payload| payload.downcast_ref::<String>().cloned())
            })
            .unwrap();
        settle().await;
        assert_eq!(count.get(), 1);

        let alice = AssembleContext {
            scope: None,
            payload: Some(Rc::new("alice".to_string())),
        };
        assert_eq!(
            sp.assemble(alice).await.unwrap().variables,
            BTreeMap::from([("who".to_string(), Some("alice".to_string()))])
        );
        // A provider returning None records "registered but no value here".
        assert_eq!(
            sp.assemble(AssembleContext::default())
                .await
                .unwrap()
                .variables,
            BTreeMap::from([("who".to_string(), None)])
        );

        handle.dispose().await;
        settle().await;
        assert_eq!(count.get(), 2);
        assert!(
            sp.assemble(AssembleContext::default())
                .await
                .unwrap()
                .variables
                .is_empty()
        );
    });
}

#[test]
fn live_iterates_variables_registered_by_an_earlier_provider() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        let added = Rc::new(Cell::new(false));
        let added2 = added.clone();
        let sp2 = sp.clone();
        let ctx2 = ctx.clone();
        sp.variable(&ctx, "first", move |_| {
            if !added2.get() {
                added2.set(true);
                sp2.variable(&ctx2, "late", |_| Some("second value".into()))
                    .unwrap();
            }
            Some("first value".into())
        })
        .unwrap();

        assert_eq!(
            sp.assemble(AssembleContext::default())
                .await
                .unwrap()
                .variables,
            BTreeMap::from([
                ("first".to_string(), Some("first value".to_string())),
                ("late".to_string(), Some("second value".to_string())),
            ])
        );
    });
}

#[test]
fn rejects_a_duplicate_variable_name_and_an_unreferenceable_name() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        sp.variable(&ctx, "model", |_| Some("m1".into())).unwrap();
        let dup = sp
            .variable(&ctx, "model", |_| Some("m2".into()))
            .unwrap_err();
        assert!(
            format!("{dup:#}").contains("prompt variable \"model\" is already registered"),
            "{dup:#}"
        );
        let bad = sp
            .variable(&ctx, "Not Valid", |_| Some("x".into()))
            .unwrap_err();
        assert!(
            format!("{bad:#}").contains("invalid prompt variable name \"Not Valid\""),
            "{bad:#}"
        );
        // Neither failed registration leaked.
        assert_eq!(
            sp.assemble(AssembleContext::default())
                .await
                .unwrap()
                .variables,
            BTreeMap::from([("model".to_string(), Some("m1".to_string()))])
        );
    });
}

#[test]
fn interpolates_variable_references_in_section_text_at_render_the_persona_included() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, json!({"persona": "You run on {{model}} in {{cwd}}."})).await;
        sp.variable(&ctx, "model", |_| Some("deepseek-v4".into()))
            .unwrap();
        sp.variable(&ctx, "cwd", |_| Some("/work".into())).unwrap();

        assert_eq!(
            render_prompt(&sp.assemble(AssembleContext::default()).await.unwrap()).unwrap(),
            format!("{IDENTITY}\n\nYou run on deepseek-v4 in /work.")
        );
    });
}

#[test]
fn lets_a_waterfall_listener_add_or_override_variables_before_render() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        sp.section(&ctx, section("s", 0.0, "{{extra}}")).unwrap();
        on_assemble(
            &ctx,
            EventOptions::default(),
            |_ctx, mut assembly, context, next| {
                assembly
                    .variables
                    .insert("extra".into(), Some("from-waterfall".into()));
                next(assembly, context)
            },
        )
        .unwrap();
        assert_eq!(
            render_prompt(&sp.assemble(AssembleContext::default()).await.unwrap()).unwrap(),
            format!("{IDENTITY}\n\nfrom-waterfall")
        );
    });
}

#[test]
fn throws_on_a_reference_to_an_unregistered_variable_listing_what_exists() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        sp.section(&ctx, section("persona", 0.0, "on {{modle}}"))
            .unwrap();
        sp.variable(&ctx, "model", |_| Some("m".into())).unwrap();
        let assembly = sp.assemble(AssembleContext::default()).await.unwrap();
        let error = render_prompt(&assembly).unwrap_err();
        assert!(
            format!("{error:#}").contains(
                "unknown prompt variable \"{{modle}}\" in section \"persona\"; registered variables: model"
            ),
            "{error:#}"
        );
    });
}

#[test]
fn names_none_when_no_variables_are_registered_at_all() {
    let error = render_prompt(&bare(vec![("s", "{{x}}")])).unwrap_err();
    assert!(
        format!("{error:#}").contains(
            "unknown prompt variable \"{{x}}\" in section \"s\"; registered variables: (none)"
        ),
        "{error:#}"
    );
}

#[test]
fn throws_when_a_referenced_variable_has_no_value_for_this_assembly() {
    let mut assembly = bare(vec![("persona", "in {{cwd}}")]);
    assembly.variables.insert("cwd".into(), None);
    let error = render_prompt(&assembly).unwrap_err();
    assert!(
        format!("{error:#}").contains(
            "prompt variable \"{{cwd}}\" has no value for this assembly (section \"persona\")"
        ),
        "{error:#}"
    );
}

#[test]
fn throws_on_a_malformed_complete_reference_with_inner_spaces() {
    let mut assembly = bare(vec![("s", "on {{ model }}")]);
    assembly.variables.insert("model".into(), Some("m".into()));
    let error = render_prompt(&assembly).unwrap_err();
    assert!(
        format!("{error:#}")
            .contains("malformed prompt variable reference \"{{ model }}\" in section \"s\""),
        "{error:#}"
    );
}

#[test]
fn leaves_a_lone_open_brace_pair_verbatim_only_when_no_close_pair_follows() {
    let assembly = bare(vec![("s", "shell ${X:-{{fallback} stays")]);
    assert_eq!(
        render_prompt(&assembly).unwrap(),
        "shell ${X:-{{fallback} stays"
    );
}

#[test]
fn throws_on_a_mangled_reference_with_a_close_pair_still_following() {
    for text in ["{{{model}}}", "x {{a{b}} y {{model}}"] {
        let mut assembly = bare(vec![("s", text)]);
        assembly.variables.insert("model".into(), Some("m".into()));
        let error = render_prompt(&assembly).unwrap_err();
        assert!(
            format!("{error:#}").contains("malformed prompt variable reference at"),
            "{text}: {error:#}"
        );
    }
}

#[test]
fn rejects_an_unregistered_prototype_like_name_as_unknown() {
    let mut assembly = bare(vec![("s", "on {{constructor}}")]);
    assembly.variables.insert("model".into(), Some("m".into()));
    let error = render_prompt(&assembly).unwrap_err();
    assert!(
        format!("{error:#}").contains("unknown prompt variable \"{{constructor}}\""),
        "{error:#}"
    );
}

#[test]
fn a_variable_named_like_a_prototype_property_works_once_actually_registered() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        sp.section(&ctx, section("s", 0.0, "{{constructor}}"))
            .unwrap();
        sp.variable(&ctx, "constructor", |_| Some("own-value".into()))
            .unwrap();
        assert_eq!(
            render_prompt(&sp.assemble(AssembleContext::default()).await.unwrap()).unwrap(),
            format!("{IDENTITY}\n\nown-value")
        );
    });
}

#[test]
fn never_rescans_substituted_values() {
    let mut assembly = bare(vec![("s", "v = {{model}}!")]);
    assembly
        .variables
        .insert("model".into(), Some("literal {{sneaky}} inside".into()));
    assert_eq!(
        render_prompt(&assembly).unwrap(),
        "v = literal {{sneaky}} inside!"
    );
}
