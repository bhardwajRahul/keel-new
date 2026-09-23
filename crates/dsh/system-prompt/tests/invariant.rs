//! Invariant companion tests ported from upstream `tests/invariant.spec.ts`.
//! The upstream rows asserting runtime type errors (non-string section or
//! context text, non-string variable values) are unrepresentable in Rust and
//! are not ported.

use dsh_cordis::{App, Context};
use dsh_invariants::InvariantsPlugin;
use dsh_llm::ToolSchema;
use dsh_scope::scope_target;
use dsh_system_prompt::{
    Assemble, AssembleContext, AssembledContext, AssembledSection, PromptAssembly, invariant,
};
use serde_json::Value;
use std::collections::BTreeMap;
use std::rc::Rc;

async fn setup(ctx: &Context) {
    ctx.plugin(Rc::new(InvariantsPlugin), Value::Null)
        .unwrap()
        .await_ready()
        .await
        .unwrap();
    ctx.plugin(Rc::new(invariant::plugin()), Value::Null)
        .unwrap()
        .await_ready()
        .await
        .unwrap();
}

fn valid() -> PromptAssembly {
    PromptAssembly {
        sections: vec![AssembledSection {
            name: "identity".into(),
            text: "prompt".into(),
        }],
        contexts: vec![AssembledContext {
            name: "policy".into(),
            text: "current policy".into(),
        }],
        tools: vec![ToolSchema {
            name: "echo".into(),
            description: "Echo".into(),
            parameters: Default::default(),
        }],
        variables: BTreeMap::from([
            ("cwd".to_string(), Some("/repo".to_string())),
            ("optional".to_string(), None),
        ]),
    }
}

/// Dispatch the assemble waterfall so it resolves to `result`.
async fn assemble(ctx: &Context, result: PromptAssembly) -> anyhow::Result<PromptAssembly> {
    ctx.waterfall::<Assemble, _, _>(
        (
            scope_target::<()>(None),
            valid(),
            AssembleContext::default(),
        ),
        move |_args| async move { Ok(result) },
    )
    .await
}

#[test]
fn accepts_a_well_formed_authoritative_assembly() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        setup(&ctx).await;
        assert_eq!(assemble(&ctx, valid()).await.unwrap(), valid());
    });
}

#[test]
fn rejects_malformed_authoritative_assemblies() {
    let sections = |sections: Vec<AssembledSection>| PromptAssembly {
        sections,
        ..valid()
    };
    let contexts = |contexts: Vec<AssembledContext>| PromptAssembly {
        contexts,
        ..valid()
    };
    let assembled_section = |name: &str, text: &str| AssembledSection {
        name: name.into(),
        text: text.into(),
    };
    let assembled_context = |name: &str, text: &str| AssembledContext {
        name: name.into(),
        text: text.into(),
    };

    let rows: Vec<(PromptAssembly, &str)> = vec![
        (
            sections(vec![assembled_section("", "x")]),
            "section names must be non-empty",
        ),
        (
            sections(vec![
                assembled_section("x", "a"),
                assembled_section("x", "b"),
            ]),
            "section name \"x\" is duplicated",
        ),
        (
            contexts(vec![assembled_context("", "x")]),
            "context names must be non-empty",
        ),
        (
            contexts(vec![
                assembled_context("x", "a"),
                assembled_context("x", "b"),
            ]),
            "context name \"x\" is duplicated",
        ),
        (
            PromptAssembly {
                tools: vec![ToolSchema {
                    name: String::new(),
                    description: "x".into(),
                    parameters: Default::default(),
                }],
                ..valid()
            },
            "tool names must be non-empty",
        ),
        (
            PromptAssembly {
                variables: BTreeMap::from([("Bad".to_string(), Some("x".to_string()))]),
                ..valid()
            },
            "variable name \"Bad\" is invalid",
        ),
    ];

    for (assembly, message) in rows {
        dsh_cordis::run(async {
            let app = App::new();
            let ctx = app.root();
            setup(&ctx).await;
            let error = assemble(&ctx, assembly).await.unwrap_err();
            assert!(
                format!("{error:#}").contains(message),
                "expected {message:?} in {error:#}"
            );
        });
    }
}
