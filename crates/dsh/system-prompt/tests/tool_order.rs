//! Tool ordering contract tests ported from upstream
//! `tests/tool-order.spec.ts`.

use dsh_cordis::{App, Context, EventOptions};
use dsh_llm::ToolSchema;
use dsh_system_prompt::{
    AssembleContext, Config, PromptAssembly, SystemPrompt, SystemPromptPlugin, TOOL_ORDER_REST,
    ToolProviderResult, on_assemble,
};
use serde_json::{Value, json};
use std::cell::RefCell;
use std::rc::Rc;

fn tool(name: &str) -> ToolSchema {
    tool_described(name, name)
}

fn tool_described(name: &str, description: &str) -> ToolSchema {
    ToolSchema {
        name: name.into(),
        description: description.into(),
        parameters: json!({"type": "object", "properties": {}})
            .as_object()
            .unwrap()
            .clone(),
    }
}

fn schemas(tools: Vec<ToolSchema>) -> ToolProviderResult {
    ToolProviderResult {
        schemas: tools,
        known_names: None,
    }
}

async fn mount(ctx: &Context, config: Value) -> Rc<SystemPrompt> {
    let fiber = ctx.plugin(Rc::new(SystemPromptPlugin), config).unwrap();
    fiber.await_ready().await.unwrap();
    ctx.service::<SystemPrompt>().unwrap()
}

fn names(assembly: &PromptAssembly) -> Vec<&str> {
    assembly.tools.iter().map(|t| t.name.as_str()).collect()
}

// The ONE place the public constant's value is pinned; everything else
// references TOOL_ORDER_REST.
#[test]
fn exports_the_rest_entry_as_unlisted_tools() {
    assert_eq!(TOOL_ORDER_REST, "<unlisted-tools>");
}

#[test]
fn assembles_tools_in_lexicographic_name_order_when_no_tool_order_is_configured() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        sp.tools(&ctx, |_| schemas(vec![tool("charlie"), tool("alpha")]))
            .unwrap();
        sp.tools(&ctx, |_| schemas(vec![tool("bravo")])).unwrap();
        let assembly = sp.assemble(AssembleContext::default()).await.unwrap();
        assert_eq!(names(&assembly), ["alpha", "bravo", "charlie"]);
    });
}

#[test]
fn assembles_the_same_order_regardless_of_provider_registration_order() {
    dsh_cordis::run(async {
        let app = App::new();
        let forward_ctx = app.root();
        let forward = mount(&forward_ctx, Value::Null).await;
        forward
            .tools(&forward_ctx, |_| schemas(vec![tool("alpha")]))
            .unwrap();
        forward
            .tools(&forward_ctx, |_| schemas(vec![tool("zulu")]))
            .unwrap();

        let backward_app = App::new();
        let backward_ctx = backward_app.root();
        let backward = mount(&backward_ctx, Value::Null).await;
        backward
            .tools(&backward_ctx, |_| schemas(vec![tool("zulu")]))
            .unwrap();
        backward
            .tools(&backward_ctx, |_| schemas(vec![tool("alpha")]))
            .unwrap();

        assert_eq!(
            names(&forward.assemble(AssembleContext::default()).await.unwrap()),
            ["alpha", "zulu"]
        );
        assert_eq!(
            names(&backward.assemble(AssembleContext::default()).await.unwrap()),
            ["alpha", "zulu"]
        );
    });
}

#[test]
fn applies_a_configured_tool_order_with_the_rest_inserted_lexicographically() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(
            &ctx,
            json!({"toolOrder": ["todo_write", TOOL_ORDER_REST, "bash"]}),
        )
        .await;
        sp.tools(&ctx, |_| {
            schemas(vec![
                tool("bash"),
                tool("echo_b"),
                tool("todo_write"),
                tool("echo_a"),
            ])
        })
        .unwrap();
        let assembly = sp.assemble(AssembleContext::default()).await.unwrap();
        assert_eq!(names(&assembly), ["todo_write", "echo_a", "echo_b", "bash"]);
    });
}

#[test]
fn rejects_the_assembly_when_tool_order_names_unregistered_tools() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(
            &ctx,
            json!({"toolOrder": ["todo_write", "ghost", TOOL_ORDER_REST, "wraith"]}),
        )
        .await;
        sp.tools(&ctx, |_| schemas(vec![tool("bash"), tool("todo_write")]))
            .unwrap();
        let error = sp.assemble(AssembleContext::default()).await.unwrap_err();
        assert!(
            format!("{error:#}").contains(
                "toolOrder lists unregistered tools \"ghost\", \"wraith\"; known tools: bash, todo_write"
            ),
            "{error:#}"
        );
    });
}

#[test]
fn names_the_single_unregistered_tool_when_no_tools_are_registered_at_all() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, json!({"toolOrder": ["ghost", TOOL_ORDER_REST]})).await;
        let error = sp.assemble(AssembleContext::default()).await.unwrap_err();
        assert!(
            format!("{error:#}")
                .contains("toolOrder lists unregistered tool \"ghost\"; known tools: (none)"),
            "{error:#}"
        );
    });
}

#[test]
fn rejects_a_provider_tool_named_like_the_reserved_rest_entry() {
    // Both without an explicit toolOrder and with only the rest entry.
    for config in [Value::Null, json!({"toolOrder": [TOOL_ORDER_REST]})] {
        dsh_cordis::run(async {
            let app = App::new();
            let ctx = app.root();
            let sp = mount(&ctx, config).await;
            sp.tools(&ctx, |_| schemas(vec![tool(TOOL_ORDER_REST)]))
                .unwrap();
            let error = sp.assemble(AssembleContext::default()).await.unwrap_err();
            assert!(
                format!("{error:#}")
                    .contains("tool provider returned reserved tool name \"<unlisted-tools>\""),
                "{error:#}"
            );
        });
    }
}

#[test]
fn keeps_collection_order_between_tools_that_share_a_name() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        sp.tools(&ctx, |_| {
            schemas(vec![
                tool_described("dup", "first"),
                tool_described("anchor", "anchor"),
                tool_described("dup", "second"),
            ])
        })
        .unwrap();
        let assembly = sp.assemble(AssembleContext::default()).await.unwrap();
        assert_eq!(
            assembly
                .tools
                .iter()
                .map(|t| t.description.as_str())
                .collect::<Vec<_>>(),
            ["anchor", "first", "second"]
        );
    });
}

#[test]
fn canonicalizes_before_the_assemble_waterfall_and_listeners_own_their_own_edits() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let sp = mount(&ctx, Value::Null).await;
        sp.tools(&ctx, |_| schemas(vec![tool("zulu"), tool("alpha")]))
            .unwrap();
        let seen: Rc<RefCell<Option<Vec<String>>>> = Rc::default();
        let seen2 = seen.clone();
        on_assemble(
            &ctx,
            EventOptions::default(),
            move |_ctx, mut assembly, context, next| {
                *seen2.borrow_mut() = Some(assembly.tools.iter().map(|t| t.name.clone()).collect());
                // A listener-appended tool is NOT re-sorted — same contract as
                // sections: canonicalization applies to registry contributions.
                assembly.tools.push(tool("aardvark"));
                next(assembly, context)
            },
        )
        .unwrap();
        let assembly = sp.assemble(AssembleContext::default()).await.unwrap();
        assert_eq!(
            seen.borrow().as_deref(),
            Some(["alpha".to_string(), "zulu".to_string()].as_slice())
        );
        assert_eq!(names(&assembly), ["alpha", "zulu", "aardvark"]);
    });
}

#[test]
fn rejects_an_order_without_the_rest_entry_at_load() {
    // An empty list and a list without the rest entry.
    for order in [json!([]), json!(["bash", "todo_write"])] {
        dsh_cordis::run(async {
            let app = App::new();
            let ctx = app.root();
            let fiber = ctx
                .plugin(Rc::new(SystemPromptPlugin), json!({"toolOrder": order}))
                .unwrap();
            let error = fiber.await_ready().await.unwrap_err();
            assert!(
                format!("{error:#}").contains("must contain the \"<unlisted-tools>\" rest entry"),
                "{error:#}"
            );
        });
    }
}

#[test]
fn rejects_duplicate_entries_at_load() {
    // A duplicate tool name and a duplicate rest entry.
    for order in [
        json!(["bash", "bash", TOOL_ORDER_REST]),
        json!([TOOL_ORDER_REST, "bash", TOOL_ORDER_REST]),
    ] {
        dsh_cordis::run(async {
            let app = App::new();
            let ctx = app.root();
            let fiber = ctx
                .plugin(Rc::new(SystemPromptPlugin), json!({"toolOrder": order}))
                .unwrap();
            let error = fiber.await_ready().await.unwrap_err();
            assert!(format!("{error:#}").contains("more than once"), "{error:#}");
        });
    }
}

#[test]
fn throws_from_direct_construction_too() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let error = SystemPrompt::new(
            &ctx,
            Config {
                tool_order: Some(vec!["bash".into()]),
                ..Default::default()
            },
        )
        .map(|_| ())
        .unwrap_err();
        assert!(format!("{error:#}").contains("rest entry"), "{error:#}");
    });
}
