//! Behavior tests for the value-schema DSL: author parsing, compilation to
//! the enforced subset, argument validation, and the `define_tool` wrappers
//! (mirroring the upstream schema and validateArgs suites).

use dsh_llm::ContentBlock;
use dsh_tools::{
    DefineToolOptions, DefineToolOutput, ParameterSchemaSpec, ToolArgsError, ValueSchemaSpec,
    define_tool, parameter_schema_spec_to_json_schema, validate_args,
    value_schema_spec_to_json_schema,
};
use futures::FutureExt;
use serde_json::{Value, json};
use std::rc::Rc;

fn parameters(spec: Value) -> ParameterSchemaSpec {
    ParameterSchemaSpec::from_author_value(&spec, "parameters").unwrap()
}

fn value_spec(spec: Value) -> ValueSchemaSpec {
    ValueSchemaSpec::from_author_value(&spec, "schema").unwrap()
}

#[test]
fn compiles_parameters_with_required_array_and_annotations() {
    let spec = parameters(json!({
        "path": { "type": "string", "required": true, "description": "Absolute path" },
        "offset": { "type": "number" },
        "limit": { "type": "number", "description": "Max lines" },
    }));
    let schema = parameter_schema_spec_to_json_schema(&spec).unwrap();
    assert_eq!(
        schema,
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Absolute path" },
                "offset": { "type": "number" },
                "limit": { "type": "number", "description": "Max lines" },
            },
            "required": ["path"],
        })
    );
}

#[test]
fn compiles_an_empty_spec_without_required() {
    assert_eq!(
        parameter_schema_spec_to_json_schema(&ParameterSchemaSpec::default()).unwrap(),
        json!({ "type": "object", "properties": {} })
    );
}

#[test]
fn compiles_nested_objects_with_per_level_required() {
    let spec = parameters(json!({
        "config": {
            "type": "object",
            "additionalProperties": true,
            "required": true,
            "properties": {
                "host": { "type": "string", "required": true },
                "port": { "type": "number" },
            },
        },
    }));
    assert_eq!(
        parameter_schema_spec_to_json_schema(&spec).unwrap(),
        json!({
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "additionalProperties": true,
                    "properties": { "host": { "type": "string" }, "port": { "type": "number" } },
                    "required": ["host"],
                },
            },
            "required": ["config"],
        })
    );
}

#[test]
fn compiles_literals_arrays_and_defaults() {
    let spec = parameters(json!({
        "color": { "type": "string", "enum": ["red", "green", "blue"], "description": "Color choice" },
        "limit": { "type": "number", "default": 25 },
        "tags": { "type": "array", "items": { "type": "string" } },
        "raw": { "type": "array" },
        "bare": { "type": "string" },
    }));
    let schema = parameter_schema_spec_to_json_schema(&spec).unwrap();
    let properties = &schema["properties"];
    assert_eq!(
        properties["color"],
        json!({ "type": "string", "enum": ["red", "green", "blue"], "description": "Color choice" })
    );
    assert_eq!(
        properties["limit"],
        json!({ "type": "number", "default": 25 })
    );
    assert_eq!(
        properties["tags"],
        json!({ "type": "array", "items": { "type": "string" } })
    );
    assert_eq!(properties["raw"], json!({ "type": "array" }));
    // Unspecified keys stay absent rather than serializing as null.
    assert_eq!(properties["bare"], json!({ "type": "string" }));
}

#[test]
fn compiles_value_schemas_including_json_and_one_of() {
    // The author-only `json` node compiles to the annotation-only schema.
    assert_eq!(
        value_schema_spec_to_json_schema(&value_spec(
            json!({ "type": "json", "description": "d" })
        ))
        .unwrap(),
        json!({ "description": "d" })
    );
    assert_eq!(
        value_schema_spec_to_json_schema(&value_spec(json!({
            "oneOf": [{ "type": "string" }, { "type": "null" }]
        })))
        .unwrap(),
        json!({ "oneOf": [{ "type": "string" }, { "type": "null" }] })
    );
    // A single-branch union fails the compiled-schema assertion.
    let single = value_spec(json!({ "oneOf": [{ "type": "string" }, { "type": "null" }] }));
    let ValueSchemaSpec::OneOf(mut one_of) = single else {
        panic!("expected oneOf")
    };
    one_of.branches.truncate(1);
    let error = value_schema_spec_to_json_schema(&ValueSchemaSpec::OneOf(one_of)).unwrap_err();
    assert_eq!(
        error.violations,
        vec!["schema.oneOf must be an array of at least two schemas"]
    );
}

#[test]
fn author_boundary_rejects_dsl_violations() {
    let unknown = ParameterSchemaSpec::from_author_value(
        &json!({ "x": { "type": "string", "minLength": 1 } }),
        "parameters",
    )
    .unwrap_err();
    assert_eq!(
        unknown.violations,
        vec!["parameters.x.minLength is not supported by the value schema DSL"]
    );

    let openness =
        ValueSchemaSpec::from_author_value(&json!({ "type": "object" }), "schema").unwrap_err();
    assert_eq!(
        openness.violations,
        vec!["schema.additionalProperties must be explicitly true or false"]
    );

    let bad_type =
        ValueSchemaSpec::from_author_value(&json!({ "type": "weird" }), "schema").unwrap_err();
    assert_eq!(
        bad_type.violations,
        vec![
            "schema.type must be string/number/integer/boolean/null/array/object/json, or use oneOf"
        ]
    );

    let bad_required = ParameterSchemaSpec::from_author_value(
        &json!({ "x": { "type": "string", "required": false } }),
        "parameters",
    )
    .unwrap_err();
    assert_eq!(
        bad_required.violations,
        vec!["parameters.x.required must be true when present"]
    );

    let bad_enum = ParameterSchemaSpec::from_author_value(
        &json!({ "n": { "type": "number", "enum": ["1", "2"] } }),
        "parameters",
    )
    .unwrap_err();
    assert_eq!(
        bad_enum.violations,
        vec!["parameters.n.enum must be a non-empty array of number values"]
    );
}

#[test]
fn validate_args_is_total_and_path_qualified() {
    let spec = parameters(json!({
        "path": { "type": "string", "required": true },
        "limit": { "type": "number" },
    }));
    assert_eq!(
        validate_args(&spec, &json!({ "path": "/tmp" })).unwrap(),
        Vec::<String>::new()
    );
    assert_eq!(
        validate_args(&spec, &json!({ "path": "/tmp", "limit": 5 })).unwrap(),
        Vec::<String>::new()
    );
    // Total over malformed argument roots.
    assert_eq!(validate_args(&spec, &json!(null)).unwrap().len(), 1);
    assert_eq!(validate_args(&spec, &json!("nope")).unwrap().len(), 1);
    assert_eq!(validate_args(&spec, &json!([])).unwrap().len(), 1);
    // Missing required, extra keys allowed, defaults not applied.
    assert_eq!(
        validate_args(&spec, &json!({})).unwrap(),
        vec!["missing required property \"path\""]
    );
    assert_eq!(
        validate_args(&spec, &json!({ "path": "/tmp", "extra": 1 })).unwrap(),
        Vec::<String>::new()
    );
}

#[test]
fn validate_args_recurses_into_containers() {
    let spec = parameters(json!({
        "config": {
            "type": "object",
            "additionalProperties": true,
            "required": true,
            "properties": { "host": { "type": "string", "required": true }, "port": { "type": "number" } },
        },
        "servers": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": true,
                "properties": { "host": { "type": "string", "required": true } },
            },
        },
    }));
    assert_eq!(
        validate_args(
            &spec,
            &json!({ "config": { "host": "h" }, "servers": [{ "host": "a" }] })
        )
        .unwrap(),
        Vec::<String>::new()
    );
    assert_eq!(
        validate_args(
            &spec,
            &json!({ "config": { "port": 9 }, "servers": [{ "host": "a" }, {}] })
        )
        .unwrap(),
        vec![
            "missing required property \"config.host\"",
            "missing required property \"servers[1].host\"",
        ]
    );
}

#[test]
fn validate_args_checks_enum_membership() {
    let spec = parameters(json!({ "color": { "type": "string", "enum": ["red", "green"] } }));
    assert_eq!(
        validate_args(&spec, &json!({ "color": "red" })).unwrap(),
        Vec::<String>::new()
    );
    assert_eq!(
        validate_args(&spec, &json!({ "color": "blue" })).unwrap(),
        vec!["\"color\" must be one of [\"red\",\"green\"]"]
    );
}

#[test]
fn tool_args_error_carries_code_and_violations() {
    let error = ToolArgsError {
        violations: vec![
            "missing required property \"a\"".into(),
            "\"b\" must be a number".into(),
        ],
    };
    assert_eq!(ToolArgsError::CODE, "INVALID_ARGS");
    assert_eq!(
        error.to_string(),
        "invalid arguments: missing required property \"a\"; \"b\" must be a number"
    );
}

fn echo_options() -> DefineToolOptions {
    DefineToolOptions {
        name: "typed-echo".into(),
        description: "A typed echo tool".into(),
        parameters: parameters(json!({
            "text": { "type": "string", "required": true },
            "uppercase": { "type": "boolean" },
        })),
        output: DefineToolOutput {
            schema: value_spec(json!({ "type": "string" })),
            render: Rc::new(|_args, value| {
                Ok(vec![ContentBlock::Text {
                    text: value.as_str().unwrap_or_default().into(),
                }])
            }),
            presentation_meta: None,
        },
        timeout_ms: None,
        is_concurrency_safe: None,
        execute: Rc::new(|args, _exec| {
            async move {
                let text = args["text"].as_str().unwrap_or_default();
                let uppercase = args["uppercase"].as_bool().unwrap_or(false);
                Ok(json!(if uppercase {
                    text.to_uppercase()
                } else {
                    text.to_string()
                }))
            }
            .boxed_local()
        }),
        finalize_content: None,
        present_call: None,
        present_result: None,
    }
}

#[test]
fn define_tool_compiles_the_model_facing_schema() {
    let tool = define_tool(echo_options()).unwrap();
    assert_eq!(tool.name, "typed-echo");
    assert_eq!(
        serde_json::Value::Object(tool.parameters.clone()),
        json!({
            "type": "object",
            "properties": { "text": { "type": "string" }, "uppercase": { "type": "boolean" } },
            "required": ["text"],
        })
    );
    assert_eq!(tool.output.schema, json!({ "type": "string" }));
    assert!(tool.timeout_ms.is_none());
}

#[test]
fn define_tool_validates_timeout() {
    for bad in [0.0, -5.0, f64::INFINITY] {
        let error = define_tool(DefineToolOptions {
            timeout_ms: Some(bad),
            ..echo_options()
        })
        .err()
        .unwrap();
        assert!(
            error
                .to_string()
                .contains("timeoutMs must be a positive finite number")
        );
    }
    let tool = define_tool(DefineToolOptions {
        timeout_ms: Some(30_000.0),
        ..echo_options()
    })
    .unwrap();
    assert_eq!(tool.timeout_ms, Some(30_000.0));
}

#[test]
fn define_tool_soft_validates_presenters_and_classifier() {
    let mut options = echo_options();
    options.present_call = Some(Rc::new(|args| {
        Some(dsh_tools::ToolCallView::Generic(
            dsh_tools::GenericCallView {
                title: args["text"].as_str().unwrap_or_default().into(),
                kind: None,
                raw_input: None,
                content: None,
                locations: None,
            },
        ))
    }));
    options.is_concurrency_safe = Some(Rc::new(|_args| true));
    let tool = define_tool(options).unwrap();

    // Valid args reach the user presenter/classifier.
    let view = (tool.present_call.as_ref().unwrap())(&json!({ "text": "hi" }));
    assert!(matches!(view, Some(dsh_tools::ToolCallView::Generic(v)) if v.title == "hi"));
    assert!((tool.is_concurrency_safe.as_ref().unwrap())(
        &json!({ "text": "hi" })
    ));

    // Malformed args soft-fail: presenters fall back, classifiers fail closed.
    assert!((tool.present_call.as_ref().unwrap())(&json!({})).is_none());
    assert!(!(tool.is_concurrency_safe.as_ref().unwrap())(&json!({})));
}

#[test]
fn serde_round_trips_the_author_syntax() {
    let author = json!({
        "oneOf": [
            { "type": "string", "enum": ["a"] },
            { "type": "object", "additionalProperties": false, "properties": { "n": { "type": "integer", "required": true } } },
        ],
        "description": "either",
    });
    let spec: ValueSchemaSpec = serde_json::from_value(author.clone()).unwrap();
    assert_eq!(serde_json::to_value(&spec).unwrap(), author);
}
