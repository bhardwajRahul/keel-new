//! Behavior tests for the enforced JSON Schema subset, mirroring the
//! upstream json-schema suite: subset acceptance, rejection wording, and
//! path-qualified value validation.

use dsh_tools::{
    assert_object_json_schema, assert_supported_json_schema, validate_json_schema_value,
};
use serde_json::{Value, json};

fn violations_of(schema: &Value) -> Vec<String> {
    match assert_supported_json_schema(schema) {
        Ok(()) => Vec::new(),
        Err(error) => error.violations,
    }
}

#[test]
fn accepts_the_documented_subset() {
    let accepted = [
        json!({}),
        json!({ "description": "anything", "title": "t" }),
        json!({ "type": "string" }),
        json!({ "type": "string", "enum": ["a", "b"] }),
        json!({ "type": "string", "enum": ["a", "b"], "const": "a" }),
        json!({ "type": "integer", "const": 3 }),
        json!({ "type": "object", "properties": { "a": { "type": "null" } }, "required": ["a"], "additionalProperties": false }),
        json!({ "type": "array", "items": { "type": "number" } }),
        json!({ "oneOf": [{ "type": "string" }, { "type": "null" }] }),
        json!({ "type": "number", "default": 5, "examples": [1, 2] }),
    ];
    for schema in accepted {
        assert_eq!(
            violations_of(&schema),
            Vec::<String>::new(),
            "schema: {schema}"
        );
    }
}

#[test]
fn rejects_non_object_schema_roots() {
    for schema in [json!(null), json!([]), json!("no")] {
        assert_eq!(
            violations_of(&schema),
            vec!["schema must be a schema object"]
        );
    }
}

#[test]
fn rejects_type_arrays_and_unknown_types() {
    assert_eq!(
        violations_of(&json!({ "type": ["string", "null"] })),
        vec!["schema.type must be a single type string (type arrays are not supported)"]
    );
    assert_eq!(
        violations_of(&json!({ "type": "weird" })),
        vec!["schema.type must be one of object/array/string/number/integer/boolean/null"]
    );
}

#[test]
fn rejects_unsupported_keywords() {
    assert_eq!(
        violations_of(&json!({ "type": "string", "minLength": 3 })),
        vec![
            "schema.minLength is not a supported keyword (subset: type/oneOf/properties/required/additionalProperties/items/enum/const + annotations)"
        ]
    );
}

#[test]
fn enforces_one_of_shape_and_exclusivity() {
    for bad in [
        json!({ "oneOf": [] }),
        json!({ "oneOf": [{}] }),
        json!({ "oneOf": "x" }),
    ] {
        assert_eq!(
            violations_of(&bad),
            vec!["schema.oneOf must be an array of at least two schemas"]
        );
    }
    assert_eq!(
        violations_of(&json!({ "type": "string", "oneOf": [{}, {}] })),
        vec!["schema cannot declare both type and oneOf"]
    );
    assert_eq!(
        violations_of(&json!({ "oneOf": [{ "type": "string" }, { "type": "null" }], "items": {} })),
        vec!["schema.items is not supported beside oneOf"]
    );
    // Branches are walked with indexed paths.
    assert_eq!(
        violations_of(&json!({ "oneOf": [{ "type": "string" }, { "type": "weird" }] })),
        vec!["schema.oneOf[1].type must be one of object/array/string/number/integer/boolean/null"]
    );
}

#[test]
fn rejects_misplaced_keywords_per_type() {
    assert_eq!(
        violations_of(&json!({ "type": "object", "items": {} })),
        vec!["schema.items is not supported on type \"object\""]
    );
    assert_eq!(
        violations_of(&json!({ "type": "array", "properties": {} })),
        vec!["schema.properties is not supported on type \"array\""]
    );
    assert_eq!(
        violations_of(&json!({ "type": "object", "enum": ["a"] })),
        vec!["schema.enum is not supported on type \"object\""]
    );
    assert_eq!(
        violations_of(&json!({ "type": "array", "const": "a" })),
        vec!["schema.const is not supported on type \"array\""]
    );
    assert_eq!(
        violations_of(&json!({ "properties": {} })),
        vec!["schema.properties requires type or oneOf"]
    );
}

#[test]
fn validates_object_schema_structure() {
    assert_eq!(
        violations_of(&json!({ "type": "object", "properties": [] })),
        vec!["schema.properties must be an object of schemas"]
    );
    assert_eq!(
        violations_of(&json!({ "type": "object", "properties": { "a": 1 } })),
        vec!["schema.properties.a must be a schema object"]
    );
    assert_eq!(
        violations_of(&json!({ "type": "object", "required": "a" })),
        vec!["schema.required must be an array of strings"]
    );
    assert_eq!(
        violations_of(&json!({ "type": "object", "required": [1] })),
        vec!["schema.required must be an array of strings"]
    );
    assert_eq!(
        violations_of(&json!({ "type": "object", "properties": {}, "required": ["missing"] })),
        vec!["schema.required names \"missing\" which is not in properties"]
    );
    assert_eq!(
        violations_of(&json!({ "type": "object", "additionalProperties": "no" })),
        vec!["schema.additionalProperties must be a boolean"]
    );
}

#[test]
fn enforces_type_correct_literal_constraints() {
    assert_eq!(
        violations_of(&json!({ "type": "string", "enum": [1] })),
        vec!["schema.enum must be a non-empty array of string values"]
    );
    assert_eq!(
        violations_of(&json!({ "type": "string", "enum": [] })),
        vec!["schema.enum must be a non-empty array of string values"]
    );
    assert_eq!(
        violations_of(&json!({ "type": "number", "enum": ["1"] })),
        vec!["schema.enum must be a non-empty array of number values"]
    );
    assert_eq!(
        violations_of(&json!({ "type": "integer", "enum": [1.5] })),
        vec!["schema.enum must be a non-empty array of integer values"]
    );
    assert_eq!(
        violations_of(&json!({ "type": "number", "const": "x" })),
        vec!["schema.const must be a number value"]
    );
    assert_eq!(
        violations_of(&json!({ "type": "boolean", "const": 1 })),
        vec!["schema.const must be a boolean value"]
    );
    assert_eq!(
        violations_of(&json!({ "type": "string", "enum": ["a"], "const": "b" })),
        vec!["schema.const must be one of schema.enum when both are declared"]
    );
}

#[test]
fn requires_string_annotations() {
    assert_eq!(
        violations_of(&json!({ "description": 1 })),
        vec!["schema.description must be a string"]
    );
    assert_eq!(
        violations_of(&json!({ "title": 1 })),
        vec!["schema.title must be a string"]
    );
}

#[test]
fn object_assertion_adds_the_root_constraint() {
    assert!(assert_object_json_schema(&json!({ "type": "object" })).is_ok());
    let error = assert_object_json_schema(&json!({ "type": "string" })).unwrap_err();
    assert_eq!(
        error.violations,
        vec!["schema.type must be \"object\" (structured output is object-rooted)"]
    );
    // Subset violations surface first, without the root nag.
    let error = assert_object_json_schema(&json!({ "type": "weird" })).unwrap_err();
    assert_eq!(
        error.violations,
        vec!["schema.type must be one of object/array/string/number/integer/boolean/null"]
    );
}

#[test]
fn json_schema_error_renders_message_and_code() {
    let error = assert_supported_json_schema(&json!("no")).unwrap_err();
    assert_eq!(
        error.to_string(),
        "unsupported JSON schema: schema must be a schema object"
    );
    assert_eq!(dsh_tools::JsonSchemaError::CODE, "UNSUPPORTED_SCHEMA");
}

// --- value validation ---

#[test]
fn validates_primitive_types_with_paths() {
    let object = json!({
        "type": "object",
        "properties": { "s": { "type": "string" }, "n": { "type": "number" }, "b": { "type": "boolean" }, "z": { "type": "null" }, "i": { "type": "integer" } }
    });
    assert_eq!(
        validate_json_schema_value(&object, &json!({ "s": 1 }), ""),
        vec!["\"s\" must be a string"]
    );
    assert_eq!(
        validate_json_schema_value(&object, &json!({ "n": "x" }), ""),
        vec!["\"n\" must be a number"]
    );
    assert_eq!(
        validate_json_schema_value(&object, &json!({ "b": "x" }), ""),
        vec!["\"b\" must be a boolean"]
    );
    assert_eq!(
        validate_json_schema_value(&object, &json!({ "z": 0 }), ""),
        vec!["\"z\" must be null"]
    );
    assert_eq!(
        validate_json_schema_value(&object, &json!({ "i": 1.5 }), ""),
        vec!["\"i\" must be an integer"]
    );
    assert_eq!(
        validate_json_schema_value(
            &object,
            &json!({ "s": "ok", "n": 1, "b": true, "z": null, "i": 2 }),
            ""
        ),
        Vec::<String>::new()
    );
}

#[test]
fn reports_missing_required_and_nested_paths() {
    let schema = json!({
        "type": "object",
        "properties": {
            "config": {
                "type": "object",
                "properties": { "host": { "type": "string" }, "port": { "type": "number" } },
                "required": ["host"]
            },
            "bag": { "type": "object" }
        },
        "required": ["config"]
    });
    assert_eq!(
        validate_json_schema_value(&schema, &json!({}), ""),
        vec!["missing required property \"config\""]
    );
    // A dependency can enable serde_json's preserve_order feature for the
    // workspace. Error order is presentation only; check the violations.
    let mut actual =
        validate_json_schema_value(&schema, &json!({ "config": { "port": 9 }, "bag": 5 }), "");
    actual.sort();
    assert_eq!(
        actual,
        vec![
            "\"bag\" must be an object",
            "missing required property \"config.host\""
        ]
    );
}

#[test]
fn validates_arrays_element_wise() {
    let schema = json!({
        "type": "object",
        "properties": { "tags": { "type": "array", "items": { "type": "string" } }, "raw": { "type": "array" } }
    });
    assert_eq!(
        validate_json_schema_value(
            &schema,
            &json!({ "tags": ["a", "b"], "raw": [1, {}, "x"] }),
            ""
        ),
        Vec::<String>::new()
    );
    assert_eq!(
        validate_json_schema_value(&schema, &json!({ "tags": ["a", 2] }), ""),
        vec!["\"tags[1]\" must be a string"]
    );
    assert_eq!(
        validate_json_schema_value(&schema, &json!({ "tags": "nope" }), ""),
        vec!["\"tags\" must be an array"]
    );
}

#[test]
fn enforces_closed_objects_and_literals() {
    let closed = json!({
        "type": "object",
        "properties": { "ok": { "type": "boolean" } },
        "additionalProperties": false
    });
    assert_eq!(
        validate_json_schema_value(&closed, &json!({ "ok": true, "extra": 1 }), "value"),
        vec!["\"value.extra\" is not a declared property (additionalProperties: false)"]
    );
    let colored = json!({ "type": "string", "enum": ["red", "green"] });
    assert_eq!(
        validate_json_schema_value(&colored, &json!("blue"), "value"),
        vec!["\"value\" must be one of [\"red\",\"green\"]"]
    );
    let pinned = json!({ "type": "integer", "const": 1 });
    assert_eq!(
        validate_json_schema_value(&pinned, &json!(2), "value"),
        vec!["\"value\" must be 1"]
    );
}

#[test]
fn one_of_requires_exactly_one_match() {
    let schema = json!({ "oneOf": [{ "type": "string" }, { "type": "null" }] });
    assert_eq!(
        validate_json_schema_value(&schema, &json!("x"), "value"),
        Vec::<String>::new()
    );
    assert_eq!(
        validate_json_schema_value(&schema, &json!(1), "value"),
        vec!["\"value\" must match exactly one oneOf branch (matched 0)"]
    );
    let ambiguous = json!({ "oneOf": [{ "type": "string" }, {}] });
    assert_eq!(
        validate_json_schema_value(&ambiguous, &json!("x"), "value"),
        vec!["\"value\" must match exactly one oneOf branch (matched 2)"]
    );
}

#[test]
fn empty_root_path_reads_as_arguments() {
    let schema = json!({ "type": "object", "properties": {} });
    assert_eq!(
        validate_json_schema_value(&schema, &json!(null), ""),
        vec!["\"arguments\" must be an object"]
    );
}
