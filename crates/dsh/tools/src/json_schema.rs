//! Enforced JSON Schema subset (port of upstream `src/json-schema.ts`):
//! any-JSON annotation-only nodes, one scalar `type`, object
//! `properties`/`required`/boolean `additionalProperties`, array `items`,
//! type-correct scalar `enum`/`const`, and exact-one `oneOf`. Anything else
//! rejects instead of passing unenforced.
//!
//! Divergences: schemas and values are `serde_json::Value`, so the lossless
//! and realm checks collapse; cycles are unrepresentable in a `Value` tree.
//! The walk is plain recursion — parsed JSON is already depth-bounded by
//! serde_json's recursion limit.

use serde_json::{Map, Value};

/// Single-type keywords accepted by the enforced subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonSchemaType {
    Object,
    Array,
    String,
    Number,
    Integer,
    Boolean,
    Null,
}

impl JsonSchemaType {
    /// The wire spelling of this type keyword.
    pub fn as_str(self) -> &'static str {
        match self {
            JsonSchemaType::Object => "object",
            JsonSchemaType::Array => "array",
            JsonSchemaType::String => "string",
            JsonSchemaType::Number => "number",
            JsonSchemaType::Integer => "integer",
            JsonSchemaType::Boolean => "boolean",
            JsonSchemaType::Null => "null",
        }
    }

    fn parse(value: &str) -> Option<JsonSchemaType> {
        Some(match value {
            "object" => JsonSchemaType::Object,
            "array" => JsonSchemaType::Array,
            "string" => JsonSchemaType::String,
            "number" => JsonSchemaType::Number,
            "integer" => JsonSchemaType::Integer,
            "boolean" => JsonSchemaType::Boolean,
            "null" => JsonSchemaType::Null,
            _ => return None,
        })
    }
}

/// A raw schema fell outside the enforced subset; `violations` lists every
/// offending path in walk order.
#[derive(Debug, Clone, thiserror::Error)]
#[error("unsupported JSON schema: {}", .violations.join("; "))]
pub struct JsonSchemaError {
    /// Individual schema violations in walk order.
    pub violations: Vec<String>,
}

impl JsonSchemaError {
    /// Stable machine-routable failure class.
    pub const CODE: &'static str = "UNSUPPORTED_SCHEMA";
}

const CONSTRAINT_KEYWORDS: [&str; 8] = [
    "type",
    "oneOf",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "const",
];
const ANNOTATION_KEYWORDS: [&str; 4] = ["description", "title", "default", "examples"];
const ONE_OF_SIBLING_KEYWORDS: [&str; 6] = [
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "const",
];

/// Lossless JSON number: serde already excludes NaN/∞; negative zero remains.
fn is_json_number(value: &Value) -> bool {
    match value.as_f64() {
        Some(f) => !(f == 0.0 && f.is_sign_negative()),
        None => false,
    }
}

fn is_integer(value: &Value) -> bool {
    match value.as_f64() {
        Some(f) => is_json_number(value) && f.fract() == 0.0,
        None => false,
    }
}

/// Whether a scalar value is valid for one declared scalar schema type.
fn scalar_matches(schema_type: JsonSchemaType, value: &Value) -> bool {
    match schema_type {
        JsonSchemaType::String => value.is_string(),
        JsonSchemaType::Number => is_json_number(value),
        JsonSchemaType::Integer => is_integer(value),
        JsonSchemaType::Boolean => value.is_boolean(),
        JsonSchemaType::Null => value.is_null(),
        JsonSchemaType::Object | JsonSchemaType::Array => false,
    }
}

/// Object-only tail checks, run after the property schemas were visited.
fn check_object_schema_tail(node: &Map<String, Value>, path: &str, violations: &mut Vec<String>) {
    if let Some(required) = node.get("required") {
        let strings: Option<Vec<&str>> = required
            .as_array()
            .and_then(|entries| entries.iter().map(Value::as_str).collect());
        match strings {
            None => violations.push(format!("{path}.required must be an array of strings")),
            Some(names) => {
                let empty = Map::new();
                let declared = node
                    .get("properties")
                    .and_then(Value::as_object)
                    .unwrap_or(&empty);
                for name in names {
                    if !declared.contains_key(name) {
                        violations.push(format!(
                            "{path}.required names \"{name}\" which is not in properties"
                        ));
                    }
                }
            }
        }
    }
    if let Some(additional) = node.get("additionalProperties") {
        if !additional.is_boolean() {
            violations.push(format!("{path}.additionalProperties must be a boolean"));
        }
    }
}

/// Collect every violation for one raw schema tree.
fn check_schema_node(node: &Value, path: &str, violations: &mut Vec<String>) {
    let Some(map) = node.as_object() else {
        violations.push(format!("{path} must be a schema object"));
        return;
    };

    for key in map.keys() {
        if CONSTRAINT_KEYWORDS.contains(&key.as_str())
            || ANNOTATION_KEYWORDS.contains(&key.as_str())
        {
            continue;
        }
        violations.push(format!(
            "{path}.{key} is not a supported keyword (subset: type/oneOf/properties/required/additionalProperties/items/enum/const + annotations)"
        ));
    }
    if let Some(description) = map.get("description") {
        if !description.is_string() {
            violations.push(format!("{path}.description must be a string"));
        }
    }
    if let Some(title) = map.get("title") {
        if !title.is_string() {
            violations.push(format!("{path}.title must be a string"));
        }
    }

    let has_type = map.contains_key("type");
    let has_one_of = map.contains_key("oneOf");
    if has_type && has_one_of {
        violations.push(format!("{path} cannot declare both type and oneOf"));
        return;
    }
    if !has_type && !has_one_of {
        for key in ONE_OF_SIBLING_KEYWORDS {
            if map.contains_key(key) {
                violations.push(format!("{path}.{key} requires type or oneOf"));
            }
        }
        return;
    }

    if has_one_of {
        match map.get("oneOf").and_then(Value::as_array) {
            Some(branches) if branches.len() >= 2 => {
                for (index, branch) in branches.iter().enumerate() {
                    check_schema_node(branch, &format!("{path}.oneOf[{index}]"), violations);
                }
            }
            _ => violations.push(format!(
                "{path}.oneOf must be an array of at least two schemas"
            )),
        }
        for key in ONE_OF_SIBLING_KEYWORDS {
            if map.contains_key(key) {
                violations.push(format!("{path}.{key} is not supported beside oneOf"));
            }
        }
        return;
    }

    let type_value = map.get("type").expect("has_type checked");
    let schema_type = type_value.as_str().and_then(JsonSchemaType::parse);
    let Some(schema_type) = schema_type else {
        violations.push(if type_value.is_array() {
            format!("{path}.type must be a single type string (type arrays are not supported)")
        } else {
            format!("{path}.type must be one of object/array/string/number/integer/boolean/null")
        });
        return;
    };

    let allowed_for: [(&str, &[JsonSchemaType]); 6] = [
        ("properties", &[JsonSchemaType::Object]),
        ("required", &[JsonSchemaType::Object]),
        ("additionalProperties", &[JsonSchemaType::Object]),
        ("items", &[JsonSchemaType::Array]),
        (
            "enum",
            &[
                JsonSchemaType::String,
                JsonSchemaType::Number,
                JsonSchemaType::Integer,
                JsonSchemaType::Boolean,
                JsonSchemaType::Null,
            ],
        ),
        (
            "const",
            &[
                JsonSchemaType::String,
                JsonSchemaType::Number,
                JsonSchemaType::Integer,
                JsonSchemaType::Boolean,
                JsonSchemaType::Null,
            ],
        ),
    ];
    for (key, types) in allowed_for {
        if map.contains_key(key) && !types.contains(&schema_type) {
            violations.push(format!(
                "{path}.{key} is not supported on type \"{}\"",
                schema_type.as_str()
            ));
        }
    }

    match schema_type {
        JsonSchemaType::Object => {
            if let Some(properties) = map.get("properties") {
                match properties.as_object() {
                    None => {
                        violations.push(format!("{path}.properties must be an object of schemas"))
                    }
                    Some(entries) => {
                        for (name, child) in entries {
                            check_schema_node(
                                child,
                                &format!("{path}.properties.{name}"),
                                violations,
                            );
                        }
                    }
                }
            }
            check_object_schema_tail(map, path, violations);
        }
        JsonSchemaType::Array => {
            if let Some(items) = map.get("items") {
                check_schema_node(items, &format!("{path}.items"), violations);
            }
        }
        scalar => {
            let type_name = scalar.as_str();
            let allowed = map.get("enum");
            let enum_valid = match allowed.and_then(Value::as_array) {
                Some(entries) => {
                    !entries.is_empty() && entries.iter().all(|entry| scalar_matches(scalar, entry))
                }
                None => false,
            };
            if allowed.is_some() && !enum_valid {
                violations.push(format!(
                    "{path}.enum must be a non-empty array of {type_name} values"
                ));
            }
            if let Some(declared) = map.get("const") {
                if !scalar_matches(scalar, declared) {
                    violations.push(format!("{path}.const must be a {type_name} value"));
                } else if enum_valid {
                    let entries = allowed.and_then(Value::as_array).expect("enum_valid");
                    if !entries.contains(declared) {
                        violations.push(format!(
                            "{path}.const must be one of {path}.enum when both are declared"
                        ));
                    }
                }
            }
        }
    }
}

/// Reject a raw schema outside the enforced subset. Annotation-only schemas
/// pass as the standard unconstrained-JSON form.
pub fn assert_supported_json_schema(schema: &Value) -> Result<(), JsonSchemaError> {
    let mut violations = Vec::new();
    check_schema_node(schema, "schema", &mut violations);
    if violations.is_empty() {
        Ok(())
    } else {
        Err(JsonSchemaError { violations })
    }
}

/// The enforced subset plus the object-root constraint kept by structured
/// outputs.
pub fn assert_object_json_schema(schema: &Value) -> Result<(), JsonSchemaError> {
    let mut violations = Vec::new();
    check_schema_node(schema, "schema", &mut violations);
    if violations.is_empty() && schema.get("type").and_then(Value::as_str) != Some("object") {
        violations
            .push("schema.type must be \"object\" (structured output is object-rooted)".into());
    }
    if violations.is_empty() {
        Ok(())
    } else {
        Err(JsonSchemaError { violations })
    }
}

/// Root-aware diagnostic label: the empty parameter-root path reads as
/// `arguments`.
fn diagnostic_path(path: &str) -> &str {
    if path.is_empty() { "arguments" } else { path }
}

/// One property step, without a leading dot at the implicit root.
fn property_path(path: &str, key: &str) -> String {
    if path.is_empty() {
        key.to_string()
    } else {
        format!("{path}.{key}")
    }
}

fn scalar_constraint_violations(
    node: &Map<String, Value>,
    value: &Value,
    path: &str,
) -> Vec<String> {
    if let Some(allowed) = node.get("enum").and_then(Value::as_array) {
        if !allowed.contains(value) {
            return vec![format!(
                "\"{}\" must be one of {}",
                diagnostic_path(path),
                serde_json::to_string(allowed).expect("enum is JSON")
            )];
        }
    }
    if let Some(declared) = node.get("const") {
        if value != declared {
            return vec![format!(
                "\"{}\" must be {}",
                diagnostic_path(path),
                serde_json::to_string(declared).expect("const is JSON")
            )];
        }
    }
    Vec::new()
}

fn check_value(schema: &Value, value: &Value, path: &str, out: &mut Vec<String>) {
    // Total over untrusted values; the schema itself was asserted. A
    // non-object schema node validates nothing rather than panicking.
    let Some(node) = schema.as_object() else {
        return;
    };

    if let Some(branches) = node.get("oneOf").and_then(Value::as_array) {
        let mut matches = 0;
        for branch in branches {
            let mut branch_violations = Vec::new();
            check_value(branch, value, path, &mut branch_violations);
            if branch_violations.is_empty() {
                matches += 1;
            }
        }
        if matches != 1 {
            out.push(format!(
                "\"{}\" must match exactly one oneOf branch (matched {matches})",
                diagnostic_path(path)
            ));
        }
        return;
    }

    let Some(schema_type) = node
        .get("type")
        .and_then(Value::as_str)
        .and_then(JsonSchemaType::parse)
    else {
        // Annotation-only node: any JSON value (losslessness is structural).
        return;
    };

    match schema_type {
        JsonSchemaType::Object => {
            let Some(record) = value.as_object() else {
                out.push(format!("\"{}\" must be an object", diagnostic_path(path)));
                return;
            };
            let empty = Map::new();
            let properties = node
                .get("properties")
                .and_then(Value::as_object)
                .unwrap_or(&empty);
            if let Some(required) = node.get("required").and_then(Value::as_array) {
                for key in required.iter().filter_map(Value::as_str) {
                    if !record.contains_key(key) {
                        out.push(format!(
                            "missing required property \"{}\"",
                            property_path(path, key)
                        ));
                    }
                }
            }
            for (key, child_schema) in properties {
                if let Some(child_value) = record.get(key) {
                    check_value(child_schema, child_value, &property_path(path, key), out);
                }
            }
            if node.get("additionalProperties") == Some(&Value::Bool(false)) {
                for key in record.keys() {
                    if !properties.contains_key(key) {
                        out.push(format!(
                            "\"{}\" is not a declared property (additionalProperties: false)",
                            property_path(path, key)
                        ));
                    }
                }
            }
        }
        JsonSchemaType::Array => {
            let Some(items) = value.as_array() else {
                out.push(format!("\"{}\" must be an array", diagnostic_path(path)));
                return;
            };
            if let Some(item_schema) = node.get("items") {
                for (index, entry) in items.iter().enumerate() {
                    check_value(item_schema, entry, &format!("{path}[{index}]"), out);
                }
            }
        }
        JsonSchemaType::String => {
            if value.is_string() {
                out.extend(scalar_constraint_violations(node, value, path));
            } else {
                out.push(format!("\"{}\" must be a string", diagnostic_path(path)));
            }
        }
        JsonSchemaType::Number => {
            if !value.is_number() {
                out.push(format!("\"{}\" must be a number", diagnostic_path(path)));
            } else if !is_json_number(value) {
                out.push(format!(
                    "\"{}\" must be a finite JSON number",
                    diagnostic_path(path)
                ));
            } else {
                out.extend(scalar_constraint_violations(node, value, path));
            }
        }
        JsonSchemaType::Integer => {
            if is_integer(value) {
                out.extend(scalar_constraint_violations(node, value, path));
            } else {
                out.push(format!("\"{}\" must be an integer", diagnostic_path(path)));
            }
        }
        JsonSchemaType::Boolean => {
            if value.is_boolean() {
                out.extend(scalar_constraint_violations(node, value, path));
            } else {
                out.push(format!("\"{}\" must be a boolean", diagnostic_path(path)));
            }
        }
        JsonSchemaType::Null => {
            if value.is_null() {
                out.extend(scalar_constraint_violations(node, value, path));
            } else {
                out.push(format!("\"{}\" must be null", diagnostic_path(path)));
            }
        }
    }
}

/// Validate a candidate value against an asserted schema. Total for
/// arbitrary values; returns path-qualified violations in walk order (empty
/// means valid). `path` labels the root in diagnostics (`""` reads as
/// `arguments`).
pub fn validate_json_schema_value(schema: &Value, value: &Value, path: &str) -> Vec<String> {
    let mut violations = Vec::new();
    check_value(schema, value, path, &mut violations);
    violations
}
