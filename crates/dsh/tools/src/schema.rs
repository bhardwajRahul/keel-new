//! Author-facing value-schema DSL compiled to the enforced JSON Schema
//! subset, plus argument validation and the typed [`define_tool`] helper
//! (port of upstream `src/schema.ts`).
//!
//! Divergences: the specs are Rust types, so upstream's runtime author
//! checks are mostly unrepresentable states; parsing untyped author input
//! (config files, tests) goes through [`ValueSchemaSpec::from_author_value`],
//! which keeps those checks. Compile-time argument/value inference does not
//! exist in Rust — [`define_tool`] closures receive validated
//! `serde_json::Value`s. Null-literal `enum`/`const` are tautological and
//! not carried.

use crate::json_schema::{
    JsonSchemaError, assert_supported_json_schema, validate_json_schema_value,
};
use crate::presentation::{ToolCallView, ToolResultView};
use crate::runtime::{
    ToolDefinition, ToolExecuteFn, ToolFinalizeFn, ToolMetaFn, ToolOutputDefinition, ToolRenderFn,
    ToolResult,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value, json};
use std::rc::Rc;

/// Annotation keywords shared by every author-facing schema node.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ValueSchemaAnnotations {
    /// Human-readable description projected into the compiled schema.
    pub description: Option<String>,
    /// Human-readable title projected into the compiled schema.
    pub title: Option<String>,
    /// Non-validating default annotation.
    pub default: Option<Value>,
    /// Non-validating examples annotation.
    pub examples: Option<Value>,
}

/// String value schema with type-correct literal constraints.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StringValueSchemaSpec {
    pub annotations: ValueSchemaAnnotations,
    pub enum_values: Option<Vec<String>>,
    pub const_value: Option<String>,
}

/// Finite JSON-number schema with type-correct literal constraints.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NumberValueSchemaSpec {
    pub annotations: ValueSchemaAnnotations,
    pub enum_values: Option<Vec<f64>>,
    pub const_value: Option<f64>,
}

/// Integer schema with type-correct literal constraints.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IntegerValueSchemaSpec {
    pub annotations: ValueSchemaAnnotations,
    pub enum_values: Option<Vec<i64>>,
    pub const_value: Option<i64>,
}

/// Boolean value schema with type-correct literal constraints.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BooleanValueSchemaSpec {
    pub annotations: ValueSchemaAnnotations,
    pub enum_values: Option<Vec<bool>>,
    pub const_value: Option<bool>,
}

/// Null value schema (literal constraints on `null` are tautological and
/// deliberately not carried).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NullValueSchemaSpec {
    pub annotations: ValueSchemaAnnotations,
}

/// Array value schema; omitted `items` accepts any JSON item.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ArrayValueSchemaSpec {
    pub annotations: ValueSchemaAnnotations,
    pub items: Option<Box<ValueSchemaSpec>>,
}

/// Explicit object value schema. Openness is a mandatory field so a nested
/// or output object never acquires an accidental JSON Schema default.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectValueSchemaSpec {
    pub annotations: ValueSchemaAnnotations,
    pub properties: Option<ParameterSchemaSpec>,
    pub additional_properties: bool,
}

/// Author-only unconstrained lossless JSON node.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct JsonValueSchemaSpec {
    pub annotations: ValueSchemaAnnotations,
}

/// Exact-one union schema; at least two branches are required (enforced by
/// the compiled-schema assertion).
#[derive(Debug, Clone, PartialEq)]
pub struct OneOfValueSchemaSpec {
    pub annotations: ValueSchemaAnnotations,
    pub branches: Vec<ValueSchemaSpec>,
}

/// One author-facing schema for any lossless JSON value root.
#[derive(Debug, Clone, PartialEq)]
pub enum ValueSchemaSpec {
    String(StringValueSchemaSpec),
    Number(NumberValueSchemaSpec),
    Integer(IntegerValueSchemaSpec),
    Boolean(BooleanValueSchemaSpec),
    Null(NullValueSchemaSpec),
    Array(ArrayValueSchemaSpec),
    Object(ObjectValueSchemaSpec),
    Json(JsonValueSchemaSpec),
    OneOf(OneOfValueSchemaSpec),
}

/// One implicit parameter-root property, optionally required.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterPropertySpec {
    pub required: bool,
    pub spec: ValueSchemaSpec,
}

/// Tool parameter schema: an implicit open object root whose requiredness is
/// per-property. Declaration order is preserved for the compiled `required`
/// array.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ParameterSchemaSpec(pub Vec<(String, ParameterPropertySpec)>);

const AUTHOR_ANNOTATION_KEYS: [&str; 4] = ["description", "title", "default", "examples"];

fn author_error(message: String) -> JsonSchemaError {
    JsonSchemaError {
        violations: vec![message],
    }
}

fn parse_annotations(
    map: &Map<String, Value>,
    path: &str,
) -> Result<ValueSchemaAnnotations, JsonSchemaError> {
    let description = match map.get("description") {
        None => None,
        Some(Value::String(text)) => Some(text.clone()),
        Some(_) => return Err(author_error(format!("{path}.description must be a string"))),
    };
    let title = match map.get("title") {
        None => None,
        Some(Value::String(text)) => Some(text.clone()),
        Some(_) => return Err(author_error(format!("{path}.title must be a string"))),
    };
    Ok(ValueSchemaAnnotations {
        description,
        title,
        default: map.get("default").cloned(),
        examples: map.get("examples").cloned(),
    })
}

fn assert_author_keys(
    map: &Map<String, Value>,
    path: &str,
    allowed: &[&str],
) -> Result<(), JsonSchemaError> {
    for key in map.keys() {
        if !AUTHOR_ANNOTATION_KEYS.contains(&key.as_str()) && !allowed.contains(&key.as_str()) {
            return Err(author_error(format!(
                "{path}.{key} is not supported by the value schema DSL"
            )));
        }
    }
    Ok(())
}

fn parse_scalar_list<T>(
    map: &Map<String, Value>,
    path: &str,
    type_name: &str,
    parse: impl Fn(&Value) -> Option<T>,
) -> Result<(Option<Vec<T>>, Option<T>), JsonSchemaError> {
    let enum_values = match map.get("enum") {
        None => None,
        Some(Value::Array(entries)) if !entries.is_empty() => {
            let parsed: Option<Vec<T>> = entries.iter().map(&parse).collect();
            match parsed {
                Some(values) => Some(values),
                None => {
                    return Err(author_error(format!(
                        "{path}.enum must be a non-empty array of {type_name} values"
                    )));
                }
            }
        }
        Some(_) => {
            return Err(author_error(format!(
                "{path}.enum must be a non-empty array of {type_name} values"
            )));
        }
    };
    let const_value = match map.get("const") {
        None => None,
        Some(value) => match parse(value) {
            Some(parsed) => Some(parsed),
            None => {
                return Err(author_error(format!(
                    "{path}.const must be a {type_name} value"
                )));
            }
        },
    };
    Ok((enum_values, const_value))
}

impl ParameterSchemaSpec {
    /// Parse one untyped author property map (`{ name: spec-with-required }`).
    pub fn from_author_value(input: &Value, path: &str) -> Result<Self, JsonSchemaError> {
        let Some(map) = input.as_object() else {
            return Err(author_error(format!(
                "{path} must be an object of value schemas"
            )));
        };
        let mut properties = Vec::with_capacity(map.len());
        for (name, property) in map {
            let property_path = format!("{path}.{name}");
            let Some(record) = property.as_object() else {
                return Err(author_error(format!(
                    "{property_path} must be a value schema object"
                )));
            };
            let required = match record.get("required") {
                None => false,
                Some(Value::Bool(true)) => true,
                Some(_) => {
                    return Err(author_error(format!(
                        "{property_path}.required must be true when present"
                    )));
                }
            };
            let mut bare = record.clone();
            bare.remove("required");
            let spec = ValueSchemaSpec::from_author_value(&Value::Object(bare), &property_path)?;
            properties.push((name.clone(), ParameterPropertySpec { required, spec }));
        }
        Ok(ParameterSchemaSpec(properties))
    }

    fn to_author_value(&self) -> Value {
        let mut map = Map::new();
        for (name, property) in &self.0 {
            let mut node = property.spec.to_author_value();
            if property.required {
                node.as_object_mut()
                    .expect("author form is an object")
                    .insert("required".into(), Value::Bool(true));
            }
            map.insert(name.clone(), node);
        }
        Value::Object(map)
    }
}

impl ValueSchemaSpec {
    /// Parse one untyped author node, with the upstream author-boundary
    /// checks (unknown keys, mandatory object openness, unsupported types).
    pub fn from_author_value(input: &Value, path: &str) -> Result<Self, JsonSchemaError> {
        let Some(map) = input.as_object() else {
            return Err(author_error(format!(
                "{path} must be a value schema object"
            )));
        };
        let annotations = parse_annotations(map, path)?;

        if map.contains_key("oneOf") {
            assert_author_keys(map, path, &["oneOf", "type"])?;
            if map.contains_key("type") {
                return Err(author_error(format!(
                    "{path} cannot declare both type and oneOf"
                )));
            }
            let Some(entries) = map.get("oneOf").and_then(Value::as_array) else {
                return Err(author_error(format!(
                    "{path}.oneOf must be an array of at least two value schemas"
                )));
            };
            let branches = entries
                .iter()
                .enumerate()
                .map(|(index, branch)| {
                    ValueSchemaSpec::from_author_value(branch, &format!("{path}.oneOf[{index}]"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(ValueSchemaSpec::OneOf(OneOfValueSchemaSpec {
                annotations,
                branches,
            }));
        }

        let type_name = map.get("type").and_then(Value::as_str);
        match type_name {
            Some("json") => {
                assert_author_keys(map, path, &["type"])?;
                Ok(ValueSchemaSpec::Json(JsonValueSchemaSpec { annotations }))
            }
            Some("object") => {
                assert_author_keys(map, path, &["type", "properties", "additionalProperties"])?;
                let Some(additional) = map.get("additionalProperties").and_then(Value::as_bool)
                else {
                    return Err(author_error(format!(
                        "{path}.additionalProperties must be explicitly true or false"
                    )));
                };
                let properties = match map.get("properties") {
                    None => None,
                    Some(value) => Some(ParameterSchemaSpec::from_author_value(
                        value,
                        &format!("{path}.properties"),
                    )?),
                };
                Ok(ValueSchemaSpec::Object(ObjectValueSchemaSpec {
                    annotations,
                    properties,
                    additional_properties: additional,
                }))
            }
            Some("array") => {
                assert_author_keys(map, path, &["type", "items"])?;
                let items = match map.get("items") {
                    None => None,
                    Some(value) => Some(Box::new(ValueSchemaSpec::from_author_value(
                        value,
                        &format!("{path}.items"),
                    )?)),
                };
                Ok(ValueSchemaSpec::Array(ArrayValueSchemaSpec {
                    annotations,
                    items,
                }))
            }
            Some("string") => {
                assert_author_keys(map, path, &["type", "enum", "const"])?;
                let (enum_values, const_value) =
                    parse_scalar_list(map, path, "string", |v| v.as_str().map(str::to_string))?;
                Ok(ValueSchemaSpec::String(StringValueSchemaSpec {
                    annotations,
                    enum_values,
                    const_value,
                }))
            }
            Some("number") => {
                assert_author_keys(map, path, &["type", "enum", "const"])?;
                let (enum_values, const_value) =
                    parse_scalar_list(map, path, "number", Value::as_f64)?;
                Ok(ValueSchemaSpec::Number(NumberValueSchemaSpec {
                    annotations,
                    enum_values,
                    const_value,
                }))
            }
            Some("integer") => {
                assert_author_keys(map, path, &["type", "enum", "const"])?;
                let (enum_values, const_value) = parse_scalar_list(map, path, "integer", |v| {
                    v.as_f64().filter(|f| f.fract() == 0.0).map(|f| f as i64)
                })?;
                Ok(ValueSchemaSpec::Integer(IntegerValueSchemaSpec {
                    annotations,
                    enum_values,
                    const_value,
                }))
            }
            Some("boolean") => {
                assert_author_keys(map, path, &["type", "enum", "const"])?;
                let (enum_values, const_value) =
                    parse_scalar_list(map, path, "boolean", Value::as_bool)?;
                Ok(ValueSchemaSpec::Boolean(BooleanValueSchemaSpec {
                    annotations,
                    enum_values,
                    const_value,
                }))
            }
            Some("null") => {
                assert_author_keys(map, path, &["type", "enum", "const"])?;
                Ok(ValueSchemaSpec::Null(NullValueSchemaSpec { annotations }))
            }
            _ => Err(author_error(format!(
                "{path}.type must be string/number/integer/boolean/null/array/object/json, or use oneOf"
            ))),
        }
    }

    /// Project this spec back onto the untyped author syntax.
    pub fn to_author_value(&self) -> Value {
        let mut node = match self {
            ValueSchemaSpec::OneOf(spec) => {
                let branches: Vec<Value> = spec
                    .branches
                    .iter()
                    .map(ValueSchemaSpec::to_author_value)
                    .collect();
                json!({ "oneOf": branches })
            }
            ValueSchemaSpec::Json(_) => json!({ "type": "json" }),
            ValueSchemaSpec::Null(_) => json!({ "type": "null" }),
            ValueSchemaSpec::Object(spec) => {
                let mut node = json!({
                    "type": "object",
                    "additionalProperties": spec.additional_properties,
                });
                if let Some(properties) = &spec.properties {
                    node["properties"] = properties.to_author_value();
                }
                node
            }
            ValueSchemaSpec::Array(spec) => {
                let mut node = json!({ "type": "array" });
                if let Some(items) = &spec.items {
                    node["items"] = items.to_author_value();
                }
                node
            }
            ValueSchemaSpec::String(spec) => {
                scalar_author_value("string", &spec.enum_values, &spec.const_value)
            }
            ValueSchemaSpec::Number(spec) => {
                scalar_author_value("number", &spec.enum_values, &spec.const_value)
            }
            ValueSchemaSpec::Integer(spec) => {
                scalar_author_value("integer", &spec.enum_values, &spec.const_value)
            }
            ValueSchemaSpec::Boolean(spec) => {
                scalar_author_value("boolean", &spec.enum_values, &spec.const_value)
            }
        };
        write_annotations(self.annotations(), &mut node);
        node
    }

    fn annotations(&self) -> &ValueSchemaAnnotations {
        match self {
            ValueSchemaSpec::String(spec) => &spec.annotations,
            ValueSchemaSpec::Number(spec) => &spec.annotations,
            ValueSchemaSpec::Integer(spec) => &spec.annotations,
            ValueSchemaSpec::Boolean(spec) => &spec.annotations,
            ValueSchemaSpec::Null(spec) => &spec.annotations,
            ValueSchemaSpec::Array(spec) => &spec.annotations,
            ValueSchemaSpec::Object(spec) => &spec.annotations,
            ValueSchemaSpec::Json(spec) => &spec.annotations,
            ValueSchemaSpec::OneOf(spec) => &spec.annotations,
        }
    }
}

fn scalar_author_value<T: Serialize>(
    type_name: &str,
    enum_values: &Option<Vec<T>>,
    const_value: &Option<T>,
) -> Value {
    let mut node = json!({ "type": type_name });
    if let Some(values) = enum_values {
        node["enum"] = serde_json::to_value(values).expect("scalars are JSON");
    }
    if let Some(value) = const_value {
        node["const"] = serde_json::to_value(value).expect("scalars are JSON");
    }
    node
}

fn write_annotations(annotations: &ValueSchemaAnnotations, node: &mut Value) {
    let map = node.as_object_mut().expect("author form is an object");
    if let Some(description) = &annotations.description {
        map.insert("description".into(), Value::String(description.clone()));
    }
    if let Some(title) = &annotations.title {
        map.insert("title".into(), Value::String(title.clone()));
    }
    if let Some(default) = &annotations.default {
        map.insert("default".into(), default.clone());
    }
    if let Some(examples) = &annotations.examples {
        map.insert("examples".into(), examples.clone());
    }
}

impl Serialize for ValueSchemaSpec {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_author_value().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ValueSchemaSpec {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        ValueSchemaSpec::from_author_value(&value, "schema").map_err(serde::de::Error::custom)
    }
}

impl Serialize for ParameterSchemaSpec {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_author_value().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ParameterSchemaSpec {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        ParameterSchemaSpec::from_author_value(&value, "parameters")
            .map_err(serde::de::Error::custom)
    }
}

/// Compile one spec node to the enforced subset (no assertion yet).
fn compile_value(spec: &ValueSchemaSpec) -> Value {
    let mut node = match spec {
        ValueSchemaSpec::OneOf(one_of) => {
            let branches: Vec<Value> = one_of.branches.iter().map(compile_value).collect();
            json!({ "oneOf": branches })
        }
        ValueSchemaSpec::Json(_) => json!({}),
        ValueSchemaSpec::Null(_) => json!({ "type": "null" }),
        ValueSchemaSpec::Object(object) => {
            let mut node = json!({
                "type": "object",
                "additionalProperties": object.additional_properties,
            });
            if let Some(properties) = &object.properties {
                let (compiled, required) = compile_property_map(properties);
                node["properties"] = compiled;
                if !required.is_empty() {
                    node["required"] = json!(required);
                }
            }
            node
        }
        ValueSchemaSpec::Array(array) => {
            let mut node = json!({ "type": "array" });
            if let Some(items) = &array.items {
                node["items"] = compile_value(items);
            }
            node
        }
        ValueSchemaSpec::String(spec) => {
            scalar_author_value("string", &spec.enum_values, &spec.const_value)
        }
        ValueSchemaSpec::Number(spec) => {
            scalar_author_value("number", &spec.enum_values, &spec.const_value)
        }
        ValueSchemaSpec::Integer(spec) => {
            scalar_author_value("integer", &spec.enum_values, &spec.const_value)
        }
        ValueSchemaSpec::Boolean(spec) => {
            scalar_author_value("boolean", &spec.enum_values, &spec.const_value)
        }
    };
    write_annotations(spec.annotations(), &mut node);
    node
}

fn compile_property_map(spec: &ParameterSchemaSpec) -> (Value, Vec<String>) {
    let mut properties = Map::new();
    let mut required = Vec::new();
    for (name, property) in &spec.0 {
        if property.required {
            required.push(name.clone());
        }
        properties.insert(name.clone(), compile_value(&property.spec));
    }
    (Value::Object(properties), required)
}

/// Compile one author-facing value schema to the enforced raw subset; the
/// author-only `json` node becomes an annotation-only schema.
pub fn value_schema_spec_to_json_schema(spec: &ValueSchemaSpec) -> Result<Value, JsonSchemaError> {
    let schema = compile_value(spec);
    assert_supported_json_schema(&schema)?;
    Ok(schema)
}

/// Compile the implicit open parameter object into an object-rooted raw
/// schema (no openness override on the root).
pub fn parameter_schema_spec_to_json_schema(
    spec: &ParameterSchemaSpec,
) -> Result<Value, JsonSchemaError> {
    let (properties, required) = compile_property_map(spec);
    let mut schema = json!({ "type": "object", "properties": properties });
    if !required.is_empty() {
        schema["required"] = json!(required);
    }
    assert_supported_json_schema(&schema)?;
    Ok(schema)
}

/// Invalid model-generated arguments for a typed tool.
#[derive(Debug, Clone, thiserror::Error)]
#[error("invalid arguments: {}", .violations.join("; "))]
pub struct ToolArgsError {
    /// Individual violations in schema-walk order.
    pub violations: Vec<String>,
}

impl ToolArgsError {
    /// Stable machine-routable failure class.
    pub const CODE: &'static str = "INVALID_ARGS";
}

/// Validate model-generated arguments against an implicit parameter schema.
/// Total over malformed argument values; path-qualified violations, empty
/// means valid.
pub fn validate_args(
    spec: &ParameterSchemaSpec,
    args: &Value,
) -> Result<Vec<String>, JsonSchemaError> {
    let schema = parameter_schema_spec_to_json_schema(spec)?;
    Ok(validate_json_schema_value(&schema, args, ""))
}

/// The output half of [`DefineToolOptions`]: canonical schema plus pure
/// projections.
pub struct DefineToolOutput {
    /// Schema enforced against every successful body or policy-replaced
    /// value.
    pub schema: ValueSchemaSpec,
    /// Pure model-facing rendering of one validated canonical value.
    pub render: ToolRenderFn,
    /// Pure replayable presentation metadata for direct top-level calls.
    pub presentation_meta: Option<ToolMetaFn>,
}

/// Options for [`define_tool`]. Closures receive validated
/// `serde_json::Value` arguments (the Rust stand-in for upstream's inferred
/// argument types).
pub struct DefineToolOptions {
    /// Tool name (unique per registry layer).
    pub name: String,
    /// Human-readable description sent to the model.
    pub description: String,
    /// Per-property parameter schema compiled to an implicit open object
    /// root.
    pub parameters: ParameterSchemaSpec,
    /// Canonical output declaration.
    pub output: DefineToolOutput,
    /// Optional positive cooperative timeout budget in milliseconds.
    pub timeout_ms: Option<f64>,
    /// Pure classifier for sibling overlap; only an explicit `true` opts in.
    pub is_concurrency_safe: Option<Rc<dyn Fn(&Value) -> bool>>,
    /// Execute the tool after argument validation.
    pub execute: ToolExecuteFn,
    /// Optional last-mile content transform for every normalized outcome.
    pub finalize_content: Option<ToolFinalizeFn>,
    /// Pure pending-state presenter.
    pub present_call: Option<Rc<dyn Fn(&Value) -> Option<ToolCallView>>>,
    /// Pure completed-state presenter.
    pub present_result: Option<Rc<dyn Fn(&Value, &ToolResult) -> Option<ToolResultView>>>,
}

/// Build a registry-ready definition with strict execute-time argument
/// validation and soft-validating presenters (display may replay obsolete
/// logged arguments and must never fail).
pub fn define_tool(options: DefineToolOptions) -> anyhow::Result<ToolDefinition> {
    if let Some(timeout_ms) = options.timeout_ms {
        if !timeout_ms.is_finite() || timeout_ms <= 0.0 {
            anyhow::bail!(
                "define_tool({}): timeoutMs must be a positive finite number",
                options.name
            );
        }
    }
    let parameters = parameter_schema_spec_to_json_schema(&options.parameters)?;
    let output_schema = value_schema_spec_to_json_schema(&options.output.schema)?;
    let parameters_map = match parameters {
        Value::Object(map) => map,
        _ => unreachable!("parameter compilation is object-rooted"),
    };

    let validate = {
        let schema = Value::Object(parameters_map.clone());
        Rc::new(move |args: &Value| validate_json_schema_value(&schema, args, ""))
    };

    let user_execute = options.execute;
    let execute: ToolExecuteFn = {
        let validate = validate.clone();
        Rc::new(move |args, exec| {
            let violations = validate(&args);
            if !violations.is_empty() {
                let error = ToolArgsError { violations };
                return Box::pin(async move { Err(error.into()) });
            }
            user_execute(args, exec)
        })
    };

    let present_call = options.present_call.map(|user| {
        let validate = validate.clone();
        let wrapped: Rc<dyn Fn(&Value) -> Option<ToolCallView>> = Rc::new(move |args| {
            if !validate(args).is_empty() {
                return None;
            }
            user(args)
        });
        wrapped
    });
    let present_result = options.present_result.map(|user| {
        let validate = validate.clone();
        let wrapped: Rc<dyn Fn(&Value, &ToolResult) -> Option<ToolResultView>> =
            Rc::new(move |args, result| {
                if !validate(args).is_empty() {
                    return None;
                }
                user(args, result)
            });
        wrapped
    });
    let is_concurrency_safe = options.is_concurrency_safe.map(|user| {
        let validate = validate.clone();
        let wrapped: Rc<dyn Fn(&Value) -> bool> = Rc::new(move |args| {
            if !validate(args).is_empty() {
                return false;
            }
            user(args)
        });
        wrapped
    });

    Ok(ToolDefinition {
        name: options.name,
        description: options.description,
        parameters: parameters_map,
        output: ToolOutputDefinition {
            schema: output_schema,
            render: options.output.render,
            presentation_meta: options.output.presentation_meta,
        },
        execute,
        finalize_content: options.finalize_content,
        timeout_ms: options.timeout_ms,
        is_concurrency_safe,
        present_call,
        present_result,
    })
}
