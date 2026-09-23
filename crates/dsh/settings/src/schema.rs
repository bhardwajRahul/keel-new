//! Runtime settings schemas, standing in for upstream schemastery schemas
//! (`z.object(...)` etc.). Only the surface the settings seam relies on is
//! modeled: default resolution, structural validation, `role("secret")`
//! metadata for the redactor, and a serializable description for
//! configuration UIs.

use serde_json::{Map, Value};

/// Structural kind of one schema node.
#[derive(Clone)]
pub enum SchemaKind {
    /// Accepts any JSON value verbatim.
    Any,
    String,
    Number,
    Boolean,
    /// Closed set of literal values (upstream `z.union([...])` of constants).
    Union(Vec<Value>),
    /// Fixed properties in declaration order; unknown input keys pass through.
    Object(Vec<(String, Schema)>),
    /// Homogeneous string-keyed map (upstream `z.dict`).
    Dict(Box<Schema>),
    /// Homogeneous list (upstream `z.array`).
    Array(Box<Schema>),
}

/// One schema node: a kind plus the metadata the seam reads (`default`,
/// `role`). Values resolve through [`Schema::resolve`]; the redactor walks
/// the node tree directly.
#[derive(Clone)]
pub struct Schema {
    pub(crate) kind: SchemaKind,
    pub(crate) role: Option<String>,
    pub(crate) default: Option<Value>,
}

impl Schema {
    fn of(kind: SchemaKind) -> Schema {
        Schema {
            kind,
            role: None,
            default: None,
        }
    }

    pub fn any() -> Schema {
        Schema::of(SchemaKind::Any)
    }

    pub fn string() -> Schema {
        Schema::of(SchemaKind::String)
    }

    pub fn number() -> Schema {
        Schema::of(SchemaKind::Number)
    }

    pub fn boolean() -> Schema {
        Schema::of(SchemaKind::Boolean)
    }

    /// Literal union: the value must equal one of the given constants.
    pub fn union(literals: impl IntoIterator<Item = Value>) -> Schema {
        Schema::of(SchemaKind::Union(literals.into_iter().collect()))
    }

    /// Object schema with declared properties in order.
    pub fn object(props: impl IntoIterator<Item = (&'static str, Schema)>) -> Schema {
        Schema::of(SchemaKind::Object(
            props.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        ))
    }

    pub fn dict(inner: Schema) -> Schema {
        Schema::of(SchemaKind::Dict(Box::new(inner)))
    }

    pub fn array(inner: Schema) -> Schema {
        Schema::of(SchemaKind::Array(Box::new(inner)))
    }

    /// Fallback used when the input omits the value.
    pub fn default(mut self, value: Value) -> Schema {
        self.default = Some(value);
        self
    }

    /// Attach a role marker; `role("secret")` fields are stripped by the
    /// redactor before values cross a wire boundary.
    pub fn role(mut self, role: &str) -> Schema {
        self.role = Some(role.to_string());
        self
    }

    pub(crate) fn is_secret(&self) -> bool {
        self.role.as_deref() == Some("secret")
    }

    /// Serialized structural description for configuration surfaces
    /// (node-shaped JSON: `type`, `dict`/`inner`, `meta`).
    pub fn to_json(&self) -> Value {
        let mut node = Map::new();
        let type_name = match &self.kind {
            SchemaKind::Any => "any",
            SchemaKind::String => "string",
            SchemaKind::Number => "number",
            SchemaKind::Boolean => "boolean",
            SchemaKind::Union(_) => "union",
            SchemaKind::Object(_) => "object",
            SchemaKind::Dict(_) => "dict",
            SchemaKind::Array(_) => "array",
        };
        node.insert("type".into(), Value::String(type_name.into()));
        match &self.kind {
            SchemaKind::Union(literals) => {
                node.insert("list".into(), Value::Array(literals.clone()));
            }
            SchemaKind::Object(props) => {
                let mut dict = Map::new();
                for (key, child) in props {
                    dict.insert(key.clone(), child.to_json());
                }
                node.insert("dict".into(), Value::Object(dict));
            }
            SchemaKind::Dict(inner) | SchemaKind::Array(inner) => {
                node.insert("inner".into(), inner.to_json());
            }
            _ => {}
        }
        let mut meta = Map::new();
        if let Some(role) = &self.role {
            meta.insert("role".into(), Value::String(role.clone()));
        }
        if let Some(default) = &self.default {
            meta.insert("default".into(), default.clone());
        }
        if !meta.is_empty() {
            node.insert("meta".into(), Value::Object(meta));
        }
        Value::Object(node)
    }

    /// Resolve an input value against this schema: apply defaults, validate
    /// structure, and materialize object properties. `None` means the value
    /// is absent (an omitted optional stays absent).
    pub fn resolve(&self, value: Option<&Value>) -> Result<Option<Value>, String> {
        self.resolve_at(value, "$")
    }

    fn resolve_at(&self, value: Option<&Value>, path: &str) -> Result<Option<Value>, String> {
        // JSON null carries no information the seam distinguishes from
        // absence; both fall back to the default.
        let value = value.filter(|v| !v.is_null());
        let fallback = || self.default.clone();
        match &self.kind {
            SchemaKind::Any => Ok(value.cloned().or_else(fallback)),
            SchemaKind::String => match value {
                None => Ok(fallback()),
                Some(v @ Value::String(_)) => Ok(Some(v.clone())),
                Some(other) => Err(type_error(path, "a string", other)),
            },
            SchemaKind::Number => match value {
                None => Ok(fallback()),
                Some(v @ Value::Number(_)) => Ok(Some(v.clone())),
                Some(other) => Err(type_error(path, "a number", other)),
            },
            SchemaKind::Boolean => match value {
                None => Ok(fallback()),
                Some(v @ Value::Bool(_)) => Ok(Some(v.clone())),
                Some(other) => Err(type_error(path, "a boolean", other)),
            },
            SchemaKind::Union(literals) => match value {
                None => Ok(fallback()),
                Some(v) if literals.contains(v) => Ok(Some(v.clone())),
                Some(other) => Err(format!("{path}: {other} is not one of the allowed values")),
            },
            SchemaKind::Object(props) => {
                let source = match value {
                    None => None,
                    Some(Value::Object(map)) => Some(map),
                    Some(other) => return Err(type_error(path, "an object", other)),
                };
                let mut out = source.cloned().unwrap_or_default();
                for (key, child) in props {
                    let entry = source.and_then(|map| map.get(key));
                    let child_path = format!("{path}.{key}");
                    match child.resolve_at(entry, &child_path)? {
                        Some(resolved) => {
                            out.insert(key.clone(), resolved);
                        }
                        None => {
                            out.remove(key);
                        }
                    }
                }
                Ok(Some(Value::Object(out)))
            }
            SchemaKind::Dict(inner) => match value {
                None => Ok(fallback()),
                Some(Value::Object(map)) => {
                    let mut out = Map::new();
                    for (key, entry) in map {
                        let child_path = format!("{path}.{key}");
                        if let Some(resolved) = inner.resolve_at(Some(entry), &child_path)? {
                            out.insert(key.clone(), resolved);
                        }
                    }
                    Ok(Some(Value::Object(out)))
                }
                Some(other) => Err(type_error(path, "a map", other)),
            },
            SchemaKind::Array(inner) => match value {
                None => Ok(fallback()),
                Some(Value::Array(items)) => {
                    let mut out = Vec::with_capacity(items.len());
                    for (index, item) in items.iter().enumerate() {
                        let child_path = format!("{path}[{index}]");
                        out.push(
                            inner
                                .resolve_at(Some(item), &child_path)?
                                .unwrap_or(Value::Null),
                        );
                    }
                    Ok(Some(Value::Array(out)))
                }
                Some(other) => Err(type_error(path, "an array", other)),
            },
        }
    }
}

fn type_error(path: &str, expected: &str, got: &Value) -> String {
    let kind = match got {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    };
    format!("{path}: expected {expected}, got {kind}")
}
