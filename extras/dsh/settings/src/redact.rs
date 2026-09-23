//! Structural secret redaction for settings values. Fields the schema marks
//! `role("secret")` are removed before a value crosses a wire boundary; a
//! sidecar lists each schema-declared secret position and whether it held a
//! value, so a form can render a write-only input without receiving the
//! secret itself.

use crate::schema::{Schema, SchemaKind};
use serde_json::{Map, Value};

/// One schema-declared secret position inside a redacted value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RedactedSecret {
    /// Path from the section root to the removed field, including concrete
    /// dict keys and array indexes.
    pub path: Vec<String>,
    /// Whether the field held a value before redaction.
    pub set: bool,
}

/// A value with every secret field removed, plus the removal record.
#[derive(Debug, Clone)]
pub struct RedactedValue {
    /// Detached copy of the input with secret fields absent; `None` when the
    /// input itself was absent (or a secret leaf).
    pub value: Option<Value>,
    /// Every reachable secret position: object properties always (even when
    /// unset, so a form knows the slot exists), dict entries and array items
    /// only where the value has them.
    pub secrets: Vec<RedactedSecret>,
}

fn walk(
    node: &Schema,
    value: Option<&Value>,
    path: &[String],
    secrets: &mut Vec<RedactedSecret>,
) -> Option<Value> {
    if node.is_secret() {
        secrets.push(RedactedSecret {
            path: path.to_vec(),
            set: value.is_some(),
        });
        return None;
    }
    match &node.kind {
        SchemaKind::Object(props) => {
            let source = match value {
                Some(Value::Object(map)) => Some(map),
                _ => None,
            };
            let mut rebuilt = Map::new();
            if let Some(source) = source {
                for (key, entry) in source {
                    if props.iter().any(|(name, _)| name == key) {
                        continue;
                    }
                    rebuilt.insert(key.clone(), entry.clone());
                }
            }
            for (key, child) in props {
                let mut child_path = path.to_vec();
                child_path.push(key.clone());
                let stripped = walk(
                    child,
                    source.and_then(|map| map.get(key)),
                    &child_path,
                    secrets,
                );
                if let Some(stripped) = stripped {
                    rebuilt.insert(key.clone(), stripped);
                }
            }
            if source.is_none() && rebuilt.is_empty() {
                // A malformed (non-object) or absent value passes through
                // verbatim rather than inventing a container.
                value.cloned()
            } else {
                Some(Value::Object(rebuilt))
            }
        }
        SchemaKind::Dict(inner) => {
            let Some(Value::Object(map)) = value else {
                return value.cloned();
            };
            let mut rebuilt = Map::new();
            // serde_json may preserve insertion order depending on workspace
            // features. Secret paths must stay stable regardless of that flag.
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by(|(left, _), (right, _)| left.cmp(right));
            for (key, entry) in entries {
                let mut child_path = path.to_vec();
                child_path.push(key.clone());
                if let Some(stripped) = walk(inner, Some(entry), &child_path, secrets) {
                    rebuilt.insert(key.clone(), stripped);
                }
            }
            Some(Value::Object(rebuilt))
        }
        SchemaKind::Array(inner) => {
            let Some(Value::Array(items)) = value else {
                return value.cloned();
            };
            let rebuilt = items
                .iter()
                .enumerate()
                .map(|(index, entry)| {
                    let mut child_path = path.to_vec();
                    child_path.push(index.to_string());
                    walk(inner, Some(entry), &child_path, secrets).unwrap_or(Value::Null)
                })
                .collect();
            Some(Value::Array(rebuilt))
        }
        _ => value.cloned(),
    }
}

/// Remove every `role("secret")` field the schema declares from a value.
///
/// The walker follows object, dict, and array containers; a secret must be
/// declared directly on a field reachable through those containers. The input
/// is never mutated (the result is a detached copy).
pub fn redact_secrets(schema: &Schema, value: Option<&Value>) -> RedactedValue {
    let mut secrets = Vec::new();
    let stripped = walk(schema, value, &[], &mut secrets);
    RedactedValue {
        value: stripped,
        secrets,
    }
}
