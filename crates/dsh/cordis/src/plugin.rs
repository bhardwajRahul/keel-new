//! Plugin entrypoints and dependency declarations, ported from
//! `vendor/cordis/src/registry.ts` (`Plugin`, `Inject`).
//!
//! Upstream accepts function, class, and `{ apply }` object plugins keyed by
//! callback identity; here a plugin is any type implementing [`Plugin`],
//! keyed by `TypeId` (two instances of one type share a runtime record, like
//! upstream class plugins). [`plugin_fn`] wraps a closure, matching upstream
//! function plugins.

use crate::core::Context;
use crate::error::{CordisError, Result};
use futures::FutureExt;
use futures::future::LocalBoxFuture;
use serde_json::Value;
use std::any::{Any, TypeId};
use std::collections::HashMap;

/// Service dependency declaration: name → optional intercept config
/// (upstream `Inject`, already `Inject.resolve`d to its map form).
#[derive(Default, Clone)]
pub struct Inject(pub HashMap<String, Option<Value>>);

impl Inject {
    /// Array form: request services without intercept config.
    pub fn names<I: IntoIterator<Item = S>, S: Into<String>>(names: I) -> Inject {
        Inject(names.into_iter().map(|name| (name.into(), None)).collect())
    }

    /// Add one requirement with intercept config.
    pub fn with(mut self, name: impl Into<String>, config: Value) -> Inject {
        self.0.insert(name.into(), Some(config));
        self
    }
}

/// Registry identity for one plugin (upstream callback identity).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PluginKey(TypeId);

/// A plugin entrypoint (upstream `Plugin` + `Plugin.Base` metadata).
///
/// Lifecycle is the body plus its registered effects: `apply` runs when every
/// injected service is available, and the effects it registered through
/// `ctx.effect` / `ctx.on` / `ctx.provide` are disposed when any of them
/// leaves (upstream has no separate unload hook either).
pub trait Plugin: Any {
    /// Display name used for fiber diagnostics and logger names.
    fn name(&self) -> Option<String> {
        None
    }

    /// Services the plugin requires; it only stays loaded while all are
    /// available.
    fn inject(&self) -> Inject {
        Inject::default()
    }

    /// Validate config before the plugin starts (upstream `Config` schema).
    fn validate_config(&self, config: Value) -> Result<Value> {
        Ok(config)
    }

    /// The plugin body. Register services, listeners, and effects on `ctx`;
    /// return `Err` to mark the fiber FAILED.
    fn apply(&self, ctx: Context, config: Value) -> LocalBoxFuture<'static, anyhow::Result<()>>;

    /// Registry identity (upstream callback identity).
    fn key(&self) -> PluginKey {
        PluginKey(self.type_id())
    }
}

/// A function plugin: name + inject + async body (upstream `Plugin.Function`).
pub struct FnPlugin<F> {
    name: String,
    inject: Inject,
    body: F,
}

impl<F, Fut> Plugin for FnPlugin<F>
where
    F: Fn(Context, Value) -> Fut + 'static,
    Fut: std::future::Future<Output = anyhow::Result<()>> + 'static,
{
    fn name(&self) -> Option<String> {
        Some(self.name.clone())
    }

    fn inject(&self) -> Inject {
        self.inject.clone()
    }

    fn apply(&self, ctx: Context, config: Value) -> LocalBoxFuture<'static, anyhow::Result<()>> {
        (self.body)(ctx, config).boxed_local()
    }
}

/// Wrap a closure as a plugin (upstream function-plugin form).
pub fn plugin_fn<F, Fut>(name: impl Into<String>, inject: Inject, body: F) -> FnPlugin<F>
where
    F: Fn(Context, Value) -> Fut + 'static,
    Fut: std::future::Future<Output = anyhow::Result<()>> + 'static,
{
    FnPlugin {
        name: name.into(),
        inject,
        body,
    }
}

/// Validate config through a serde deserialization, mirroring upstream
/// standard-schema validation: deserialize to `T`, then re-serialize so the
/// stored config carries defaults.
pub fn validate_as<T: serde::de::DeserializeOwned + serde::Serialize>(
    config: Value,
) -> Result<Value> {
    let typed: T = serde_json::from_value(if config.is_null() {
        Value::Object(serde_json::Map::new())
    } else {
        config
    })
    .map_err(|error| CordisError::InvalidConfig(format!("  - {error}")))?;
    serde_json::to_value(&typed).map_err(|error| CordisError::InvalidConfig(format!("  - {error}")))
}
