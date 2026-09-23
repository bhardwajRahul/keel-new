//! Typed service sugar over the string-keyed store, ported from
//! `vendor/cordis/src/service.ts`.
//!
//! Upstream services subclass `Service` and register in their constructor;
//! here a service is any type implementing [`Service`], registered with
//! [`Context::provide_service`] and read with [`Context::service`]. Intercept
//! config resolution (`Service[symbols.resolveConfig]`) is exposed as
//! [`Context::resolve_service_config`].

use crate::core::Context;
use crate::core::EffectHandle;
use crate::error::{CordisError, Result};
use serde_json::Value;
use std::rc::Rc;

/// A named service exposed on the context.
pub trait Service: 'static {
    /// The service name this type registers under (upstream `provide`).
    const NAME: &'static str;

    /// Availability predicate consulted before dependents may load
    /// (upstream `Service.check`).
    fn check(&self, _ctx: &Context) -> bool {
        true
    }
}

impl Context {
    /// Register a typed service owned by the current fiber.
    pub fn provide_service<S: Service>(&self, service: Rc<S>) -> Result<EffectHandle> {
        let for_check = service.clone();
        self.provide(
            S::NAME,
            service,
            Some(Rc::new(move |ctx| for_check.check(ctx))),
        )
    }

    /// Read a typed service through the inject walk (upstream property read).
    pub fn service<S: Service>(&self) -> Result<Rc<S>> {
        let value = self.get_raw(S::NAME)?;
        value
            .downcast::<S>()
            .map_err(|_| CordisError::ServiceMissing(S::NAME.to_string()))
    }

    /// Read a typed service without the inject requirement (upstream
    /// `ctx.get`); `None` when unprovided, inactive, or of another type.
    pub fn try_service<S: Service>(&self) -> Option<Rc<S>> {
        self.try_get_raw(S::NAME, true)?.downcast::<S>().ok()
    }

    /// Merge intercept config for a service from ancestor contexts, with
    /// optional lowest-precedence `base` and highest-precedence `head`
    /// (upstream `Service[symbols.resolveConfig]`; shallow object merge).
    pub fn resolve_service_config(
        &self,
        name: &str,
        base: Option<Value>,
        head: Option<Value>,
    ) -> Value {
        let mut configs = Vec::new();
        if let Some(base) = base {
            configs.push(base);
        }
        configs.extend(self.intercept_configs(name));
        if let Some(head) = head {
            configs.push(head);
        }
        let mut merged = serde_json::Map::new();
        for config in configs {
            if let Value::Object(map) = config {
                for (key, value) in map {
                    merged.insert(key, value);
                }
            }
        }
        Value::Object(merged)
    }
}
