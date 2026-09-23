//! Rust port of `packages/runtime-diagnostics/invariants`: the `invariants`
//! service — a registry through which every workspace package contributes its
//! runtime invariant checks, with global and per-package-name selection.
//!
//! Divergences from the TypeScript package (contract-level; forced by the
//! Rust host or the dsh-cordis port):
//! - Package filters are anchored-literal patterns, not JavaScript regexes
//!   (no regex engine in the workspace): an optional leading `^` and trailing
//!   `$` anchor an otherwise literal, case-sensitive substring match; `.` is
//!   literal; any other regex metacharacter (`\\[](){}|*+?`) is rejected as an
//!   invalid pattern.
//! - `fail` cannot throw through an event dispatch (the Rust port's `emit` is
//!   fire-and-forget), so it *constructs* the typed [`InvariantError`] for the
//!   installer to return; event-time violations must surface through the
//!   event's own return channel.
//! - [`InvariantRegistry::register`] is `async` and returns the registration's
//!   [`EffectHandle`] (the cordis disposer) instead of a callable-thenable.
//!   All validation and the ownership reservation happen before its first
//!   await, preserving the synchronous duplicate check.
//! - Upstream's rollback for a throwing `internal/plugin` observer has no
//!   counterpart: listeners cannot throw into `ctx.plugin` here.
//! - Constructing the registry directly does not self-provide the service;
//!   loading [`InvariantsPlugin`] provides it under the name `invariants`.

use dsh_cordis::{
    Context, Disposer, Effect, EffectHandle, Fiber, FnPlugin, Inject, Plugin, Service, plugin_fn,
    validate_as,
};
use futures::FutureExt;
use futures::future::LocalBoxFuture;
use serde_json::Value;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

/// Runtime invariant selection configured on the service plugin.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Config {
    /// Global switch; defaults to `true`.
    pub enabled: bool,
    /// Case-sensitive anchored-literal patterns that admit package names;
    /// empty admits all.
    pub package_allowlist: Vec<String>,
    /// Case-sensitive anchored-literal patterns that exclude package names
    /// after allowlist matching.
    pub package_blocklist: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            enabled: true,
            package_allowlist: Vec::new(),
            package_blocklist: Vec::new(),
        }
    }
}

/// Raised when a package-owned runtime invariant is violated.
#[derive(Debug, Clone, thiserror::Error)]
#[error("invariant violated by \"{package_name}\": {message}")]
pub struct InvariantError {
    /// Full package name that owns the violated invariant.
    pub package_name: String,
    /// The violated contract, without the standard prefix.
    pub message: String,
}

impl InvariantError {
    /// Stable machine-readable invariant failure code.
    pub const CODE: &'static str = "INVARIANT";

    /// Build a package-attributed invariant failure.
    pub fn new(package_name: impl Into<String>, message: impl Into<String>) -> Self {
        InvariantError {
            package_name: package_name.into(),
            message: message.into(),
        }
    }
}

/// Reporter bound to the registering package name: turns a violated-contract
/// message into the package-attributed [`InvariantError`].
pub type InvariantFailure = Rc<dyn Fn(&str) -> InvariantError>;

type InstallFn =
    Rc<dyn Fn(Context, InvariantFailure) -> LocalBoxFuture<'static, anyhow::Result<()>>>;

/// One package's invariant contribution: an installer run inside a child
/// fiber owned by the registration, plus the services that fiber may access.
pub struct InvariantInstaller {
    /// Services the child installer fiber may access.
    pub inject: Inject,
    install: InstallFn,
}

impl InvariantInstaller {
    /// Wrap an installer body with no service requirements.
    pub fn new<F, Fut>(install: F) -> Self
    where
        F: Fn(Context, InvariantFailure) -> Fut + 'static,
        Fut: std::future::Future<Output = anyhow::Result<()>> + 'static,
    {
        InvariantInstaller {
            inject: Inject::default(),
            install: Rc::new(move |ctx, fail| install(ctx, fail).boxed_local()),
        }
    }

    /// Declare the services the child installer fiber requires.
    pub fn with_inject(mut self, inject: Inject) -> Self {
        self.inject = inject;
        self
    }
}

/// Anchored-literal package filter (the port's replacement for a JS regex).
struct Pattern {
    start: bool,
    end: bool,
    literal: String,
}

impl Pattern {
    fn parse(source: &str) -> Option<Pattern> {
        let (start, rest) = match source.strip_prefix('^') {
            Some(rest) => (true, rest),
            None => (false, source),
        };
        let (end, literal) = match rest.strip_suffix('$') {
            Some(literal) => (true, literal),
            None => (false, rest),
        };
        const UNSUPPORTED: &[char] = &[
            '\\', '[', ']', '(', ')', '{', '}', '|', '*', '+', '?', '^', '$',
        ];
        if literal.chars().any(|c| UNSUPPORTED.contains(&c)) {
            return None;
        }
        Some(Pattern {
            start,
            end,
            literal: literal.to_string(),
        })
    }

    fn matches(&self, name: &str) -> bool {
        match (self.start, self.end) {
            (true, true) => name == self.literal,
            (true, false) => name.starts_with(&self.literal),
            (false, true) => name.ends_with(&self.literal),
            (false, false) => name.contains(&self.literal),
        }
    }

    /// Compile and validate one package-filter list.
    fn compile(field: &str, values: &[String]) -> anyhow::Result<Vec<Pattern>> {
        let mut seen = HashSet::new();
        values
            .iter()
            .map(|value| {
                if value.is_empty() || value.trim() != value {
                    anyhow::bail!(
                        "invariants: {field} entries must be non-blank and have no surrounding whitespace"
                    );
                }
                if !seen.insert(value.as_str()) {
                    anyhow::bail!("invariants: {field} contains duplicate regex {value:?}");
                }
                Pattern::parse(value)
                    .ok_or_else(|| anyhow::anyhow!("invariants: {field} contains invalid regex {value:?}"))
            })
            .collect()
    }
}

struct InstallerEntry {
    package: String,
    install: InstallFn,
    /// Slot preserving the installer's typed error across the fiber's
    /// stringly-typed failure storage.
    failure: Rc<RefCell<Option<anyhow::Error>>>,
}

thread_local! {
    /// Live installer bodies by registration id. Global so that every
    /// [`InstallerPlugin`] fiber resolves its own installer even though all
    /// instances share one cordis runtime record (plugins are keyed by type).
    static INSTALLERS: RefCell<HashMap<u64, InstallerEntry>> = RefCell::new(HashMap::new());
    static NEXT_INSTALLER: Cell<u64> = const { Cell::new(0) };
}

/// Child plugin backing one registration. The registration id travels in the
/// fiber config because every instance shares one runtime record; the stored
/// first instance dispatches through the thread-local installer table.
struct InstallerPlugin {
    inject: Inject,
}

impl Plugin for InstallerPlugin {
    fn name(&self) -> Option<String> {
        Some("invariant-installer".into())
    }

    fn inject(&self) -> Inject {
        self.inject.clone()
    }

    fn apply(&self, ctx: Context, config: Value) -> LocalBoxFuture<'static, anyhow::Result<()>> {
        let entry = config.as_u64().and_then(|id| {
            INSTALLERS.with(|table| {
                table.borrow().get(&id).map(|entry| {
                    (
                        entry.package.clone(),
                        entry.install.clone(),
                        entry.failure.clone(),
                    )
                })
            })
        });
        async move {
            // Registration already disposed: nothing to install.
            let Some((package, install, failure)) = entry else {
                return Ok(());
            };
            let fail: InvariantFailure = {
                let package = package.clone();
                Rc::new(move |message| InvariantError::new(package.clone(), message))
            };
            match install(ctx, fail).await {
                Ok(()) => Ok(()),
                Err(error) => {
                    let text = format!("{error:#}");
                    *failure.borrow_mut() = Some(error);
                    Err(anyhow::anyhow!(text))
                }
            }
        }
        .boxed_local()
    }
}

/// Package-owned invariant registry with global and pattern-based selection.
pub struct InvariantRegistry {
    owner_ctx: Context,
    enabled: bool,
    allow: Vec<Pattern>,
    block: Vec<Pattern>,
    registrations: Rc<RefCell<HashSet<String>>>,
}

impl Service for InvariantRegistry {
    const NAME: &'static str = "invariants";
}

impl InvariantRegistry {
    /// Build the registry; fails when a filter list is malformed. The
    /// registry does not provide itself — load [`InvariantsPlugin`] for that.
    pub fn new(ctx: Context, config: Config) -> anyhow::Result<Self> {
        Ok(InvariantRegistry {
            owner_ctx: ctx,
            enabled: config.enabled,
            allow: Pattern::compile("package_allowlist", &config.package_allowlist)?,
            block: Pattern::compile("package_blocklist", &config.package_blocklist)?,
            registrations: Rc::default(),
        })
    }

    /// Whether one full package name passes the configured filters.
    fn selected(&self, package_name: &str) -> bool {
        self.enabled
            && (self.allow.is_empty() || self.allow.iter().any(|p| p.matches(package_name)))
            && !self.block.iter().any(|p| p.matches(package_name))
    }

    fn release(&self, package_name: &str, installer_id: Option<u64>) {
        if let Some(id) = installer_id {
            INSTALLERS.with(|table| {
                table.borrow_mut().remove(&id);
            });
        }
        self.registrations.borrow_mut().remove(package_name);
    }

    /// Register one package's invariant installer. The package name is
    /// reserved even when filtering disables its checks; an enabled installer
    /// runs in a child fiber owned by the service. Failure disposes that
    /// fiber, rolls back its effects, and releases the reservation. On
    /// success the returned handle is the registration's disposer; disposal
    /// keeps the name reserved until the child fiber finished tearing down.
    pub async fn register(
        &self,
        package_name: &str,
        installer: InvariantInstaller,
    ) -> anyhow::Result<EffectHandle> {
        if package_name.is_empty() || package_name.chars().any(char::is_whitespace) {
            anyhow::bail!("invariants: packageName must be non-blank and contain no whitespace");
        }
        if !self
            .registrations
            .borrow_mut()
            .insert(package_name.to_string())
        {
            anyhow::bail!("invariants: package {package_name:?} is already registered");
        }
        let label = format!("invariants.register({package_name:?})");
        let package = package_name.to_string();

        if !self.selected(package_name) {
            let registrations = self.registrations.clone();
            let released = package.clone();
            return match self.owner_ctx.effect_labeled(&label, move |_| {
                Ok(Effect::One(Disposer::sync(move || {
                    registrations.borrow_mut().remove(&released);
                })))
            }) {
                Ok(handle) => Ok(handle),
                Err(error) => {
                    self.release(&package, None);
                    Err(error.into())
                }
            };
        }

        let id = NEXT_INSTALLER.with(|next| {
            let id = next.get() + 1;
            next.set(id);
            id
        });
        let failure: Rc<RefCell<Option<anyhow::Error>>> = Rc::default();
        INSTALLERS.with(|table| {
            table.borrow_mut().insert(
                id,
                InstallerEntry {
                    package: package.clone(),
                    install: installer.install.clone(),
                    failure: failure.clone(),
                },
            );
        });

        let child: Fiber = match self.owner_ctx.plugin(
            Rc::new(InstallerPlugin {
                inject: installer.inject.clone(),
            }),
            serde_json::json!(id),
        ) {
            Ok(fiber) => fiber,
            Err(error) => {
                self.release(&package, Some(id));
                return Err(error.into());
            }
        };

        if let Err(error) = child.await_ready().await {
            child.dispose().await;
            self.release(&package, Some(id));
            return Err(failure.borrow_mut().take().unwrap_or_else(|| error.into()));
        }

        let registrations = self.registrations.clone();
        let disposed_child = child.clone();
        let released = package.clone();
        match self.owner_ctx.effect_labeled(&label, move |_| {
            Ok(Effect::One(Disposer::asynchronous(move || async move {
                disposed_child.dispose().await;
                INSTALLERS.with(|table| {
                    table.borrow_mut().remove(&id);
                });
                registrations.borrow_mut().remove(&released);
            })))
        }) {
            Ok(handle) => Ok(handle),
            Err(error) => {
                child.dispose().await;
                self.release(&package, Some(id));
                Err(error.into())
            }
        }
    }
}

/// Service plugin providing `ctx.invariants` (upstream default export).
pub struct InvariantsPlugin;

impl Plugin for InvariantsPlugin {
    fn name(&self) -> Option<String> {
        Some("invariants".into())
    }

    fn validate_config(&self, config: Value) -> dsh_cordis::Result<Value> {
        validate_as::<Config>(config)
    }

    fn apply(&self, ctx: Context, config: Value) -> LocalBoxFuture<'static, anyhow::Result<()>> {
        async move {
            let config: Config = serde_json::from_value(config)?;
            let registry = Rc::new(InvariantRegistry::new(ctx.clone(), config)?);
            ctx.provide_service(registry)?;
            Ok(())
        }
        .boxed_local()
    }
}

pub mod invariant {
    //! This package's own invariant companion (upstream `./invariant`).
    //!
    //! No runtime invariant: registration ownership and the child-fiber
    //! lifecycle are the service's own mutation boundary; observing them from
    //! the same registry would only restate its implementation.

    use super::*;

    /// Package name registered by the companion.
    pub const PACKAGE_NAME: &str = "dsh-invariants";
    /// Cordis companion plugin name.
    pub const NAME: &str = "invariants-invariant";

    /// Build the companion plugin: requires the `invariants` service and
    /// reserves this package's ownership with an empty installer.
    pub fn plugin()
    -> FnPlugin<impl Fn(Context, Value) -> LocalBoxFuture<'static, anyhow::Result<()>> + 'static>
    {
        plugin_fn(NAME, Inject::names(["invariants"]), |ctx, _config| {
            async move {
                let registry = ctx.service::<InvariantRegistry>()?;
                let registration = registry
                    .register(
                        PACKAGE_NAME,
                        InvariantInstaller::new(|_ctx, _fail| async { Ok(()) }),
                    )
                    .await?;
                ctx.effect_labeled(NAME, move |_| {
                    Ok(Effect::One(Disposer::asynchronous(move || {
                        registration.dispose()
                    })))
                })?;
                Ok(())
            }
            .boxed_local()
        })
    }
}
