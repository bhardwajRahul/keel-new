//! The `llm` service, ported from `packages/llm/llm/src/index.ts`: an adapter
//! registry plus a streaming model-call API, interceptable via the
//! `llm/stream` waterfall.
//!
//! Divergences:
//! - Upstream throws `LlmError` out of registration calls; the port returns
//!   `Result<_, LlmError>` (same synchronous timing — upstream effect bodies
//!   run synchronously inside `ctx.effect`).
//! - `emitAdaptersUpdated`'s per-listener containment is inherent here: the
//!   port's `emit` runs every listener's future independently.
//! - Registration handles are structs with `dispose()`/`replace()` instead of
//!   callable functions.

use crate::adapter_failure::normalize_llm_failure;
use crate::brand::ReasoningEffortId;
use crate::call_config::{LlmCallConfig, LlmCallConfigAdapterDefaults, options_match_config};
use crate::error::LlmError;
use crate::message::MessageSource;
use crate::retry::{ResolvedRetryPolicy, resolve_retry_policy};
use crate::types::{
    FinishReason, GenerateOptions, LlmConfigurableProvider, LlmDiscoveredModel, LlmModelContext,
    LlmModelDiscoveryRequest, LlmModelInfo, LlmProviderInfo, LlmResolvedModelInfo, StreamChunk,
};
use async_stream::stream;
use dsh_cordis::{Context, Event, Service};
use futures::future::LocalBoxFuture;
use futures::stream::LocalBoxStream;
use futures::{FutureExt, StreamExt};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

/// Consumer-facing chunk stream: failures are already normalized into
/// terminal finish chunks.
pub type ChunkStream = LocalBoxStream<'static, StreamChunk>;

/// Adapter-facing chunk stream: a failed item ends the stream and becomes a
/// terminal finish chunk at the runtime boundary (upstream adapters throw).
pub type AdapterStream = LocalBoxStream<'static, anyhow::Result<StreamChunk>>;

/// The provider topology changed: an adapter registered or unregistered
/// routes, or the configurable-provider directory changed. Payload-free;
/// consumers re-read the list methods (upstream `llm/adapters-updated`).
pub struct AdaptersUpdated;
impl Event for AdaptersUpdated {
    const NAME: &'static str = "llm/adapters-updated";
    type Args = ();
    type Ret = ();
}

/// Waterfall around every streaming model call (retry, replay, routing).
/// Call `next(options)` to reach the resolved adapter's stream, or yield your
/// own stream to short-circuit (upstream `llm/stream`). Loop-built requests
/// are pure functions of the session log: listeners read them, never rewrite
/// them.
pub struct LlmStream;
impl Event for LlmStream {
    const NAME: &'static str = "llm/stream";
    type Args = GenerateOptions;
    type Ret = ChunkStream;
}

/// Provider-wire adapter for the harness message and stream vocabulary
/// (upstream `LlmAdapter`). Register with `LlmRuntime::register_adapter`.
/// Every provider HTTP request must include `attribution_headers()`.
pub trait LlmAdapter: 'static {
    /// Describe one provider route owned by this adapter; the id must equal
    /// `provider`.
    fn provider_info(&self, provider: &str) -> LlmProviderInfo {
        LlmProviderInfo {
            id: provider.to_string(),
            name: provider.to_string(),
        }
    }

    /// Provider-owned retry policy for one route; `None` selects defaults.
    fn provider_retry_policy(&self, _provider: &str) -> Option<ResolvedRetryPolicy> {
        None
    }

    /// List models this adapter can currently advertise for one owned
    /// provider. Advisory: consumers must not turn absence into rejection.
    fn list_models(
        &self,
        _provider: &str,
    ) -> LocalBoxFuture<'_, anyhow::Result<Vec<LlmModelInfo>>> {
        async { Ok(Vec::new()) }.boxed_local()
    }

    /// Resolve all metadata available for one exact model. Independent of the
    /// advisory catalog; does not validate request routing.
    fn resolve_model<'a>(
        &'a self,
        provider: &'a str,
        model: &'a str,
        _signal: Option<dsh_timeout::AbortSignal>,
    ) -> LocalBoxFuture<'a, anyhow::Result<LlmResolvedModelInfo>> {
        async move { Ok(LlmResolvedModelInfo::bare(provider, model)) }.boxed_local()
    }

    /// Stream one model call as raw chunks. The only required method;
    /// implementations must honor `options.signal`.
    fn stream(&self, options: GenerateOptions) -> AdapterStream;
}

#[derive(Clone)]
struct AdapterRegistration {
    adapter: Rc<dyn LlmAdapter>,
    provider: LlmProviderInfo,
    retry_policy: ResolvedRetryPolicy,
}

type DiscoverFn = dyn Fn(
    LlmModelDiscoveryRequest,
) -> LocalBoxFuture<'static, anyhow::Result<Vec<LlmDiscoveredModel>>>;

/// The abstract `llm` service (upstream `LlmRuntime`).
pub struct LlmRuntime {
    ctx: Context,
    adapters: RefCell<HashMap<String, AdapterRegistration>>,
    adapter_order: RefCell<Vec<String>>,
    directory: RefCell<HashMap<String, LlmConfigurableProvider>>,
    directory_order: RefCell<Vec<String>>,
    discoveries: RefCell<HashMap<String, Rc<DiscoverFn>>>,
}

impl Service for LlmRuntime {
    const NAME: &'static str = "llm";
}

/// What `register_adapter` returns: disposal plus atomic route replacement
/// for the same adapter instance (upstream `AdapterRegistrationHandle`).
pub struct AdapterRegistrationHandle {
    runtime: Rc<LlmRuntime>,
    adapter: Rc<dyn LlmAdapter>,
    owned: Rc<RefCell<HashSet<String>>>,
    released: Rc<Cell<bool>>,
    effect: dsh_cordis::EffectHandle,
}

impl AdapterRegistrationHandle {
    /// Release every route this registration currently holds.
    pub async fn dispose(&self) {
        self.effect.dispose().await;
    }

    /// Replace this registration's routes with `providers`, keeping the same
    /// adapter instance. All-or-nothing: a conflict, invalid name, or bad
    /// metadata fails and leaves current routes untouched. An empty array is
    /// legal here, unlike an empty initial registration.
    pub fn replace(&self, providers: &[String]) -> Result<(), LlmError> {
        if self.released.get() {
            return Err(LlmError::new(
                "a disposed adapter registration cannot replace its routes",
                "REGISTRATION_DISPOSED",
            ));
        }
        let owned = self.owned.borrow().clone();
        let prepared = self
            .runtime
            .prepare_routes(providers, &self.adapter, &owned)?;
        self.runtime.commit_routes(&self.owned, prepared);
        Ok(())
    }
}

/// A live configurable-provider registration (upstream
/// `DirectoryRegistrationHandle`).
pub struct DirectoryRegistrationHandle {
    runtime: Rc<LlmRuntime>,
    held: Rc<RefCell<Vec<LlmConfigurableProvider>>>,
    disposed: Rc<Cell<bool>>,
    effect: dsh_cordis::EffectHandle,
}

impl DirectoryRegistrationHandle {
    /// Withdraw every entry this registration currently holds.
    pub async fn dispose(&self) {
        self.effect.dispose().await;
    }

    /// Replace this registration's entries. All-or-nothing; an empty array is
    /// legal here, unlike an empty initial registration.
    pub fn replace(&self, entries: &[LlmConfigurableProvider]) -> Result<(), LlmError> {
        if self.disposed.get() {
            return Err(LlmError::new(
                "this configurable-provider registration was disposed",
                "REGISTRATION_DISPOSED",
            ));
        }
        self.runtime.commit_directory(&self.held, entries)
    }
}

/// One model call whose config and adapter registration were resolved
/// together (upstream `PreparedLlmCall`). Dispatch at most once through the
/// captured registration.
pub struct PreparedLlmCall {
    /// Detached config with any adapter-owned default materialized.
    pub config: LlmCallConfig,
    /// Immutable retry policy captured with the adapter registration.
    pub retry_policy: ResolvedRetryPolicy,
    /// Detached context metadata resolved with the registration-bound call.
    pub context: Option<LlmModelContext>,
    /// Config fields materialized by the captured adapter rather than
    /// proposed by the caller.
    pub adapter_defaults: LlmCallConfigAdapterDefaults,
    runtime: Rc<LlmRuntime>,
    registration: AdapterRegistration,
    dispatched: Cell<bool>,
}

impl PreparedLlmCall {
    /// Dispatch this call once through the registration captured during
    /// preparation. The request's call-config fields must match `config`;
    /// reuse or mismatch fails with `INVALID_PREPARED_CALL`.
    pub fn stream(&self, options: GenerateOptions) -> Result<ChunkStream, LlmError> {
        if self.dispatched.get() {
            return Err(LlmError::new(
                "a prepared LLM call can only be dispatched once",
                "INVALID_PREPARED_CALL",
            ));
        }
        if !options_match_config(&options, &self.config) {
            return Err(LlmError::new(
                "prepared LLM call config changed before adapter dispatch",
                "INVALID_PREPARED_CALL",
            ));
        }
        self.dispatched.set(true);
        Ok(self.runtime.stream_with_registration(
            options,
            Some((self.registration.clone(), self.config.clone())),
        ))
    }
}

impl LlmRuntime {
    /// Create and register the service in `ctx` (upstream constructor calls
    /// `super(ctx, 'llm')`).
    pub fn provide(ctx: &Context) -> dsh_cordis::Result<Rc<LlmRuntime>> {
        let runtime = Rc::new(LlmRuntime {
            ctx: ctx.clone(),
            adapters: RefCell::new(HashMap::new()),
            adapter_order: RefCell::new(Vec::new()),
            directory: RefCell::new(HashMap::new()),
            directory_order: RefCell::new(Vec::new()),
            discoveries: RefCell::new(HashMap::new()),
        });
        ctx.provide_service(runtime.clone())?;
        Ok(runtime)
    }

    fn emit_adapters_updated(&self) {
        self.ctx.emit::<AdaptersUpdated>(&());
    }

    /// Register an adapter for the given provider routes. All-or-nothing:
    /// `DUPLICATE_ADAPTER` if any provider already has an adapter. Disposed
    /// with the fiber.
    pub fn register_adapter(
        self: &Rc<Self>,
        providers: &[String],
        adapter: Rc<dyn LlmAdapter>,
    ) -> Result<AdapterRegistrationHandle, LlmError> {
        if providers.is_empty() {
            return Err(LlmError::new(
                "an adapter must register at least one provider",
                "INVALID_ADAPTER",
            ));
        }
        let owned: Rc<RefCell<HashSet<String>>> = Rc::default();
        let released: Rc<Cell<bool>> = Rc::default();
        let prepared = self.prepare_routes(providers, &adapter, &owned.borrow())?;
        self.commit_routes(&owned, prepared);

        let runtime = self.clone();
        let owned_for_dispose = owned.clone();
        let released_for_dispose = released.clone();
        let effect = self
            .ctx
            .effect_labeled("llm.registerAdapter()", move |_| {
                Ok(dsh_cordis::Effect::One(dsh_cordis::Disposer::sync(
                    move || {
                        released_for_dispose.set(true);
                        {
                            let mut adapters = runtime.adapters.borrow_mut();
                            let mut order = runtime.adapter_order.borrow_mut();
                            for provider in owned_for_dispose.borrow().iter() {
                                adapters.remove(provider);
                                order.retain(|id| id != provider);
                            }
                        }
                        owned_for_dispose.borrow_mut().clear();
                        runtime.emit_adapters_updated();
                    },
                )))
            })
            .map_err(|error| LlmError::new(error.to_string(), "INVALID_ADAPTER"))?;

        Ok(AdapterRegistrationHandle {
            runtime: self.clone(),
            adapter,
            owned,
            released,
            effect,
        })
    }

    /// Validate one candidate route set, treating routes this registration
    /// already holds as available. Nothing is mutated on rejection.
    fn prepare_routes(
        &self,
        providers: &[String],
        adapter: &Rc<dyn LlmAdapter>,
        owned: &HashSet<String>,
    ) -> Result<Vec<AdapterRegistration>, LlmError> {
        let mut unique = HashSet::new();
        let mut registrations = Vec::new();
        let adapters = self.adapters.borrow();
        for provider in providers {
            if provider.is_empty() {
                return Err(LlmError::new(
                    "adapter provider names must be non-empty",
                    "INVALID_ADAPTER",
                ));
            }
            if unique.contains(provider)
                || (adapters.contains_key(provider) && !owned.contains(provider))
            {
                return Err(LlmError::new(
                    format!("an adapter for provider \"{provider}\" is already registered"),
                    "DUPLICATE_ADAPTER",
                ));
            }
            let info = adapter.provider_info(provider);
            if info.id != *provider || info.name.is_empty() {
                return Err(LlmError::new(
                    format!(
                        "adapter metadata for provider \"{provider}\" must preserve its id and have a non-empty name"
                    ),
                    "INVALID_ADAPTER",
                ));
            }
            unique.insert(provider.clone());
            let retry_policy = match adapter.provider_retry_policy(provider) {
                Some(policy) => policy,
                None => {
                    resolve_retry_policy(None, &format!("llm: provider \"{provider}\" retryPolicy"))
                        .map_err(|reason| LlmError::new(reason, "INVALID_ADAPTER"))?
                }
            };
            registrations.push(AdapterRegistration {
                adapter: adapter.clone(),
                provider: info,
                retry_policy,
            });
        }
        Ok(registrations)
    }

    /// Swap this registration's routes for the prepared ones in one
    /// synchronous section, then announce the topology change.
    fn commit_routes(
        &self,
        owned: &Rc<RefCell<HashSet<String>>>,
        registrations: Vec<AdapterRegistration>,
    ) {
        {
            let mut adapters = self.adapters.borrow_mut();
            let mut order = self.adapter_order.borrow_mut();
            for provider in owned.borrow().iter() {
                adapters.remove(provider);
                order.retain(|id| id != provider);
            }
            let mut next_owned = owned.borrow_mut();
            next_owned.clear();
            for registration in registrations {
                let id = registration.provider.id.clone();
                adapters.insert(id.clone(), registration);
                order.push(id.clone());
                next_owned.insert(id);
            }
        }
        self.emit_adapters_updated();
    }

    /// Describe provider routes with a registered adapter, in registration
    /// order.
    pub fn list_providers(&self) -> Vec<LlmProviderInfo> {
        let adapters = self.adapters.borrow();
        self.adapter_order
            .borrow()
            .iter()
            .filter_map(|id| adapters.get(id).map(|r| r.provider.clone()))
            .collect()
    }

    /// Declare provider routes an adapter plugin can activate through
    /// configuration. All-or-nothing; disposed with the fiber.
    pub fn register_configurable_providers(
        self: &Rc<Self>,
        entries: &[LlmConfigurableProvider],
    ) -> Result<DirectoryRegistrationHandle, LlmError> {
        if entries.is_empty() {
            return Err(LlmError::new(
                "a configurable-provider registration must declare at least one provider",
                "INVALID_DIRECTORY",
            ));
        }
        let held: Rc<RefCell<Vec<LlmConfigurableProvider>>> = Rc::default();
        let disposed: Rc<Cell<bool>> = Rc::default();
        self.commit_directory(&held, entries)?;

        let runtime = self.clone();
        let held_for_dispose = held.clone();
        let disposed_flag = disposed.clone();
        let effect = self
            .ctx
            .effect_labeled("llm.registerConfigurableProviders()", move |_| {
                Ok(dsh_cordis::Effect::One(dsh_cordis::Disposer::sync(
                    move || {
                        disposed_flag.set(true);
                        {
                            let mut directory = runtime.directory.borrow_mut();
                            let mut order = runtime.directory_order.borrow_mut();
                            for entry in held_for_dispose.borrow().iter() {
                                directory.remove(&entry.provider);
                                order.retain(|id| id != &entry.provider);
                            }
                        }
                        held_for_dispose.borrow_mut().clear();
                        runtime.emit_adapters_updated();
                    },
                )))
            })
            .map_err(|error| LlmError::new(error.to_string(), "INVALID_DIRECTORY"))?;

        Ok(DirectoryRegistrationHandle {
            runtime: self.clone(),
            held,
            disposed,
            effect,
        })
    }

    /// Validate a candidate directory set in full, then publish it — the swap
    /// property that keeps `replace` from stranding the directory empty.
    fn commit_directory(
        &self,
        held: &Rc<RefCell<Vec<LlmConfigurableProvider>>>,
        candidates: &[LlmConfigurableProvider],
    ) -> Result<(), LlmError> {
        let mut detached: Vec<LlmConfigurableProvider> = Vec::new();
        {
            let directory = self.directory.borrow();
            let own: HashSet<String> = held
                .borrow()
                .iter()
                .map(|entry| entry.provider.clone())
                .collect();
            for entry in candidates {
                if entry.provider.is_empty()
                    || entry.display_name.is_empty()
                    || entry.settings_ns.is_empty()
                {
                    return Err(LlmError::new(
                        "configurable providers need a non-empty provider, displayName, and settingsNs",
                        "INVALID_DIRECTORY",
                    ));
                }
                if entry.settings_path.iter().any(|segment| segment.is_empty()) {
                    return Err(LlmError::new(
                        format!(
                            "configurable provider \"{}\" has an empty settingsPath segment",
                            entry.provider
                        ),
                        "INVALID_DIRECTORY",
                    ));
                }
                if (directory.contains_key(&entry.provider) && !own.contains(&entry.provider))
                    || detached.iter().any(|seen| seen.provider == entry.provider)
                {
                    return Err(LlmError::new(
                        format!(
                            "configurable provider \"{}\" is already declared",
                            entry.provider
                        ),
                        "DUPLICATE_DIRECTORY",
                    ));
                }
                detached.push(entry.clone());
            }
        }
        {
            let mut directory = self.directory.borrow_mut();
            let mut order = self.directory_order.borrow_mut();
            for entry in held.borrow().iter() {
                directory.remove(&entry.provider);
                order.retain(|id| id != &entry.provider);
            }
            for entry in &detached {
                directory.insert(entry.provider.clone(), entry.clone());
                order.push(entry.provider.clone());
            }
        }
        *held.borrow_mut() = detached;
        self.emit_adapters_updated();
        Ok(())
    }

    /// List every declared configurable provider, registered or dormant, in
    /// declaration order.
    pub fn list_configurable_providers(&self) -> Vec<LlmConfigurableProvider> {
        let directory = self.directory.borrow();
        self.directory_order
            .borrow()
            .iter()
            .filter_map(|id| directory.get(id).cloned())
            .collect()
    }

    /// Offer to interrogate provider endpoints on behalf of the settings
    /// namespace this plugin owns. Disposed with the fiber.
    pub fn register_model_discovery(
        self: &Rc<Self>,
        settings_ns: &str,
        discover: impl Fn(
            LlmModelDiscoveryRequest,
        ) -> LocalBoxFuture<'static, anyhow::Result<Vec<LlmDiscoveredModel>>>
        + 'static,
    ) -> Result<dsh_cordis::EffectHandle, LlmError> {
        if settings_ns.is_empty() {
            return Err(LlmError::new(
                "model discovery needs a non-empty settings namespace",
                "INVALID_DISCOVERY",
            ));
        }
        if self.discoveries.borrow().contains_key(settings_ns) {
            return Err(LlmError::new(
                format!("model discovery for \"{settings_ns}\" is already registered"),
                "DUPLICATE_DISCOVERY",
            ));
        }
        let runtime = self.clone();
        let ns = settings_ns.to_string();
        self.ctx
            .effect_labeled("llm.registerModelDiscovery()", move |_| {
                runtime
                    .discoveries
                    .borrow_mut()
                    .insert(ns.clone(), Rc::new(discover));
                let runtime = runtime.clone();
                Ok(dsh_cordis::Effect::One(dsh_cordis::Disposer::sync(
                    move || {
                        runtime.discoveries.borrow_mut().remove(&ns);
                    },
                )))
            })
            .map_err(|error| LlmError::new(error.to_string(), "INVALID_DISCOVERY"))
    }

    /// Interrogate one provider endpoint for the models it advertises.
    /// Nothing here reads or writes settings or credentials — the caller owns
    /// both; the reply is candidate metadata, deduplicated in endpoint order.
    pub async fn discover_models(
        &self,
        settings_ns: &str,
        request: LlmModelDiscoveryRequest,
    ) -> Result<Vec<LlmDiscoveredModel>, LlmError> {
        let discover = self.discoveries.borrow().get(settings_ns).cloned();
        let Some(discover) = discover else {
            return Err(LlmError::new(
                format!("no model discovery is registered for \"{settings_ns}\""),
                "NO_DISCOVERY",
            ));
        };
        if request.provider.as_deref().unwrap_or("").is_empty()
            && request.base_url.as_deref().unwrap_or("").is_empty()
        {
            return Err(LlmError::new(
                "model discovery needs a provider route or a baseURL",
                "INVALID_DISCOVERY",
            ));
        }
        let discovered = discover(request)
            .await
            .map_err(|error| LlmError::new(error.to_string(), "DISCOVERY_FAILED"))?;
        let mut seen = HashSet::new();
        let mut models = Vec::new();
        for model in discovered {
            if model.id.is_empty() || seen.contains(&model.id) {
                continue;
            }
            seen.insert(model.id.clone());
            models.push(model);
        }
        Ok(models)
    }

    fn registration(&self, provider: &str) -> Result<AdapterRegistration, LlmError> {
        self.adapters
            .borrow()
            .get(provider)
            .cloned()
            .ok_or_else(|| {
                LlmError::new(
                    format!("no adapter registered for provider \"{provider}\""),
                    "NO_ADAPTER",
                )
            })
    }

    /// Resolve the retry policy captured when one provider route was
    /// registered.
    pub fn provider_retry_policy(&self, provider: &str) -> Result<ResolvedRetryPolicy, LlmError> {
        Ok(self.registration(provider)?.retry_policy)
    }

    /// Discover models advertised by one registered provider. Advisory only.
    pub async fn list_models(&self, provider: &str) -> Result<Vec<LlmModelInfo>, LlmError> {
        let registration = self.registration(provider)?;
        let models = registration
            .adapter
            .list_models(provider)
            .await
            .map_err(|error| LlmError::new(error.to_string(), "INVALID_CATALOG"))?;
        let mut seen = HashSet::new();
        for model in &models {
            if model.provider != provider
                || model.id.is_empty()
                || model.name.is_empty()
                || seen.contains(&model.id)
            {
                return Err(LlmError::new(
                    format!(
                        "adapter returned invalid or duplicate model metadata for provider \"{provider}\""
                    ),
                    "INVALID_CATALOG",
                ));
            }
            seen.insert(model.id.clone());
        }
        Ok(models)
    }

    /// Resolve and validate all metadata from the adapter that owns one exact
    /// route (upstream `resolveModelInfo`).
    pub async fn resolve_model_info(
        &self,
        provider: &str,
        model: &str,
        signal: Option<dsh_timeout::AbortSignal>,
    ) -> Result<LlmResolvedModelInfo, LlmError> {
        let registration = self.registration(provider)?;
        self.resolve_model_info_for(&registration, model, signal)
            .await
    }

    async fn resolve_model_info_for(
        &self,
        registration: &AdapterRegistration,
        model: &str,
        signal: Option<dsh_timeout::AbortSignal>,
    ) -> Result<LlmResolvedModelInfo, LlmError> {
        let provider = registration.provider.id.clone();
        let resolved = registration
            .adapter
            .resolve_model(&provider, model, signal)
            .await
            .map_err(|error| LlmError::new(error.to_string(), "INVALID_MODEL_INFO"))?;
        if resolved.provider != provider || resolved.id != model || resolved.name.is_empty() {
            return Err(LlmError::new(
                format!(
                    "adapter returned invalid exact model metadata for provider \"{provider}\" model \"{model}\""
                ),
                "INVALID_MODEL_INFO",
            ));
        }
        if let Some(context) = &resolved.context {
            if context.context_window == 0 {
                return Err(LlmError::new(
                    format!(
                        "adapter returned invalid context metadata for provider \"{provider}\" model \"{model}\""
                    ),
                    "INVALID_MODEL_CONTEXT",
                ));
            }
        }
        if let Some(default_max_tokens) = resolved.default_max_tokens {
            if default_max_tokens == 0 {
                return Err(LlmError::new(
                    format!(
                        "adapter returned invalid default maxTokens for provider \"{provider}\" model \"{model}\""
                    ),
                    "INVALID_MODEL_MAX_TOKENS",
                ));
            }
        }
        if let Some(reasoning) = &resolved.reasoning {
            if reasoning.efforts.is_empty() {
                return Err(LlmError::new(
                    format!(
                        "adapter returned invalid reasoning metadata for provider \"{provider}\" model \"{model}\""
                    ),
                    "INVALID_MODEL_REASONING",
                ));
            }
            let mut seen = HashSet::new();
            for effort in &reasoning.efforts {
                if effort.id.as_str().is_empty()
                    || effort.name.is_empty()
                    || seen.contains(effort.id.as_str())
                {
                    return Err(LlmError::new(
                        format!(
                            "adapter returned invalid or duplicate reasoning effort metadata for provider \"{provider}\" model \"{model}\""
                        ),
                        "INVALID_MODEL_REASONING",
                    ));
                }
                seen.insert(effort.id.as_str().to_string());
            }
            if let Some(default_effort) = &reasoning.default_effort {
                if !seen.contains(default_effort.as_str()) {
                    return Err(LlmError::new(
                        format!(
                            "adapter returned an unknown default reasoning effort for provider \"{provider}\" model \"{model}\""
                        ),
                        "INVALID_MODEL_REASONING",
                    ));
                }
            }
        }
        Ok(resolved)
    }

    /// Validate a conversation call config against exact-model capability and
    /// materialize adapter defaults. This standalone query does not bind a
    /// later dispatch; use [`LlmRuntime::prepare_call`] when logging and
    /// streaming must share one adapter registration.
    pub async fn resolve_call_config(
        &self,
        config: &LlmCallConfig,
        signal: Option<dsh_timeout::AbortSignal>,
    ) -> Result<LlmCallConfig, LlmError> {
        let registration = self.registration(&config.provider)?;
        Ok(self
            .resolve_call_for(&registration, config, signal)
            .await?
            .0)
    }

    async fn resolve_call_for(
        &self,
        registration: &AdapterRegistration,
        config: &LlmCallConfig,
        signal: Option<dsh_timeout::AbortSignal>,
    ) -> Result<(LlmCallConfig, Option<LlmModelContext>), LlmError> {
        let info = self
            .resolve_model_info_for(registration, &config.model, signal)
            .await?;
        let mut resolved = config.clone();
        if resolved.max_tokens.is_none() {
            resolved.max_tokens = info.default_max_tokens;
        }
        let requested = resolved.reasoning_effort.clone();
        match &info.reasoning {
            None => {
                if let Some(requested) = requested {
                    return Err(LlmError::new(
                        format!(
                            "provider \"{}\" model \"{}\" does not support reasoning effort \"{}\"",
                            config.provider,
                            config.model,
                            requested.as_str()
                        ),
                        "UNSUPPORTED_REASONING_EFFORT",
                    ));
                }
            }
            Some(reasoning) => {
                let effective: Option<ReasoningEffortId> = requested
                    .clone()
                    .or_else(|| reasoning.default_effort.clone());
                if let Some(effective) = effective {
                    if !reasoning
                        .efforts
                        .iter()
                        .any(|effort| effort.id == effective)
                    {
                        return Err(LlmError::new(
                            format!(
                                "provider \"{}\" model \"{}\" does not support reasoning effort \"{}\"",
                                config.provider,
                                config.model,
                                effective.as_str()
                            ),
                            "UNSUPPORTED_REASONING_EFFORT",
                        ));
                    }
                    resolved.reasoning_effort = Some(effective);
                }
            }
        }
        Ok((resolved, info.context))
    }

    /// Resolve one call under its current adapter registration. The returned
    /// one-shot handle keeps that registration across header logging and
    /// dispatch, so a reload cannot combine one adapter's capability result
    /// with another adapter.
    pub async fn prepare_call(
        self: &Rc<Self>,
        config: &LlmCallConfig,
        signal: Option<dsh_timeout::AbortSignal>,
    ) -> Result<PreparedLlmCall, LlmError> {
        let registration = self.registration(&config.provider)?;
        let (resolved, context) = self.resolve_call_for(&registration, config, signal).await?;
        let adapter_defaults = LlmCallConfigAdapterDefaults {
            reasoning_effort: (config.reasoning_effort.is_none()
                && resolved.reasoning_effort.is_some())
            .then_some(true),
            max_tokens: (config.max_tokens.is_none() && resolved.max_tokens.is_some())
                .then_some(true),
        };
        Ok(PreparedLlmCall {
            config: resolved,
            retry_policy: registration.retry_policy.clone(),
            context,
            adapter_defaults,
            runtime: self.clone(),
            registration,
            dispatched: Cell::new(false),
        })
    }

    /// Remove replay state whose historical route is owned by another adapter
    /// (upstream `forAdapter`).
    fn for_adapter(
        &self,
        mut options: GenerateOptions,
        adapter: &Rc<dyn LlmAdapter>,
    ) -> GenerateOptions {
        let adapters = self.adapters.borrow();
        for message in &mut options.messages {
            if let MessageSource::Model(provenance) = &mut message.source {
                if provenance.replay_state.is_none() {
                    continue;
                }
                let same_adapter = adapters
                    .get(&provenance.provider)
                    .map(|registration| Rc::ptr_eq(&registration.adapter, adapter))
                    .unwrap_or(false);
                if !same_adapter {
                    provenance.replay_state = None;
                }
            }
        }
        options
    }

    /// Final adapter boundary: adapter selection, dispatch, and iteration
    /// failures become one terminal failure chunk. Middleware and downstream
    /// consumer failures remain ordinary errors.
    fn adapter_stream(
        self: Rc<Self>,
        options: GenerateOptions,
        prepared: Option<(AdapterRegistration, LlmCallConfig)>,
    ) -> ChunkStream {
        let signal = options.signal.clone();
        stream! {
            let setup: Result<AdapterStream, LlmError> = async {
                let (registration, resolved_config) = match &prepared {
                    Some((registration, config)) => {
                        if !options_match_config(&options, config) {
                            return Err(LlmError::new(
                                "prepared LLM call config changed before adapter dispatch",
                                "INVALID_PREPARED_CALL",
                            ));
                        }
                        (registration.clone(), config.clone())
                    }
                    None => {
                        let registration = self.registration(&options.provider)?;
                        let (config, _) = self
                            .resolve_call_for(&registration, &LlmCallConfig::of(&options), options.signal.clone())
                            .await?;
                        (registration, config)
                    }
                };
                let mut resolved_options = options.clone();
                resolved_config.apply_to(&mut resolved_options);
                let adapter = registration.adapter.clone();
                let request = self.for_adapter(resolved_options, &adapter);
                Ok(adapter.stream(request))
            }
            .await;

            let mut inner = match setup {
                Ok(stream) => stream,
                Err(error) => {
                    yield failure_chunk(&anyhow::Error::new(error), signal.as_ref());
                    return;
                }
            };
            while let Some(item) = inner.next().await {
                match item {
                    Ok(chunk) => yield chunk,
                    Err(error) => {
                        yield failure_chunk(&error, signal.as_ref());
                        return;
                    }
                }
            }
        }
        .boxed_local()
    }

    /// Stream one model call as raw chunks, wrapped by `llm/stream` waterfall
    /// listeners (upstream `stream`).
    pub fn stream(self: &Rc<Self>, options: GenerateOptions) -> ChunkStream {
        self.stream_with_registration(options, None)
    }

    fn stream_with_registration(
        self: &Rc<Self>,
        options: GenerateOptions,
        prepared: Option<(AdapterRegistration, LlmCallConfig)>,
    ) -> ChunkStream {
        let signal = options.signal.clone();
        let runtime = self.clone();
        let chain = self
            .ctx
            .waterfall::<LlmStream, _, _>(options, move |options| {
                let runtime = runtime.clone();
                async move { Ok(runtime.adapter_stream(options, prepared)) }
            });
        stream! {
            match chain.await {
                Ok(mut inner) => {
                    while let Some(chunk) = inner.next().await {
                        yield chunk;
                    }
                }
                Err(error) => {
                    yield failure_chunk(&error, signal.as_ref());
                }
            }
        }
        .boxed_local()
    }
}

/// Convert one adapter failure into the stream protocol's terminal outcome
/// (upstream `adapterFailureChunk`).
fn failure_chunk(error: &anyhow::Error, signal: Option<&dsh_timeout::AbortSignal>) -> StreamChunk {
    let failure = normalize_llm_failure(error);
    let aborted =
        signal.map(|signal| signal.aborted()).unwrap_or(false) || failure.code == "ABORTED";
    StreamChunk::Finish {
        reason: if aborted {
            FinishReason::Aborted { failure }
        } else {
            FinishReason::Error { failure }
        },
        replay_state: None,
    }
}
