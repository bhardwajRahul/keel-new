//! Rust port of `packages/core/system-prompt`: the `systemPrompt` registry
//! service collecting ordered prompt sections, dynamic runtime context, tool
//! schemas, and `{{variable}}` values, assembled before each model step and
//! rendered by [`render_prompt`] / [`render_context_snapshot`].
//!
//! Divergences from the TypeScript package (contract-level; forced by the
//! Rust host or the dsh ports):
//! - No service property proxy: registration methods take the calling
//!   [`Context`] explicitly (for example [`SystemPrompt::section`]), which
//!   supplies the scope tag and the effect ownership upstream reads off
//!   `this.ctx`.
//! - Registrations return the cordis [`EffectHandle`] instead of a bare
//!   disposer function, and fallible APIs return `Result` instead of
//!   throwing.
//! - `system-prompt/change` is fire-and-forget (`ctx.emit` spawns listener
//!   futures), so a failing change listener cannot roll a registration back;
//!   upstream's rollback-on-listener-throw path has no counterpart.
//! - Scope-filtered dispatch of the assemble waterfall is library-owned
//!   (as in dsh-scope): the event args carry a [`Scoped`] carrier and
//!   listeners register through [`on_assemble`], which applies the
//!   admission check and hides the carrier.
//! - [`AssembleContext`] is a closed struct: upstream's merge-extensible
//!   fields collapse into the opaque `payload` slot, and `signal` is not
//!   ported (upstream never reads it).
//! - One assembly's variable pass re-reads the table between entries
//!   instead of using upstream's generation-scoped live Map iterators; the
//!   observable contract holds (a variable registered mid-pass joins the
//!   same assembly, a replacement for an already-visited name defers to the
//!   next one).
//! - `variables` is a `BTreeMap`, so the unknown-variable diagnostic lists
//!   registered names sorted rather than in insertion order.
//! - Upstream's `structuredClone` detach of tool parameters is subsumed by
//!   ownership: providers hand over owned `ToolSchema`s.

pub mod invariant;

use dsh_cordis::{Context, EffectHandle, Event, EventOptions, Plugin, Service, validate_as};
use dsh_llm::{ContextSnapshotSection, ToolSchema};
use dsh_scope::{
    AnonymousEntries, NamedEntries, ScopeKey, ScopeLayer, Scoped, ScopedLayers, scope_target,
};
use futures::FutureExt;
use futures::future::LocalBoxFuture;
use serde_json::Value;
use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::rc::Rc;

/// The deployment persona's reserved section name. Exported so a composition
/// can claim the same slot: a preset that registers a scoped section under
/// this name replaces the deployment persona instead of duplicating it.
pub const PERSONA_SECTION: &str = "deployment:persona";

/// Prompt order of the persona slot; the first section a model reads.
pub const PERSONA_ORDER: f64 = 0.0;

/// Reserved [`Config::tool_order`] marker naming the position of every
/// unlisted tool.
pub const TOOL_ORDER_REST: &str = "<unlisted-tools>";

/// Human-readable form of the variable-name rule, used in diagnostics.
const VARIABLE_NAME_PATTERN: &str = "/^[a-z][a-z0-9_]*$/";

/// Built-in harness identity section text (order −100).
const HARNESS_IDENTITY: &str = "You are an AI agent powered by DeepSeek Harness.";

/// Whether `name` is writable between the braces of a `{{name}}` reference.
fn is_variable_name(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some('a'..='z'))
        && chars.all(|c| matches!(c, 'a'..='z' | '0'..='9' | '_'))
}

/// Per-assembly inputs handed to every provider and waterfall listener.
#[derive(Clone, Default)]
pub struct AssembleContext {
    /// Scope whose providers and scoped listeners participate; `None` keeps
    /// the assembly to global providers and untagged listeners.
    pub scope: Option<ScopeKey>,
    /// Caller-defined per-assembly data (upstream's merge-extensible fields);
    /// providers downcast it to what they and the caller agreed on.
    pub payload: Option<Rc<dyn Any>>,
}

/// Section text: fixed prose or a provider evaluated per assembly. Either
/// form may contain `{{variable}}` references, interpolated later by
/// [`render_prompt`].
#[derive(Clone)]
pub enum PromptText {
    Fixed(String),
    Provider(Rc<dyn Fn(&AssembleContext) -> String>),
}

impl PromptText {
    /// Wrap fixed text.
    pub fn fixed(text: impl Into<String>) -> PromptText {
        PromptText::Fixed(text.into())
    }

    /// Wrap a provider evaluated with each assembly's context.
    pub fn provider(f: impl Fn(&AssembleContext) -> String + 'static) -> PromptText {
        PromptText::Provider(Rc::new(f))
    }

    fn resolve(&self, context: &AssembleContext) -> String {
        match self {
            PromptText::Fixed(text) => text.clone(),
            PromptText::Provider(f) => f(context),
        }
    }
}

/// One contributed system-prompt section (registry input).
pub struct PromptSection {
    /// Unique per layer; a duplicate registration fails.
    pub name: String,
    /// Sections concatenate in ascending order. Convention: −100 is the
    /// harness identity, 0 the deployment persona, 100–199 tool guidance;
    /// other negative orders also precede the persona.
    pub order: f64,
    /// Fixed text or a per-assembly provider.
    pub text: PromptText,
    /// Treat this contribution as the complete system prompt: assembly still
    /// runs the waterfall (tools, contexts, and variables resolve), then this
    /// exact section is restored as the sole prompt section. Two effective
    /// complete sections fail the assembly.
    pub complete: bool,
}

/// One contributed dynamic-context entry, materialized into the runtime
/// snapshot (registry input).
pub struct PromptContext {
    /// Unique per layer; a duplicate registration fails.
    pub name: String,
    /// Contexts join in ascending order.
    pub order: f64,
    /// Fixed text or a per-assembly provider; empty text contributes nothing.
    pub text: PromptText,
}

/// A [`PromptSection`] with its text resolved (not yet interpolated).
#[derive(Clone, Debug, PartialEq)]
pub struct AssembledSection {
    pub name: String,
    pub text: String,
}

/// A [`PromptContext`] with its text resolved (not yet interpolated).
#[derive(Clone, Debug, PartialEq)]
pub struct AssembledContext {
    pub name: String,
    pub text: String,
}

/// One tool provider's contribution to an assembly.
pub struct ToolProviderResult {
    /// Schemas visible in THIS assembly.
    pub schemas: Vec<ToolSchema>,
    /// Pre-restriction name universe for `tool_order` validation; `None`
    /// defaults to the schemas' names.
    pub known_names: Option<Vec<String>>,
}

/// Assembled model input. Sections and contexts stay uninterpolated until
/// rendered; tools are already in canonical order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PromptAssembly {
    pub sections: Vec<AssembledSection>,
    pub contexts: Vec<AssembledContext>,
    pub tools: Vec<ToolSchema>,
    /// Registered variable values; `None` marks a name registered without a
    /// value for this assembly (referencing it fails at render).
    pub variables: BTreeMap<String, Option<String>>,
}

/// Waterfall over the assembled prompt (upstream `system-prompt/assemble`).
/// Args carry the scope carrier; register through [`on_assemble`] for the
/// filtered listener contract. The resolved value is authoritative, except
/// that an effective complete section is restored afterwards.
pub struct Assemble;
impl Event for Assemble {
    const NAME: &'static str = "system-prompt/assemble";
    type Args = (Scoped<()>, PromptAssembly, AssembleContext);
    type Ret = PromptAssembly;
}

/// Emitted when any prompt provider registers or is disposed. Unfiltered:
/// a registry change affects every scope (upstream `system-prompt/change`).
pub struct Change;
impl Event for Change {
    const NAME: &'static str = "system-prompt/change";
    type Args = ();
    type Ret = ();
}

/// Continuation delegating to the rest of the assemble waterfall.
pub type AssembleNext = Box<
    dyn FnOnce(
        PromptAssembly,
        AssembleContext,
    ) -> LocalBoxFuture<'static, anyhow::Result<PromptAssembly>>,
>;

/// Register an assemble-waterfall listener owned by `ctx`'s fiber. A listener
/// on a scope-tagged context receives only that scope's assemblies (others
/// pass through untouched); an untagged listener receives every assembly, and
/// `options.global` skips filtering entirely. The listener MUST call `next`
/// to delegate; returning without it makes its value authoritative.
pub fn on_assemble<F, Fut>(
    ctx: &Context,
    options: EventOptions,
    listener: F,
) -> dsh_cordis::Result<EffectHandle>
where
    F: Fn(&Context, PromptAssembly, AssembleContext, AssembleNext) -> Fut + 'static,
    Fut: std::future::Future<Output = anyhow::Result<PromptAssembly>> + 'static,
{
    let registered = ctx.clone();
    ctx.on_waterfall::<Assemble, _, _>(options, move |ctx, (carrier, assembly, context), next| {
        if options.global || carrier.admits(&registered) {
            let wrapped: AssembleNext =
                Box::new(move |assembly, context| next((carrier, assembly, context)));
            futures::future::Either::Left(listener(ctx, assembly, context, wrapped))
        } else {
            futures::future::Either::Right(next((carrier, assembly, context)))
        }
    })
}

/// Plugin config: the deployment-authored fragment of the system prompt.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Config {
    /// Prepend the fixed harness identity before the persona (default true).
    pub include_harness_identity: bool,
    /// Materialize dynamic runtime-context snapshots (default true).
    pub include_runtime_context: bool,
    /// Deployment-wide order-0 persona template; a scoped section named
    /// [`PERSONA_SECTION`] shadows it. `{{variable}}` references are strict.
    pub persona: String,
    /// Model-facing tool names in order, containing [`TOOL_ORDER_REST`]
    /// exactly once. Structural mistakes fail at load, unknown names at
    /// assembly; a known name hidden for one scope is simply absent there.
    /// `None` means lexicographic order.
    pub tool_order: Option<Vec<String>>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            include_harness_identity: true,
            include_runtime_context: true,
            persona: String::new(),
            tool_order: None,
        }
    }
}

/// Interpolate strict `{{variable}}` references in every section, drop empty
/// results, and join the rest with blank lines. Malformed, unknown, or
/// valueless references fail; a lone `{{` with no later `}}` stays literal,
/// and substituted values are never re-scanned.
pub fn render_prompt(assembly: &PromptAssembly) -> anyhow::Result<String> {
    let mut rendered = Vec::new();
    for section in &assembly.sections {
        let text = interpolate(&section.name, &section.text, &assembly.variables, "section")?;
        if !text.is_empty() {
            rendered.push(text);
        }
    }
    Ok(rendered.join("\n\n"))
}

/// Render the complete dynamic-context snapshot, or `""` when no context is
/// active.
pub fn render_context_snapshot(assembly: &PromptAssembly) -> anyhow::Result<String> {
    Ok(join_context_sections(&render_context_sections(assembly)?))
}

/// The snapshot as the named contributions it was assembled from, one entry
/// per context that rendered non-empty; lets a consumer attribute each part
/// without re-splitting the joined prose.
pub fn render_context_sections(
    assembly: &PromptAssembly,
) -> anyhow::Result<Vec<ContextSnapshotSection>> {
    let mut sections = Vec::new();
    for context in &assembly.contexts {
        let text = interpolate(&context.name, &context.text, &assembly.variables, "context")?;
        if !text.is_empty() {
            sections.push(ContextSnapshotSection {
                name: context.name.clone(),
                text,
            });
        }
    }
    Ok(sections)
}

/// The model-facing snapshot text for already-rendered sections, so a caller
/// needing both does not interpolate twice; `""` when no context is active.
pub fn join_context_sections(sections: &[ContextSnapshotSection]) -> String {
    let body = sections
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    if body.is_empty() {
        return String::new();
    }
    format!(
        "Current runtime context. This snapshot supersedes earlier runtime-context snapshots.\n\n{body}"
    )
}

/// Byte length and inner name of a complete simple `{{name}}` group at the
/// start of `text`; `None` when no complete group starts here.
fn group_at(text: &str) -> Option<(&str, usize)> {
    let rest = text.strip_prefix("{{")?;
    let special = rest.find(['{', '}'])?;
    if rest[special..].starts_with("}}") {
        Some((&rest[..special], 2 + special + 2))
    } else {
        None
    }
}

/// Interpolate one section or context, attributing diagnostics to `owner`.
fn interpolate(
    owner: &str,
    text: &str,
    variables: &BTreeMap<String, Option<String>>,
    kind: &str,
) -> anyhow::Result<String> {
    let mut result = String::new();
    let mut last = 0usize;
    while let Some(found) = text[last..].find("{{") {
        let open = last + found;
        let Some((name, len)) = group_at(&text[open..]) else {
            // A later closing pair makes this malformed; otherwise it is
            // literal prose.
            if text[open + 2..].contains("}}") {
                let snippet: String = text[open..].chars().take(16).collect();
                anyhow::bail!(
                    "malformed prompt variable reference at \"{snippet}…\" in {kind} {owner:?} (references are complete simple {{{{name}}}} groups)"
                );
            }
            result.push_str(&text[last..open + 2]);
            last = open + 2;
            continue;
        };
        // `{{}}` yields the empty name and lands on the malformed path.
        if !is_variable_name(name) {
            anyhow::bail!(
                "malformed prompt variable reference \"{{{{{name}}}}}\" in {kind} {owner:?} (variable names match {VARIABLE_NAME_PATTERN})"
            );
        }
        match variables.get(name) {
            None => {
                let known = variables.keys().cloned().collect::<Vec<_>>().join(", ");
                let known = if known.is_empty() {
                    "(none)".to_string()
                } else {
                    known
                };
                anyhow::bail!(
                    "unknown prompt variable \"{{{{{name}}}}}\" in {kind} {owner:?}; registered variables: {known}"
                );
            }
            Some(None) => {
                anyhow::bail!(
                    "prompt variable \"{{{{{name}}}}}\" has no value for this assembly ({kind} {owner:?})"
                );
            }
            Some(Some(value)) => {
                result.push_str(&text[last..open]);
                result.push_str(value);
                last = open + len;
            }
        }
    }
    result.push_str(&text[last..]);
    Ok(result)
}

/// Reject duplicate names and require the [`TOOL_ORDER_REST`] marker.
/// Registered-name checks wait for assembly, when providers exist.
fn validate_tool_order(tool_order: Option<Vec<String>>) -> anyhow::Result<Option<Vec<String>>> {
    let Some(order) = tool_order else {
        return Ok(None);
    };
    let mut seen = HashSet::new();
    for name in &order {
        if !seen.insert(name.as_str()) {
            anyhow::bail!("toolOrder lists {name:?} more than once");
        }
    }
    if !seen.contains(TOOL_ORDER_REST) {
        anyhow::bail!(
            "toolOrder must contain the {TOOL_ORDER_REST:?} rest entry (where unlisted tools are inserted)"
        );
    }
    Ok(Some(order))
}

/// Apply the configured order, inserting unlisted tools lexicographically at
/// the rest marker. Unknown configured names fail; known-but-restricted names
/// may be absent. Without an order, sort lexicographically (stable, so tools
/// sharing a name keep collection order).
fn order_tools(
    mut tools: Vec<ToolSchema>,
    tool_order: Option<&[String]>,
    known: &BTreeSet<String>,
) -> anyhow::Result<Vec<ToolSchema>> {
    if tools.iter().any(|tool| tool.name == TOOL_ORDER_REST) {
        anyhow::bail!(
            "tool provider returned reserved tool name {TOOL_ORDER_REST:?} (reserved for toolOrder's rest entry)"
        );
    }
    let Some(order) = tool_order else {
        tools.sort_by(|a, b| a.name.cmp(&b.name));
        return Ok(tools);
    };
    let unknown: Vec<&String> = order
        .iter()
        .filter(|name| name.as_str() != TOOL_ORDER_REST && !known.contains(name.as_str()))
        .collect();
    if !unknown.is_empty() {
        let listed = unknown
            .iter()
            .map(|name| format!("{name:?}"))
            .collect::<Vec<_>>()
            .join(", ");
        let known_list = if known.is_empty() {
            "(none)".to_string()
        } else {
            known.iter().cloned().collect::<Vec<_>>().join(", ")
        };
        anyhow::bail!(
            "toolOrder lists unregistered tool{} {listed}; known tools: {known_list}",
            if unknown.len() > 1 { "s" } else { "" }
        );
    }
    let listed: HashSet<&str> = order.iter().map(String::as_str).collect();
    let mut rest: Vec<ToolSchema> = tools
        .iter()
        .filter(|tool| !listed.contains(tool.name.as_str()))
        .cloned()
        .collect();
    rest.sort_by(|a, b| a.name.cmp(&b.name));
    let mut result = Vec::new();
    for name in order {
        if name == TOOL_ORDER_REST {
            result.append(&mut rest);
        } else {
            result.extend(tools.iter().filter(|tool| tool.name == *name).cloned());
        }
    }
    Ok(result)
}

type ToolProvider = Rc<dyn Fn(&AssembleContext) -> ToolProviderResult>;
type VariableProvider = Rc<dyn Fn(&AssembleContext) -> Option<String>>;

/// All prompt registrations owned by one global or scoped layer.
struct PromptLayer {
    sections: NamedEntries<Rc<PromptSection>>,
    contexts: NamedEntries<Rc<PromptContext>>,
    runtime_context_suppressors: AnonymousEntries<()>,
    tool_providers: AnonymousEntries<ToolProvider>,
    variables: NamedEntries<VariableProvider>,
}

impl PromptLayer {
    fn new(scoped: bool) -> PromptLayer {
        fn dup(
            kind: &'static str,
            hint: &'static str,
            scoped: bool,
        ) -> impl Fn(&str) -> anyhow::Error {
            move |name| {
                if scoped {
                    anyhow::anyhow!("prompt {kind} {name:?} is already registered in this scope")
                } else {
                    anyhow::anyhow!(
                        "prompt {kind} {name:?} is already registered (for a per-agent {hint}, register through that agent's `agent.ctx` instead)"
                    )
                }
            }
        }
        PromptLayer {
            sections: NamedEntries::new(dup("section", "override", scoped)),
            contexts: NamedEntries::new(dup("context", "override", scoped)),
            runtime_context_suppressors: AnonymousEntries::new(),
            tool_providers: AnonymousEntries::new(),
            variables: NamedEntries::new(dup("variable", "value", scoped)),
        }
    }
}

impl ScopeLayer for PromptLayer {
    fn is_empty(&self) -> bool {
        self.sections.is_empty()
            && self.contexts.is_empty()
            && self.runtime_context_suppressors.is_empty()
            && self.tool_providers.is_empty()
            && self.variables.is_empty()
    }
}

/// Evaluate one layer's variable providers into `variables`, re-reading the
/// table between entries so a variable registered by an earlier provider
/// joins this assembly, while a replacement for an already-visited name
/// defers to the next one.
// ponytail: O(n²) snapshot refresh per pass; registries hold tens of entries.
fn collect_variables(
    table: &NamedEntries<VariableProvider>,
    context: &AssembleContext,
    variables: &mut BTreeMap<String, Option<String>>,
) {
    let mut index = 0;
    loop {
        let Some((name, provider)) = table.entries().into_iter().nth(index) else {
            break;
        };
        variables.insert(name, provider(context));
        index += 1;
    }
}

/// Registry service for the prompt inputs assembled before each model step.
pub struct SystemPrompt {
    ctx: Context,
    layers: ScopedLayers<PromptLayer>,
    tool_order: Option<Vec<String>>,
}

impl Service for SystemPrompt {
    const NAME: &'static str = "systemPrompt";
}

impl SystemPrompt {
    /// Build the service and register its built-in sections (the fixed
    /// harness identity at order −100 unless disabled, and the configured
    /// persona at [`PERSONA_ORDER`]) on `ctx`, the service's own context.
    /// Fails on a structurally invalid `tool_order`.
    pub fn new(ctx: &Context, config: Config) -> anyhow::Result<Rc<SystemPrompt>> {
        let tool_order = validate_tool_order(config.tool_order)?;
        let emit_ctx = ctx.clone();
        let layers = ScopedLayers::new(
            |scope| Ok(PromptLayer::new(scope.is_some())),
            move || {
                emit_ctx.emit::<Change>(&());
                Ok(())
            },
        )?;
        let service = Rc::new(SystemPrompt {
            ctx: ctx.clone(),
            layers,
            tool_order,
        });
        // Harness-owned openers stay independent of the selected loop plugin.
        if config.include_harness_identity {
            service.section(
                ctx,
                PromptSection {
                    name: "harness:identity".into(),
                    order: -100.0,
                    text: PromptText::fixed(HARNESS_IDENTITY),
                    complete: false,
                },
            )?;
        }
        service.section(
            ctx,
            PromptSection {
                name: PERSONA_SECTION.into(),
                order: PERSONA_ORDER,
                text: PromptText::fixed(config.persona),
                complete: false,
            },
        )?;
        if !config.include_runtime_context {
            service.suppress_runtime_context(ctx)?;
        }
        Ok(service)
    }

    /// Register an ordered prompt section in `ctx`'s scope. A scoped section
    /// shadows a same-named global one; a duplicate within one layer or a
    /// non-finite order fails. Registration and disposal emit
    /// `system-prompt/change`; the handle is the exact cordis disposer.
    pub fn section(&self, ctx: &Context, section: PromptSection) -> anyhow::Result<EffectHandle> {
        if !section.order.is_finite() {
            anyhow::bail!(
                "prompt section {:?} order must be a finite number",
                section.name
            );
        }
        let section = Rc::new(section);
        self.layers.effect(
            ctx,
            |layer| layer.sections.insert(section.name.clone(), section.clone()),
            "systemPrompt.section()",
            true,
        )
    }

    /// Register ordered dynamic context in `ctx`'s scope; scoped entries
    /// shadow same-named global ones. Returns the exact cordis disposer.
    pub fn context(&self, ctx: &Context, context: PromptContext) -> anyhow::Result<EffectHandle> {
        if !context.order.is_finite() {
            anyhow::bail!(
                "prompt context {:?} order must be a finite number",
                context.name
            );
        }
        let context = Rc::new(context);
        self.layers.effect(
            ctx,
            |layer| layer.contexts.insert(context.name.clone(), context.clone()),
            "systemPrompt.context()",
            true,
        )
    }

    /// Suppress every dynamic runtime-context contribution for `ctx`'s scope
    /// without touching the services that own those facts. Suppressors stack
    /// and dispose independently. Returns the exact cordis disposer.
    pub fn suppress_runtime_context(&self, ctx: &Context) -> anyhow::Result<EffectHandle> {
        self.layers.effect(
            ctx,
            |layer| Ok(layer.runtime_context_suppressors.append(())),
            "systemPrompt.suppressRuntimeContext()",
            true,
        )
    }

    /// Register a tool-schema provider in `ctx`'s scope; global and matching
    /// scoped providers both contribute. A provider returning the reserved
    /// [`TOOL_ORDER_REST`] name fails the assembly. Returns the exact cordis
    /// disposer.
    pub fn tools(
        &self,
        ctx: &Context,
        provider: impl Fn(&AssembleContext) -> ToolProviderResult + 'static,
    ) -> anyhow::Result<EffectHandle> {
        let provider: ToolProvider = Rc::new(provider);
        self.layers.effect(
            ctx,
            |layer| Ok(layer.tool_providers.append(provider)),
            "systemPrompt.tools()",
            true,
        )
    }

    /// Register a prompt variable in `ctx`'s scope; scoped values shadow
    /// global name-twins, and invalid or duplicate names fail. A provider may
    /// return `None`, but rendering a reference to it then fails. Returns the
    /// exact cordis disposer.
    pub fn variable(
        &self,
        ctx: &Context,
        name: &str,
        provider: impl Fn(&AssembleContext) -> Option<String> + 'static,
    ) -> anyhow::Result<EffectHandle> {
        if !is_variable_name(name) {
            anyhow::bail!(
                "invalid prompt variable name {name:?} (must match {VARIABLE_NAME_PATTERN})"
            );
        }
        let provider: VariableProvider = Rc::new(provider);
        self.layers.effect(
            ctx,
            |layer| layer.variables.insert(name, provider),
            "systemPrompt.variable()",
            true,
        )
    }

    /// Assemble global plus scoped providers, apply canonical tool ordering,
    /// then run the assemble waterfall. Scoped sections and variables shadow
    /// globals. The waterfall's value is authoritative, except that an
    /// effective complete section is restored afterwards as the sole prompt
    /// section, and a suppressed scope keeps its contexts empty.
    pub async fn assemble(&self, context: AssembleContext) -> anyhow::Result<PromptAssembly> {
        let scope = context.scope.clone();
        let chain = self.layers.chain_layers(scope.as_ref());
        let suppressed = !self.layers.global().runtime_context_suppressors.is_empty()
            || chain
                .iter()
                .any(|layer| !layer.runtime_context_suppressors.is_empty());

        // Chain layers come farthest-ancestor first, so the nearest scope
        // wins a variable name.
        let mut variables: BTreeMap<String, Option<String>> = BTreeMap::new();
        collect_variables(&self.layers.global().variables, &context, &mut variables);
        for layer in &chain {
            collect_variables(&layer.variables, &context, &mut variables);
        }

        let section_by_name = self.layers.merge(scope.as_ref(), |layer| &layer.sections);
        let context_by_name = self.layers.merge(scope.as_ref(), |layer| &layer.contexts);

        // Snapshot provider membership before evaluation, then validate the
        // order against pre-restriction names while collecting schemas.
        let providers: Vec<ToolProvider> = self
            .layers
            .global()
            .tool_providers
            .values()
            .into_iter()
            .chain(chain.iter().flat_map(|layer| layer.tool_providers.values()))
            .collect();
        let mut collected: Vec<ToolSchema> = Vec::new();
        let mut known: BTreeSet<String> = BTreeSet::new();
        for provider in providers {
            let result = provider(&context);
            match &result.known_names {
                Some(names) => known.extend(names.iter().cloned()),
                None => known.extend(result.schemas.iter().map(|tool| tool.name.clone())),
            }
            collected.extend(result.schemas);
        }

        let mut definitions: Vec<Rc<PromptSection>> = section_by_name
            .into_iter()
            .map(|(_, section)| section)
            .collect();
        definitions.sort_by(|a, b| a.order.total_cmp(&b.order));
        let complete_definitions: Vec<&Rc<PromptSection>> = definitions
            .iter()
            .filter(|section| section.complete)
            .collect();
        if complete_definitions.len() > 1 {
            anyhow::bail!(
                "multiple complete prompt sections are active: {}",
                complete_definitions
                    .iter()
                    .map(|section| format!("{:?}", section.name))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        let mut complete_section: Option<AssembledSection> = None;
        let sections: Vec<AssembledSection> = definitions
            .iter()
            .map(|definition| {
                let assembled = AssembledSection {
                    name: definition.name.clone(),
                    text: definition.text.resolve(&context),
                };
                if definition.complete {
                    complete_section = Some(assembled.clone());
                }
                assembled
            })
            .collect();

        let contexts: Vec<AssembledContext> = if suppressed {
            Vec::new()
        } else {
            let mut entries: Vec<Rc<PromptContext>> = context_by_name
                .into_iter()
                .map(|(_, entry)| entry)
                .collect();
            entries.sort_by(|a, b| a.order.total_cmp(&b.order));
            entries
                .iter()
                .map(|entry| AssembledContext {
                    name: entry.name.clone(),
                    text: entry.text.resolve(&context),
                })
                .collect()
        };

        let assembly = PromptAssembly {
            sections,
            contexts,
            tools: order_tools(collected, self.tool_order.as_deref(), &known)?,
            variables,
        };

        let carrier = scope_target::<()>(scope.as_ref());
        let transformed = self
            .ctx
            .waterfall::<Assemble, _, _>(
                (carrier, assembly, context),
                |(_carrier, assembly, _context)| async move { Ok(assembly) },
            )
            .await?;

        if complete_section.is_none() && !suppressed {
            return Ok(transformed);
        }
        Ok(PromptAssembly {
            sections: match complete_section {
                Some(section) => vec![section],
                None => transformed.sections,
            },
            contexts: if suppressed {
                Vec::new()
            } else {
                transformed.contexts
            },
            tools: transformed.tools,
            variables: transformed.variables,
        })
    }
}

/// Service plugin providing `systemPrompt` (upstream default export).
pub struct SystemPromptPlugin;

impl Plugin for SystemPromptPlugin {
    fn name(&self) -> Option<String> {
        Some("systemPrompt".into())
    }

    fn validate_config(&self, config: Value) -> dsh_cordis::Result<Value> {
        validate_as::<Config>(config)
    }

    fn apply(&self, ctx: Context, config: Value) -> LocalBoxFuture<'static, anyhow::Result<()>> {
        async move {
            let config: Config = serde_json::from_value(config)?;
            let service = SystemPrompt::new(&ctx, config)?;
            ctx.provide_service(service)?;
            Ok(())
        }
        .boxed_local()
    }
}
