//! The `tools` registry service and execution pipeline (port of upstream
//! `src/index.ts`): scoped registration with shadowing, restrictions and
//! monotonic guards, model-facing schema projection, and the
//! pre/guard/around/post/result dispatch pipeline with cancellation
//! semantics.

use crate::json_schema::{
    JsonSchemaError, assert_supported_json_schema, validate_json_schema_value,
};
use crate::presentation::{ToolCallView, ToolResultView};
use crate::schema::ToolArgsError;
use dsh_agent::AgentRef;
use dsh_cordis::{Context, EffectHandle, Event, EventOptions, Next, Service};
use dsh_llm::{CallId, ContentBlock, HarnessError, Message, ToolSchema};
use dsh_scope::{
    AnonymousEntries, NamedEntries, ScopeKey, ScopeLayer, Scoped, ScopedEvent, ScopedEvents,
    ScopedLayers, scope_of, scope_target,
};
use dsh_timeout::AbortSignal;
use futures::FutureExt;
use futures::future::LocalBoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::rc::Rc;

/// Canonical error code for cancellation after the tool body was invoked.
pub const TOOL_ABORTED: &str = "ABORTED";

/// Canonical error code for cancellation before the tool body was invoked.
pub const TOOL_ABORTED_BEFORE_DISPATCH: &str = "ABORTED_BEFORE_DISPATCH";

/// Reserved for the deferred Code Mode presentation transport: keeping the
/// name unregistrable now means its later arrival cannot collide.
pub const RUN_CODE_NAME: &str = "run_code";

/// The model requested a tool that is not visible to it
/// (code `UNKNOWN_TOOL`).
#[derive(Debug, Clone, thiserror::Error)]
pub struct ToolNotFoundError {
    /// The name the caller asked for.
    pub tool_name: String,
    /// How the model reaches the tool instead, when the name IS visible and
    /// only presentation denies the direct call.
    pub reachable_from: Option<String>,
}

impl std::fmt::Display for ToolNotFoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.reachable_from {
            None => write!(f, "unknown tool \"{}\"", self.tool_name),
            Some(route) => write!(f, "unknown tool \"{}\": {route}", self.tool_name),
        }
    }
}

impl ToolNotFoundError {
    /// Stable machine-routable failure class.
    pub const CODE: &'static str = "UNKNOWN_TOOL";

    pub fn new(tool_name: impl Into<String>) -> Self {
        ToolNotFoundError {
            tool_name: tool_name.into(),
            reachable_from: None,
        }
    }
}

/// A tool body or post-policy value violated its declared output contract
/// (code `INVALID_TOOL_OUTPUT`).
#[derive(Debug, Clone, thiserror::Error)]
#[error("tool \"{tool_name}\" returned invalid output: {}", .violations.join("; "))]
pub struct ToolOutputError {
    pub tool_name: String,
    /// Schema/value violations in validation order.
    pub violations: Vec<String>,
}

impl ToolOutputError {
    /// Stable machine-routable failure class.
    pub const CODE: &'static str = "INVALID_TOOL_OUTPUT";
}

/// Structured error metadata beside the model-facing failure text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolErrorInfo {
    pub name: String,
    pub code: String,
}

/// Canonical failure detail; routing metadata stays optional.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolFailure {
    /// Human-readable message without the `Error: ` envelope.
    pub message: String,
    /// Internal error class/code for policy and durable diagnostics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub info: Option<ToolErrorInfo>,
}

/// Structured `{name, code}` for known error classes in the chain.
fn error_info(error: &anyhow::Error) -> Option<ToolErrorInfo> {
    if error.downcast_ref::<ToolNotFoundError>().is_some() {
        return Some(ToolErrorInfo {
            name: "ToolNotFoundError".into(),
            code: ToolNotFoundError::CODE.into(),
        });
    }
    if error.downcast_ref::<ToolOutputError>().is_some() {
        return Some(ToolErrorInfo {
            name: "ToolOutputError".into(),
            code: ToolOutputError::CODE.into(),
        });
    }
    if error.downcast_ref::<ToolArgsError>().is_some() {
        return Some(ToolErrorInfo {
            name: "ToolArgsError".into(),
            code: ToolArgsError::CODE.into(),
        });
    }
    if error.downcast_ref::<JsonSchemaError>().is_some() {
        return Some(ToolErrorInfo {
            name: "JsonSchemaError".into(),
            code: JsonSchemaError::CODE.into(),
        });
    }
    if let Some(harness) = error.downcast_ref::<HarnessError>() {
        return Some(ToolErrorInfo {
            name: "HarnessError".into(),
            code: harness.code.clone(),
        });
    }
    None
}

fn text_block(text: impl Into<String>) -> ContentBlock {
    ContentBlock::Text { text: text.into() }
}

/// Normalize an arbitrary error into the canonical failure result.
fn tool_error_result(error: &anyhow::Error) -> ToolExecutionResult {
    let message = error.to_string();
    ToolExecutionResult::Failure {
        content: vec![text_block(format!("Error: {message}"))],
        error: ToolFailure {
            message,
            info: error_info(error),
        },
        meta: None,
        additional_contexts: Vec::new(),
    }
}

/// Canonical result when cancellation supersedes success after body
/// invocation.
fn tool_aborted_result(prior: Option<&ToolExecutionResult>) -> ToolExecutionResult {
    ToolExecutionResult::Failure {
        content: vec![text_block("Error: tool call aborted")],
        error: ToolFailure {
            message: "tool call aborted".into(),
            info: Some(ToolErrorInfo {
                name: "AbortError".into(),
                code: TOOL_ABORTED.into(),
            }),
        },
        meta: None,
        additional_contexts: prior
            .map(|p| p.additional_contexts().to_vec())
            .unwrap_or_default(),
    }
}

/// Canonical result when cancellation prevents tool body invocation.
fn tool_aborted_before_dispatch_result(prior: Option<&ToolExecutionResult>) -> ToolExecutionResult {
    ToolExecutionResult::Failure {
        content: vec![text_block("Error: tool call aborted before dispatch")],
        error: ToolFailure {
            message: "tool call aborted before dispatch".into(),
            info: Some(ToolErrorInfo {
                name: "AbortError".into(),
                code: TOOL_ABORTED_BEFORE_DISPATCH.into(),
            }),
        },
        meta: None,
        additional_contexts: prior
            .map(|p| p.additional_contexts().to_vec())
            .unwrap_or_default(),
    }
}

/// One failure message derived from policy feedback without changing its
/// rendered blocks.
fn failure_message_from_content(content: &[ContentBlock]) -> String {
    let joined = content
        .iter()
        .map(|block| match block {
            ContentBlock::Text { text } => text.clone(),
            other => {
                let type_name = serde_json::to_value(other)
                    .ok()
                    .and_then(|v| v.get("type").and_then(Value::as_str).map(str::to_string))
                    .unwrap_or_else(|| "unknown".into());
                format!("[{type_name} content]")
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    if joined.is_empty() {
        "tool result blocked by post-execute policy".into()
    } else {
        joined
    }
}

/// The tool body callback: validated arguments plus the live run context.
pub type ToolExecuteFn =
    Rc<dyn Fn(Value, ToolRunContext) -> LocalBoxFuture<'static, anyhow::Result<Value>>>;
/// Pure `(args, value) -> content` projection.
pub type ToolRenderFn = Rc<dyn Fn(&Value, &Value) -> anyhow::Result<Vec<ContentBlock>>>;
/// Pure `(args, value) -> presentation meta` projection.
pub type ToolMetaFn = Rc<dyn Fn(&Value, &Value) -> anyhow::Result<Value>>;
/// Total last-mile content transform; `None` preserves the content.
pub type ToolFinalizeFn =
    Rc<dyn Fn(&ToolExecution, &ToolExecutionResult) -> Option<Vec<ContentBlock>>>;
/// Monotonic guard: a returned reason denies the call, `None` abstains.
pub type ToolGuardFn = Rc<dyn Fn(&ToolExecution) -> Option<String>>;

/// Tool-owned canonical output contract applied after the body returns.
#[derive(Clone)]
pub struct ToolOutputDefinition {
    /// Raw supported JSON Schema enforced against every successful canonical
    /// value.
    pub schema: Value,
    /// Pure projection from validated arguments and value to model content.
    pub render: ToolRenderFn,
    /// Pure replayable presentation projection, computed only for top-level
    /// calls.
    pub presentation_meta: Option<ToolMetaFn>,
}

/// A registered tool: model-facing schema plus execution and presentation
/// callbacks.
#[derive(Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    /// JSON Schema object for the arguments (the model-facing wire form).
    pub parameters: Map<String, Value>,
    /// Mandatory canonical output declaration.
    pub output: ToolOutputDefinition,
    /// Run one accepted call; must observe `exec.signal()` and settle only
    /// after its owned work quiesces — the registry never abandons the
    /// future but cannot hard-kill same-process code.
    pub execute: ToolExecuteFn,
    /// Snapshot-at-start last-mile content transform, invoked exactly once
    /// per normalized outcome (pipeline failures included).
    pub finalize_content: Option<ToolFinalizeFn>,
    /// Cooperative timeout budget in ms; enforced by a `tools/execute`
    /// wrapper policy and NEVER sent to the model.
    pub timeout_ms: Option<f64>,
    /// Pure overlap classifier; only `true` opts into parallel dispatch.
    pub is_concurrency_safe: Option<Rc<dyn Fn(&Value) -> bool>>,
    /// Pure pending-state presenter (replay-safe).
    pub present_call: Option<Rc<dyn Fn(&Value) -> Option<ToolCallView>>>,
    /// Pure completed-state presenter (replay-safe).
    pub present_result: Option<Rc<dyn Fn(&Value, &ToolResult) -> Option<ToolResultView>>>,
}

/// The completed outcome handed to `present_result`.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolResult {
    /// Final model-facing content (the rendered error text on failure).
    pub content: Vec<ContentBlock>,
    /// Whether the call failed.
    pub is_error: bool,
    /// Tool-private presentation payload from the output projector, when
    /// declared and top-level.
    pub meta: Option<Value>,
}

/// Opaque same-process correlation identity for one execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ToolExecutionToken(u64);

thread_local! {
    static NEXT_TOKEN: Cell<u64> = const { Cell::new(0) };
}

fn mint_token() -> ToolExecutionToken {
    NEXT_TOKEN.with(|next| {
        let id = next.get() + 1;
        next.set(id);
        ToolExecutionToken(id)
    })
}

/// Caller-supplied description of one tool call; the registry mints the
/// correlation token itself.
#[derive(Clone)]
pub struct ToolExecutionInput {
    pub call_id: CallId,
    /// Root model-requested call owning this execution tree; omitted for a
    /// root execution.
    pub root_call_id: Option<CallId>,
    pub name: String,
    /// Parsed arguments (lossless JSON by construction).
    pub arguments: Value,
    /// The agent on whose behalf the call runs (set by the agent loop).
    pub agent: Option<AgentRef>,
    /// Token of the enclosing composite execution for nested sub-dispatches.
    pub parent: Option<ToolExecutionToken>,
    /// Required caller-owned cancellation for this invocation.
    pub signal: AbortSignal,
}

/// Scheduling mode for one pending call: `Parallel` may overlap with
/// siblings, `Exclusive` runs alone as an ordering barrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolExecutionMode {
    Parallel,
    Exclusive,
}

struct ExecInner {
    token: ToolExecutionToken,
    call_id: CallId,
    root_call_id: CallId,
    name: String,
    arguments: Value,
    agent: Option<AgentRef>,
    parent: Option<ToolExecutionToken>,
    /// The wrapper-visible signal; around-dispatch wrappers may replace it
    /// for their delegated lifetime.
    signal: RefCell<AbortSignal>,
    /// The original caller cancellation, immune to wrapper replacement.
    caller_signal: AbortSignal,
    body_invoked: Cell<bool>,
    concludes: Cell<bool>,
    deferred_contexts: RefCell<Vec<Message>>,
    /// Definition-owned final content transform, snapshotted at creation.
    finalizer: Option<ToolFinalizeFn>,
}

/// One pending call inside the registry pipeline. Cheap to clone (shared
/// state); identity is the registry-minted [`token`](Self::token).
#[derive(Clone)]
pub struct ToolExecution {
    inner: Rc<ExecInner>,
}

/// The runtime context handed to a tool body; the same shared execution
/// object the pipeline observes.
pub type ToolRunContext = ToolExecution;

impl ToolExecution {
    /// Registry-assigned correlation identity, shared with nested calls only
    /// as their opaque `parent` token.
    pub fn token(&self) -> ToolExecutionToken {
        self.inner.token
    }

    pub fn call_id(&self) -> &CallId {
        &self.inner.call_id
    }

    /// Root model-requested call, resolved for root and nested executions.
    pub fn root_call_id(&self) -> &CallId {
        &self.inner.root_call_id
    }

    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// Parsed arguments (immutable for the execution's lifetime).
    pub fn arguments(&self) -> &Value {
        &self.inner.arguments
    }

    pub fn agent(&self) -> Option<AgentRef> {
        self.inner.agent.clone()
    }

    pub fn parent(&self) -> Option<ToolExecutionToken> {
        self.inner.parent
    }

    /// The cancellation signal currently visible to the next wrapper or the
    /// tool body.
    pub fn signal(&self) -> AbortSignal {
        self.inner.signal.borrow().clone()
    }

    /// Replace the visible signal for a wrapper's delegated lifetime. The
    /// registry fuses every replacement with the captured caller signal, so
    /// replacement can never detach caller cancellation.
    pub fn set_signal(&self, signal: AbortSignal) {
        *self.inner.signal.borrow_mut() = signal;
    }

    /// The original caller cancellation signal.
    pub fn caller_signal(&self) -> AbortSignal {
        self.inner.caller_signal.clone()
    }

    /// Defer one context until this tool's final result reaches the agent
    /// loop; contexts keep their source and are emitted in call order.
    pub fn defer_context(&self, context: Message) {
        self.inner.deferred_contexts.borrow_mut().push(context);
    }

    /// Mark a successful final result as terminal for the current agent
    /// turn. The marker rides only successful results; a composite forwards
    /// it from a nested success like `additional_contexts`.
    pub fn conclude_turn(&self) {
        self.inner.concludes.set(true);
    }
}

/// The discriminated, execution-local outcome of one tool call.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolExecutionResult {
    Success {
        /// Execution-local canonical value validated by the output schema.
        value: Value,
        content: Vec<ContentBlock>,
        meta: Option<Value>,
        additional_contexts: Vec<Message>,
        /// The agent loop stops after committing this successful batch.
        concludes_turn: bool,
    },
    Failure {
        error: ToolFailure,
        content: Vec<ContentBlock>,
        meta: Option<Value>,
        additional_contexts: Vec<Message>,
    },
}

impl ToolExecutionResult {
    pub fn is_error(&self) -> bool {
        matches!(self, ToolExecutionResult::Failure { .. })
    }

    pub fn content(&self) -> &[ContentBlock] {
        match self {
            ToolExecutionResult::Success { content, .. } => content,
            ToolExecutionResult::Failure { content, .. } => content,
        }
    }

    pub fn additional_contexts(&self) -> &[Message] {
        match self {
            ToolExecutionResult::Success {
                additional_contexts,
                ..
            } => additional_contexts,
            ToolExecutionResult::Failure {
                additional_contexts,
                ..
            } => additional_contexts,
        }
    }

    /// The canonical value of a successful outcome.
    pub fn value(&self) -> Option<&Value> {
        match self {
            ToolExecutionResult::Success { value, .. } => Some(value),
            ToolExecutionResult::Failure { .. } => None,
        }
    }

    pub fn error(&self) -> Option<&ToolFailure> {
        match self {
            ToolExecutionResult::Success { .. } => None,
            ToolExecutionResult::Failure { error, .. } => Some(error),
        }
    }

    pub fn meta(&self) -> Option<&Value> {
        match self {
            ToolExecutionResult::Success { meta, .. } => meta.as_ref(),
            ToolExecutionResult::Failure { meta, .. } => meta.as_ref(),
        }
    }

    pub fn concludes_turn(&self) -> bool {
        matches!(
            self,
            ToolExecutionResult::Success {
                concludes_turn: true,
                ..
            }
        )
    }

    fn with_content(mut self, new_content: Vec<ContentBlock>) -> Self {
        match &mut self {
            ToolExecutionResult::Success { content, .. } => *content = new_content,
            ToolExecutionResult::Failure { content, .. } => *content = new_content,
        }
        self
    }

    fn with_additional_contexts(mut self, contexts: Vec<Message>) -> Self {
        match &mut self {
            ToolExecutionResult::Success {
                additional_contexts,
                ..
            } => *additional_contexts = contexts,
            ToolExecutionResult::Failure {
                additional_contexts,
                ..
            } => *additional_contexts = contexts,
        }
        self
    }
}

/// Pre-dispatch decision: `Allow` runs the call, `Deny` materializes an
/// error, `Ask` runs only after an approval grant and otherwise denies.
/// Input rewriting is excluded — arguments are already logged and presented.
#[derive(Debug, Clone, PartialEq)]
pub enum PreToolDecision {
    Allow,
    Deny { reason: String },
    Ask { reason: Option<String> },
}

/// The one projection an accepting post decision may replace. Replacing both
/// at once is unrepresentable (upstream rejects it at runtime).
#[derive(Debug, Clone, PartialEq)]
pub enum PostAcceptReplacement {
    Content(Vec<ContentBlock>),
    Value(Value),
}

/// Post-dispatch decision: accept (optionally replacing one projection or
/// attaching context) or block by turning corrective feedback into an error
/// result.
#[derive(Debug, Clone, PartialEq)]
pub enum PostToolDecision {
    Accept {
        replace: Option<PostAcceptReplacement>,
        additional_contexts: Vec<Message>,
    },
    Block {
        feedback: Vec<ContentBlock>,
        additional_contexts: Vec<Message>,
    },
}

impl PostToolDecision {
    /// Accept the result unchanged.
    pub fn accept() -> Self {
        PostToolDecision::Accept {
            replace: None,
            additional_contexts: Vec::new(),
        }
    }
}

/// Allow/deny/ask before dispatch (waterfall; `next` delegates toward
/// `Allow`). Scope-filtered: agent-scoped listeners receive only that
/// agent's calls. Register with [`ScopedWaterfall::on_waterfall_scoped`].
pub struct ToolsPreExecute;
impl Event for ToolsPreExecute {
    const NAME: &'static str = "tools/pre-execute";
    type Args = ToolExecution;
    type Ret = PreToolDecision;
}

/// Around-dispatch waterfall for timeout/retry/metrics. `next` returns a
/// normalized result; a wrapper may replace only the execution's signal (the
/// registry re-fuses caller cancellation before the body) and must restore
/// it. Scope-filtered like [`ToolsPreExecute`].
pub struct ToolsExecute;
impl Event for ToolsExecute {
    const NAME: &'static str = "tools/execute";
    type Args = ToolExecution;
    type Ret = ToolExecutionResult;
}

/// Accept, replace, enrich, or block a normalized dispatch result
/// (waterfall; `next` accepts unchanged). Tool failures also flow through
/// here. Scope-filtered like [`ToolsPreExecute`].
pub struct ToolsPostExecute;
impl Event for ToolsPostExecute {
    const NAME: &'static str = "tools/post-execute";
    type Args = (ToolExecution, ToolExecutionResult);
    type Ret = PostToolDecision;
}

/// Observe one final outcome (emit; listener failures are contained by the
/// fire-and-forget dispatch). Scope-filtered: listen with
/// [`ScopedEvents::on_scoped`].
pub struct ToolsResult;
impl Event for ToolsResult {
    const NAME: &'static str = "tools/result";
    type Args = (ToolExecution, ToolExecutionResult);
    type Ret = ();
}

/// The available tool set changed (registration, disposal, or a scoped
/// restriction). Deliberately unfiltered: a global change concerns every
/// agent's next assembly.
pub struct ToolsChange;
impl Event for ToolsChange {
    const NAME: &'static str = "tools/change";
    type Args = ();
    type Ret = ();
}

/// Scope-filtered waterfall registration over the cordis bus, the waterfall
/// twin of `dsh-scope`'s `on_scoped`. A listener registered through a scoped
/// context composes only around dispatches whose carrier admits that scope;
/// non-admitted dispatches pass straight through to the rest of the chain.
// ponytail: lives here because dsh-tools is its only consumer; move to
// dsh-scope when a second registry needs it.
pub trait ScopedWaterfall {
    fn on_waterfall_scoped<E, F, Fut>(
        &self,
        options: EventOptions,
        listener: F,
    ) -> dsh_cordis::Result<EffectHandle>
    where
        E: Event,
        F: Fn(&Context, E::Args, Next<E>) -> Fut + 'static,
        Fut: std::future::Future<Output = anyhow::Result<E::Ret>> + 'static;
}

impl ScopedWaterfall for Context {
    fn on_waterfall_scoped<E, F, Fut>(
        &self,
        options: EventOptions,
        listener: F,
    ) -> dsh_cordis::Result<EffectHandle>
    where
        E: Event,
        F: Fn(&Context, E::Args, Next<E>) -> Fut + 'static,
        Fut: std::future::Future<Output = anyhow::Result<E::Ret>> + 'static,
    {
        let registered = self.clone();
        self.on_waterfall::<ScopedEvent<E>, _, _>(options, move |ctx, (carrier, args), next| {
            let admitted = options.global || carrier.admits(&registered);
            if admitted {
                let rewrap = carrier;
                let user_next: Next<E> = Box::new(move |args| next((rewrap, args)));
                listener(ctx, args, user_next).boxed_local()
            } else {
                next((carrier, args))
            }
        })
    }
}

/// How the registry presents its tools to the model (see [`Config`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolPresentationMode {
    Native,
    Code,
    Both,
}

/// Service config. `Code`/`Both` are recognized but deferred with the Code
/// Mode transport (see the crate doc).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Config {
    /// Model presentation; `Native` (default) sends every visible schema.
    pub mode: Option<ToolPresentationMode>,
    /// Concurrency cap for a `run_code` program's overlapping sub-calls
    /// (default 10); validated now so the field survives the Code Mode port.
    pub max_parallel_sub_calls: Option<u32>,
}

/// Per-scope filter over inherited tools. Restrictions intersect and never
/// mask the scope's own registrations.
#[derive(Debug, Clone, Default)]
pub struct ToolRestriction {
    /// Inherited tool names that stay visible; everything else is removed.
    pub allow: Option<Vec<String>>,
    /// Inherited tool names removed from visibility.
    pub deny: Option<Vec<String>>,
}

/// One restriction compiled at registration for repeated lookup.
struct CompiledToolRestriction {
    allow: Option<HashSet<String>>,
    deny: Option<HashSet<String>>,
}

/// One scope's complete registry contribution.
struct ToolLayer {
    tools: NamedEntries<ToolDefinition>,
    restrictions: AnonymousEntries<Rc<CompiledToolRestriction>>,
    guards: AnonymousEntries<ToolGuardFn>,
}

impl ToolLayer {
    fn new(scope: Option<&ScopeKey>) -> ToolLayer {
        let scoped = scope.is_some();
        ToolLayer {
            tools: NamedEntries::new(move |name| {
                if scoped {
                    anyhow::anyhow!("tool \"{name}\" is already registered in this scope")
                } else {
                    anyhow::anyhow!(
                        "tool \"{name}\" is already registered (for a per-agent variant, register through that agent's `agent.ctx` instead)"
                    )
                }
            }),
            restrictions: AnonymousEntries::new(),
            guards: AnonymousEntries::new(),
        }
    }

    /// Whether every compiled restriction in this layer admits a name.
    fn admits(&self, name: &str) -> bool {
        self.restrictions.values().iter().all(|filter| {
            filter
                .allow
                .as_ref()
                .is_none_or(|allow| allow.contains(name))
                && filter.deny.as_ref().is_none_or(|deny| !deny.contains(name))
        })
    }

    /// First monotonic denial from this layer's guards.
    fn guard_reason(&self, exec: &ToolExecution) -> Option<String> {
        self.guards.values().iter().find_map(|guard| guard(exec))
    }
}

impl ScopeLayer for ToolLayer {
    fn is_empty(&self) -> bool {
        self.tools.is_empty() && self.restrictions.is_empty() && self.guards.is_empty()
    }
}

/// One scope's complete registry view, derived in one layer traversal.
struct ToolView {
    /// Visible definitions after restrictions and scoped shadowing.
    visible: Vec<(String, ToolDefinition)>,
    /// Pre-restriction capability names (prompt-order validation), in
    /// insertion order.
    known_names: Vec<String>,
    /// Inherited names a scoped restriction may name.
    restrictable_names: HashSet<String>,
}

/// Wire schemas plus prompt-order names for one scope; the payload of the
/// system-prompt integration hook.
#[derive(Debug, Clone)]
pub struct ToolProviderResult {
    pub schemas: Vec<ToolSchema>,
    pub known_names: Vec<String>,
}

/// The approval request routed through the optional [`ToolApproval`] seam.
pub struct ApprovalRequest {
    pub agent: AgentRef,
    pub tool_name: String,
    pub call_id: CallId,
    pub reason: Option<String>,
    /// The execution's live signal; answerers should observe it.
    pub signal: AbortSignal,
}

/// The answerer's verdict on one approval request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalOutcome {
    AllowedOnce,
    Rejected,
    Cancelled,
    Unavailable,
}

/// Optional approval seam consumed opportunistically by `Ask` decisions
/// (hosted here until the user-approval package is ported). A deployment
/// that provides none keeps the degrade-to-deny behavior.
pub struct ToolApproval {
    handler: Box<dyn Fn(ApprovalRequest) -> LocalBoxFuture<'static, ApprovalOutcome>>,
}

impl Service for ToolApproval {
    const NAME: &'static str = "approval";
}

impl ToolApproval {
    pub fn new(
        handler: impl Fn(ApprovalRequest) -> LocalBoxFuture<'static, ApprovalOutcome> + 'static,
    ) -> Self {
        ToolApproval {
            handler: Box::new(handler),
        }
    }

    /// Route one request to the answerer.
    pub fn request(&self, request: ApprovalRequest) -> LocalBoxFuture<'static, ApprovalOutcome> {
        (self.handler)(request)
    }
}

/// Scheduler stage after ordered pre-execute and guards: a `PostResult`
/// still receives post-execute, a `FinalResult` bypasses it.
pub enum ScheduledToolPreparation {
    Dispatch {
        exec: ToolExecution,
    },
    PostResult {
        exec: ToolExecution,
        result: ToolExecutionResult,
    },
    FinalResult {
        exec: ToolExecution,
        result: ToolExecutionResult,
    },
}

/// Scheduler stage after around-dispatch.
pub enum ScheduledToolDispatch {
    PostResult { result: ToolExecutionResult },
    FinalResult { result: ToolExecutionResult },
}

/// Tool registry and execution pipeline: scoped registrations shadow
/// globals, and one visibility resolver feeds presentation, lookup, and
/// dispatch.
pub struct ToolRuntime {
    ctx: Context,
    layers: ScopedLayers<ToolLayer>,
    max_parallel_sub_calls: u32,
}

impl Service for ToolRuntime {
    const NAME: &'static str = "tools";
}

impl std::fmt::Debug for ToolRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRuntime")
            .field("max_parallel_sub_calls", &self.max_parallel_sub_calls)
            .finish_non_exhaustive()
    }
}

fn agent_scope(agent: &Option<AgentRef>) -> Option<ScopeKey> {
    agent.as_ref().and_then(|agent| scope_of(&agent.ctx()))
}

impl ToolRuntime {
    /// Create and register the service. `Code`/`Both` presentation modes
    /// fail here: Code Mode is deferred with its embedded JS runtime.
    pub fn provide(ctx: &Context, config: Config) -> anyhow::Result<Rc<ToolRuntime>> {
        match config.mode.unwrap_or(ToolPresentationMode::Native) {
            ToolPresentationMode::Native => {}
            other => anyhow::bail!(
                "dsh-tools: mode {other:?} is not implemented in this port — Code Mode needs an embedded JS runtime; use native"
            ),
        }
        let max_parallel_sub_calls = config.max_parallel_sub_calls.unwrap_or(10);
        if max_parallel_sub_calls < 1 {
            anyhow::bail!("maxParallelSubCalls must be a positive integer");
        }
        let change_ctx = ctx.clone();
        let runtime = Rc::new(ToolRuntime {
            ctx: ctx.clone(),
            layers: ScopedLayers::new(
                |scope| Ok(ToolLayer::new(scope)),
                move || {
                    change_ctx.emit::<ToolsChange>(&());
                    Ok(())
                },
            )?,
            max_parallel_sub_calls,
        });
        ctx.provide_service(runtime.clone())?;
        Ok(runtime)
    }

    /// The validated `run_code` overlap cap (consumed by the deferred Code
    /// Mode transport and the loop scheduler).
    pub fn max_parallel_sub_calls(&self) -> u32 {
        self.max_parallel_sub_calls
    }

    /// Register a tool globally (unscoped `ctx`) or into the calling agent
    /// scope (a scoped `ctx`). Scoped tools shadow globals; duplicates
    /// within one layer and the reserved `run_code` name fail. Returns the
    /// exact effect disposer.
    pub fn register(
        &self,
        ctx: &Context,
        definition: ToolDefinition,
    ) -> anyhow::Result<EffectHandle> {
        assert_supported_json_schema(&definition.output.schema)?;
        if let Some(timeout_ms) = definition.timeout_ms {
            if !timeout_ms.is_finite() || timeout_ms <= 0.0 {
                anyhow::bail!(
                    "tool \"{}\" timeoutMs must be a positive finite number",
                    definition.name
                );
            }
        }
        // Reserved unconditionally so the deferred Code Mode transport can
        // never collide with a user registration when it arrives.
        if definition.name == RUN_CODE_NAME {
            anyhow::bail!(
                "tool name \"{RUN_CODE_NAME}\" is reserved for the Code Mode presentation transport and cannot be registered or shadowed"
            );
        }
        let name = definition.name.clone();
        self.layers.effect(
            ctx,
            move |layer| layer.tools.insert(name, definition),
            "tools.register()",
            true,
        )
    }

    /// Restrict inherited tools for the calling agent scope. Unscoped
    /// contexts, empty filters, reserved names, and names the scope does not
    /// inherit fail. Restrictions intersect; the scope's own registrations
    /// stay visible. Returns the exact effect disposer.
    pub fn restrict(&self, ctx: &Context, filter: ToolRestriction) -> anyhow::Result<EffectHandle> {
        let scope = scope_of(ctx);
        if scope.is_none() {
            anyhow::bail!(
                "tools.restrict() requires a scoped context (agent.ctx): a context-global restriction would mask every agent — deny the tool for the intended agent instead"
            );
        }
        if filter.allow.is_none() && filter.deny.is_none() {
            anyhow::bail!(
                "tools.restrict() with no filter is a no-op: pass allow and/or deny (an empty filter is almost always a materialized-empty-config bug)"
            );
        }
        let named: Vec<&String> = filter
            .allow
            .iter()
            .flatten()
            .chain(filter.deny.iter().flatten())
            .collect();
        if named.iter().any(|name| *name == RUN_CODE_NAME) {
            anyhow::bail!(
                "tools.restrict() cannot name reserved Code Mode presentation transport \"{RUN_CODE_NAME}\"; restrict end-capability tools instead"
            );
        }
        let known = self.view(scope.as_ref()).restrictable_names;
        let unknown: Vec<String> = named
            .iter()
            .filter(|name| !known.contains(**name))
            .map(|name| format!("\"{name}\""))
            .collect();
        if !unknown.is_empty() {
            let plural = if unknown.len() > 1 { "s" } else { "" };
            let mut known_sorted: Vec<String> = known.into_iter().collect();
            known_sorted.sort();
            let known_text = if known_sorted.is_empty() {
                "(none)".to_string()
            } else {
                known_sorted.join(", ")
            };
            anyhow::bail!(
                "tools.restrict() names unknown global tool{plural} {}; known global tools: {known_text}",
                unknown.join(", ")
            );
        }
        let compiled = Rc::new(CompiledToolRestriction {
            allow: filter.allow.map(|names| names.into_iter().collect()),
            deny: filter.deny.map(|names| names.into_iter().collect()),
        });
        self.layers.effect(
            ctx,
            move |layer| Ok(layer.restrictions.append(compiled)),
            "tools.restrict()",
            true,
        )
    }

    /// Register a monotonic guard evaluated after the extensible
    /// `tools/pre-execute` waterfall: any guard may deny with a reason, none
    /// can force-allow. Unscoped `ctx` applies globally; a scoped `ctx`
    /// applies only to that agent. Returns the exact effect disposer.
    pub fn register_guard(
        &self,
        ctx: &Context,
        guard: impl Fn(&ToolExecution) -> Option<String> + 'static,
    ) -> anyhow::Result<EffectHandle> {
        let guard: ToolGuardFn = Rc::new(guard);
        self.layers.effect(
            ctx,
            move |layer| Ok(layer.guards.append(guard)),
            "tools.guard()",
            false,
        )
    }

    /// First monotonic denial: the global layer, then the agent's scope
    /// chain, farthest ancestor first.
    fn guard_reason(&self, exec: &ToolExecution) -> Option<String> {
        if let Some(reason) = self.layers.global().guard_reason(exec) {
            return Some(reason);
        }
        let scope = agent_scope(&exec.inner.agent)?;
        self.layers
            .chain_layers(Some(&scope))
            .iter()
            .find_map(|layer| layer.guard_reason(exec))
    }

    /// Resolve every registry fact one scope needs in one traversal: the
    /// visible map applies restrictions to the INHERITED surface (global +
    /// ancestor layers) and then the scope's own registrations — a scope's
    /// filter never strips what that scope registered for itself.
    fn view(&self, scope: Option<&ScopeKey>) -> ToolView {
        let chain = self.layers.chain_layers(scope);
        let own = self.layers.peek(scope);

        // Inherited surface: global first, then ancestors nearest-last so a
        // nearer scope's same-name entry shadows a farther one.
        let mut inherited: Vec<(String, ToolDefinition)> = self.layers.global().tools.entries();
        for layer in &chain {
            if let Some(own) = &own {
                if Rc::ptr_eq(layer, own) {
                    continue;
                }
            }
            for (name, definition) in layer.tools.entries() {
                match inherited.iter_mut().find(|(existing, _)| *existing == name) {
                    Some(slot) => slot.1 = definition,
                    None => inherited.push((name, definition)),
                }
            }
        }

        let mut visible: Vec<(String, ToolDefinition)> = Vec::new();
        let mut known_names: Vec<String> = Vec::new();
        let mut restrictable_names: HashSet<String> = HashSet::new();
        for (name, definition) in inherited {
            if !known_names.contains(&name) {
                known_names.push(name.clone());
            }
            restrictable_names.insert(name.clone());
            // Restrictions intersect across the whole chain: any scope on it
            // may mask an inherited name for everything nested inside it.
            if chain.iter().all(|layer| layer.admits(&name)) {
                visible.push((name, definition));
            }
        }
        // The scope's own registrations last: they shadow inherited names
        // and sit outside the restriction filter above.
        if let Some(own) = own {
            for (name, definition) in own.tools.entries() {
                if !known_names.contains(&name) {
                    known_names.push(name.clone());
                }
                match visible.iter_mut().find(|(existing, _)| *existing == name) {
                    Some(slot) => slot.1 = definition,
                    None => visible.push((name, definition)),
                }
            }
        }
        ToolView {
            visible,
            known_names,
            restrictable_names,
        }
    }

    /// Look up a tool as one scope sees it: scoped shadows global, and a
    /// restricted-away inherited tool reads as absent.
    pub fn get(&self, name: &str, scope: Option<&ScopeKey>) -> Option<ToolDefinition> {
        self.view(scope)
            .visible
            .into_iter()
            .find_map(|(existing, definition)| (existing == name).then_some(definition))
    }

    /// Project one definition onto the allowlisted model-facing fields —
    /// never callbacks, never `timeout_ms`.
    fn schema_of(&self, definition: &ToolDefinition) -> ToolSchema {
        ToolSchema {
            name: definition.name.clone(),
            description: definition.description.clone(),
            parameters: definition.parameters.clone(),
        }
    }

    /// The visible model-facing schemas for one scope (owned snapshots).
    pub fn schemas(&self, scope: Option<&ScopeKey>) -> Vec<ToolSchema> {
        self.view(scope)
            .visible
            .iter()
            .map(|(_, definition)| self.schema_of(definition))
            .collect()
    }

    /// One scope's wire schemas plus prompt-order names — the payload the
    /// system-prompt wiring consumes (this port exposes the hook instead of
    /// depending on the concurrently ported dsh-system-prompt).
    pub fn wire_schemas(&self, scope: Option<&ScopeKey>) -> ToolProviderResult {
        let view = self.view(scope);
        ToolProviderResult {
            schemas: view
                .visible
                .iter()
                .map(|(_, definition)| self.schema_of(definition))
                .collect(),
            known_names: view.known_names,
        }
    }

    /// A shareable closure over [`Self::wire_schemas`] for the system-prompt
    /// integration to register as its tool provider.
    pub fn schemas_provider(
        self: &Rc<Self>,
    ) -> Rc<dyn Fn(Option<&ScopeKey>) -> ToolProviderResult> {
        let runtime = self.clone();
        Rc::new(move |scope| runtime.wire_schemas(scope))
    }

    /// Fail-closed scheduling classification: only an explicit `true` from a
    /// visible tool's classifier is `Parallel`; unknown, hidden, and
    /// undeclared classifiers are `Exclusive`.
    pub fn execution_mode(&self, input: &ToolExecutionInput) -> ToolExecutionMode {
        let scope = agent_scope(&input.agent);
        let Some(tool) = self.get(&input.name, scope.as_ref()) else {
            return ToolExecutionMode::Exclusive;
        };
        match &tool.is_concurrency_safe {
            Some(classify) if classify(&input.arguments) => ToolExecutionMode::Parallel,
            _ => ToolExecutionMode::Exclusive,
        }
    }

    fn carrier_for(&self, exec: &ToolExecution) -> Scoped<()> {
        scope_target::<()>(agent_scope(&exec.inner.agent).as_ref())
    }

    /// Execute through pre-policy, guards, around-dispatch, post-policy,
    /// content finalization, and final notification. Tool and listener
    /// failures resolve as error results; an invisible tool reports
    /// `UNKNOWN_TOOL`. Cancellation after entry skips a not-yet-started body
    /// with `ABORTED_BEFORE_DISPATCH` or replaces a successful started
    /// outcome with `ABORTED`; started work is always drained first.
    pub async fn execute(self: &Rc<Self>, input: ToolExecutionInput) -> ToolExecutionResult {
        match self.prepare(input).await {
            ScheduledToolPreparation::Dispatch { exec } => match self.dispatch(&exec).await {
                ScheduledToolDispatch::PostResult { result } => self.finalize(&exec, result).await,
                ScheduledToolDispatch::FinalResult { result } => self.finish(&exec, result),
            },
            ScheduledToolPreparation::PostResult { exec, result } => {
                self.finalize(&exec, result).await
            }
            ScheduledToolPreparation::FinalResult { exec, result } => self.finish(&exec, result),
        }
    }

    fn create_execution(&self, input: ToolExecutionInput) -> ToolExecution {
        let scope = agent_scope(&input.agent);
        // Snapshot the definition-owned finalizer when the call starts: it
        // stays bound through disposal and pipeline failures alike.
        let finalizer = self
            .get(&input.name, scope.as_ref())
            .and_then(|tool| tool.finalize_content);
        let root_call_id = input.root_call_id.unwrap_or_else(|| input.call_id.clone());
        ToolExecution {
            inner: Rc::new(ExecInner {
                token: mint_token(),
                call_id: input.call_id,
                root_call_id,
                name: input.name,
                arguments: input.arguments,
                agent: input.agent,
                parent: input.parent,
                signal: RefCell::new(input.signal.clone()),
                caller_signal: input.signal,
                body_invoked: Cell::new(false),
                concludes: Cell::new(false),
                deferred_contexts: RefCell::new(Vec::new()),
                finalizer,
            }),
        }
    }

    /// Scheduler stage 1: materialize the input and run the ordered
    /// pre-execute/guard gate.
    pub async fn prepare(self: &Rc<Self>, input: ToolExecutionInput) -> ScheduledToolPreparation {
        let exec = self.create_execution(input);
        if exec.caller_signal().aborted() {
            return ScheduledToolPreparation::FinalResult {
                exec,
                result: tool_aborted_before_dispatch_result(None),
            };
        }
        let carrier = self.carrier_for(&exec);
        let gate = self
            .ctx
            .waterfall::<ScopedEvent<ToolsPreExecute>, _, _>(
                (carrier, exec.clone()),
                |_args| async { Ok(PreToolDecision::Allow) },
            )
            .await;
        let gate = match gate {
            Ok(gate) => gate,
            Err(error) => {
                return ScheduledToolPreparation::FinalResult {
                    exec,
                    result: tool_error_result(&error),
                };
            }
        };
        let (deny_reason, approval_cancelled) = match gate {
            PreToolDecision::Allow => (None, false),
            PreToolDecision::Deny { reason } => (Some(reason), false),
            PreToolDecision::Ask { reason } => self.service_ask(&exec, reason).await,
        };
        if exec.caller_signal().aborted() && approval_cancelled {
            return ScheduledToolPreparation::PostResult {
                exec,
                result: tool_aborted_before_dispatch_result(None),
            };
        }
        // Guards run only over a call the extensible policy allowed; a
        // denial is monotonic — nothing later can turn it back.
        let denial = deny_reason.or_else(|| self.guard_reason(&exec));
        if let Some(reason) = denial {
            return ScheduledToolPreparation::PostResult {
                exec,
                result: ToolExecutionResult::Failure {
                    content: vec![text_block(format!("Error: {reason}"))],
                    error: ToolFailure {
                        message: reason,
                        info: None,
                    },
                    meta: None,
                    additional_contexts: Vec::new(),
                },
            };
        }
        if exec.caller_signal().aborted() {
            return ScheduledToolPreparation::PostResult {
                exec,
                result: tool_aborted_before_dispatch_result(None),
            };
        }
        ScheduledToolPreparation::Dispatch { exec }
    }

    /// Resolve an `Ask` through the optional approval seam: a missing seam,
    /// a missing agent, and every non-grant deny with distinct reasons so
    /// the model can tell a human "no" from an absent channel.
    async fn service_ask(
        &self,
        exec: &ToolExecution,
        reason: Option<String>,
    ) -> (Option<String>, bool) {
        let name = exec.name();
        let Some(approval) = self.ctx.try_service::<ToolApproval>() else {
            let fallback = reason.unwrap_or_else(|| {
                format!("tool \"{name}\" requires approval (not yet supported)")
            });
            return (Some(fallback), false);
        };
        let Some(agent) = exec.agent() else {
            return (
                Some(format!(
                    "tool \"{name}\" requires approval, but the call has no agent to route it through"
                )),
                false,
            );
        };
        let outcome = approval
            .request(ApprovalRequest {
                agent,
                tool_name: name.to_string(),
                call_id: exec.call_id().clone(),
                reason,
                signal: exec.signal(),
            })
            .await;
        match outcome {
            ApprovalOutcome::AllowedOnce => (None, false),
            ApprovalOutcome::Rejected => {
                (Some(format!("the user rejected tool \"{name}\"")), false)
            }
            ApprovalOutcome::Cancelled => (
                Some(format!("approval for tool \"{name}\" was cancelled")),
                true,
            ),
            ApprovalOutcome::Unavailable => (
                Some(format!(
                    "tool \"{name}\" requires approval, but no approval channel is available"
                )),
                false,
            ),
        }
    }

    /// Canonical cancellation outcome selected by whether the body started.
    fn cancellation_result(
        &self,
        exec: &ToolExecution,
        prior: Option<&ToolExecutionResult>,
    ) -> ToolExecutionResult {
        if exec.inner.body_invoked.get() {
            tool_aborted_result(prior)
        } else {
            tool_aborted_before_dispatch_result(prior)
        }
    }

    /// Dispatch the registered body under the caller signal fused with any
    /// wrapper replacement. A started body is always drained to quiescence
    /// before its outcome becomes `ABORTED`.
    async fn dispatch_tool_body(&self, exec: &ToolExecution) -> ToolExecutionResult {
        let caller = exec.caller_signal();
        let wrapper = exec.signal();
        // Always fuse: replacement can never detach caller cancellation.
        let fused = AbortSignal::any([caller, wrapper.clone()]);
        if fused.aborted() {
            return tool_aborted_before_dispatch_result(None);
        }
        exec.set_signal(fused.clone());
        let outcome: anyhow::Result<ToolExecutionResult> = async {
            let scope = agent_scope(&exec.inner.agent);
            let Some(tool) = self.get(exec.name(), scope.as_ref()) else {
                return Err(ToolNotFoundError::new(exec.name()).into());
            };
            exec.inner.body_invoked.set(true);
            let returned = (tool.execute)(exec.arguments().clone(), exec.clone()).await?;
            let result = self.create_success_result(exec, &tool, returned)?;
            Ok(if fused.aborted() {
                tool_aborted_result(Some(&result))
            } else {
                result
            })
        }
        .await;
        exec.set_signal(wrapper);
        match outcome {
            Ok(result) => result,
            Err(error) => tool_error_result(&error),
        }
    }

    /// Scheduler stage 2: run only the around-dispatch/body stage. Tool and
    /// unknown-tool failures still receive post-execute; pipeline failures
    /// are already final.
    pub async fn dispatch(self: &Rc<Self>, exec: &ToolExecution) -> ScheduledToolDispatch {
        let carrier = self.carrier_for(exec);
        let runtime = self.clone();
        let outcome = self
            .ctx
            .waterfall::<ScopedEvent<ToolsExecute>, _, _>(
                (carrier, exec.clone()),
                move |(_carrier, exec)| async move { Ok(runtime.dispatch_tool_body(&exec).await) },
            )
            .await;
        let result = match outcome {
            Ok(result) => result,
            Err(error) => {
                return ScheduledToolDispatch::FinalResult {
                    result: tool_error_result(&error),
                };
            }
        };
        let normalized = match self.normalize_dispatch_result(exec, result) {
            Ok(normalized) => normalized,
            Err(error) => {
                return ScheduledToolDispatch::FinalResult {
                    result: tool_error_result(&error),
                };
            }
        };
        let deferred = exec.inner.deferred_contexts.borrow().clone();
        let with_contexts = if deferred.is_empty() {
            normalized
        } else {
            let mut contexts = deferred;
            contexts.extend(normalized.additional_contexts().to_vec());
            normalized.with_additional_contexts(contexts)
        };
        let result = if exec.caller_signal().aborted() && !with_contexts.is_error() {
            self.cancellation_result(exec, Some(&with_contexts))
        } else {
            with_contexts
        };
        ScheduledToolDispatch::PostResult { result }
    }

    /// Snapshot, validate, render, and optionally project one successful
    /// body or policy value.
    fn create_success_result(
        &self,
        exec: &ToolExecution,
        tool: &ToolDefinition,
        value: Value,
    ) -> anyhow::Result<ToolExecutionResult> {
        let violations = validate_json_schema_value(&tool.output.schema, &value, "value");
        if !violations.is_empty() {
            return Err(ToolOutputError {
                tool_name: tool.name.clone(),
                violations,
            }
            .into());
        }
        let content =
            (tool.output.render)(exec.arguments(), &value).map_err(|error| ToolOutputError {
                tool_name: tool.name.clone(),
                violations: vec![format!("output.render failed: {error}")],
            })?;
        // Presentation metadata is a top-level replay affordance: nested
        // composite sub-dispatches never project it.
        let meta =
            match (&exec.inner.parent, &tool.output.presentation_meta) {
                (None, Some(project)) => Some(project(exec.arguments(), &value).map_err(
                    |error| ToolOutputError {
                        tool_name: tool.name.clone(),
                        violations: vec![format!("output.presentationMeta failed: {error}")],
                    },
                )?),
                _ => None,
            };
        Ok(ToolExecutionResult::Success {
            value,
            content,
            meta,
            additional_contexts: Vec::new(),
            concludes_turn: exec.inner.concludes.get(),
        })
    }

    /// Normalize an around-wrapper's authored result through the owning
    /// output contract. Results carry no object identity in Rust, so every
    /// success re-normalizes (render is pure per its contract).
    fn normalize_dispatch_result(
        &self,
        exec: &ToolExecution,
        result: ToolExecutionResult,
    ) -> anyhow::Result<ToolExecutionResult> {
        match result {
            failure @ ToolExecutionResult::Failure { .. } => Ok(failure),
            ToolExecutionResult::Success {
                value,
                additional_contexts,
                ..
            } => {
                let scope = agent_scope(&exec.inner.agent);
                let Some(tool) = self.get(exec.name(), scope.as_ref()) else {
                    return Err(ToolNotFoundError::new(exec.name()).into());
                };
                let normalized = self.create_success_result(exec, &tool, value)?;
                Ok(normalized.with_additional_contexts(additional_contexts))
            }
        }
    }

    /// Run the post-execute waterfall over one normalized dispatch result
    /// and apply its decision. Context deferred by the tool body survives an
    /// accepted result but is discarded when the call is blocked — a block
    /// exposes only what the blocking decision supplied.
    async fn post_execute(
        self: &Rc<Self>,
        exec: &ToolExecution,
        result: ToolExecutionResult,
    ) -> anyhow::Result<ToolExecutionResult> {
        let carrier = self.carrier_for(exec);
        let decision = self
            .ctx
            .waterfall::<ScopedEvent<ToolsPostExecute>, _, _>(
                (carrier, (exec.clone(), result.clone())),
                |_args| async { Ok(PostToolDecision::accept()) },
            )
            .await?;
        match decision {
            PostToolDecision::Block {
                feedback,
                additional_contexts,
            } => {
                let message = failure_message_from_content(&feedback);
                Ok(ToolExecutionResult::Failure {
                    content: feedback,
                    error: ToolFailure {
                        message,
                        info: None,
                    },
                    meta: None,
                    additional_contexts,
                })
            }
            PostToolDecision::Accept {
                replace,
                additional_contexts: decision_contexts,
            } => {
                let mut contexts = result.additional_contexts().to_vec();
                contexts.extend(decision_contexts);
                match replace {
                    None => Ok(result.with_additional_contexts(contexts)),
                    Some(PostAcceptReplacement::Content(content)) => Ok(result
                        .with_content(content)
                        .with_additional_contexts(contexts)),
                    Some(PostAcceptReplacement::Value(value)) => {
                        if result.is_error() {
                            anyhow::bail!(
                                "tools/post-execute cannot replace the value of a failed result"
                            );
                        }
                        let scope = agent_scope(&exec.inner.agent);
                        let Some(tool) = self.get(exec.name(), scope.as_ref()) else {
                            return Err(ToolNotFoundError::new(exec.name()).into());
                        };
                        let replaced = self.create_success_result(exec, &tool, value)?;
                        Ok(replaced.with_additional_contexts(contexts))
                    }
                }
            }
        }
    }

    /// Scheduler stage 3: run ordered post-execute, then finalize content,
    /// notify, and return the authoritative outcome.
    pub async fn finalize(
        self: &Rc<Self>,
        exec: &ToolExecution,
        result: ToolExecutionResult,
    ) -> ToolExecutionResult {
        match self.post_execute(exec, result).await {
            Ok(post) => {
                let result = if exec.caller_signal().aborted() && !post.is_error() {
                    self.cancellation_result(exec, Some(&post))
                } else {
                    post
                };
                self.finish(exec, result)
            }
            Err(error) => self.finish(exec, tool_error_result(&error)),
        }
    }

    /// Scheduler stage 4: apply the snapshotted definition-owned content
    /// transform and notify final observers without post-execute.
    pub fn finish(&self, exec: &ToolExecution, result: ToolExecutionResult) -> ToolExecutionResult {
        let finalized = match &exec.inner.finalizer {
            Some(finalize) => match finalize(exec, &result) {
                Some(content) => result.with_content(content),
                None => result,
            },
            None => result,
        };
        let carrier = self.carrier_for(exec);
        self.ctx
            .emit_scoped::<ToolsResult, ()>(&carrier, (exec.clone(), finalized.clone()));
        finalized
    }
}
