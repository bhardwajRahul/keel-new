//! Rust port of `@deepseek-ai/dsh-tools` (`packages/core/tools`): the
//! enforced JSON Schema subset, the author-facing value-schema DSL, the
//! render-intent presentation vocabulary, and the `tools` registry service
//! with its pre/guard/around/post/result execution pipeline.
//!
//! Deliberately deferred (not ported):
//! - **Code Mode** (`code-mode.ts`, `ts-types.ts`, `py-types.ts`,
//!   `types.ts`): the `run_code` presentation transport needs an embedded JS
//!   runtime that does not exist in this workspace yet. [`Config`] keeps the
//!   `mode` field; selecting `code` or `both` fails with a clear
//!   not-implemented error. The `run_code` name stays reserved so its later
//!   arrival cannot collide with a user registration, and the
//!   `tools/code-dispatch-log` waterfall plus `presentAs` are omitted with
//!   the transport they serve.
//!
//! Divergences from the TypeScript package (contract-level; forced by the
//! Rust host or the sibling dsh ports):
//! - Arguments, schemas, and canonical values are `serde_json::Value`, which
//!   is lossless JSON by construction: the snapshot/freeze machinery, the
//!   "arguments must be losslessly JSON-serializable" failure class, and the
//!   schema-walk circularity/realm checks all collapse into the type system.
//! - Compile-time inference (`InferArgs`/`InferValue`) has no Rust
//!   counterpart: [`define_tool`] closures receive validated
//!   `serde_json::Value` arguments, and the DSL's illegal author states are
//!   unrepresentable in the typed specs (parse untyped author input with
//!   [`ValueSchemaSpec::from_author_value`], which keeps the runtime author
//!   checks).
//! - Registry methods take the registration [`Context`] explicitly (cordis-rs
//!   services are not caller-context traced) and return the exact cordis
//!   [`EffectHandle`] instead of a disposer function; failures are `Result`s
//!   instead of throws.
//! - `tools/change` is a fire-and-forget `emit`: a listener cannot fail the
//!   registration that notified it (upstream rolls back on a throwing
//!   listener). `tools/result` observers run on the local task queue, not
//!   synchronously inside the pipeline.
//! - Results are values without object identity, so the canonical-result
//!   cache is gone: every around-wrapper success is re-normalized through the
//!   owning output contract (render is pure per its contract, so the repeat
//!   projection is unobservable).
//! - The registry store iterates snapshots (see `dsh-scope`), so a guard
//!   registered by a running guard is seen from the next call onward.
//! - The system-prompt integration is decoupled: instead of depending on the
//!   concurrently ported `dsh-system-prompt`, the runtime exposes
//!   [`ToolRuntime::wire_schemas`]/[`ToolRuntime::schemas_provider`] for that
//!   wiring to consume.
//! - The approval seam (`ctx.get('approval')`) is the [`ToolApproval`]
//!   service hosted here until a dedicated user-approval crate is ported.
//! - Scope-filtered waterfall dispatch is provided locally by
//!   [`ScopedWaterfall`]; `dsh-scope` covers only notify events today.

mod json_schema;
mod presentation;
mod runtime;
mod schema;

pub use json_schema::{
    JsonSchemaError, JsonSchemaType, assert_object_json_schema, assert_supported_json_schema,
    validate_json_schema_value,
};
pub use presentation::{
    DiffCallView, DiffResultView, FileDiff, FileLocation, GenericCallView, GenericResultView,
    ReadFileLine, ReadResultView, SearchFileMatches, SearchLineMatch, SearchMatchesResultView,
    SearchPathsResultView, SearchResultView, TerminalCallView, TerminalResultView, ToolCallKind,
    ToolCallView, ToolResultView, WebFetchResultView, WebResultView, WebSearchResultView,
    WebSource,
};
pub use runtime::{
    ApprovalOutcome, ApprovalRequest, Config, PostAcceptReplacement, PostToolDecision,
    PreToolDecision, RUN_CODE_NAME, ScheduledToolDispatch, ScheduledToolPreparation,
    ScopedWaterfall, TOOL_ABORTED, TOOL_ABORTED_BEFORE_DISPATCH, ToolApproval, ToolDefinition,
    ToolErrorInfo, ToolExecuteFn, ToolExecution, ToolExecutionInput, ToolExecutionMode,
    ToolExecutionResult, ToolExecutionToken, ToolFailure, ToolFinalizeFn, ToolGuardFn, ToolMetaFn,
    ToolNotFoundError, ToolOutputDefinition, ToolOutputError, ToolPresentationMode,
    ToolProviderResult, ToolRenderFn, ToolRestriction, ToolResult, ToolRunContext, ToolRuntime,
    ToolsChange, ToolsExecute, ToolsPostExecute, ToolsPreExecute, ToolsResult,
};
pub use schema::{
    ArrayValueSchemaSpec, BooleanValueSchemaSpec, DefineToolOptions, DefineToolOutput,
    IntegerValueSchemaSpec, JsonValueSchemaSpec, NullValueSchemaSpec, NumberValueSchemaSpec,
    ObjectValueSchemaSpec, OneOfValueSchemaSpec, ParameterPropertySpec, ParameterSchemaSpec,
    StringValueSchemaSpec, ToolArgsError, ValueSchemaAnnotations, ValueSchemaSpec, define_tool,
    parameter_schema_spec_to_json_schema, validate_args, value_schema_spec_to_json_schema,
};
