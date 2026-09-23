//! Package-owned prompt-assembly invariant companion (port of upstream
//! `src/invariant.ts`): validates the authoritative assembly resolved by the
//! `system-prompt/assemble` waterfall.
//!
//! Divergence: upstream's runtime type checks (section/context text must be a
//! string, variable values string-or-undefined) are unrepresentable here —
//! the Rust types already guarantee them — so only the value-level checks
//! (non-empty names, duplicates, valid variable names) are ported.

use crate::{Assemble, PromptAssembly, VARIABLE_NAME_PATTERN, is_variable_name};
use dsh_cordis::{Context, Disposer, Effect, EventOptions, FnPlugin, Inject, plugin_fn};
use dsh_invariants::{InvariantError, InvariantFailure, InvariantInstaller, InvariantRegistry};
use futures::FutureExt;
use futures::future::LocalBoxFuture;
use serde_json::Value;
use std::collections::HashSet;

/// Package name registered by the companion.
pub const PACKAGE_NAME: &str = "dsh-system-prompt";
/// Cordis companion plugin name.
pub const NAME: &str = "system-prompt-invariant";

/// Check the authoritative assembly; the first violation wins.
fn validate_assembly(
    assembly: &PromptAssembly,
    fail: &InvariantFailure,
) -> Result<(), InvariantError> {
    let mut section_names = HashSet::new();
    for section in &assembly.sections {
        if section.name.is_empty() {
            return Err(fail("assembled section names must be non-empty"));
        }
        if !section_names.insert(section.name.as_str()) {
            return Err(fail(&format!(
                "assembled section name {:?} is duplicated",
                section.name
            )));
        }
    }

    let mut context_names = HashSet::new();
    for context in &assembly.contexts {
        if context.name.is_empty() {
            return Err(fail("assembled context names must be non-empty"));
        }
        if !context_names.insert(context.name.as_str()) {
            return Err(fail(&format!(
                "assembled context name {:?} is duplicated",
                context.name
            )));
        }
    }

    for tool in &assembly.tools {
        if tool.name.is_empty() {
            return Err(fail("assembled tool names must be non-empty"));
        }
    }

    for name in assembly.variables.keys() {
        if !is_variable_name(name) {
            return Err(fail(&format!(
                "assembled variable name {name:?} is invalid (must match {VARIABLE_NAME_PATTERN})"
            )));
        }
    }
    Ok(())
}

/// Build the companion plugin: requires the `invariants` service and installs
/// validation around the authoritative assemble-waterfall result.
pub fn plugin()
-> FnPlugin<impl Fn(Context, Value) -> LocalBoxFuture<'static, anyhow::Result<()>> + 'static> {
    plugin_fn(NAME, Inject::names(["invariants"]), |ctx, _config| {
        async move {
            let registry = ctx.service::<InvariantRegistry>()?;
            let registration = registry
                .register(
                    PACKAGE_NAME,
                    InvariantInstaller::new(|ctx: Context, fail: InvariantFailure| async move {
                        ctx.on_waterfall::<Assemble, _, _>(
                            EventOptions {
                                global: true,
                                prepend: true,
                            },
                            move |_ctx, args, next| {
                                let fail = fail.clone();
                                async move {
                                    let assembled = next(args).await?;
                                    validate_assembly(&assembled, &fail)?;
                                    Ok(assembled)
                                }
                            },
                        )?;
                        Ok(())
                    }),
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
