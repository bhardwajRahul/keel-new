//! Typed routing at a clean session boundary. The registry supplies eligible
//! provider/model pairs; local Laya or optional TypeSafe Jev returns only an
//! opaque ID from that set. The coding harness retains its own tool authority.

use std::path::Path;
use std::time::Duration;

use futures::future::join_all;
use jev_core::{Candidate, DecisionBudget, DecisionState, JevConfig, JevSelector, SelectionInput};
use keel_proto::{
    DecisionBackend, DecisionCandidate, DecisionEvent, DecisionResult, DecisionStage,
    DecisionValidation, HarnessId, ReasoningLevel, RunRequest,
};
use laya_local::LayaSelector;

use crate::registry::{HarnessRegistry, descriptor_enabled};
use crate::workflow::{self, PreparedAction, TaskState, ValidationContext};

const MAX_ROUTES: usize = 16;
const MAX_MODELS_PER_HARNESS: usize = 3;
const MAX_LOCAL_ROUTES: usize = 8;

#[derive(Clone, Debug)]
pub struct RouteChoice {
    pub harness: HarnessId,
    pub model: Option<String>,
}

/// The selected route and its factual decision receipt. A receipt exists only
/// when a selector was actually called; host-only eligibility failures do not
/// appear as model decisions.
pub struct RouteDecision {
    pub choice: Option<RouteChoice>,
    pub event: Option<DecisionEvent>,
}

impl RouteDecision {
    fn skipped() -> Self {
        Self {
            choice: None,
            event: None,
        }
    }
}

struct PreparedRoute {
    id: String,
    choice: RouteChoice,
    description: String,
    provider_name: String,
    model_label: String,
    reasoning_levels: Vec<ReasoningLevel>,
}

pub enum RouteBackend<'a> {
    LocalLaya(&'a LayaSelector),
    TypeSafeJev(&'a Path),
}

/// The composer marks an unpinned, new task with `jevAuto`. This host-only
/// option is removed before the request reaches an ACP coding provider.
pub fn requested(request: &RunRequest) -> bool {
    request
        .model_options
        .get("jevAuto")
        .and_then(|v| v.as_bool())
        == Some(true)
        && request
            .model_options
            .get("jevRouted")
            .and_then(|v| v.as_bool())
            != Some(true)
        // Model options such as Fast and Plan belong to the originally
        // selected provider. Never discard or transplant them to another.
        && request
            .model_options
            .keys()
            .all(|key| key == "jevAuto" || key == "jevRouted")
}

/// Route one new coding task. Every error or abstention leaves the normal
/// Avid-selected harness untouched. The caller must recheck the returned
/// route before execution and must not reroute a live or resumed session.
pub async fn choose(
    registry: &HarnessRegistry,
    request: &RunRequest,
    backend: RouteBackend<'_>,
) -> Option<RouteChoice> {
    choose_with_activity(registry, request, backend, |_| {}).await
}

/// The callback marks only the selector's evaluation window, after the host
/// has prepared a nonempty candidate set and before its result is validated.
pub async fn choose_with_activity(
    registry: &HarnessRegistry,
    request: &RunRequest,
    backend: RouteBackend<'_>,
    activity: impl FnMut(bool),
) -> Option<RouteChoice> {
    choose_with_report(registry, request, backend, activity)
        .await
        .choice
}

/// Return the receipt alongside the route so the caller can write the user
/// message first, then place the decision in the transcript in causal order.
pub async fn choose_with_report(
    registry: &HarnessRegistry,
    request: &RunRequest,
    backend: RouteBackend<'_>,
    mut activity: impl FnMut(bool),
) -> RouteDecision {
    if !requested(request) {
        return RouteDecision::skipped();
    }
    let prepared = prepare_routes(registry, request.reasoning).await;
    if prepared.is_empty() {
        tracing::info!("Jev router: no eligible installed models; using selected harness");
        return RouteDecision::skipped();
    }
    let state = TaskState::new(crate::now_ms().max(0) as u64);
    let Some(fingerprint) = route_fingerprint(&prepared) else {
        return RouteDecision::skipped();
    };
    let actions = prepared
        .iter()
        .map(|route| PreparedAction {
            id: route.id.clone(),
            description: route.description.clone(),
            payload: route.choice.clone(),
            prepared_at_revision: state.revision,
            preconditions: Vec::new(),
            read_set_fingerprint: Some(fingerprint.clone()),
            expires_at_ms: Some(crate::now_ms().saturating_add(45_000)),
        })
        .collect::<Vec<_>>();
    let initial_context = ValidationContext {
        now_ms: crate::now_ms(),
        current_read_set_fingerprint: Some(fingerprint.clone()),
        authorized: true,
    };
    if !actions
        .iter()
        .all(|action| workflow::eligible(&state, action, &initial_context))
    {
        return RouteDecision::skipped();
    }
    let (
        backend_name,
        considered,
        selected,
        selector_fallback,
        confidence,
        selected_probability,
        fit,
    ) = match backend {
        RouteBackend::TypeSafeJev(key_path) => {
            let Ok(selector) = JevSelector::new(JevConfig::app_owned(key_path.to_path_buf()))
            else {
                return RouteDecision::skipped();
            };
            let considered = prepared
                .iter()
                .map(|route| DecisionCandidate::new(&route.id, &route.description))
                .collect();
            let input = SelectionInput {
                state: DecisionState {
                    task: request.prompt.chars().take(6_000).collect(),
                    context: format!(
                        "Coding task. Sandbox: {:?}. Image attachments: {}. Select one available coding provider/model that can address the task. The host retains execution and permission authority.",
                        request.sandbox,
                        request.attachments.len()
                    ),
                    state_version: state.revision,
                },
                candidates: prepared
                    .iter()
                    .map(|route| Candidate {
                        id: route.id.clone(),
                        description: route.description.clone(),
                    })
                    .collect(),
            };
            activity(true);
            let outcome = selector.select(input, &DecisionBudget::new(1)).await;
            activity(false);
            tracing::info!(trace = ?outcome.trace, "TypeSafe Jev route decision");
            (
                DecisionBackend::Jev,
                considered,
                outcome.selected_id,
                outcome.trace.fallback.map(|reason| format!("{reason:?}")),
                outcome.trace.confidence,
                None,
                outcome.trace.fit,
            )
        }
        RouteBackend::LocalLaya(selector) => {
            // Laya's question-prefix budget is smaller than its 1024-token
            // context. Interleave providers so eight compact options give
            // every installed agent a chance before adding second models.
            let local = compact_local_routes(&prepared);
            let considered = local
                .iter()
                .map(|route| {
                    DecisionCandidate::new(
                        &route.id,
                        format!("{}: {}", route.provider_name, route.model_label),
                    )
                })
                .collect();
            let input = SelectionInput {
                state: DecisionState {
                    task: request.prompt.chars().take(900).collect(),
                    context: format!(
                        "Choose a coding agent/model. Sandbox {:?}; {} attachments. Host enforces permissions.",
                        request.sandbox,
                        request.attachments.len()
                    ),
                    state_version: state.revision,
                },
                candidates: local
                    .iter()
                    .map(|route| Candidate {
                        id: route.id.clone(),
                        description: format!("{}: {}", route.provider_name, route.model_label)
                            .chars()
                            .take(90)
                            .collect(),
                    })
                    .collect(),
            };
            activity(true);
            let outcome = selector.select(input).await;
            activity(false);
            tracing::info!(trace = ?outcome.trace, "Local Laya route decision");
            (
                DecisionBackend::Laya,
                considered,
                outcome.selected_id,
                outcome.trace.fallback.map(|reason| format!("{reason:?}")),
                outcome.trace.confidence,
                outcome.trace.selected_probability,
                None,
            )
        }
    };
    let mut event = DecisionEvent::new(
        format!("intake-{}", state.revision),
        state.revision,
        backend_name,
        DecisionStage::Intake,
        considered,
        match &selected {
            Some(candidate_id) => DecisionResult::Selected {
                candidate_id: candidate_id.clone(),
            },
            None => DecisionResult::Abstained,
        },
        DecisionValidation::Accepted,
        crate::now_ms(),
    );
    event.confidence = confidence;
    event.selected_probability = selected_probability;
    event.fit = fit;
    event = event.bounded();
    if let Some(reason) = selector_fallback {
        event = event.with_fallback(reason);
    }
    let Some(selected) = selected else {
        return RouteDecision {
            choice: None,
            event: Some(event.with_observed_outcome("Original harness retained")),
        };
    };
    let Some(action) = actions.iter().find(|action| action.id == selected) else {
        event.validation = DecisionValidation::Rejected;
        return RouteDecision {
            choice: None,
            event: Some(
                event
                    .with_fallback("Unknown route ID")
                    .with_observed_outcome("Original harness retained"),
            ),
        };
    };
    let Some(live_fingerprint) =
        route_fingerprint(&prepare_routes(registry, request.reasoning).await)
    else {
        event.validation = DecisionValidation::Stale;
        return RouteDecision {
            choice: None,
            event: Some(
                event
                    .with_fallback("Route catalog unavailable")
                    .with_observed_outcome("Original harness retained"),
            ),
        };
    };
    // The registry can change while TypeSafe responds. A selected opaque ID
    // never grants eligibility after the underlying provider was disabled.
    let still_eligible = registry.descriptors().into_iter().any(|descriptor| {
        descriptor.id == action.payload.harness
            && descriptor.installed
            && descriptor_enabled(&descriptor)
    });
    let validation = workflow::validate_selected(
        &state,
        action,
        &selected,
        &ValidationContext {
            now_ms: crate::now_ms(),
            current_read_set_fingerprint: Some(live_fingerprint),
            authorized: still_eligible,
        },
    );
    if !validation.accepted {
        tracing::info!(trace = ?validation, "Decision route became ineligible");
        event.validation = match validation.rejection {
            Some(workflow::RejectReason::Expired) => DecisionValidation::Expired,
            Some(workflow::RejectReason::Unauthorized) => DecisionValidation::Unauthorized,
            Some(workflow::RejectReason::StaleReadSet | workflow::RejectReason::StaleRevision) => {
                DecisionValidation::Stale
            }
            _ => DecisionValidation::Rejected,
        };
        return RouteDecision {
            choice: None,
            event: Some(
                event
                    .with_fallback("Route changed before dispatch")
                    .with_observed_outcome("Original harness retained"),
            ),
        };
    }
    let current_models = match registry.resolve(action.payload.harness) {
        Ok(harness) => tokio::time::timeout(Duration::from_secs(5), harness.models())
            .await
            .ok()
            .and_then(Result::ok),
        Err(_) => None,
    };
    let model_valid = current_models
        .as_ref()
        .and_then(|models| {
            models
                .iter()
                .find(|model| Some(&model.id) == action.payload.model.as_ref())
        })
        .is_some_and(|model| reasoning_compatible(request.reasoning, &model.reasoning_levels));
    if !model_valid {
        event.validation = DecisionValidation::Stale;
        return RouteDecision {
            choice: None,
            event: Some(
                event
                    .with_fallback("Selected model unavailable")
                    .with_observed_outcome("Original harness retained"),
            ),
        };
    }
    RouteDecision {
        choice: Some(action.payload.clone()),
        event: Some(event),
    }
}

fn reasoning_compatible(requested: Option<ReasoningLevel>, offered: &[ReasoningLevel]) -> bool {
    requested.is_none_or(|level| offered.contains(&level))
}

fn route_fingerprint(routes: &[PreparedRoute]) -> Option<String> {
    workflow::fingerprint_of(
        &routes
            .iter()
            .map(|route| {
                (
                    &route.id,
                    &route.description,
                    &route.choice.harness,
                    &route.choice.model,
                    &route.reasoning_levels,
                )
            })
            .collect::<Vec<_>>(),
    )
    .ok()
}

fn compact_local_routes(prepared: &[PreparedRoute]) -> Vec<&PreparedRoute> {
    let mut selected = Vec::new();
    for model_index in 0..MAX_MODELS_PER_HARNESS {
        let mut providers = Vec::new();
        for route in prepared {
            if !providers.contains(&route.choice.harness) {
                providers.push(route.choice.harness);
            }
        }
        for provider in providers {
            let route = prepared
                .iter()
                .filter(|route| route.choice.harness == provider)
                .nth(model_index);
            if let Some(route) = route {
                selected.push(route);
                if selected.len() == MAX_LOCAL_ROUTES {
                    return selected;
                }
            }
        }
    }
    selected
}

async fn prepare_routes(
    registry: &HarnessRegistry,
    requested_reasoning: Option<ReasoningLevel>,
) -> Vec<PreparedRoute> {
    let mut routes = Vec::new();
    let probes = registry
        .descriptors()
        .into_iter()
        .filter(|descriptor| {
            descriptor.installed
                && descriptor_enabled(descriptor)
                && descriptor.id != HarnessId::Mock
                // Dsh is in-process but still needs a generation credential.
                && (descriptor.id != HarnessId::Dsh
                    || std::env::var("DEEPSEEK_API_KEY")
                        .ok()
                        .is_some_and(|key| !key.trim().is_empty()))
        })
        .map(|descriptor| async move {
            let harness = registry.resolve(descriptor.id).ok()?;
            let models = tokio::time::timeout(Duration::from_secs(5), harness.models())
                .await
                .ok()?
                .ok()?;
            Some((descriptor, models))
        });
    for (descriptor, models) in join_all(probes).await.into_iter().flatten() {
        for model in models.into_iter().take(MAX_MODELS_PER_HARNESS) {
            if routes.len() >= MAX_ROUTES {
                break;
            }
            if !reasoning_compatible(requested_reasoning, &model.reasoning_levels) {
                continue;
            }
            let id = format!("route_{}", routes.len());
            routes.push(PreparedRoute {
                id,
                choice: RouteChoice {
                    harness: descriptor.id,
                    model: Some(model.id),
                },
                description: format!(
                    "{} coding agent, model {}. {}",
                    descriptor.name,
                    model.label,
                    model.description.unwrap_or_default()
                ),
                provider_name: descriptor.name.clone(),
                model_label: model.label,
                reasoning_levels: model.reasoning_levels,
            });
        }
    }
    routes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_snapshot_changes_when_a_model_changes() {
        let mut routes = vec![PreparedRoute {
            id: "route_0".into(),
            choice: RouteChoice {
                harness: HarnessId::Codex,
                model: Some("model-a".into()),
            },
            description: "Codex model A".into(),
            provider_name: "Codex".into(),
            model_label: "A".into(),
            reasoning_levels: vec![ReasoningLevel::High],
        }];
        let original = route_fingerprint(&routes).unwrap();
        routes[0].choice.model = Some("model-b".into());
        assert_ne!(route_fingerprint(&routes).unwrap(), original);
    }

    #[test]
    fn requested_reasoning_must_exist_on_routed_model() {
        assert!(reasoning_compatible(None, &[]));
        assert!(reasoning_compatible(
            Some(ReasoningLevel::High),
            &[ReasoningLevel::High],
        ));
        assert!(!reasoning_compatible(
            Some(ReasoningLevel::High),
            &[ReasoningLevel::Medium],
        ));
    }

    #[test]
    fn only_new_auto_tasks_are_routed() {
        let mut request = RunRequest {
            prompt: "fix test".into(),
            harness: Some(HarnessId::Codex),
            model: None,
            reasoning: None,
            model_options: Default::default(),
            cwd: "/tmp".into(),
            sandbox: keel_proto::SandboxLevel::WorkspaceWrite,
            auto_approve: false,
            resume: None,
            attachments: Vec::new(),
        };
        assert!(!requested(&request));
        request.model_options.insert("jevAuto".into(), true.into());
        assert!(requested(&request));
        request.model_options.insert("fast".into(), true.into());
        assert!(!requested(&request));
        request.model_options.remove("fast");
        request
            .model_options
            .insert("jevRouted".into(), true.into());
        assert!(!requested(&request));
    }
}
