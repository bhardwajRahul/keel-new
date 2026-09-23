//! One bounded computer-use decision. The caller owns the actual native action,
//! permission checks, and a fresh eligibility check after selection.

use std::collections::HashSet;
use std::io::Read;
use std::path::PathBuf;

use clap::ValueEnum;
use jev_core::{Candidate, DecisionBudget, DecisionState, JevConfig, JevSelector, SelectionInput};
use laya_local::{LayaConfig, LayaSelector};
use serde::{Deserialize, Serialize};

const MAX_INPUT_BYTES: u64 = 12_000;
const MAX_CANDIDATES: usize = 8;
const SECRET_PATH: &str = ".codex/codex-router/typesafe-api-key.secret";

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum ComputerUseMode {
    Laya,
    Jev,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    state: ObservedState,
    candidates: Vec<PreparedCandidate>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservedState {
    task: String,
    context: String,
    state_version: u64,
    /// The native host must strip secrets and irrelevant personal data first.
    redacted: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PreparedCandidate {
    id: String,
    description: String,
    action: ActionKind,
    risk: Risk,
    reversible: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum ActionKind {
    Observe,
    Scroll,
    Focus,
    Click,
    Dismiss,
    NavigateBack,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Risk {
    None,
    Low,
    Medium,
    High,
}

#[derive(Serialize)]
struct Reply {
    selected_id: Option<String>,
    reason: Option<String>,
}

impl Reply {
    fn abstain(reason: impl Into<String>) -> Self {
        Self {
            selected_id: None,
            reason: Some(reason.into()),
        }
    }

    fn selected(id: String) -> Self {
        Self {
            selected_id: Some(id),
            reason: None,
        }
    }
}

pub fn decide_cli(mode: ComputerUseMode) -> anyhow::Result<()> {
    let mut body = Vec::new();
    std::io::stdin()
        .lock()
        .take(MAX_INPUT_BYTES + 1)
        .read_to_end(&mut body)?;
    let reply = if body.len() as u64 > MAX_INPUT_BYTES {
        Reply::abstain("input_too_large")
    } else {
        let runtime = tokio::runtime::Runtime::new()?;
        runtime.block_on(decide(mode, &body))
    };
    println!("{}", serde_json::to_string(&reply)?);
    Ok(())
}

async fn decide(mode: ComputerUseMode, body: &[u8]) -> Reply {
    let request: Request = match serde_json::from_slice(body) {
        Ok(request) => request,
        Err(_) => return Reply::abstain("invalid_json"),
    };
    let input = match validate(request) {
        Ok(input) => input,
        Err(reason) => return Reply::abstain(reason),
    };
    let eligible: HashSet<String> = input.candidates.iter().map(|c| c.id.clone()).collect();
    let selected = match mode {
        ComputerUseMode::Laya => {
            let Some((worker, model)) = keel_engine::decision_mode::laya_assets() else {
                return Reply::abstain("local_model_unavailable");
            };
            let Ok(selector) = LayaSelector::new(LayaConfig::standalone(worker, model)) else {
                return Reply::abstain("local_model_unavailable");
            };
            let outcome = selector.select(input).await;
            if outcome.selected_id.is_none() {
                return Reply::abstain(format!("laya_{:?}", outcome.trace.fallback));
            }
            outcome.selected_id
        }
        ComputerUseMode::Jev => {
            let Some(home) = std::env::var_os("HOME") else {
                return Reply::abstain("credential_unavailable");
            };
            // app_owned reads only this protected local secret; it never uses
            // ambient env credentials or an OpenRouter endpoint.
            let key = PathBuf::from(home).join(SECRET_PATH);
            let Ok(selector) = JevSelector::new(JevConfig::app_owned(key)) else {
                return Reply::abstain("jev_unavailable");
            };
            let outcome = selector.select(input, &DecisionBudget::new(1)).await;
            if outcome.selected_id.is_none() {
                return Reply::abstain(format!("jev_{:?}", outcome.trace.fallback));
            }
            outcome.selected_id
        }
    };
    match selected.filter(|id| eligible.contains(id)) {
        Some(id) => Reply::selected(id),
        None => Reply::abstain("selector_abstained"),
    }
}

fn validate(request: Request) -> Result<SelectionInput, &'static str> {
    let state = request.state;
    if !state.redacted
        || state.task.trim().is_empty()
        || state.task.len() > 600
        || state.context.len() > 500
        || contains_secret_marker(&state.task)
        || contains_secret_marker(&state.context)
    {
        return Err("unsafe_observation");
    }
    if request.candidates.is_empty() || request.candidates.len() > MAX_CANDIDATES {
        return Err("invalid_candidates");
    }
    let mut seen = HashSet::new();
    let mut candidates = Vec::with_capacity(request.candidates.len());
    for candidate in request.candidates {
        if candidate.id.is_empty()
            || candidate.id.len() > 64
            || candidate.id == "escalate"
            || !candidate
                .id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
            || !seen.insert(candidate.id.clone())
        {
            return Err("invalid_candidates");
        }
        if !matches!(candidate.risk, Risk::None | Risk::Low)
            || !candidate.reversible
            || candidate.description.trim().is_empty()
            || candidate.description.len() > 180
            || contains_unsafe_action(&candidate.description)
        {
            return Err("unsafe_candidate");
        }
        let action = match candidate.action {
            ActionKind::Observe => "Observe",
            ActionKind::Scroll => "Scroll",
            ActionKind::Focus => "Focus",
            ActionKind::Click => "Click",
            ActionKind::Dismiss => "Dismiss",
            ActionKind::NavigateBack => "Navigate back",
        };
        candidates.push(Candidate {
            id: candidate.id,
            description: format!("{action}: {}", candidate.description),
        });
    }
    Ok(SelectionInput {
        state: DecisionState {
            task: state.task,
            context: state.context,
            state_version: state.state_version,
        },
        candidates,
    })
}

fn contains_secret_marker(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    [
        "api_key",
        "api key",
        "access_token",
        "bearer ",
        "password",
        "sk-",
        "secret=",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn contains_unsafe_action(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    if lower.contains("api key") {
        return true;
    }
    [
        "delete",
        "remove",
        "send",
        "submit",
        "publish",
        "purchase",
        "pay",
        "install",
        "uninstall",
        "upload",
        "download",
        "approve",
        "permission",
        "terminal",
        "shell",
        "password",
        "token",
    ]
    .iter()
    .any(|word| {
        lower
            .split(|c: char| !c.is_ascii_alphanumeric())
            .any(|part| part == *word)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(candidates: serde_json::Value) -> Vec<u8> {
        serde_json::to_vec(&json!({
            "state": {"task":"Inspect the next panel", "context":"The settings pane is visible", "state_version":1, "redacted":true},
            "candidates":candidates
        }))
        .unwrap()
    }

    fn candidate(id: &str) -> serde_json::Value {
        json!({"id":id,"description":"Open the appearance panel","action":"click","risk":"low","reversible":true})
    }

    #[test]
    fn accepts_only_bounded_reversible_prepared_actions() {
        let body = request(json!([candidate("step_1")]));
        let input = validate(serde_json::from_slice(&body).unwrap()).unwrap();
        assert_eq!(input.candidates[0].id, "step_1");
        assert_eq!(
            input.candidates[0].description,
            "Click: Open the appearance panel"
        );
        let no_risk = json!([{"id":"step_2","description":"Observe panel title","action":"observe","risk":"none","reversible":true}]);
        assert!(validate(serde_json::from_slice(&request(no_risk)).unwrap()).is_ok());
    }

    #[tokio::test]
    async fn bad_candidates_abstain_before_any_backend_call() {
        for candidates in [
            json!([candidate("same"), candidate("same")]),
            json!([
                candidate("a"),
                candidate("b"),
                candidate("c"),
                candidate("d"),
                candidate("e"),
                candidate("f"),
                candidate("g"),
                candidate("h"),
                candidate("i")
            ]),
            json!([{"id":"bad","description":"Delete the account","action":"click","risk":"low","reversible":true}]),
            json!([{"id":"bad","description":"Open panel","action":"click","risk":"high","reversible":true}]),
            json!([{"id":"bad","description":"Open panel","action":"click","risk":"low","reversible":false}]),
            json!([{"id":"bad","description":"Open panel","action":"keypress","risk":"low","reversible":true}]),
        ] {
            let reply = decide(ComputerUseMode::Jev, &request(candidates)).await;
            assert!(reply.selected_id.is_none());
            assert_ne!(reply.reason.as_deref(), Some("selector_abstained"));
        }
    }

    #[test]
    fn unredacted_or_secret_observation_is_refused() {
        let mut body: serde_json::Value =
            serde_json::from_slice(&request(json!([candidate("step_1")]))).unwrap();
        body["state"]["redacted"] = json!(false);
        assert_eq!(
            validate(serde_json::from_value(body.clone()).unwrap()).err(),
            Some("unsafe_observation")
        );
        body["state"]["redacted"] = json!(true);
        body["state"]["context"] = json!("Bearer example credential");
        assert_eq!(
            validate(serde_json::from_value(body).unwrap()).err(),
            Some("unsafe_observation")
        );
        let mut body: serde_json::Value =
            serde_json::from_slice(&request(json!([candidate("step_1")]))).unwrap();
        body["state"]["context"] = json!("x".repeat(501));
        assert_eq!(
            validate(serde_json::from_value(body).unwrap()).err(),
            Some("unsafe_observation")
        );
    }
}
