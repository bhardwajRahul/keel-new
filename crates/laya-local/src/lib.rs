//! Local typed selection using Laya-CoreML. The host supplies eligible opaque
//! candidate IDs; this crate cannot create or execute coding actions.

mod install;

pub use install::{
    InstallError, InstallProgress, MODEL_REVISION, install_model, install_model_with_progress,
    model_is_installed,
};

use jev_core::SelectionInput;
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    path::PathBuf,
    process::Stdio,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::Mutex,
    time::timeout,
};

const WORKER_SCRIPT: &str = include_str!("../worker.py");
const PROTOCOL_PREFIX: &str = "KEEL_LAYA_JSON:";
const MAX_REQUEST_BYTES: usize = 90_000;
const MAX_RESPONSE_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug)]
pub struct LayaConfig {
    pub runtime: LayaRuntime,
    pub model_dir: PathBuf,
    /// Core ML compilation can take tens of seconds on the first launch.
    pub startup_timeout: Duration,
    pub decision_timeout: Duration,
    pub min_probability: f64,
    pub min_confidence: f64,
    pub min_margin: f64,
}

#[derive(Clone, Debug)]
pub enum LayaRuntime {
    Python(PathBuf),
    Standalone(PathBuf),
}

impl LayaConfig {
    pub fn new(python_executable: PathBuf, model_dir: PathBuf) -> Self {
        Self {
            runtime: LayaRuntime::Python(python_executable),
            model_dir,
            startup_timeout: Duration::from_secs(120),
            decision_timeout: Duration::from_secs(25),
            min_probability: 0.4,
            min_confidence: 0.1,
            min_margin: 0.05,
        }
    }

    pub fn standalone(worker_executable: PathBuf, model_dir: PathBuf) -> Self {
        let mut config = Self::new(worker_executable.clone(), model_dir);
        config.runtime = LayaRuntime::Standalone(worker_executable);
        config
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LayaFallback {
    NoCandidates,
    InvalidInput,
    ModelUnavailable,
    WorkerUnavailable,
    Timeout,
    CapacityExceeded,
    InferenceError,
    InvalidResponse,
    Abstained,
    LowProbability,
    LowConfidence,
    Ambiguous,
}

/// Safe to log: contains no task text, candidate descriptions, or local path.
#[derive(Clone, Debug, Serialize)]
pub struct LayaTrace {
    pub decision_schema: &'static str,
    pub model: &'static str,
    pub state_version: u64,
    pub state_fingerprint: String,
    pub candidates_fingerprint: String,
    pub selected_id: Option<String>,
    pub selected_probability: Option<f64>,
    pub confidence: Option<f64>,
    pub fallback: Option<LayaFallback>,
    pub latency_ms: Option<u128>,
}

#[derive(Clone, Debug)]
pub struct LayaOutcome {
    pub selected_id: Option<String>,
    pub trace: LayaTrace,
}

#[derive(Clone)]
pub struct LayaSelector {
    config: LayaConfig,
    worker: Arc<Mutex<Option<Worker>>>,
    script: &'static str,
}

impl LayaSelector {
    pub fn new(config: LayaConfig) -> Result<Self, &'static str> {
        if config.runtime.executable().as_os_str().is_empty()
            || config.model_dir.as_os_str().is_empty()
            || config.startup_timeout.is_zero()
            || config.decision_timeout.is_zero()
            || !valid_probability(config.min_probability)
            || !valid_probability(config.min_confidence)
            || !valid_probability(config.min_margin)
        {
            return Err("invalid Laya configuration");
        }
        Ok(Self {
            config,
            worker: Arc::new(Mutex::new(None)),
            script: WORKER_SCRIPT,
        })
    }

    pub async fn select(&self, input: SelectionInput) -> LayaOutcome {
        let mut trace = LayaTrace {
            decision_schema: "laya-selection-v1",
            model: "laya-coreml",
            state_version: input.state.state_version,
            state_fingerprint: fingerprint(&input.state),
            candidates_fingerprint: fingerprint(&input.candidates),
            selected_id: None,
            selected_probability: None,
            confidence: None,
            fallback: None,
            latency_ms: None,
        };
        let result = self.select_inner(&input, &mut trace).await;
        match result {
            Ok(id) => {
                trace.selected_id = Some(id.clone());
                LayaOutcome {
                    selected_id: Some(id),
                    trace,
                }
            }
            Err(reason) => {
                trace.fallback = Some(reason);
                LayaOutcome {
                    selected_id: None,
                    trace,
                }
            }
        }
    }

    async fn select_inner(
        &self,
        input: &SelectionInput,
        trace: &mut LayaTrace,
    ) -> Result<String, LayaFallback> {
        if input.candidates.is_empty() {
            return Err(LayaFallback::NoCandidates);
        }
        if !valid_input(input) {
            return Err(LayaFallback::InvalidInput);
        }
        // Requiring a real directory prevents the Hub ID form from ever reaching
        // the Python loader. `local_files_only=True` is enforced in worker.py.
        if !self.config.model_dir.is_dir()
            || !self.config.model_dir.join("coreml_config.json").is_file()
        {
            return Err(LayaFallback::ModelUnavailable);
        }
        let state = format!(
            "task: {}\ncontext: {}",
            input.state.task, input.state.context
        );
        let mut criteria = Vec::with_capacity(input.candidates.len() + 1);
        for candidate in &input.candidates {
            criteria.push((candidate.id.clone(), candidate.description.clone()));
        }
        criteria.push((
            "escalate".into(),
            "None of these steps helps the task".into(),
        ));
        let mut request = serde_json::to_vec(&json!({"state":state,"criteria":criteria}))
            .map_err(|_| LayaFallback::InvalidInput)?;
        if request.len() > MAX_REQUEST_BYTES {
            return Err(LayaFallback::InvalidInput);
        }
        request.push(b'\n');

        let started = Instant::now();
        let mut worker = self.worker.lock().await;
        if worker.is_none() {
            let model_dir = self
                .config
                .model_dir
                .canonicalize()
                .map_err(|_| LayaFallback::ModelUnavailable)?;
            let launched = timeout(
                self.config.startup_timeout,
                Worker::launch(&self.config.runtime, &model_dir, self.script),
            )
            .await
            .map_err(|_| LayaFallback::Timeout)??;
            *worker = Some(launched);
        }
        let reply = timeout(
            self.config.decision_timeout,
            worker.as_mut().expect("worker just set").query(&request),
        )
        .await;
        trace.latency_ms = Some(started.elapsed().as_millis());
        let value = match reply {
            Ok(Ok(value)) => value,
            Ok(Err(reason)) => {
                *worker = None;
                return Err(reason);
            }
            Err(_) => {
                *worker = None;
                return Err(LayaFallback::Timeout);
            }
        };
        parse_reply(&value, input, &self.config, trace)
    }

    #[cfg(test)]
    fn with_test_script(mut self, script: &'static str) -> Self {
        self.script = script;
        self
    }
}

struct Worker {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Worker {
    async fn launch(
        runtime: &LayaRuntime,
        model_dir: &PathBuf,
        script: &'static str,
    ) -> Result<Self, LayaFallback> {
        let mut command = Command::new(runtime.executable());
        match runtime {
            LayaRuntime::Python(_) => {
                command.arg("-u").arg("-c").arg(script).arg(model_dir);
            }
            LayaRuntime::Standalone(_) => {
                command.arg("--model").arg(model_dir);
            }
        }
        let mut child = command
            .env("HF_HUB_OFFLINE", "1")
            .env("TRANSFORMERS_OFFLINE", "1")
            .env("HF_HUB_DISABLE_TELEMETRY", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .map_err(|_| LayaFallback::WorkerUnavailable)?;
        let stdin = child.stdin.take().ok_or(LayaFallback::WorkerUnavailable)?;
        let stdout = child.stdout.take().ok_or(LayaFallback::WorkerUnavailable)?;
        let mut worker = Self {
            _child: child,
            stdin,
            stdout: BufReader::new(stdout),
        };
        let ready = worker.read_reply().await?;
        if ready.get("ready").and_then(Value::as_bool) != Some(true)
            || ready.get("model").and_then(Value::as_str) != Some("laya-coreml")
        {
            return Err(LayaFallback::ModelUnavailable);
        }
        Ok(worker)
    }

    async fn query(&mut self, request: &[u8]) -> Result<Value, LayaFallback> {
        self.stdin
            .write_all(request)
            .await
            .map_err(|_| LayaFallback::WorkerUnavailable)?;
        self.stdin
            .flush()
            .await
            .map_err(|_| LayaFallback::WorkerUnavailable)?;
        self.read_reply().await
    }

    async fn read_reply(&mut self) -> Result<Value, LayaFallback> {
        // Native Core ML diagnostics occasionally reach stdout. Only framed
        // worker messages are accepted; skip a bounded amount of chatter.
        let mut skipped_bytes = 0;
        for _ in 0..8 {
            let mut line = String::new();
            if self
                .stdout
                .read_line(&mut line)
                .await
                .map_err(|_| LayaFallback::WorkerUnavailable)?
                == 0
            {
                return Err(LayaFallback::WorkerUnavailable);
            }
            if line.len() > MAX_RESPONSE_BYTES {
                return Err(LayaFallback::InvalidResponse);
            }
            if let Some(payload) = line.strip_prefix(PROTOCOL_PREFIX) {
                return serde_json::from_str(payload).map_err(|_| LayaFallback::InvalidResponse);
            }
            skipped_bytes += line.len();
            if skipped_bytes > 8 * 1024 {
                return Err(LayaFallback::InvalidResponse);
            }
        }
        Err(LayaFallback::InvalidResponse)
    }
}

fn parse_reply(
    reply: &Value,
    input: &SelectionInput,
    config: &LayaConfig,
    trace: &mut LayaTrace,
) -> Result<String, LayaFallback> {
    if let Some(error) = reply.get("error").and_then(Value::as_str) {
        return Err(match error {
            "capacity" => LayaFallback::CapacityExceeded,
            _ => LayaFallback::InferenceError,
        });
    }
    let selected = reply
        .get("choice")
        .and_then(Value::as_str)
        .ok_or(LayaFallback::InvalidResponse)?;
    let probabilities = reply
        .get("probabilities")
        .and_then(Value::as_object)
        .ok_or(LayaFallback::InvalidResponse)?;
    let expected: HashSet<&str> = input
        .candidates
        .iter()
        .map(|c| c.id.as_str())
        .chain(["escalate"])
        .collect();
    if probabilities.len() != expected.len()
        || !probabilities
            .keys()
            .all(|id| expected.contains(id.as_str()))
        || !expected.contains(selected)
    {
        return Err(LayaFallback::InvalidResponse);
    }
    let mut sum = 0.0;
    let mut highest = 0.0_f64;
    let mut runner_up = 0.0_f64;
    for probability in probabilities.values() {
        let p = probability
            .as_f64()
            .filter(|p| valid_probability(*p))
            .ok_or(LayaFallback::InvalidResponse)?;
        sum += p;
        if p >= highest {
            runner_up = highest;
            highest = p;
        } else {
            runner_up = runner_up.max(p);
        }
    }
    if (sum - 1.0).abs() > 0.01 {
        return Err(LayaFallback::InvalidResponse);
    }
    let selected_probability = probabilities
        .get(selected)
        .and_then(Value::as_f64)
        .ok_or(LayaFallback::InvalidResponse)?;
    if selected_probability + 1e-8 < highest {
        return Err(LayaFallback::InvalidResponse);
    }
    let confidence = reply
        .get("confidence")
        .and_then(Value::as_f64)
        .filter(|p| valid_probability(*p))
        .ok_or(LayaFallback::InvalidResponse)?;
    trace.selected_probability = Some(selected_probability);
    trace.confidence = Some(confidence);
    if selected == "escalate" {
        return Err(LayaFallback::Abstained);
    }
    if selected_probability < config.min_probability {
        return Err(LayaFallback::LowProbability);
    }
    if confidence < config.min_confidence {
        return Err(LayaFallback::LowConfidence);
    }
    if selected_probability - runner_up < config.min_margin {
        return Err(LayaFallback::Ambiguous);
    }
    Ok(selected.to_owned())
}

impl LayaRuntime {
    fn executable(&self) -> &PathBuf {
        match self {
            Self::Python(path) | Self::Standalone(path) => path,
        }
    }
}

fn valid_probability(p: f64) -> bool {
    p.is_finite() && (0.0..=1.0).contains(&p)
}

fn valid_input(input: &SelectionInput) -> bool {
    if input.state.task.trim().is_empty()
        || input.state.task.len() > 12_000
        || input.state.context.len() > 40_000
        || input.candidates.len() > 16
    {
        return false;
    }
    let mut ids = HashSet::new();
    input.candidates.iter().all(|candidate| {
        !candidate.id.is_empty()
            && candidate.id.len() <= 64
            && candidate.id != "escalate"
            && candidate
                .id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
            && !candidate.description.trim().is_empty()
            && candidate.description.len() <= 2_000
            && ids.insert(candidate.id.as_str())
    })
}

fn fingerprint<T: Serialize>(value: &T) -> String {
    let data = serde_json::to_vec(value).expect("serializable decision input");
    format!("v1:{:x}", Sha256::digest(data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use jev_core::{Candidate, DecisionState};

    fn input() -> SelectionInput {
        SelectionInput {
            state: DecisionState {
                task: "Fix a failing discount test".into(),
                context: "Expected 90, got 100".into(),
                state_version: 4,
            },
            candidates: vec![
                Candidate {
                    id: "inspect_discount".into(),
                    description: "Inspect discount implementation".into(),
                },
                Candidate {
                    id: "inspect_network".into(),
                    description: "Inspect network settings".into(),
                },
            ],
        }
    }

    fn config() -> LayaConfig {
        let dir = tempfile::tempdir().expect("temp model dir");
        std::fs::write(dir.path().join("coreml_config.json"), "{}").expect("manifest");
        // The worker only checks for a real directory in this protocol test.
        LayaConfig::new(PathBuf::from("python3"), dir.keep())
    }

    const MOCK_WORKER: &str = r#"
import json,os,sys
print('\nKEEL_LAYA_JSON:'+json.dumps({'ready': True, 'model': 'laya-coreml'}), flush=True)
for line in sys.stdin:
    request=json.loads(line)
    ids=[pair[0] for pair in request['criteria']]
    selected='inspect_discount'
    print('\nKEEL_LAYA_JSON:'+json.dumps({'choice':selected,'probabilities':{k:(1.0 if k==selected else 0.0) for k in ids},'confidence':0.99}),flush=True)
"#;

    #[tokio::test]
    async fn warm_worker_and_exact_opaque_id() {
        let selector = LayaSelector::new(config())
            .expect("selector")
            .with_test_script(MOCK_WORKER);
        let first = selector.select(input()).await;
        let second = selector.select(input()).await;
        assert_eq!(first.selected_id.as_deref(), Some("inspect_discount"));
        assert_eq!(second.selected_id.as_deref(), Some("inspect_discount"));
        assert!(first.trace.latency_ms.is_some());
        assert_eq!(first.trace.selected_probability, Some(1.0));
        assert!(selector.worker.lock().await.is_some());
    }

    #[test]
    fn invalid_or_unlisted_model_choice_is_rejected() {
        let mut trace = LayaTrace {
            decision_schema: "test",
            model: "test",
            state_version: 0,
            state_fingerprint: String::new(),
            candidates_fingerprint: String::new(),
            selected_id: None,
            selected_probability: None,
            confidence: None,
            fallback: None,
            latency_ms: None,
        };
        let forged = json!({"choice":"run_shell","probabilities":{"run_shell":1.0,"escalate":0.0},"confidence":1.0});
        assert_eq!(
            parse_reply(&forged, &input(), &config(), &mut trace),
            Err(LayaFallback::InvalidResponse)
        );
        let ambiguous = json!({
            "choice":"inspect_discount",
            "probabilities":{"inspect_discount":0.46,"inspect_network":0.44,"escalate":0.10},
            "confidence":0.5
        });
        assert_eq!(
            parse_reply(&ambiguous, &input(), &config(), &mut trace),
            Err(LayaFallback::Ambiguous)
        );
    }

    #[tokio::test]
    async fn absent_model_falls_back_without_starting_python() {
        let selector = LayaSelector::new(LayaConfig::new(
            PathBuf::from("python3"),
            PathBuf::from("/definitely/missing/laya/model"),
        ))
        .expect("selector");
        let outcome = selector.select(input()).await;
        assert_eq!(outcome.trace.fallback, Some(LayaFallback::ModelUnavailable));
        assert!(selector.worker.lock().await.is_none());
    }

    #[tokio::test]
    async fn real_model_when_explicitly_configured() {
        let Ok(python) = std::env::var("LAYA_PYTHON") else {
            return;
        };
        let Ok(model) = std::env::var("LAYA_MODEL_DIR") else {
            return;
        };
        let selector =
            LayaSelector::new(LayaConfig::new(python.into(), model.into())).expect("selector");
        let outcome = selector.select(input()).await;
        eprintln!(
            "local Laya decision: id={:?}, probability={:?}, confidence={:?}, fallback={:?}, latency_ms={:?}",
            outcome.selected_id,
            outcome.trace.selected_probability,
            outcome.trace.confidence,
            outcome.trace.fallback,
            outcome.trace.latency_ms
        );
        let warm = selector.select(input()).await;
        eprintln!(
            "warm local Laya decision: id={:?}, fallback={:?}, latency_ms={:?}",
            warm.selected_id, warm.trace.fallback, warm.trace.latency_ms
        );
        assert_eq!(warm.selected_id, outcome.selected_id);
        assert!(
            outcome.selected_id.is_some()
                || matches!(
                    outcome.trace.fallback,
                    Some(
                        LayaFallback::Abstained
                            | LayaFallback::LowProbability
                            | LayaFallback::LowConfidence
                            | LayaFallback::CapacityExceeded
                    )
                ),
            "unexpected local inference failure: {:?}",
            outcome.trace.fallback
        );
    }
}
