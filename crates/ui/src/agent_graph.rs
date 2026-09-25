//! Agent graph — every active session as an agent node, the subagent tasks it
//! spawned as child nodes, and the files it reads or writes as context nodes
//! shared between agents. Select a node to steer (or stop) the agent that owns
//! it; double-click an agent to open its session.
//!
//! Layered layout, no physics: agents on the top row, subagents under their
//! parent, context files on the bottom row at the mean x of the agents that
//! touch them — so a file two agents both edit sits between them with both
//! edges converging on it.
//!
//! Each shown chat gets its own `WatchDocMessages` subscription while the
//! graph is open (the app state only watches the selected chat); the entity
//! is dropped on close, which drops the watches.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use chrono::Utc;
use gpui::{
    AnyElement, Context, Entity, EventEmitter, FocusHandle, Focusable as _, Hsla, PathBuilder,
    Render, SharedString, Subscription, Task, Window, canvas, div, point, prelude::*, px,
};
use keel_doc::{
    MessagePart, MessageRole, SessionCommandPayload, SessionMessageEntry, TranscriptFrame,
};
use keel_proto::{Chat, ChatIndicator, HarnessId, ToolCall};
use keel_rpc::methods;

use crate::composer::{ComposerInput, ComposerInputEvent};
use crate::icons::{self, icon};
use crate::motion::{self, KEEL_PULSE};
use crate::popover;
use crate::state::{AppState, EngineHandle};
use crate::theme::Theme;

// ---------------------------------------------------------------------------
// Model (pure — unit-tested below)
// ---------------------------------------------------------------------------

/// Most recent subagents shown per agent.
const MAX_SUBAGENTS: usize = 6;
/// Most recently touched files shown per agent (files shared with another
/// agent are always shown).
const MAX_FILES: usize = 5;
/// Latest steps shown on an agent card.
const CARD_STEPS: usize = 3;
/// Steps listed in the process panel (newest kept).
const PANEL_STEPS: usize = 300;
/// Character cap for one step's detail line.
const STEP_CHARS: usize = 220;
/// Agents shown at most (most recent first).
const MAX_AGENTS: usize = 12;
/// Idle sessions stay on the graph this long after their last message.
const IDLE_WINDOW_MIN: i64 = 60;

const PAD: f32 = 32.0;
const H_GAP: f32 = 20.0;
const ROW_GAP: f32 = 72.0;
const AGENT_W: f32 = 232.0;
const AGENT_H: f32 = 118.0;
const SUB_W: f32 = 172.0;
const SUB_H: f32 = 58.0;
const CTX_W: f32 = 164.0;
const CTX_H: f32 = 42.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    Agent,
    Subagent,
    Context,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStatus {
    Working,
    Waiting,
    Errored,
    Done,
    Idle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    Spawn,
    Read,
    Write,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphNode {
    /// Stable selection key: `a:{chat}`, `s:{chat}:{part}`, `f:{path}`.
    pub key: String,
    pub kind: NodeKind,
    /// Owning chat (context nodes: the first agent that touched the file).
    pub chat_id: String,
    pub label: String,
    pub detail: String,
    pub status: NodeStatus,
    pub harness: Option<HarnessId>,
    /// Context nodes: how many agents touch this file.
    pub shared_by: usize,
    /// Agent nodes: the latest steps, oldest first (the card's live log).
    pub recent: Vec<Step>,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraphEdge {
    pub from: usize,
    pub to: usize,
    pub kind: EdgeKind,
    /// The tool call behind this edge is in flight right now.
    pub live: bool,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Graph {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub width: f32,
    pub height: f32,
}

/// One agent's input to [`build_graph`].
pub struct AgentSnapshot<'a> {
    pub chat_id: &'a str,
    pub title: String,
    pub harness: Option<HarnessId>,
    pub status: ChatIndicator,
    pub cwd: Option<&'a str>,
    pub transcript: &'a [SessionMessageEntry],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepKind {
    Prompt,
    Message,
    Tool,
    Spawn,
    Question,
    Error,
    Result,
}

/// One entry of an agent's process: a prompt, a message, a tool call, a
/// subagent spawn, a question to the user, or an error.
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    pub kind: StepKind,
    pub label: String,
    pub detail: String,
    pub status: NodeStatus,
}

fn snippet(text: &str) -> String {
    let line = crate::transcript::single_line(text.trim());
    if line.chars().count() > STEP_CHARS {
        let cut: String = line.chars().take(STEP_CHARS).collect();
        format!("{}…", cut.trim_end())
    } else {
        line
    }
}

fn tool_status(resolved: bool, is_error: bool, working: bool) -> NodeStatus {
    match (resolved, is_error, working) {
        (true, true, _) => NodeStatus::Errored,
        (true, false, _) => NodeStatus::Done,
        (false, _, true) => NodeStatus::Working,
        // Unresolved in a run that is no longer live: interrupted.
        (false, _, false) => NodeStatus::Idle,
    }
}

/// Every step of a transcript, in order. `working` marks unresolved tool
/// calls as running (else they were interrupted).
pub fn process_steps(transcript: &[SessionMessageEntry], working: bool) -> Vec<Step> {
    let spawned: HashMap<String, NodeStatus> = subagents(transcript, working)
        .into_iter()
        .map(|(id, _, status)| (id, status))
        .collect();
    let mut steps = Vec::new();
    for entry in transcript {
        for part in &entry.parts {
            let step = match part {
                MessagePart::Text { text, .. } if text.trim().is_empty() => continue,
                MessagePart::Text { text, .. } if entry.role == MessageRole::User => Step {
                    kind: StepKind::Prompt,
                    label: "You".into(),
                    detail: snippet(text),
                    status: NodeStatus::Done,
                },
                MessagePart::Text { text, .. } => Step {
                    kind: StepKind::Message,
                    label: "Said".into(),
                    detail: snippet(text),
                    status: NodeStatus::Done,
                },
                MessagePart::Tool {
                    id,
                    call,
                    is_error,
                    resolved,
                    output,
                    ..
                } => {
                    let status = tool_status(*resolved, *is_error, working);
                    match subagent_label(call, output.as_deref()) {
                        Some(label) => Step {
                            kind: StepKind::Spawn,
                            label: "Spawned subagent".into(),
                            detail: label,
                            status: spawned.get(id).copied().unwrap_or(status),
                        },
                        None => {
                            let (label, detail) = keel_proto::view::tool_chip_content(call);
                            Step {
                                kind: StepKind::Tool,
                                label: label.into(),
                                detail: snippet(&detail),
                                status,
                            }
                        }
                    }
                }
                MessagePart::Input {
                    questions,
                    resolved,
                    ..
                } => Step {
                    kind: StepKind::Question,
                    label: "Asked you".into(),
                    detail: snippet(
                        &questions
                            .iter()
                            .map(|q| q.question.as_str())
                            .collect::<Vec<_>>()
                            .join(" · "),
                    ),
                    status: if *resolved {
                        NodeStatus::Done
                    } else {
                        NodeStatus::Waiting
                    },
                },
                MessagePart::Error { message, .. } => Step {
                    kind: StepKind::Error,
                    label: "Error".into(),
                    detail: snippet(message),
                    status: NodeStatus::Errored,
                },
                MessagePart::Decision { .. } => continue,
            };
            steps.push(step);
        }
    }
    steps
}

/// First line of a Grok spawn that returned before its subagent finished.
const BACKGROUND_SPAWN: &str = "Subagent started in background";

/// The call a parent makes to collect background subagents' results.
fn is_subagent_wait(call: &ToolCall) -> bool {
    matches!(call, ToolCall::Unknown { name, .. }
        if name == "get_command_or_subagent_output"
            || name.starts_with("Get task output")
            || name.starts_with("multi-wait"))
}

/// Every subagent in a transcript: `(part id, label, status)`, in order.
///
/// A foreground spawn's status is its tool call's. A background spawn
/// resolves at once, so it stays running until a later wait call resolves.
// ponytail: the doc strips the wait's task ids, so a resolved wait marks
// every earlier background spawn finished — exact for wait-all, early for a
// wait on a subset. Keep the ids through sanitize if that matters.
pub fn subagents(
    transcript: &[SessionMessageEntry],
    working: bool,
) -> Vec<(String, String, NodeStatus)> {
    let mut out: Vec<(String, String, NodeStatus)> = Vec::new();
    let mut pending: Vec<usize> = Vec::new();
    for part in transcript.iter().flat_map(|e| e.parts.iter()) {
        let MessagePart::Tool {
            id,
            call,
            is_error,
            resolved,
            output,
            ..
        } = part
        else {
            continue;
        };
        if let Some(label) = subagent_label(call, output.as_deref()) {
            let background = *resolved
                && !*is_error
                && output
                    .as_deref()
                    .is_some_and(|o| o.starts_with(BACKGROUND_SPAWN));
            let status = match (background, working) {
                (true, true) => NodeStatus::Working,
                (true, false) => NodeStatus::Idle,
                (false, _) => tool_status(*resolved, *is_error, working),
            };
            if background {
                pending.push(out.len());
            }
            out.push((id.clone(), label, status));
        } else if is_subagent_wait(call) && *resolved {
            let done = if *is_error {
                NodeStatus::Errored
            } else {
                NodeStatus::Done
            };
            for ix in pending.drain(..) {
                out[ix].2 = done;
            }
        }
    }
    out
}

/// A subagent's own process: what it was asked, and what it returned.
/// Adapters report the spawn and its result only — the subagent's inner tool
/// calls are not streamed to the host.
pub fn subagent_steps(
    transcript: &[SessionMessageEntry],
    part_id: &str,
    status: NodeStatus,
) -> Vec<Step> {
    let Some(MessagePart::Tool { call, output, .. }) = transcript
        .iter()
        .flat_map(|e| e.parts.iter())
        .find(|p| p.id() == part_id)
    else {
        return Vec::new();
    };
    let ToolCall::Unknown { input, .. } = call else {
        return Vec::new();
    };
    let field = |key: &str| {
        input
            .as_ref()
            .and_then(|i| i.get(key))
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
    };
    let mut steps = Vec::new();
    if let Some(kind) = field("subagent_type") {
        steps.push(Step {
            kind: StepKind::Spawn,
            label: "Type".into(),
            detail: kind,
            status: NodeStatus::Done,
        });
    }
    if let Some(prompt) = field("prompt").or_else(|| field("description")) {
        steps.push(Step {
            kind: StepKind::Prompt,
            label: "Asked to".into(),
            detail: snippet(&prompt),
            status: NodeStatus::Done,
        });
    }
    let background = output
        .as_deref()
        .is_some_and(|o| o.starts_with(BACKGROUND_SPAWN));
    if background {
        steps.push(Step {
            kind: StepKind::Spawn,
            label: "Started in background".into(),
            detail: String::new(),
            status: NodeStatus::Done,
        });
    }
    steps.push(Step {
        kind: StepKind::Result,
        label: match status {
            NodeStatus::Working => "Running",
            NodeStatus::Errored => "Failed",
            NodeStatus::Done => "Returned",
            _ => "Interrupted",
        }
        .into(),
        detail: if background {
            String::new()
        } else {
            output.as_deref().map(snippet).unwrap_or_default()
        },
        status,
    });
    steps
}

fn node_status(status: ChatIndicator) -> NodeStatus {
    match status {
        ChatIndicator::Working => NodeStatus::Working,
        ChatIndicator::AwaitingInput => NodeStatus::Waiting,
        ChatIndicator::Errored => NodeStatus::Errored,
        ChatIndicator::Completed => NodeStatus::Done,
        ChatIndicator::Idle => NodeStatus::Idle,
    }
}

/// A subagent spawn: ACP `task` tools arrive as `Unknown { name: "Task: …" }`
/// (see `acp::normalize`); other adapters name the tool `Task`/`Agent` or
/// carry a `subagent_type` input.
pub fn subagent_label(call: &ToolCall, output: Option<&str>) -> Option<String> {
    let ToolCall::Unknown { name, input } = call else {
        return None;
    };
    // Grok spawns report "Subagent started in background" and the doc keeps
    // only the description as the name (inputs are stripped before sync).
    if output.is_some_and(|o| o.starts_with(BACKGROUND_SPAWN)) && !name.starts_with("Task:") {
        return Some(crate::transcript::single_line(name));
    }
    let field = |key: &str| {
        input
            .as_ref()
            .and_then(|i| i.get(key))
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
    };
    let named = name == "Task"
        || name.starts_with("Task:")
        || name.eq_ignore_ascii_case("agent")
        || name == "spawn_agent"
        || name == "spawn_subagent";
    if !named && field("subagent_type").is_none() {
        return None;
    }
    let label = name
        .strip_prefix("Task:")
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .or_else(|| field("description"))
        .or_else(|| field("subagent_type"))
        .unwrap_or_else(|| "Subagent".into());
    Some(crate::transcript::single_line(&label))
}

/// The file a tool call reads or writes: `(path, wrote)`.
fn touched_path(call: &ToolCall) -> Option<(&str, bool)> {
    let (path, wrote) = match call {
        ToolCall::ReadFile { path } => (path.as_str(), false),
        ToolCall::WriteFile { path, .. } | ToolCall::EditFile { path, .. } => (path.as_str(), true),
        ToolCall::ApplyPatch { path: Some(path) } => (path.as_str(), true),
        ToolCall::Search {
            path: Some(path), ..
        } => (path.as_str(), false),
        _ => return None,
    };
    let path = path.trim();
    (!path.is_empty()).then_some((path, wrote))
}

/// Absolute key for a touched path so two agents naming the same file
/// (one relative, one absolute) meet on one node.
fn normalize_path(path: &str, cwd: Option<&str>) -> String {
    let path = path.strip_prefix("./").unwrap_or(path);
    let joined = match cwd {
        Some(cwd) if !path.starts_with('/') && !path.starts_with('~') => {
            format!("{}/{}", cwd.trim_end_matches('/'), path)
        }
        _ => path.to_string(),
    };
    // Resolve `.` and `..` lexically so `src/../lib.rs` meets `lib.rs`.
    let absolute = joined.starts_with('/');
    let mut parts: Vec<&str> = Vec::new();
    for seg in joined.split('/') {
        match seg {
            "" | "." => {}
            ".." if parts.last().is_some_and(|p| *p != "..") => {
                parts.pop();
            }
            // Above the root stays at the root; a relative path keeps it.
            ".." if absolute => {}
            seg => parts.push(seg),
        }
    }
    let rest = parts.join("/");
    if absolute { format!("/{rest}") } else { rest }
}

/// `(file name, parent shown relative to cwd)`.
fn split_display(path: &str, cwd: Option<&str>) -> (String, String) {
    let (parent, name) = match path.rsplit_once('/') {
        Some((parent, name)) => (parent, name),
        None => ("", path),
    };
    let parent = match cwd.map(|c| c.trim_end_matches('/')) {
        Some(cwd) if parent == cwd => ".".to_string(),
        Some(cwd) => parent
            .strip_prefix(cwd)
            .and_then(|rest| rest.strip_prefix('/'))
            .map(str::to_owned)
            .unwrap_or_else(|| parent.to_string()),
        None => parent.to_string(),
    };
    (name.to_string(), parent)
}

struct FileTouch {
    wrote: bool,
    order: usize,
    live: bool,
}

struct Scan {
    activity: String,
    /// `(part id, label, status)` in transcript order.
    subagents: Vec<(String, String, NodeStatus)>,
    files: HashMap<String, FileTouch>,
}

fn scan(agent: &AgentSnapshot<'_>) -> Scan {
    let working = agent.status == ChatIndicator::Working;
    let mut subagents = subagents(agent.transcript, working);
    let mut files: HashMap<String, FileTouch> = HashMap::new();
    let mut last_tool: Option<(&ToolCall, bool)> = None;
    let mut order = 0usize;
    for part in agent.transcript.iter().flat_map(|e| e.parts.iter()) {
        order += 1;
        let MessagePart::Tool { call, resolved, .. } = part else {
            last_tool = None;
            continue;
        };
        last_tool = Some((call, *resolved));
        if let Some((path, wrote)) = touched_path(call) {
            let key = normalize_path(path, agent.cwd);
            let touch = files.entry(key).or_insert(FileTouch {
                wrote: false,
                order,
                live: false,
            });
            touch.wrote |= wrote;
            touch.order = order;
            // Any unresolved touch keeps the file live.
            touch.live |= !*resolved && working;
        }
    }
    let activity = match (agent.status, last_tool) {
        (ChatIndicator::Working, Some((call, false))) => {
            let (label, detail) = keel_proto::view::tool_chip_content(call);
            format!("{label} {detail}")
        }
        (ChatIndicator::Working, _) => "Thinking…".into(),
        (ChatIndicator::AwaitingInput, _) => "Waiting for your answer".into(),
        (ChatIndicator::Errored, _) => "Stopped with an error".into(),
        (ChatIndicator::Completed, _) => "Finished — not seen yet".into(),
        (ChatIndicator::Idle, _) => "Idle".into(),
    };
    if subagents.len() > MAX_SUBAGENTS {
        subagents.drain(..subagents.len() - MAX_SUBAGENTS);
    }
    Scan {
        activity,
        subagents,
        files,
    }
}

/// Build and lay out the graph for `agents` (already ordered and capped).
pub fn build_graph(agents: &[AgentSnapshot<'_>]) -> Graph {
    let scans: Vec<Scan> = agents.iter().map(scan).collect();

    // Which agents touch each file.
    let mut touchers: HashMap<&str, Vec<usize>> = HashMap::new();
    for (ai, scan) in scans.iter().enumerate() {
        for path in scan.files.keys() {
            touchers.entry(path.as_str()).or_default().push(ai);
        }
    }
    // Per agent: its most recent files, plus every file it shares.
    let mut kept: HashSet<&str> = HashSet::new();
    for scan in &scans {
        let mut recent: Vec<(&str, usize)> = scan
            .files
            .iter()
            .map(|(path, touch)| (path.as_str(), touch.order))
            .collect();
        recent.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        kept.extend(recent.iter().take(MAX_FILES).map(|(path, _)| *path));
        kept.extend(
            scan.files
                .keys()
                .map(String::as_str)
                .filter(|path| touchers.get(path).is_some_and(|t| t.len() > 1)),
        );
    }

    let any_subagents = scans.iter().any(|s| !s.subagents.is_empty());
    let y_agent = PAD;
    let y_sub = y_agent + AGENT_H + ROW_GAP;
    let y_ctx = if any_subagents {
        y_sub + SUB_H + ROW_GAP
    } else {
        y_sub
    };

    let mut graph = Graph::default();
    let mut agent_center: Vec<f32> = Vec::with_capacity(agents.len());
    let mut cursor = PAD;
    for (agent, scan) in agents.iter().zip(&scans) {
        let n = scan.subagents.len() as f32;
        let subs_w = if n > 0.0 {
            n * SUB_W + (n - 1.0) * H_GAP
        } else {
            0.0
        };
        let col_w = AGENT_W.max(subs_w);
        let agent_ix = graph.nodes.len();
        graph.nodes.push(GraphNode {
            key: format!("a:{}", agent.chat_id),
            kind: NodeKind::Agent,
            chat_id: agent.chat_id.to_string(),
            label: agent.title.clone(),
            detail: scan.activity.clone(),
            status: node_status(agent.status),
            harness: agent.harness,
            shared_by: 0,
            recent: {
                let steps = process_steps(agent.transcript, agent.status == ChatIndicator::Working);
                let skip = steps.len().saturating_sub(CARD_STEPS);
                steps.into_iter().skip(skip).collect()
            },
            x: cursor + (col_w - AGENT_W) / 2.0,
            y: y_agent,
            w: AGENT_W,
            h: AGENT_H,
        });
        agent_center.push(cursor + col_w / 2.0);
        let mut sx = cursor + (col_w - subs_w) / 2.0;
        for (part_id, label, status) in &scan.subagents {
            let ix = graph.nodes.len();
            graph.nodes.push(GraphNode {
                key: format!("s:{}:{}", agent.chat_id, part_id),
                kind: NodeKind::Subagent,
                chat_id: agent.chat_id.to_string(),
                label: label.clone(),
                detail: {
                    let state = match status {
                        NodeStatus::Working => "Running",
                        NodeStatus::Errored => "Failed",
                        NodeStatus::Done => "Done",
                        _ => "Interrupted",
                    };
                    let asked = subagent_steps(agent.transcript, part_id, *status)
                        .into_iter()
                        .find(|s| s.kind == StepKind::Prompt)
                        .map(|s| s.detail);
                    match asked {
                        Some(asked) => format!("{state} · {asked}"),
                        None => state.into(),
                    }
                },
                status: *status,
                harness: agent.harness,
                shared_by: 0,
                recent: Vec::new(),
                x: sx,
                y: y_sub,
                w: SUB_W,
                h: SUB_H,
            });
            graph.edges.push(GraphEdge {
                from: agent_ix,
                to: ix,
                kind: EdgeKind::Spawn,
                live: *status == NodeStatus::Working,
            });
            sx += SUB_W + H_GAP;
        }
        cursor += col_w + H_GAP * 2.0;
    }
    let mut right = cursor - H_GAP * 2.0;

    // Context row: sort by barycenter, then sweep right so nothing overlaps.
    let mut files: Vec<(&str, f32)> = kept
        .iter()
        .map(|path| {
            let owners = &touchers[path];
            let bary = owners.iter().map(|&ai| agent_center[ai]).sum::<f32>() / owners.len() as f32;
            (*path, bary)
        })
        .collect();
    files.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(b.0)));
    let agent_ix_of: Vec<usize> = graph
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.kind == NodeKind::Agent)
        .map(|(ix, _)| ix)
        .collect();
    let mut prev_right = PAD - H_GAP;
    for (path, bary) in &files {
        let owners = &touchers[path];
        let first = owners[0];
        let x = (bary - CTX_W / 2.0).max(prev_right + H_GAP);
        prev_right = x + CTX_W;
        let (label, detail) = split_display(path, agents[first].cwd);
        let status = if owners.iter().any(|&ai| scans[ai].files[*path].live) {
            NodeStatus::Working
        } else {
            NodeStatus::Idle
        };
        let ix = graph.nodes.len();
        graph.nodes.push(GraphNode {
            key: format!("f:{path}"),
            kind: NodeKind::Context,
            chat_id: agents[first].chat_id.to_string(),
            label,
            detail,
            status,
            harness: None,
            shared_by: owners.len(),
            recent: Vec::new(),
            x,
            y: y_ctx,
            w: CTX_W,
            h: CTX_H,
        });
        for &ai in owners {
            let touch = &scans[ai].files[*path];
            graph.edges.push(GraphEdge {
                from: agent_ix_of[ai],
                to: ix,
                kind: if touch.wrote {
                    EdgeKind::Write
                } else {
                    EdgeKind::Read
                },
                live: touch.live,
            });
        }
    }
    right = right.max(prev_right);
    let bottom = graph.nodes.iter().map(|n| n.y + n.h).fold(0.0f32, f32::max);
    graph.width = right + PAD;
    graph.height = bottom + PAD;
    graph
}

/// Edge endpoints + bezier controls: bottom-center of `from` to top-center of
/// `to`, vertical tangents at both ends.
fn edge_curve(from: &GraphNode, to: &GraphNode) -> [(f32, f32); 4] {
    let p0 = (from.x + from.w / 2.0, from.y + from.h);
    let p3 = (to.x + to.w / 2.0, to.y);
    let mid = (p3.1 - p0.1) / 2.0;
    [p0, (p0.0, p0.1 + mid), (p3.0, p3.1 - mid), p3]
}

fn bezier_at(c: &[(f32, f32); 4], t: f32) -> (f32, f32) {
    let u = 1.0 - t;
    let (a, b, cc, d) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
    (
        a * c[0].0 + b * c[1].0 + cc * c[2].0 + d * c[3].0,
        a * c[0].1 + b * c[1].1 + cc * c[2].1 + d * c[3].1,
    )
}

// ---------------------------------------------------------------------------
// View
// ---------------------------------------------------------------------------

pub enum AgentGraphEvent {
    OpenChat(String),
    Close,
}

struct ChatWatch {
    entries: Vec<SessionMessageEntry>,
    _task: Task<()>,
}

pub struct AgentGraph {
    state: Entity<AppState>,
    watches: HashMap<String, ChatWatch>,
    selected: Option<String>,
    input: Entity<ComposerInput>,
    notice: Option<(SharedString, bool)>,
    send_task: Option<Task<()>>,
    _subs: Vec<Subscription>,
}

impl EventEmitter<AgentGraphEvent> for AgentGraph {}

/// The sessions the graph shows: everything not idle, plus idle sessions
/// active within [`IDLE_WINDOW_MIN`], most recent first.
fn shown_chats(state: &AppState) -> Vec<(ChatIndicator, Chat)> {
    let now = Utc::now();
    state
        .overview_chats(now)
        .into_iter()
        .filter(|(status, chat)| {
            *status != ChatIndicator::Idle
                || chat
                    .last_message_at
                    .is_some_and(|at| (now - at).num_minutes() < IDLE_WINDOW_MIN)
        })
        .take(MAX_AGENTS)
        .map(|(status, chat)| (status, chat.clone()))
        .collect()
}

impl AgentGraph {
    pub fn new(state: Entity<AppState>, cx: &mut Context<Self>) -> Self {
        let input = cx.new(|cx| ComposerInput::new("Select an agent to steer it", cx));
        let subs = vec![
            cx.observe(&state, |this: &mut Self, _, cx| {
                this.sync_watches(cx);
                cx.notify();
            }),
            cx.subscribe(&input, |this: &mut Self, _, event, cx| {
                if matches!(event, ComposerInputEvent::Submitted) {
                    this.steer(cx);
                }
            }),
        ];
        let mut this = Self {
            state,
            watches: HashMap::new(),
            selected: None,
            input,
            notice: None,
            send_task: None,
            _subs: subs,
        };
        this.sync_watches(cx);
        this
    }

    pub fn input_focus(&self, cx: &gpui::App) -> FocusHandle {
        self.input.focus_handle(cx)
    }

    fn sync_watches(&mut self, cx: &mut Context<Self>) {
        let Some(engine) = self.state.read(cx).engine().cloned() else {
            return;
        };
        let wanted: HashSet<String> = shown_chats(self.state.read(cx))
            .into_iter()
            .map(|(_, c)| c.id)
            .collect();
        self.watches.retain(|id, _| wanted.contains(id));
        for id in wanted {
            if !self.watches.contains_key(&id) {
                let task = spawn_watch(cx, engine.clone(), id.clone());
                self.watches.insert(
                    id,
                    ChatWatch {
                        entries: Vec::new(),
                        _task: task,
                    },
                );
            }
        }
    }

    fn graph(&self, cx: &gpui::App) -> (Graph, HashMap<String, ChatIndicator>) {
        let chats = shown_chats(self.state.read(cx));
        let empty: Vec<SessionMessageEntry> = Vec::new();
        let snapshots: Vec<AgentSnapshot<'_>> = chats
            .iter()
            .map(|(status, chat)| AgentSnapshot {
                chat_id: &chat.id,
                title: crate::transcript::single_line(
                    chat.title.as_deref().unwrap_or("New session"),
                ),
                harness: chat.config.as_ref().map(|c| c.harness),
                status: *status,
                cwd: chat.cwd.as_deref(),
                transcript: self
                    .watches
                    .get(&chat.id)
                    .map(|w| w.entries.as_slice())
                    .unwrap_or(&empty),
            })
            .collect();
        let graph = build_graph(&snapshots);
        let statuses = chats.iter().map(|(s, c)| (c.id.clone(), *s)).collect();
        (graph, statuses)
    }

    fn select(&mut self, key: String, cx: &mut Context<Self>) {
        self.selected = Some(key);
        self.notice = None;
        cx.notify();
    }

    /// `(chat id, chat title, subagent label)` of the steer target.
    fn target(&self, graph: &Graph) -> Option<(String, String, Option<String>)> {
        let key = self.selected.as_deref()?;
        let node = graph
            .nodes
            .iter()
            .find(|n| n.key == key && n.kind != NodeKind::Context)?;
        let agent = graph
            .nodes
            .iter()
            .find(|n| n.kind == NodeKind::Agent && n.chat_id == node.chat_id)?;
        let sub = (node.kind == NodeKind::Subagent).then(|| node.label.clone());
        Some((node.chat_id.clone(), agent.label.clone(), sub))
    }

    fn queue(
        &mut self,
        chat_id: String,
        command: serde_json::Value,
        ok: String,
        cx: &mut Context<Self>,
    ) {
        let Some(engine) = self.state.read(cx).engine().cloned() else {
            self.notice = Some(("Engine not connected".into(), true));
            cx.notify();
            return;
        };
        let params = serde_json::json!({ "chatId": chat_id, "command": command });
        self.send_task = Some(cx.spawn(async move |this, cx| {
            let result = engine.client().call(methods::QUEUE_COMMAND, params).await;
            this.update(cx, |this, cx| {
                this.notice = Some(match result {
                    Ok(_) => (ok.into(), false),
                    Err(err) => (format!("Failed: {err}").into(), true),
                });
                cx.notify();
            })
            .ok();
        }));
    }

    fn steer(&mut self, cx: &mut Context<Self>) {
        let (graph, _) = self.graph(cx);
        let Some((chat_id, title, sub)) = self.target(&graph) else {
            return;
        };
        let text = self.input.read(cx).text().trim().to_string();
        if text.is_empty() {
            return;
        }
        // Harnesses expose one steering mailbox per run — the parent's. A
        // subagent is addressed through its parent, named explicitly.
        let prompt = match &sub {
            Some(label) => format!("Regarding your subagent task \"{label}\": {text}"),
            None => text,
        };
        let payload = SessionCommandPayload::Steer {
            prompt,
            message_id: Some(uuid::Uuid::new_v4().to_string()),
        };
        let Ok(command) = serde_json::to_value(&payload) else {
            return;
        };
        self.input.update(cx, |input, cx| input.set_text("", cx));
        self.queue(chat_id, command, format!("Sent to {title}"), cx);
    }

    fn stop(&mut self, cx: &mut Context<Self>) {
        let (graph, _) = self.graph(cx);
        let Some((chat_id, title, _)) = self.target(&graph) else {
            return;
        };
        self.queue(
            chat_id,
            serde_json::json!({ "kind": "interrupt" }),
            format!("Stop sent to {title}"),
            cx,
        );
    }
}

/// Per-chat transcript pump: same resubscribe contract as the app state's
/// selected-chat watch, writing into this graph's copy.
fn spawn_watch(cx: &mut Context<AgentGraph>, handle: EngineHandle, chat_id: String) -> Task<()> {
    const RETRY_DELAY: Duration = Duration::from_secs(2);
    cx.spawn(async move |this, cx| {
        loop {
            let params = serde_json::json!({ "chatId": chat_id });
            if let Ok(mut rx) = handle
                .client()
                .subscribe(methods::WATCH_DOC_MESSAGES, params)
                .await
            {
                while let Some(value) = rx.recv().await {
                    let Ok(frame) = serde_json::from_value::<TranscriptFrame>(value) else {
                        break;
                    };
                    let mut desync = false;
                    let alive = this.update(cx, |this, cx| {
                        if let Some(watch) = this.watches.get_mut(&chat_id) {
                            desync = keel_doc::apply_transcript_frame(&mut watch.entries, frame)
                                .is_err();
                            cx.notify();
                        }
                    });
                    if alive.is_err() {
                        return;
                    }
                    if desync {
                        break;
                    }
                }
            }
            if this.update(cx, |_, _| {}).is_err() {
                return;
            }
            cx.background_executor().timer(RETRY_DELAY).await;
        }
    })
}

fn plural(n: usize, noun: &str) -> String {
    if n == 1 {
        format!("1 {noun}")
    } else {
        format!("{n} {noun}s")
    }
}

fn status_color(status: NodeStatus, theme: &Theme) -> Hsla {
    match status {
        NodeStatus::Working => theme.busy,
        NodeStatus::Waiting => theme.warning,
        NodeStatus::Errored => theme.danger,
        NodeStatus::Done => theme.success,
        NodeStatus::Idle => theme.text_faint,
    }
}

fn edge_color(kind: EdgeKind, theme: &Theme) -> Hsla {
    match kind {
        EdgeKind::Spawn => theme.accent,
        EdgeKind::Write => theme.warning,
        EdgeKind::Read => theme.text_faint,
    }
}

/// One compact line of an agent card's live log.
fn step_line(step: &Step, theme: &Theme) -> impl IntoElement {
    div()
        .flex()
        .items_center()
        .gap(px(5.0))
        .min_w_0()
        .child(
            div()
                .flex_none()
                .size(px(5.0))
                .rounded_full()
                .bg(status_color(step.status, theme)),
        )
        .child(
            div()
                .min_w_0()
                .truncate()
                .text_size(px(10.5))
                .text_color(if step.status == NodeStatus::Working {
                    theme.text
                } else {
                    theme.text_muted
                })
                .child(format!("{} {}", step.label, step.detail)),
        )
}

/// One row of the process panel: status rail, label, wrapped detail.
fn step_row(step: &Step, theme: &Theme) -> impl IntoElement {
    let label_color = match step.kind {
        StepKind::Prompt => theme.accent,
        StepKind::Spawn => theme.accent,
        StepKind::Error => theme.danger,
        StepKind::Question => theme.warning,
        _ => theme.text,
    };
    div()
        .flex()
        .gap(px(10.0))
        .py(px(6.0))
        .border_b_1()
        .border_color(theme.border.opacity(0.5))
        .child(
            div()
                .flex_none()
                .mt(px(5.0))
                .size(px(7.0))
                .rounded_full()
                .bg(status_color(step.status, theme)),
        )
        .child(
            div()
                .flex_1()
                .min_w_0()
                .flex()
                .flex_col()
                .gap(px(2.0))
                .child(
                    div()
                        .text_size(px(12.0))
                        .font_weight(gpui::FontWeight::MEDIUM)
                        .text_color(label_color)
                        .child(match step.status {
                            NodeStatus::Working => format!("{} · running", step.label),
                            NodeStatus::Errored if step.kind != StepKind::Error => {
                                format!("{} · failed", step.label)
                            }
                            NodeStatus::Waiting => format!("{} · waiting", step.label),
                            _ => step.label.clone(),
                        }),
                )
                .when(!step.detail.is_empty(), |el| {
                    el.child(
                        div()
                            .text_size(px(11.5))
                            .text_color(theme.text_muted)
                            .child(step.detail.clone()),
                    )
                }),
        )
}

fn legend_item(color: Hsla, label: &'static str, theme: &Theme) -> impl IntoElement {
    div()
        .flex()
        .items_center()
        .gap(px(6.0))
        .child(div().w(px(14.0)).h(px(2.0)).rounded(px(1.0)).bg(color))
        .child(
            div()
                .text_size(px(11.0))
                .text_color(theme.text_muted)
                .child(label),
        )
}

impl Render for AgentGraph {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = Theme::of(cx).clone();
        let (graph, statuses) = self.graph(cx);
        let any_live = graph.edges.iter().any(|e| e.live)
            || graph.nodes.iter().any(|n| n.status == NodeStatus::Working);
        let phase = if any_live {
            motion::pulse_delta(&KEEL_PULSE, cx.entity_id(), cx)
        } else {
            0.0
        };
        if self
            .selected
            .as_ref()
            .is_some_and(|key| !graph.nodes.iter().any(|n| &n.key == key))
        {
            self.selected = None;
        }

        let agents = graph
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Agent)
            .count();
        let working = graph
            .nodes
            .iter()
            .filter(|n| n.kind != NodeKind::Context && n.status == NodeStatus::Working)
            .count();
        let subs = graph
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Subagent)
            .count();
        let files = graph
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Context)
            .count();
        let shared = graph
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Context && n.shared_by > 1)
            .count();

        let header = div()
            .flex_none()
            .flex()
            .items_center()
            .gap(px(16.0))
            .px(px(Theme::SPACE_LG))
            .py(px(10.0))
            .border_b_1()
            .border_color(theme.border)
            .child(
                div()
                    .flex()
                    .flex_col()
                    .child(
                        div()
                            .text_size(px(14.0))
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .child("Agents"),
                    )
                    .child(
                        div()
                            .text_size(px(11.0))
                            .text_color(theme.text_muted)
                            .child(format!(
                                "{} · {working} working · {} · {} ({shared} shared)",
                                plural(agents, "agent"),
                                plural(subs, "subagent"),
                                plural(files, "file"),
                            )),
                    ),
            )
            .child(div().flex_1())
            .child(legend_item(theme.accent, "spawned", &theme))
            .child(legend_item(theme.warning, "writes", &theme))
            .child(legend_item(theme.text_faint, "reads", &theme))
            .child(
                div()
                    .id("agent-graph-close")
                    .size(px(28.0))
                    .flex()
                    .items_center()
                    .justify_center()
                    .rounded(px(6.0))
                    .cursor_pointer()
                    .hover(|s| s.bg(crate::theme::wash(0.11)))
                    .on_click(cx.listener(|_, _, _, cx| cx.emit(AgentGraphEvent::Close)))
                    .child(
                        icon(icons::CLOSE)
                            .size(px(14.0))
                            .text_color(theme.text_muted),
                    ),
            );

        let body: AnyElement = if graph.nodes.is_empty() {
            div()
                .flex_1()
                .flex()
                .items_center()
                .justify_center()
                .text_size(px(13.0))
                .text_color(theme.text_muted)
                .child("No active agents. Start a session and it shows up here.")
                .into_any_element()
        } else {
            let canvas = self.render_canvas(&graph, phase, &theme, cx);
            let panel = self.render_process_panel(&graph, &statuses, &theme, cx);
            div()
                .flex_1()
                .min_h_0()
                .flex()
                .flex_row()
                .child(canvas)
                .children(panel)
                .into_any_element()
        };

        let footer = self.render_steer_bar(&graph, &statuses, &theme, cx);

        div()
            .size_full()
            .flex()
            .flex_col()
            .on_key_down(cx.listener(|_, ev: &gpui::KeyDownEvent, _, cx| {
                if ev.keystroke.key == "escape" {
                    cx.emit(AgentGraphEvent::Close);
                }
            }))
            .child(header)
            .child(body)
            .child(footer)
    }
}

impl AgentGraph {
    fn render_canvas(
        &mut self,
        graph: &Graph,
        phase: f32,
        theme: &Theme,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let curves: Vec<([(f32, f32); 4], Hsla, bool)> = graph
            .edges
            .iter()
            .map(|e| {
                let color = edge_color(e.kind, theme);
                (
                    edge_curve(&graph.nodes[e.from], &graph.nodes[e.to]),
                    color,
                    e.live,
                )
            })
            .collect();
        let selected_chat = self.selected.as_deref().and_then(|key| {
            graph
                .nodes
                .iter()
                .find(|n| n.key == key)
                .map(|n| n.chat_id.clone())
        });
        // Edges of the selected agent draw strong; the rest recede.
        let emphasis: Vec<bool> = graph
            .edges
            .iter()
            .map(|e| {
                selected_chat
                    .as_deref()
                    .is_none_or(|chat| graph.nodes[e.from].chat_id == chat)
            })
            .collect();
        let paint_curves = curves.clone();
        let edges = canvas(
            |_, _, _| (),
            move |bounds, _, window, _| {
                for ((c, color, live), strong) in paint_curves.iter().zip(&emphasis) {
                    let at =
                        |p: (f32, f32)| point(bounds.origin.x + px(p.0), bounds.origin.y + px(p.1));
                    let mut builder = PathBuilder::stroke(px(if *live { 2.0 } else { 1.25 }));
                    builder.move_to(at(c[0]));
                    builder.cubic_bezier_to(at(c[3]), at(c[1]), at(c[2]));
                    let alpha = match (*strong, *live) {
                        (_, true) => 1.0,
                        (true, false) => 0.7,
                        (false, false) => 0.18,
                    };
                    if let Ok(path) = builder.build() {
                        window.paint_path(path, color.opacity(alpha));
                    }
                }
            },
        )
        .absolute()
        .inset_0();

        // A dot travels each live edge — the agent is acting on that node now.
        let pulses = curves
            .iter()
            .filter(|(_, _, live)| *live)
            .map(|(c, color, _)| {
                let (x, y) = bezier_at(c, phase);
                div()
                    .absolute()
                    .left(px(x - 3.5))
                    .top(px(y - 3.5))
                    .size(px(7.0))
                    .rounded_full()
                    .bg(*color)
            });

        let mut canvas_box = div()
            .relative()
            .flex_none()
            .w(px(graph.width))
            .h(px(graph.height))
            .child(edges)
            .children(pulses);
        for (ix, node) in graph.nodes.iter().enumerate() {
            canvas_box = canvas_box.child(self.render_node(ix, node, phase, theme, cx));
        }

        div()
            .id("agent-graph-scroll")
            .flex_1()
            .min_w_0()
            .min_h_0()
            .overflow_scroll()
            .child(canvas_box)
            .into_any_element()
    }

    fn render_node(
        &mut self,
        ix: usize,
        node: &GraphNode,
        phase: f32,
        theme: &Theme,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let selected = self.selected.as_deref() == Some(node.key.as_str());
        let dot = status_color(node.status, theme);
        let working = node.status == NodeStatus::Working;
        // Working nodes breathe: the ring's alpha follows the shared pulse.
        let ring = if selected {
            theme.accent
        } else if working {
            theme
                .busy
                .opacity(0.35 + 0.45 * (phase * std::f32::consts::TAU).sin().abs())
        } else if node.kind == NodeKind::Context && node.shared_by > 1 {
            theme.warning.opacity(0.6)
        } else {
            theme.border
        };
        let (bg, radius) = match node.kind {
            NodeKind::Agent => (theme.surface_raised, 10.0),
            NodeKind::Subagent => (theme.surface_card, 8.0),
            NodeKind::Context => (theme.surface, 6.0),
        };
        let key = node.key.clone();
        let chat_id = node.chat_id.clone();
        let is_agent = node.kind == NodeKind::Agent;
        let title_row = div()
            .flex()
            .items_center()
            .gap(px(6.0))
            .min_w_0()
            .child(div().flex_none().size(px(7.0)).rounded_full().bg(dot))
            .when_some(node.harness.filter(|_| is_agent), |el, harness| {
                let (path, tint) = crate::pickers::harness_brand_icon(harness);
                el.child(
                    icon(path)
                        .size(px(13.0))
                        .flex_none()
                        .text_color(tint.unwrap_or(theme.text_muted)),
                )
            })
            .when(node.kind == NodeKind::Context, |el| {
                el.child(
                    icon(icons::DOCUMENT)
                        .size(px(12.0))
                        .flex_none()
                        .text_color(theme.text_muted),
                )
            })
            .child(
                div()
                    .min_w_0()
                    .truncate()
                    .text_size(px(if is_agent { 13.0 } else { 12.0 }))
                    .font_weight(if is_agent {
                        gpui::FontWeight::SEMIBOLD
                    } else {
                        gpui::FontWeight::MEDIUM
                    })
                    .child(node.label.clone()),
            )
            .when(node.kind == NodeKind::Context && node.shared_by > 1, |el| {
                el.child(
                    div()
                        .flex_none()
                        .text_size(px(10.0))
                        .text_color(theme.warning)
                        .child(format!("×{}", node.shared_by)),
                )
            });
        div()
            .id(("agent-graph-node", ix))
            .absolute()
            .left(px(node.x))
            .top(px(node.y))
            .w(px(node.w))
            .h(px(node.h))
            .px(px(10.0))
            .flex()
            .flex_col()
            .justify_center()
            .gap(px(3.0))
            .rounded(px(radius))
            .bg(bg)
            .border_1()
            .border_color(ring)
            .cursor_pointer()
            .on_click(cx.listener(move |this, event: &gpui::ClickEvent, _, cx| {
                if is_agent && event.click_count() == 2 {
                    cx.emit(AgentGraphEvent::OpenChat(chat_id.clone()));
                } else {
                    this.select(key.clone(), cx);
                }
            }))
            .child(title_row)
            .child(
                div()
                    .min_w_0()
                    .truncate()
                    .text_size(px(11.0))
                    .text_color(if working {
                        theme.text
                    } else {
                        theme.text_muted
                    })
                    .child(node.detail.clone()),
            )
            // The agent's live log: its latest steps, newest last.
            .when(!node.recent.is_empty(), |el| {
                el.child(
                    div()
                        .mt(px(3.0))
                        .pt(px(5.0))
                        .border_t_1()
                        .border_color(theme.border)
                        .flex()
                        .flex_col()
                        .gap(px(2.0))
                        .children(node.recent.iter().map(|step| step_line(step, theme))),
                )
            })
            .into_any_element()
    }

    /// The clicked node's process, newest first: every step an agent took,
    /// what a subagent was asked and returned, or who touched a file.
    fn render_process_panel(
        &mut self,
        graph: &Graph,
        statuses: &HashMap<String, ChatIndicator>,
        theme: &Theme,
        cx: &mut Context<Self>,
    ) -> Option<AnyElement> {
        let key = self.selected.as_deref()?;
        let node = graph.nodes.iter().find(|n| n.key == key)?;
        let working = statuses.get(&node.chat_id) == Some(&ChatIndicator::Working);
        let transcript = |chat: &str| {
            self.watches
                .get(chat)
                .map(|w| w.entries.as_slice())
                .unwrap_or(&[])
        };
        let (kind, mut steps) = match node.kind {
            NodeKind::Agent => ("Agent", process_steps(transcript(&node.chat_id), working)),
            NodeKind::Subagent => {
                let part_id = node
                    .key
                    .strip_prefix(&format!("s:{}:", node.chat_id))
                    .unwrap_or_default();
                (
                    "Subagent",
                    subagent_steps(transcript(&node.chat_id), part_id, node.status),
                )
            }
            NodeKind::Context => {
                let path = node.key.strip_prefix("f:").unwrap_or_default();
                // Touches from several agents interleave by entry time.
                let mut timed: Vec<(i64, Step)> = Vec::new();
                for agent in graph.nodes.iter().filter(|n| n.kind == NodeKind::Agent) {
                    let live = statuses.get(&agent.chat_id) == Some(&ChatIndicator::Working);
                    let cwd = self
                        .state
                        .read(cx)
                        .chats
                        .iter()
                        .find(|c| c.id == agent.chat_id)
                        .and_then(|c| c.cwd.clone());
                    for (at, part) in transcript(&agent.chat_id)
                        .iter()
                        .flat_map(|e| e.parts.iter().map(|p| (e.created_at, p)))
                    {
                        let MessagePart::Tool {
                            call,
                            is_error,
                            resolved,
                            ..
                        } = part
                        else {
                            continue;
                        };
                        let Some((touched, _)) = touched_path(call) else {
                            continue;
                        };
                        if normalize_path(touched, cwd.as_deref()) != path {
                            continue;
                        }
                        let (label, _) = keel_proto::view::tool_chip_content(call);
                        timed.push((
                            at,
                            Step {
                                kind: StepKind::Tool,
                                label: label.into(),
                                detail: format!("by {}", agent.label),
                                status: tool_status(*resolved, *is_error, live),
                            },
                        ));
                    }
                }
                // Stable: parts of one entry keep their order.
                timed.sort_by_key(|(at, _)| *at);
                let steps: Vec<Step> = timed.into_iter().map(|(_, step)| step).collect();
                ("File", steps)
            }
        };
        let total = steps.len();
        steps.reverse();
        steps.truncate(PANEL_STEPS);
        let subtitle = match (node.kind, total) {
            (NodeKind::Context, _) => format!("{kind} · {} · {total} touches", node.detail),
            (_, 0) => format!("{kind} · no steps yet"),
            _ => format!("{kind} · {total} steps · latest first"),
        };
        Some(
            div()
                .flex_none()
                .w(px(380.0))
                .h_full()
                .flex()
                .flex_col()
                .border_l_1()
                .border_color(theme.border)
                .child(
                    div()
                        .flex_none()
                        .px(px(Theme::SPACE_LG))
                        .py(px(10.0))
                        .border_b_1()
                        .border_color(theme.border)
                        .flex()
                        .items_center()
                        .gap(px(8.0))
                        .child(
                            div()
                                .flex_none()
                                .size(px(8.0))
                                .rounded_full()
                                .bg(status_color(node.status, theme)),
                        )
                        .child(
                            div()
                                .flex_1()
                                .min_w_0()
                                .flex()
                                .flex_col()
                                .child(
                                    div()
                                        .truncate()
                                        .text_size(px(13.0))
                                        .font_weight(gpui::FontWeight::SEMIBOLD)
                                        .child(node.label.clone()),
                                )
                                .child(
                                    div()
                                        .truncate()
                                        .text_size(px(11.0))
                                        .text_color(theme.text_muted)
                                        .child(subtitle),
                                ),
                        )
                        .child(
                            div()
                                .id("agent-graph-panel-close")
                                .flex_none()
                                .size(px(24.0))
                                .flex()
                                .items_center()
                                .justify_center()
                                .rounded(px(6.0))
                                .cursor_pointer()
                                .hover(|s| s.bg(crate::theme::wash(0.11)))
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.selected = None;
                                    cx.notify();
                                }))
                                .child(
                                    icon(icons::CLOSE)
                                        .size(px(12.0))
                                        .text_color(theme.text_muted),
                                ),
                        ),
                )
                .child(
                    div()
                        .id("agent-graph-process")
                        .flex_1()
                        .min_h_0()
                        .overflow_y_scroll()
                        .px(px(Theme::SPACE_LG))
                        .py(px(8.0))
                        .flex()
                        .flex_col()
                        .when(steps.is_empty(), |el| {
                            el.child(
                                div()
                                    .py(px(8.0))
                                    .text_size(px(12.0))
                                    .text_color(theme.text_muted)
                                    .child("Nothing recorded yet."),
                            )
                        })
                        .children(steps.iter().map(|step| step_row(step, theme))),
                )
                .into_any_element(),
        )
    }

    fn render_steer_bar(
        &mut self,
        graph: &Graph,
        statuses: &HashMap<String, ChatIndicator>,
        theme: &Theme,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let target = self.target(graph);
        let bar = div()
            .flex_none()
            .flex()
            .flex_col()
            .gap(px(8.0))
            .px(px(Theme::SPACE_LG))
            .py(px(12.0))
            .border_t_1()
            .border_color(theme.border);
        let Some((chat_id, title, sub)) = target else {
            return bar
                .child(
                    div()
                        .text_size(px(12.0))
                        .text_color(theme.text_muted)
                        .child("Select an agent or subagent to steer it. Double-click an agent to open its session."),
                )
                .into_any_element();
        };
        let live = statuses.get(&chat_id) == Some(&ChatIndicator::Working);
        let heading = match &sub {
            Some(label) => {
                format!("Steer \u{201C}{title}\u{201D} about subagent \u{201C}{label}\u{201D}")
            }
            None => format!("Steer \u{201C}{title}\u{201D}"),
        };
        let hint = match (&sub, live) {
            (Some(_), _) => {
                "Subagents share their parent's steering channel — your message goes to the parent, naming this subagent."
            }
            (None, true) => "Delivered into the running turn at the next step boundary.",
            (None, false) => "Not running — your message starts a new turn.",
        };
        self.input.update(cx, |input, cx| {
            input.set_placeholder(
                if live {
                    "Tell the agent what to change…"
                } else {
                    "Send the agent a new instruction…"
                },
                cx,
            )
        });
        let open_id = chat_id.clone();
        bar.child(
            div()
                .flex()
                .items_center()
                .gap(px(8.0))
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .truncate()
                        .text_size(px(12.0))
                        .font_weight(gpui::FontWeight::MEDIUM)
                        .child(heading),
                )
                .when_some(self.notice.clone(), |el, (msg, error)| {
                    el.child(
                        div()
                            .flex_none()
                            .text_size(px(11.0))
                            .text_color(if error { theme.danger } else { theme.success })
                            .child(msg),
                    )
                }),
        )
        .child(
            div()
                .flex()
                .items_end()
                .gap(px(8.0))
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .child(popover::dialog_field(self.input.clone().into_any_element())),
                )
                .child(
                    popover::btn_primary(theme, if live { "Steer" } else { "Send" })
                        .id("agent-graph-steer")
                        .on_click(cx.listener(|this, _, _, cx| this.steer(cx))),
                )
                // Stop interrupts the parent run, so a subagent selection hides it.
                .when(live && sub.is_none(), |el| {
                    el.child(
                        popover::btn_ghost(theme, "Stop", "agent-graph-stop")
                            .id("agent-graph-stop")
                            .on_click(cx.listener(|this, _, _, cx| this.stop(cx))),
                    )
                })
                .child(
                    popover::btn_ghost(theme, "Open", "agent-graph-open")
                        .id("agent-graph-open")
                        .on_click(cx.listener(move |_, _, _, cx| {
                            cx.emit(AgentGraphEvent::OpenChat(open_id.clone()))
                        })),
                ),
        )
        .child(
            div()
                .text_size(px(11.0))
                .text_color(theme.text_faint)
                .child(hint),
        )
        .into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use keel_doc::MessageRole;

    fn tool(id: &str, call: ToolCall, resolved: bool) -> MessagePart {
        MessagePart::Tool {
            id: id.into(),
            call,
            is_error: false,
            resolved,
            output: None,
            diff: None,
            output_ref: None,
            output_bytes: None,
            diff_ref: None,
            diff_stats: None,
        }
    }

    fn entry(parts: Vec<MessagePart>) -> SessionMessageEntry {
        SessionMessageEntry {
            id: "m1".into(),
            role: MessageRole::Assistant,
            parts,
            created_at: 0,
            device_id: "d".into(),
            status: None,
            continuation_of: None,
        }
    }

    fn task(desc: &str) -> ToolCall {
        ToolCall::Unknown {
            name: format!("Task: {desc}"),
            input: None,
        }
    }

    #[test]
    fn detects_subagents_across_adapters() {
        assert_eq!(
            subagent_label(&task("scan repo"), None).as_deref(),
            Some("scan repo")
        );
        let claude = ToolCall::Unknown {
            name: "Explore auth".into(),
            input: Some(
                serde_json::json!({ "subagent_type": "Explore", "description": "Explore auth" }),
            ),
        };
        assert_eq!(
            subagent_label(&claude, None).as_deref(),
            Some("Explore auth")
        );
        let bare = ToolCall::Unknown {
            name: "Task".into(),
            input: Some(serde_json::json!({ "description": "fix tests" })),
        };
        assert_eq!(subagent_label(&bare, None).as_deref(), Some("fix tests"));
        let other = ToolCall::Unknown {
            name: "lint".into(),
            input: None,
        };
        assert_eq!(subagent_label(&other, None), None);
        assert_eq!(
            subagent_label(
                &ToolCall::Exec {
                    command: "ls".into()
                },
                None
            ),
            None
        );
    }

    /// Grok as the doc stores it: the spawn's name is its description, the
    /// input is stripped, the output is the background notice; the parent
    /// later waits with `multi-wait (wait_all)`.
    #[test]
    fn grok_background_subagents_run_until_the_wait_resolves() {
        let spawn = |id: &str, name: &str| MessagePart::Tool {
            id: id.into(),
            call: ToolCall::Unknown {
                name: name.into(),
                input: None,
            },
            is_error: false,
            resolved: true,
            output: Some("Subagent started in background.\nsubagent_id: 01a0".into()),
            output_ref: None,
            output_bytes: None,
            diff: None,
            diff_ref: None,
            diff_stats: None,
        };
        let wait = |resolved: bool| {
            tool(
                "w",
                ToolCall::Unknown {
                    name: "multi-wait (wait_all)".into(),
                    input: None,
                },
                resolved,
            )
        };
        let spawns = vec![
            spawn("a", "Subagent one check-in"),
            spawn("b", "Subagent two check-in"),
        ];

        let mut waiting = spawns.clone();
        waiting.push(wait(false));
        let t = vec![entry(waiting)];
        let subs = subagents(&t, true);
        assert_eq!(subs.len(), 2);
        assert_eq!(subs[0].1, "Subagent one check-in");
        assert!(subs.iter().all(|s| s.2 == NodeStatus::Working));

        // Connected in the graph: a live spawn edge per subagent.
        let graph = build_graph(&[AgentSnapshot {
            chat_id: "G",
            title: "Grok".into(),
            harness: Some(HarnessId::Grok),
            status: ChatIndicator::Working,
            cwd: None,
            transcript: &t,
        }]);
        let spawn_edges: Vec<_> = graph
            .edges
            .iter()
            .filter(|e| e.kind == EdgeKind::Spawn)
            .collect();
        assert_eq!(spawn_edges.len(), 2);
        assert!(spawn_edges.iter().all(|e| e.live));

        let mut collected = spawns.clone();
        collected.push(wait(true));
        let t = vec![entry(collected)];
        assert!(subagents(&t, true).iter().all(|s| s.2 == NodeStatus::Done));
        let steps = process_steps(&t, true);
        assert_eq!(steps[0].kind, StepKind::Spawn);
        assert_eq!(steps[0].status, NodeStatus::Done);

        // Never collected and the run ended: not claimed as running.
        let t = vec![entry(spawns)];
        assert!(subagents(&t, false).iter().all(|s| s.2 == NodeStatus::Idle));
        let sub = subagent_steps(&t, "a", NodeStatus::Idle);
        assert_eq!(sub[0].label, "Started in background");
    }

    #[test]
    fn paths_meet_on_one_node_relative_or_absolute() {
        assert_eq!(normalize_path("./src/a.rs", Some("/r")), "/r/src/a.rs");
        assert_eq!(normalize_path("/r/src/a.rs", Some("/x")), "/r/src/a.rs");
        assert_eq!(normalize_path("src/", None), "src");
        assert_eq!(normalize_path("src/../lib.rs", Some("/r")), "/r/lib.rs");
        assert_eq!(normalize_path("/r/./a//b.rs", None), "/r/a/b.rs");
        assert_eq!(normalize_path("../x.rs", None), "../x.rs");
        assert_eq!(normalize_path("/../x.rs", None), "/x.rs");
        assert_eq!(
            split_display("/r/src/a.rs", Some("/r")),
            ("a.rs".into(), "src".into())
        );
        assert_eq!(
            split_display("/r/a.rs", Some("/r/")),
            ("a.rs".into(), ".".into())
        );
    }

    #[test]
    fn builds_agents_subagents_and_shared_context() {
        let a = vec![entry(vec![
            tool("t1", task("research"), true),
            tool(
                "t2",
                ToolCall::ReadFile {
                    path: "src/lib.rs".into(),
                },
                true,
            ),
            tool(
                "t3",
                ToolCall::EditFile {
                    path: "src/main.rs".into(),
                    old_string: None,
                    new_string: None,
                },
                false,
            ),
        ])];
        let b = vec![entry(vec![tool(
            "u1",
            ToolCall::EditFile {
                path: "/repo/src/lib.rs".into(),
                old_string: None,
                new_string: None,
            },
            true,
        )])];
        let graph = build_graph(&[
            AgentSnapshot {
                chat_id: "A",
                title: "Agent A".into(),
                harness: None,
                status: ChatIndicator::Working,
                cwd: Some("/repo"),
                transcript: &a,
            },
            AgentSnapshot {
                chat_id: "B",
                title: "Agent B".into(),
                harness: None,
                status: ChatIndicator::Idle,
                cwd: Some("/repo"),
                transcript: &b,
            },
        ]);
        let count = |k| graph.nodes.iter().filter(|n| n.kind == k).count();
        assert_eq!(count(NodeKind::Agent), 2);
        assert_eq!(count(NodeKind::Subagent), 1);
        assert_eq!(count(NodeKind::Context), 2);

        let a_node = &graph.nodes[0];
        assert_eq!(a_node.detail, "Edit src/main.rs");

        let lib = graph
            .nodes
            .iter()
            .find(|n| n.key == "f:/repo/src/lib.rs")
            .unwrap();
        assert_eq!(lib.shared_by, 2);
        let lib_ix = graph.nodes.iter().position(|n| n.key == lib.key).unwrap();
        let into_lib: Vec<_> = graph.edges.iter().filter(|e| e.to == lib_ix).collect();
        assert_eq!(into_lib.len(), 2);
        assert!(into_lib.iter().any(|e| e.kind == EdgeKind::Read));
        assert!(into_lib.iter().any(|e| e.kind == EdgeKind::Write));

        // The in-flight edit is the live edge.
        let main_ix = graph
            .nodes
            .iter()
            .position(|n| n.key == "f:/repo/src/main.rs")
            .unwrap();
        assert!(graph.edges.iter().any(|e| e.to == main_ix && e.live));

        // Nothing overlaps within a row.
        for (i, x) in graph.nodes.iter().enumerate() {
            for y in graph.nodes.iter().skip(i + 1) {
                let overlap = x.y == y.y && x.x < y.x + y.w && y.x < x.x + x.w;
                assert!(!overlap, "{} overlaps {}", x.key, y.key);
            }
        }
        // Shared file sits between its two agents.
        let (ax, bx) = (
            graph.nodes[0].x,
            graph.nodes.iter().find(|n| n.key == "a:B").unwrap().x,
        );
        assert!(lib.x + lib.w / 2.0 > ax.min(bx) && lib.x < ax.max(bx) + AGENT_W);
    }

    #[test]
    fn subagents_cap_to_most_recent() {
        let parts: Vec<_> = (0..9)
            .map(|i| tool(&format!("t{i}"), task(&format!("job {i}")), true))
            .collect();
        let t = vec![entry(parts)];
        let graph = build_graph(&[AgentSnapshot {
            chat_id: "A",
            title: "A".into(),
            harness: None,
            status: ChatIndicator::Idle,
            cwd: None,
            transcript: &t,
        }]);
        let subs: Vec<_> = graph
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Subagent)
            .collect();
        assert_eq!(subs.len(), MAX_SUBAGENTS);
        assert_eq!(subs.last().unwrap().label, "job 8");
        assert_eq!(subs[0].label, "job 3");
    }

    #[test]
    fn process_lists_prompt_message_tools_and_spawns() {
        let user = SessionMessageEntry {
            role: MessageRole::User,
            ..entry(vec![MessagePart::Text {
                id: "p".into(),
                text: "fix the build".into(),
            }])
        };
        let assistant = entry(vec![
            MessagePart::Text {
                id: "x".into(),
                text: "On it.".into(),
            },
            tool(
                "t1",
                ToolCall::Exec {
                    command: "cargo build".into(),
                },
                true,
            ),
            tool(
                "t2",
                ToolCall::Unknown {
                    name: "Task: audit deps".into(),
                    input: Some(serde_json::json!({ "prompt": "check Cargo.lock" })),
                },
                false,
            ),
        ]);
        let t = vec![user, assistant];
        let steps = process_steps(&t, true);
        let kinds: Vec<_> = steps.iter().map(|s| s.kind).collect();
        assert_eq!(
            kinds,
            [
                StepKind::Prompt,
                StepKind::Message,
                StepKind::Tool,
                StepKind::Spawn
            ]
        );
        assert_eq!(steps[2].detail, "cargo build");
        assert_eq!(steps[2].status, NodeStatus::Done);
        assert_eq!(steps[3].status, NodeStatus::Working);
        // Not live any more: the open spawn reads as interrupted.
        assert_eq!(process_steps(&t, false)[3].status, NodeStatus::Idle);

        let sub = subagent_steps(&t, "t2", NodeStatus::Working);
        assert_eq!(sub[0].kind, StepKind::Prompt);
        assert_eq!(sub[0].detail, "check Cargo.lock");
        assert_eq!(sub.last().unwrap().label, "Running");

        let graph = build_graph(&[AgentSnapshot {
            chat_id: "A",
            title: "A".into(),
            harness: None,
            status: ChatIndicator::Working,
            cwd: None,
            transcript: &t,
        }]);
        assert_eq!(graph.nodes[0].recent.len(), CARD_STEPS);
        assert_eq!(graph.nodes[0].recent[2].kind, StepKind::Spawn);
        let sub_node = graph
            .nodes
            .iter()
            .find(|n| n.kind == NodeKind::Subagent)
            .unwrap();
        assert_eq!(sub_node.detail, "Running · check Cargo.lock");
    }

    #[test]
    fn long_step_text_is_capped() {
        let long = "x".repeat(STEP_CHARS + 50);
        let s = snippet(&long);
        assert_eq!(s.chars().count(), STEP_CHARS + 1);
        assert!(s.ends_with('…'));
    }

    #[test]
    fn bezier_hits_endpoints() {
        let c = [(0.0, 0.0), (0.0, 5.0), (10.0, 5.0), (10.0, 10.0)];
        assert_eq!(bezier_at(&c, 0.0), (0.0, 0.0));
        assert_eq!(bezier_at(&c, 1.0), (10.0, 10.0));
    }
}
