//! Rust port of `@deepseek-ai/dsh-session-persistence-jsonl`
//! (`packages/session/session-persistence-jsonl`): one append-only JSONL
//! artifact per session — a tagged header line followed by one event per
//! line — with injective path encoding for unvalidated session ids,
//! torn-tail truncation, and durable interrupted-turn recovery on load.
//!
//! Divergences:
//! - The zstd physical encoding is deferred (no zstd crate in the
//!   workspace); artifacts are plaintext `.jsonl`. The logical line format
//!   is identical, so an encoding layer can wrap it later.
//! - Chunk-run packing (`chunk-rows.ts`) is deferred with it: events write
//!   one per line, which every upstream reader also accepts.
//! - The Windows reserved-name table (`win32.ts`) collapses into the
//!   injective segment encoder, which already escapes every risky unit.

use dsh_session::{
    SESSION_FORMAT_VERSION, SessionEvent, SessionHeader, SessionId, interrupted_turn_closers,
};
use dsh_session_persistence::{
    SessionInspection, SessionLocation, SessionPersistence, SessionPersistenceRevision,
    SessionPersistenceSnapshot, session_format_version_refusal,
};
use futures::FutureExt;
use futures::future::LocalBoxFuture;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Encode an arbitrary string as one safe path segment, injectively: safe
/// units stay literal; every other unit — `~` included — becomes `~XXXX`
/// (code-unit hex), and the whole segments `.` / `..` are escaped so an
/// unvalidated id can never traverse (upstream `encodePathSegment`).
pub fn encode_path_segment(raw: &str) -> String {
    assert!(!raw.is_empty(), "path segment must be non-empty");
    if raw != "." && raw != ".." {
        let safe = raw
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
        if safe {
            return raw.to_string();
        }
    }
    let mut encoded = String::new();
    for unit in raw.encode_utf16() {
        let c = char::from_u32(unit as u32);
        match c {
            Some(c) if c.is_ascii_alphanumeric() || matches!(c, '-' | '_') => encoded.push(c),
            _ => encoded.push_str(&format!("~{unit:04x}")),
        }
    }
    if encoded.is_empty() {
        // All-escaped input still yields a non-empty segment by construction.
        encoded.push('~');
    }
    encoded
}

/// The first JSONL record of a session artifact: the immutable header tagged
/// as a `session` record (upstream `HeaderLine`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct HeaderLine {
    #[serde(rename = "type")]
    kind: String,
    version: u64,
    id: SessionId,
    created_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    cwd: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parent_session: Option<SessionId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed_length: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    origin: Option<dsh_session::SessionOrigin>,
    /// Always written; absent in the header means zero.
    delegation_depth: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    agent_preset: Option<String>,
}

fn to_header_line(header: &SessionHeader) -> HeaderLine {
    HeaderLine {
        kind: "session".into(),
        version: header.version,
        id: header.id.clone(),
        created_at: header.created_at,
        cwd: header.cwd.clone(),
        parent_session: header.parent_session.clone(),
        seed_length: header.seed_length,
        origin: header.origin,
        delegation_depth: header.delegation_depth.unwrap_or(0),
        agent_preset: header.agent_preset.clone(),
    }
}

fn from_header_line(line: HeaderLine) -> anyhow::Result<SessionHeader> {
    if line.kind != "session" {
        anyhow::bail!("session artifact does not start with a session header line");
    }
    if line.version != SESSION_FORMAT_VERSION {
        anyhow::bail!(session_format_version_refusal(line.version));
    }
    Ok(SessionHeader {
        version: line.version,
        id: line.id,
        created_at: line.created_at,
        cwd: line.cwd,
        parent_session: line.parent_session,
        seed_length: line.seed_length,
        origin: line.origin,
        delegation_depth: (line.delegation_depth > 0).then_some(line.delegation_depth),
        agent_preset: line.agent_preset,
    })
}

/// One session artifact per file under `{root}/{encoded-id}.jsonl`.
pub struct JsonlPersistence {
    root: PathBuf,
    /// Distinguishes independently backed stores so backend-local revisions
    /// cannot compare equal across sources.
    source_tag: String,
}

impl JsonlPersistence {
    /// Open (creating the directory) a JSONL store rooted at `root`.
    pub fn new(root: impl Into<PathBuf>) -> anyhow::Result<JsonlPersistence> {
        let root: PathBuf = root.into();
        std::fs::create_dir_all(&root)?;
        let canonical = root.canonicalize().unwrap_or_else(|_| root.clone());
        let source_tag = format!("{:x}", {
            use sha2::Digest;
            sha2::Sha256::digest(canonical.to_string_lossy().as_bytes())
        });
        Ok(JsonlPersistence {
            root,
            source_tag: source_tag[..12].to_string(),
        })
    }

    fn path_for(&self, id: &SessionId) -> PathBuf {
        self.root
            .join(format!("{}.jsonl", encode_path_segment(id.as_str())))
    }

    /// Parse one artifact: header first line, events after; a torn final line
    /// (no trailing newline or invalid JSON at the physical tail) is
    /// discarded, corruption in the committed prefix rejects.
    fn read_artifact(&self, path: &Path) -> anyhow::Result<(SessionHeader, Vec<SessionEvent>)> {
        let content = std::fs::read_to_string(path)?;
        let mut lines = content.split_inclusive('\n');
        let header_line = lines
            .next()
            .ok_or_else(|| anyhow::anyhow!("session artifact is empty"))?;
        if !header_line.ends_with('\n') {
            anyhow::bail!("session artifact header line is torn");
        }
        let header: HeaderLine = serde_json::from_str(header_line.trim_end())?;
        let header = from_header_line(header)?;
        let mut events: Vec<SessionEvent> = Vec::new();
        for line in lines {
            let terminated = line.ends_with('\n');
            let text = line.trim_end_matches(['\n', '\r']);
            if text.is_empty() {
                continue;
            }
            match serde_json::from_str::<SessionEvent>(text) {
                Ok(event) => {
                    if event.seq != events.len() as u64 {
                        anyhow::bail!(
                            "session artifact event seq {} is not contiguous (expected {})",
                            event.seq,
                            events.len()
                        );
                    }
                    // A type outside this build's vocabulary without the
                    // ignorable marker means a newer writer: refuse rather
                    // than silently reconstruct a gutted session.
                    if !dsh_session::is_known_session_event_type(event.event_type())
                        && event.ignorable != Some(true)
                    {
                        anyhow::bail!(
                            "session artifact contains unknown required event type \"{}\"; \
                             this log was likely written by a newer harness",
                            event.event_type()
                        );
                    }
                    events.push(event);
                }
                Err(error) => {
                    if terminated {
                        // Corruption inside the committed prefix: reject.
                        return Err(anyhow::Error::new(error)
                            .context("session artifact contains a corrupt committed event line"));
                    }
                    // A torn final record (crash mid-write) is discarded.
                    break;
                }
            }
        }
        Ok((header, events))
    }

    fn append_lines(&self, path: &Path, events: &[SessionEvent]) -> anyhow::Result<()> {
        let mut file = std::fs::OpenOptions::new().append(true).open(path)?;
        let mut buffer = String::new();
        for event in events {
            buffer.push_str(&serde_json::to_string(event)?);
            buffer.push('\n');
        }
        file.write_all(buffer.as_bytes())?;
        file.sync_data()?;
        Ok(())
    }

    fn materialize(&self, meta: &SessionHeader) -> anyhow::Result<PathBuf> {
        let path = self.path_for(&meta.id);
        if !path.exists() {
            let mut file = std::fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&path)?;
            let mut line = serde_json::to_string(&to_header_line(meta))?;
            line.push('\n');
            file.write_all(line.as_bytes())?;
            file.sync_data()?;
        }
        Ok(path)
    }

    fn revision_for(&self, path: &Path) -> SessionPersistenceRevision {
        let token = std::fs::metadata(path)
            .map(|meta| {
                let modified = meta
                    .modified()
                    .ok()
                    .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|duration| duration.as_nanos())
                    .unwrap_or(0);
                format!("{}:{}:{}", self.source_tag, meta.len(), modified)
            })
            .unwrap_or_else(|_| format!("{}:absent", self.source_tag));
        SessionPersistenceRevision(token)
    }

    fn inspect_sync(
        &self,
        id: &SessionId,
        commit_repair: bool,
    ) -> anyhow::Result<SessionInspection> {
        let path = self.path_for(id);
        let (meta, mut events) = self.read_artifact(&path)?;
        let closers = interrupted_turn_closers(&events);
        if !closers.is_empty() {
            if commit_repair {
                // Durably close the interrupted turn before handing out the
                // balanced view; committed events are never rewritten.
                self.append_lines(&path, &closers)?;
            }
            events.extend(closers);
        }
        Ok(SessionInspection { meta, events })
    }
}

impl SessionPersistence for JsonlPersistence {
    fn locate(&self, meta: &SessionHeader) -> Option<SessionLocation> {
        Some(SessionLocation {
            kind: "jsonl".into(),
            path: self.path_for(&meta.id),
        })
    }

    fn create(&self, meta: &SessionHeader) -> LocalBoxFuture<'_, anyhow::Result<()>> {
        // Materialize eagerly: simpler than lazy materialization, and an
        // abandoned session costs one header line. (Upstream defers to the
        // first append; documented divergence.)
        let result = self.materialize(meta).map(|_| ());
        async move { result }.boxed_local()
    }

    fn append(
        &self,
        id: &SessionId,
        events: Vec<SessionEvent>,
    ) -> LocalBoxFuture<'_, anyhow::Result<()>> {
        let result = (|| {
            let path = self.path_for(id);
            if !path.exists() {
                anyhow::bail!(
                    "session \"{}\" was never created in this store",
                    id.as_str()
                );
            }
            // Contiguity: the batch must continue the stored log.
            let (_, stored) = self.read_artifact(&path)?;
            if let Some(first) = events.first() {
                if first.seq != stored.len() as u64 {
                    anyhow::bail!(
                        "append batch starts at seq {} but stored next-seq is {}",
                        first.seq,
                        stored.len()
                    );
                }
            }
            self.append_lines(&path, &events)
        })();
        async move { result }.boxed_local()
    }

    fn load(&self, id: &SessionId) -> LocalBoxFuture<'_, anyhow::Result<SessionInspection>> {
        let result = self.inspect_sync(id, true);
        async move { result }.boxed_local()
    }

    fn inspect(&self, id: &SessionId) -> LocalBoxFuture<'_, anyhow::Result<SessionInspection>> {
        let result = self.inspect_sync(id, false);
        async move { result }.boxed_local()
    }

    fn read_from(
        &self,
        id: &SessionId,
        from_seq: u64,
    ) -> LocalBoxFuture<'_, anyhow::Result<(SessionHeader, Vec<SessionEvent>)>> {
        let result = (|| {
            let path = self.path_for(id);
            let (meta, events) = self.read_artifact(&path)?;
            let suffix = events
                .into_iter()
                .filter(|event| event.seq >= from_seq)
                .collect();
            Ok((meta, suffix))
        })();
        async move { result }.boxed_local()
    }

    fn list(&self) -> LocalBoxFuture<'_, anyhow::Result<Vec<SessionHeader>>> {
        let result = (|| {
            let mut headers = Vec::new();
            for entry in std::fs::read_dir(&self.root)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().and_then(|ext| ext.to_str()) != Some("jsonl") {
                    continue;
                }
                // Header is the first line; no full-log parse.
                let content = std::fs::read_to_string(&path)?;
                let Some(first) = content.split_inclusive('\n').next() else {
                    continue;
                };
                if !first.ends_with('\n') {
                    continue;
                }
                let Ok(line) = serde_json::from_str::<HeaderLine>(first.trim_end()) else {
                    continue;
                };
                if let Ok(header) = from_header_line(line) {
                    headers.push(header);
                }
            }
            headers.sort_by_key(|header| header.created_at);
            Ok(headers)
        })();
        async move { result }.boxed_local()
    }

    fn list_snapshots(
        &self,
    ) -> LocalBoxFuture<'_, anyhow::Result<Vec<SessionPersistenceSnapshot>>> {
        async move {
            let headers = self.list().await?;
            Ok(headers
                .into_iter()
                .map(|header| {
                    let revision = self.revision_for(&self.path_for(&header.id));
                    SessionPersistenceSnapshot { header, revision }
                })
                .collect())
        }
        .boxed_local()
    }
}
