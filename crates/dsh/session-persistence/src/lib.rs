//! Rust port of `@deepseek-ai/dsh-session-persistence`
//! (`packages/session/session-persistence`): the durable session-storage
//! seam (`sessionPersistence`) plus the write-behind batching controller
//! first-party backends compose.
//!
//! Divergences:
//! - The abstract service class becomes the [`SessionPersistence`] trait plus
//!   a [`PersistenceService`] newtype registered under the seam name.
//! - The coordinator's prepared-session reuse cache and revision-stability
//!   retry loop are deferred: `prepare` loads fresh (correct, less cached).
//! - Cancellation signals are dropped-future based.

use dsh_cordis::{Context, Service};
use dsh_session::{Session, SessionError, SessionEvent, SessionHeader, SessionId};
use futures::FutureExt;
use futures::future::LocalBoxFuture;
use std::cell::RefCell;
use std::rc::Rc;

/// Opaque source-qualified token that changes whenever a stored log changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionPersistenceRevision(pub String);

/// Lightweight immutable source identity returned without loading a full log.
#[derive(Debug, Clone)]
pub struct SessionPersistenceSnapshot {
    pub header: SessionHeader,
    pub revision: SessionPersistenceRevision,
}

/// Immutable logical session prepared from persistence.
#[derive(Debug, Clone)]
pub struct SessionInspection {
    /// Validated immutable session metadata.
    pub meta: SessionHeader,
    /// Validated contiguous logical event log (balanced on load).
    pub events: Vec<SessionEvent>,
}

/// A backend-resolved, per-session local artifact location: a location hint,
/// never an authorization token. The path may name an artifact that has not
/// materialized yet.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionLocation {
    /// Backend-specific artifact kind, e.g. `jsonl`.
    pub kind: String,
    /// Absolute path to this session's backend-owned artifact.
    pub path: std::path::PathBuf,
}

/// Stable refusal message for a log written under an unknown format version.
pub fn session_format_version_refusal(found: u64) -> String {
    format!(
        "session log format version {found} is not supported by this build (expected {})",
        dsh_session::SESSION_FORMAT_VERSION
    )
}

/// Durable append-only session storage (upstream `SessionPersistence`).
///
/// Implementations preserve contiguous, JSON-serializable events; `append`
/// resolves only after durability, and `load` balances a complete interrupted
/// tail (via synthetic closers) without rewriting committed events.
pub trait SessionPersistence: 'static {
    /// Resolve this backend's per-session artifact without materializing it;
    /// `None` for backends without one artifact per session.
    fn locate(&self, meta: &SessionHeader) -> Option<SessionLocation>;

    /// Register a new session's metadata. A backend MAY defer the physical
    /// write until the first `append` (lazy materialization), so an
    /// abandoned created-but-never-appended session leaves nothing behind.
    fn create(&self, meta: &SessionHeader) -> LocalBoxFuture<'_, anyhow::Result<()>>;

    /// Durably persist a contiguous batch: the first event's `seq` MUST equal
    /// the stored next-seq.
    fn append(
        &self,
        id: &SessionId,
        events: Vec<SessionEvent>,
    ) -> LocalBoxFuture<'_, anyhow::Result<()>>;

    /// Load an immutable balanced logical view, durably committing cold
    /// interrupted-turn recovery; only a torn final record is discarded.
    fn load(&self, id: &SessionId) -> LocalBoxFuture<'_, anyhow::Result<SessionInspection>>;

    /// Inspect without committing recovery or publishing: a cold interrupted
    /// turn receives synthetic closers in memory only.
    fn inspect(&self, id: &SessionId) -> LocalBoxFuture<'_, anyhow::Result<SessionInspection>>;

    /// Read stored events with `seq >= from_seq` — the watermark-resume
    /// primitive. Beyond-prefix `from_seq` returns empty, never an error.
    fn read_from(
        &self,
        id: &SessionId,
        from_seq: u64,
    ) -> LocalBoxFuture<'_, anyhow::Result<(SessionHeader, Vec<SessionEvent>)>>;

    /// Lightweight listing from metadata, without full-log parses.
    fn list(&self) -> LocalBoxFuture<'_, anyhow::Result<Vec<SessionHeader>>>;

    /// List materialized sessions with cheap per-log change tokens; an
    /// unchanged log returns the same revision.
    fn list_snapshots(&self)
    -> LocalBoxFuture<'_, anyhow::Result<Vec<SessionPersistenceSnapshot>>>;
}

/// The registered seam service: a trait object under the upstream name.
pub struct PersistenceService {
    pub backend: Rc<dyn SessionPersistence>,
}

impl Service for PersistenceService {
    const NAME: &'static str = "sessionPersistence";
}

impl PersistenceService {
    /// Register a backend as `ctx.sessionPersistence`.
    pub fn provide(
        ctx: &Context,
        backend: Rc<dyn SessionPersistence>,
    ) -> dsh_cordis::Result<Rc<PersistenceService>> {
        let service = Rc::new(PersistenceService { backend });
        ctx.provide_service(service.clone())?;
        Ok(service)
    }

    /// Prepare the exact unpublished [`Session`] used by resume (upstream
    /// `prepare`): load the balanced log and seed a restored session through
    /// the store-equivalent validation path.
    pub async fn prepare(&self, id: &SessionId) -> anyhow::Result<Rc<Session>> {
        let loaded = self.backend.load(id).await?;
        Session::create(id.clone(), loaded.events, Some(loaded.meta))
            .map_err(|error: SessionError| anyhow::Error::new(error))
    }
}

/// Default maximum intentional batching wait after an idle queue gets work.
pub const DEFAULT_WRITE_BATCH_MAX_DELAY_MS: u64 = 200;
/// Upper bound a configuration may select.
pub const MAX_WRITE_BATCH_DELAY_MS: u64 = 60_000;

type WriteFn = dyn Fn(Vec<SessionEvent>) -> LocalBoxFuture<'static, anyhow::Result<()>>;

/// Bounded per-session write batching (upstream `SessionWriteBehind`): owns
/// pending events, a fixed batching deadline, the active write, and an
/// explicit quiescence barrier. Failures of background writes are reported,
/// never bubbled to the producer.
pub struct SessionWriteBehind {
    max_delay_ms: u64,
    write: Rc<WriteFn>,
    pending: Rc<RefCell<Vec<SessionEvent>>>,
    /// Signals the batching task; drain awaits in-flight work directly.
    writing: Rc<RefCell<Option<futures::future::Shared<LocalBoxFuture<'static, ()>>>>>,
    timer_armed: Rc<std::cell::Cell<bool>>,
    /// Seq the durable log is known to need next. Every batch is filtered
    /// against it, so an event that is already durable — a resumed session's
    /// replayed seed, or a reconcile racing the live feed — is dropped
    /// instead of corrupting the artifact with a duplicate.
    durable_next: Rc<std::cell::Cell<u64>>,
}

impl SessionWriteBehind {
    pub fn new(
        max_delay_ms: u64,
        write: impl Fn(Vec<SessionEvent>) -> LocalBoxFuture<'static, anyhow::Result<()>> + 'static,
    ) -> SessionWriteBehind {
        SessionWriteBehind {
            max_delay_ms,
            write: Rc::new(write),
            pending: Rc::default(),
            writing: Rc::default(),
            timer_armed: Rc::new(std::cell::Cell::new(false)),
            durable_next: Rc::new(std::cell::Cell::new(0)),
        }
    }

    /// Declare how far the durable log already reaches. Called once when
    /// persistence attaches to a session; a resumed session starts with its
    /// stored length rather than zero.
    pub fn set_durable_next(&self, seq: u64) {
        if seq > self.durable_next.get() {
            self.durable_next.set(seq);
        }
    }

    /// Order one batch, drop what is already durable, and refuse a batch that
    /// would tear a hole in the log. Returns `None` when nothing is left to
    /// write.
    fn prepare_batch(
        durable_next: &std::cell::Cell<u64>,
        mut batch: Vec<SessionEvent>,
    ) -> Option<Vec<SessionEvent>> {
        batch.sort_by_key(|event| event.seq);
        batch.dedup_by_key(|event| event.seq);
        let next = durable_next.get();
        batch.retain(|event| event.seq >= next);
        let first = batch.first()?;
        if first.seq > next {
            // A gap here means an event never reached this controller;
            // writing anyway would corrupt the artifact irrecoverably.
            tracing::error!(
                "session write-behind refused a batch starting at seq {} with durable next-seq {next}",
                first.seq
            );
            return None;
        }
        durable_next.set(batch.last().expect("non-empty").seq + 1);
        Some(batch)
    }

    /// Whether this controller owns queued events or an active durable write.
    pub fn has_work(&self) -> bool {
        !self.pending.borrow().is_empty() || self.writing.borrow().is_some()
    }

    /// Queue one event into the persistence-owned copy and arm the batching
    /// deadline when idle.
    pub fn enqueue(&self, event: SessionEvent) {
        self.pending.borrow_mut().push(event);
        if self.timer_armed.get() {
            return;
        }
        self.timer_armed.set(true);
        let delay = std::time::Duration::from_millis(self.max_delay_ms);
        let pending = self.pending.clone();
        let write = self.write.clone();
        let timer_armed = self.timer_armed.clone();
        let durable_next = self.durable_next.clone();
        let flush: LocalBoxFuture<'static, ()> = async move {
            tokio::time::sleep(delay).await;
            timer_armed.set(false);
            let batch: Vec<SessionEvent> = pending.borrow_mut().drain(..).collect();
            let Some(batch) = Self::prepare_batch(&durable_next, batch) else {
                return;
            };
            if let Err(error) = write(batch).await {
                tracing::warn!("session write-behind background write failed: {error:#}");
            }
        }
        .boxed_local();
        let shared = flush.shared();
        *self.writing.borrow_mut() = Some(shared.clone());
        let writing_slot = self.writing.clone();
        tokio::task::spawn_local(async move {
            shared.await;
            // Only clear our own registration; a drain may have replaced it.
            let mut slot = writing_slot.borrow_mut();
            if slot.is_some() {
                *slot = None;
            }
        });
    }

    /// Cancel the batching wait and durably drain through a quiescent point;
    /// the returned future resolves after backend durability. Drain failures
    /// ARE surfaced (this is the flush path, a caller-owned boundary).
    pub async fn drain(&self) -> anyhow::Result<()> {
        // Join any in-flight background write first.
        let active = self.writing.borrow().clone();
        if let Some(active) = active {
            active.await;
        }
        let batch: Vec<SessionEvent> = self.pending.borrow_mut().drain(..).collect();
        self.timer_armed.set(false);
        let Some(batch) = Self::prepare_batch(&self.durable_next, batch) else {
            return Ok(());
        };
        (self.write)(batch).await
    }
}

/// Wire the write-behind persistence pipeline for every live session: create
/// on `session/created`, enqueue on `session/event`, drain on `session/flush`
/// (upstream: the coordinator-composed plugin body).
pub fn install_write_behind(
    ctx: &Context,
    backend: Rc<dyn SessionPersistence>,
    max_delay_ms: u64,
) -> dsh_cordis::Result<()> {
    use dsh_session::{SessionCreated, SessionEventPublished, SessionFlush};
    let controllers: Rc<RefCell<std::collections::HashMap<String, Rc<SessionWriteBehind>>>> =
        Rc::default();

    let controller_for = {
        let controllers = controllers.clone();
        let backend = backend.clone();
        move |session: &Rc<Session>| -> Rc<SessionWriteBehind> {
            let id = session.id();
            let key = id.as_str().to_string();
            if let Some(existing) = controllers.borrow().get(&key) {
                return existing.clone();
            }
            let backend = backend.clone();
            let controller = Rc::new(SessionWriteBehind::new(max_delay_ms, move |events| {
                let backend = backend.clone();
                let id = id.clone();
                async move { backend.append(&id, events).await }.boxed_local()
            }));
            controllers.borrow_mut().insert(key, controller.clone());
            controller
        }
    };

    {
        let backend = backend.clone();
        let controller_for = controller_for.clone();
        ctx.on::<SessionCreated, _, _>(Default::default(), move |_ctx, session| {
            let backend = backend.clone();
            let session = session.clone();
            // Register metadata; the physical write may defer to first append.
            let controller = controller_for(&session);
            async move {
                if let Err(error) = backend.create(&session.header).await {
                    tracing::warn!("session persistence create failed: {error:#}");
                    return None;
                }
                // Reconcile the durable tail. A session built on a seed
                // appends its `session/end-seed` marker inside the
                // constructor — before this attachment exists — so on resume
                // the live log starts one event ahead of disk. Left alone
                // that gap makes every later append fail contiguity for the
                // life of the session. The watermark makes the replay
                // idempotent: live events racing this read are filtered by
                // seq, never duplicated into the artifact.
                let stored = match backend.read_from(&session.id(), 0).await {
                    Ok((_, events)) => events.len() as u64,
                    Err(error) => {
                        tracing::warn!("session persistence tail read failed: {error:#}");
                        return None;
                    }
                };
                controller.set_durable_next(stored);
                for event in session.events().into_iter().skip(stored as usize) {
                    controller.enqueue(event);
                }
                None
            }
        })?;
    }
    {
        let controller_for = controller_for.clone();
        ctx.on::<SessionEventPublished, _, _>(
            Default::default(),
            move |_ctx, (session, event)| {
                controller_for(session).enqueue(event.clone());
                async { None }
            },
        )?;
    }
    {
        ctx.on::<SessionFlush, _, _>(Default::default(), move |_ctx, session| {
            let controller = controller_for(session);
            async move {
                if let Err(error) = controller.drain().await {
                    tracing::warn!("session flush drain failed: {error:#}");
                }
                None
            }
        })?;
    }
    Ok(())
}
