//! The event-sourced session and its in-memory store, ported from
//! `packages/core/session/src/index.ts`.
//!
//! Divergences:
//! - Upstream deep-freezes events at acceptance; Rust ownership provides the
//!   immutability (the log hands out clones/borrows, never mutable aliases).
//! - The upstream store hangs publication hooks on a module-private WeakMap;
//!   here the store wires a publication callback into the session directly.
//! - `session/created` veto semantics: upstream lets a synchronous listener
//!   throw to roll back publication. The port keeps announce-order semantics
//!   through the cordis event bus; a failing listener is contained and
//!   logged (the Rust bus has no synchronous-throw channel).

use crate::header_fold::fold_request_header;
use crate::surface::{SurfaceManager, derive_event_message};
use crate::types::{
    EpochHeader, RequestContext, SESSION_FORMAT_VERSION, SessionEvent, SessionEventData,
    SessionHeader, SessionId, SessionMeta, SurfaceIntent,
};
use dsh_cordis::{Context, Event, Service};
use dsh_llm::Message;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

/// Post-commit, fire-and-forget append feed (upstream `session/event`).
pub struct SessionEventPublished;
impl Event for SessionEventPublished {
    const NAME: &'static str = "session/event";
    type Args = (Rc<Session>, SessionEvent);
    type Ret = ();
}

/// Creation announcement during session publication (upstream
/// `session/created`).
pub struct SessionCreated;
impl Event for SessionCreated {
    const NAME: &'static str = "session/created";
    type Args = Rc<Session>;
    type Ret = ();
}

/// Emitted once when an announced session leaves the store (upstream
/// `session/disposed`).
pub struct SessionDisposed;
impl Event for SessionDisposed {
    const NAME: &'static str = "session/disposed";
    type Args = Rc<Session>;
    type Ret = ();
}

/// Awaited parallel durability checkpoint (upstream `session/flush`).
pub struct SessionFlush;
impl Event for SessionFlush {
    const NAME: &'static str = "session/flush";
    type Args = Rc<Session>;
    type Ret = ();
}

/// Session errors with the same message vocabulary as upstream throws.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct SessionError(pub String);

fn now_ms() -> u64 {
    chrono::Utc::now().timestamp_millis().max(0) as u64
}

fn validate_header(id: &SessionId, header: &SessionHeader) -> Result<(), SessionError> {
    if header.version != SESSION_FORMAT_VERSION {
        return Err(SessionError(format!(
            "session header version must be {SESSION_FORMAT_VERSION}, got {}",
            header.version
        )));
    }
    if header.id != *id {
        return Err(SessionError(format!(
            "session header id \"{}\" does not match session id \"{}\"",
            header.id.as_str(),
            id.as_str()
        )));
    }
    if let Some(cwd) = &header.cwd {
        if !Path::new(cwd).is_absolute() {
            return Err(SessionError(format!(
                "session header cwd must be an absolute path, got \"{cwd}\""
            )));
        }
    }
    Ok(())
}

/// An event-sourced session: an append-only log of [`SessionEvent`]s.
/// Message history is derived from the log; persistence subscribes to the
/// `session/event` feed and drains on `session/flush`.
pub struct Session {
    log: RefCell<Vec<SessionEvent>>,
    surface: RefCell<SurfaceManager>,
    /// Detached creation metadata; a storage concern kept out of the log.
    pub header: SessionHeader,
    /// The first seq appended IN THIS PROCESS: the constructor seed length
    /// (0 without one). Constructor seeds do not publish on the event feed.
    /// Distinct from `header.seed_length` (the durable fork-lineage
    /// boundary): a resumed session's constructor seed is its full stored log.
    pub first_live_seq: u64,
    /// Store-attached publication hook: notified after every committed append.
    publisher: RefCell<Option<Rc<dyn Fn(&Rc<Session>, &SessionEvent)>>>,
    /// Set while an append publishes, to reject reentrant appends.
    appending: Cell<bool>,
    /// Self-reference for publication (set by the store on enter).
    self_ref: RefCell<Option<std::rc::Weak<Session>>>,
    // Derived-history cache: node count + replace generation it was built at.
    derived: RefCell<Vec<Message>>,
    derived_nodes: Cell<usize>,
    derived_generation: Cell<u64>,
    // Incremental request-header fold.
    header_fold: RefCell<Option<EpochHeader>>,
    header_fold_seq: Cell<usize>,
    // Incremental request-context fold.
    context_fold: RefCell<Option<RequestContext>>,
    context_fold_seq: Cell<usize>,
}

impl Session {
    /// Create a detached session, validating any seed to the SAME invariants
    /// `append` enforces (contiguous seq from 0, valid surface transitions) so
    /// a replay/fork cannot construct a live log no backend could store.
    /// A seeded session ends its seed with a `session/end-seed` marker unless
    /// the seed already ends in one (reopening an untouched session must not
    /// grow its log per pickup).
    pub fn create(
        id: SessionId,
        seed: Vec<SessionEvent>,
        header: Option<SessionHeader>,
    ) -> Result<Rc<Session>, SessionError> {
        let seeded = !seed.is_empty();
        let mut log = Vec::with_capacity(seed.len());
        let mut surface = SurfaceManager::default();
        for (index, event) in seed.into_iter().enumerate() {
            if event.seq != index as u64 {
                return Err(SessionError(format!(
                    "seed event at index {index} has seq {} (expected {index}); seed must be contiguous from 0",
                    event.seq
                )));
            }
            surface.validate_next(&log, &event).map_err(|reason| {
                SessionError(format!("invalid seed event at index {index}: {reason}"))
            })?;
            log.push(event);
            surface.commit(&log);
        }
        let first_live_seq = log.len() as u64;
        let header = match header {
            Some(header) => {
                validate_header(&id, &header)?;
                header
            }
            None => SessionHeader {
                version: SESSION_FORMAT_VERSION,
                id: id.clone(),
                created_at: now_ms(),
                cwd: None,
                parent_session: None,
                seed_length: None,
                origin: None,
                delegation_depth: None,
                agent_preset: None,
            },
        };
        let session = Rc::new(Session {
            log: RefCell::new(log),
            surface: RefCell::new(surface),
            header,
            first_live_seq,
            publisher: RefCell::new(None),
            appending: Cell::new(false),
            self_ref: RefCell::new(None),
            derived: RefCell::new(Vec::new()),
            derived_nodes: Cell::new(0),
            derived_generation: Cell::new(0),
            header_fold: RefCell::new(None),
            header_fold_seq: Cell::new(0),
            context_fold: RefCell::new(None),
            context_fold_seq: Cell::new(0),
        });
        *session.self_ref.borrow_mut() = Some(Rc::downgrade(&session));
        let ends_with_marker = matches!(
            session.log.borrow().last().map(|event| &event.data),
            Some(SessionEventData::SessionEndSeed {})
        );
        if seeded && !ends_with_marker {
            session
                .append(SessionEventData::SessionEndSeed {}, None)
                .map_err(|error| SessionError(error.0))?;
        }
        Ok(session)
    }

    /// The session identity, from its durable header.
    pub fn id(&self) -> SessionId {
        self.header.id.clone()
    }

    /// An immutable snapshot of the append-only event log.
    pub fn events(&self) -> Vec<SessionEvent> {
        self.log.borrow().clone()
    }

    /// Run a reader over the log without cloning it.
    pub fn with_events<R>(&self, reader: impl FnOnce(&[SessionEvent]) -> R) -> R {
        reader(&self.log.borrow())
    }

    /// The next event's sequence number — always the log length (the
    /// `seq = log length` contiguity contract).
    pub fn seq(&self) -> u64 {
        self.log.borrow().len() as u64
    }

    /// Current surface event sequences in model-visible order.
    pub fn surface_nodes(&self) -> Vec<u64> {
        self.surface.borrow().nodes().to_vec()
    }

    /// Monotonic count of committed positional replacements.
    pub fn replace_generation(&self) -> u64 {
        self.surface.borrow().replace_generation()
    }

    /// Append one typed event and synchronously notify the store's feed. The
    /// hot path never blocks on I/O — persistence buffers asynchronously.
    /// Once the event enters the log the append is committed: observer
    /// failures are contained and cannot un-commit it.
    ///
    /// `intent` is REQUIRED for the three message-producing event types
    /// (every such event must declare how it joins the surface — the sole
    /// source of derived model history) and rejected on log-only types; both
    /// rules are enforced by the surface validator. A reentrant append while
    /// publication is open is rejected before the log changes.
    pub fn append(
        &self,
        data: SessionEventData,
        intent: Option<SurfaceIntent>,
    ) -> Result<SessionEvent, SessionError> {
        if self.appending.get() {
            return Err(SessionError(
                "session append cannot reenter while another append is being published".into(),
            ));
        }
        let intent = intent.unwrap_or_default();
        let event = SessionEvent {
            seq: self.log.borrow().len() as u64,
            time: now_ms(),
            data,
            source_event_seqs: intent.source_event_seqs,
            surface_op: intent.surface_op,
            ignorable: None,
        };
        {
            let log = self.log.borrow();
            self.surface
                .borrow_mut()
                .validate_next(&log, &event)
                .map_err(SessionError)?;
        }
        self.log.borrow_mut().push(event.clone());
        self.surface.borrow_mut().commit(&self.log.borrow());

        // Publish after commit, contained per listener.
        self.appending.set(true);
        let publisher = self.publisher.borrow().clone();
        if let Some(publisher) = publisher {
            let this = self
                .self_ref
                .borrow()
                .as_ref()
                .and_then(|weak| weak.upgrade());
            if let Some(this) = this {
                publisher(&this, &event);
            }
        }
        self.appending.set(false);
        Ok(event)
    }

    /// The header in force after the log's last `request/header` event — what
    /// the NEXT request is compared against — or `None` before the first
    /// snapshot. Incremental: each header event folds once.
    pub fn request_header(&self) -> Option<EpochHeader> {
        let log = self.log.borrow();
        if self.header_fold_seq.get() < log.len() {
            let current = self.header_fold.borrow().clone();
            let folded = fold_request_header(&log[self.header_fold_seq.get()..], current);
            *self.header_fold.borrow_mut() = folded;
            self.header_fold_seq.set(log.len());
        }
        self.header_fold.borrow().clone()
    }

    /// The latest resolved route metadata, or `None` before the first
    /// `request/context` event. Incremental.
    pub fn request_context(&self) -> Option<RequestContext> {
        let log = self.log.borrow();
        if self.context_fold_seq.get() < log.len() {
            for event in &log[self.context_fold_seq.get()..] {
                if let SessionEventData::RequestContext(context) = &event.data {
                    *self.context_fold.borrow_mut() = Some(context.clone());
                }
            }
            self.context_fold_seq.set(log.len());
        }
        self.context_fold.borrow().clone()
    }

    /// Derive the LLM message history by folding [`derive_event_message`] over
    /// the ordered surface — the single source of derived history. Cached per
    /// node; a surface rewrite (replace) rebuilds. Returns a fresh Vec per
    /// call (later appends never grow a caller's copy).
    pub fn derive_messages(&self) -> Vec<Message> {
        let generation = self.replace_generation();
        if generation != self.derived_generation.get() {
            self.derived.borrow_mut().clear();
            self.derived_nodes.set(0);
            self.derived_generation.set(generation);
        }
        let nodes = self.surface_nodes();
        let log = self.log.borrow();
        {
            let mut derived = self.derived.borrow_mut();
            for seq in &nodes[self.derived_nodes.get()..] {
                let event = &log[*seq as usize];
                // An empty-content assistant/message (max-tokens usage host)
                // derives to nothing and must not enter the transcript.
                if let Some(message) = derive_event_message(event) {
                    derived.push(message.clone());
                }
            }
        }
        self.derived_nodes.set(nodes.len());
        self.derived.borrow().clone()
    }
}

struct StoreEntry {
    session: Rc<Session>,
    announced: bool,
}

/// In-memory session store (`ctx.sessions`). Persistence is deliberately not
/// implemented here — persistence plugins subscribe to `session/event` and
/// flush on `session/flush`.
pub struct SessionStore {
    ctx: Context,
    store: RefCell<HashMap<SessionId, StoreEntry>>,
    order: RefCell<Vec<SessionId>>,
    counter: Cell<u64>,
}

impl Service for SessionStore {
    const NAME: &'static str = "sessions";
}

impl SessionStore {
    /// Create and register the store in `ctx`.
    pub fn provide(ctx: &Context) -> dsh_cordis::Result<Rc<SessionStore>> {
        let store = Rc::new(SessionStore {
            ctx: ctx.clone(),
            store: RefCell::new(HashMap::new()),
            order: RefCell::new(Vec::new()),
            counter: Cell::new(0),
        });
        ctx.provide_service(store.clone())?;
        Ok(store)
    }

    fn mint_id(&self) -> SessionId {
        loop {
            self.counter.set(self.counter.get() + 1);
            let id = SessionId::new(format!("session-{}", self.counter.get()));
            if !self.store.borrow().contains_key(&id) {
                return id;
            }
        }
    }

    /// Build a session WITHOUT entering it into the store (upstream
    /// `prepare`): validate the id/meta and construct the [`Session`] with
    /// its immutable header.
    pub fn prepare(
        &self,
        id: Option<SessionId>,
        seed: Vec<SessionEvent>,
        meta: SessionMeta,
    ) -> Result<Rc<Session>, SessionError> {
        let session_id = match id {
            None => self.mint_id(),
            Some(id) => id,
        };
        if self.store.borrow().contains_key(&session_id) {
            return Err(SessionError(format!(
                "session \"{}\" already exists",
                session_id.as_str()
            )));
        }
        let header = SessionHeader {
            version: SESSION_FORMAT_VERSION,
            id: session_id.clone(),
            created_at: meta.created_at.unwrap_or_else(now_ms),
            cwd: meta.cwd,
            parent_session: meta.parent_session,
            seed_length: meta.seed_length,
            origin: meta.origin,
            delegation_depth: meta.delegation_depth,
            agent_preset: meta.agent_preset,
        };
        Session::create(session_id, seed, Some(header))
    }

    /// Enter a prepared session into the store: install the append
    /// publication hook and add it. Does NOT emit `session/created` — the
    /// caller registers the returned effect first, then calls
    /// [`SessionStore::announce`], so teardown pairs correctly.
    pub fn enter(
        self: &Rc<Self>,
        session: Rc<Session>,
    ) -> Result<dsh_cordis::EffectHandle, SessionError> {
        let id = session.id();
        if self.store.borrow().contains_key(&id) {
            return Err(SessionError(format!(
                "session \"{}\" already exists",
                id.as_str()
            )));
        }
        if session.publisher.borrow().is_some() {
            return Err(SessionError(format!(
                "session \"{}\" is already attached to a store",
                id.as_str()
            )));
        }
        let ctx = self.ctx.clone();
        *session.publisher.borrow_mut() = Some(Rc::new(move |session, event| {
            ctx.emit::<SessionEventPublished>(&(session.clone(), event.clone()));
        }));
        self.store.borrow_mut().insert(
            id.clone(),
            StoreEntry {
                session: session.clone(),
                announced: false,
            },
        );
        self.order.borrow_mut().push(id.clone());

        let store = self.clone();
        self.ctx
            .effect_labeled("sessions.enter()", move |_| {
                Ok(dsh_cordis::Effect::One(dsh_cordis::Disposer::sync(
                    move || {
                        store.detach(&id);
                    },
                )))
            })
            .map_err(|error| SessionError(error.to_string()))
    }

    fn detach(&self, id: &SessionId) {
        let entry = self.store.borrow_mut().remove(id);
        self.order.borrow_mut().retain(|existing| existing != id);
        if let Some(entry) = entry {
            *entry.session.publisher.borrow_mut() = None;
            if entry.announced {
                self.ctx.emit::<SessionDisposed>(&entry.session);
            }
        }
    }

    /// Emit `session/created` exactly once for an entered session.
    pub fn announce(&self, session: &Rc<Session>) -> Result<(), SessionError> {
        let id = session.id();
        let mut store = self.store.borrow_mut();
        let entry = store.get_mut(&id).ok_or_else(|| {
            SessionError(format!(
                "session \"{}\" is not live in this store",
                id.as_str()
            ))
        })?;
        if entry.announced {
            return Err(SessionError(format!(
                "session \"{}\" was already announced",
                id.as_str()
            )));
        }
        entry.announced = true;
        drop(store);
        self.ctx.emit::<SessionCreated>(session);
        Ok(())
    }

    /// Create a session owned by the calling fiber: prepare, enter, announce
    /// (upstream `create`). Disposing the fiber removes the session.
    pub fn create(
        self: &Rc<Self>,
        id: Option<SessionId>,
        seed: Vec<SessionEvent>,
        meta: SessionMeta,
    ) -> Result<Rc<Session>, SessionError> {
        let session = self.prepare(id, seed, meta)?;
        self.enter(session.clone())?;
        self.announce(&session)?;
        Ok(session)
    }

    /// Dispatch the awaited `session/flush` durability checkpoint — THE flush
    /// entry point. Returns once every listener settled.
    pub async fn flush(&self, session: &Rc<Session>) -> Result<(), SessionError> {
        let id = session.id();
        if !self.store.borrow().contains_key(&id) {
            return Err(SessionError(format!(
                "session \"{}\" is not live in this store",
                id.as_str()
            )));
        }
        self.ctx.parallel::<SessionFlush>(session).await;
        Ok(())
    }

    /// Look up a live session.
    pub fn get(&self, id: &SessionId) -> Option<Rc<Session>> {
        self.store
            .borrow()
            .get(id)
            .map(|entry| entry.session.clone())
    }

    /// All live sessions, in creation order.
    pub fn list(&self) -> Vec<Rc<Session>> {
        let store = self.store.borrow();
        self.order
            .borrow()
            .iter()
            .filter_map(|id| store.get(id).map(|entry| entry.session.clone()))
            .collect()
    }

    /// Create a live child session from a stable prefix of a live source
    /// (upstream `fork`). `boundary` is an inclusive source seq; omitted means
    /// the current last event. The selected prefix must not end inside an
    /// open turn.
    pub fn fork(
        self: &Rc<Self>,
        source_id: &SessionId,
        boundary: Option<u64>,
        child_id: Option<SessionId>,
    ) -> Result<Rc<Session>, SessionForkError> {
        if let Some(child_id) = &child_id {
            if self.get(child_id).is_some() {
                return Err(SessionForkError {
                    message: format!("session \"{}\" already exists", child_id.as_str()),
                    code: SessionForkErrorCode::SessionAlreadyExists,
                });
            }
        }
        let source = self.get(source_id).ok_or_else(|| SessionForkError {
            message: format!("session \"{}\" not found", source_id.as_str()),
            code: SessionForkErrorCode::SessionNotFound,
        })?;
        let seed = fork_seed(&source, boundary)?;
        let seed_length = seed.len() as u64;
        self.create(
            child_id,
            seed,
            SessionMeta {
                cwd: source.header.cwd.clone(),
                parent_session: Some(source.id()),
                seed_length: Some(seed_length),
                ..Default::default()
            },
        )
        .map_err(|error| SessionForkError {
            message: error.0,
            code: SessionForkErrorCode::InvalidBoundary,
        })
    }
}

/// Rejection codes for session forking (upstream `SessionForkErrorCode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionForkErrorCode {
    SessionNotFound,
    SessionNotLive,
    SessionAlreadyExists,
    InvalidBoundary,
    OpenTurn,
}

/// Typed error for session fork rejections.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct SessionForkError {
    pub message: String,
    pub code: SessionForkErrorCode,
}

fn fork_seed(
    session: &Session,
    boundary: Option<u64>,
) -> Result<Vec<SessionEvent>, SessionForkError> {
    session.with_events(|events| {
        let boundary = match boundary {
            Some(boundary) => boundary,
            None => match events.last() {
                None => return Ok(Vec::new()),
                Some(last) => last.seq,
            },
        };
        if boundary >= events.len() as u64 {
            let last_seq = events.last().map(|event| event.seq.to_string());
            return Err(SessionForkError {
                message: format!(
                    "fork boundary {boundary} does not exist in session \"{}\" (last seq: {})",
                    session.header.id.as_str(),
                    last_seq.unwrap_or_else(|| "none".into())
                ),
                code: SessionForkErrorCode::InvalidBoundary,
            });
        }
        // The prefix may end between turns but never inside an open turn.
        let last_turn_boundary = events[..=(boundary as usize)].iter().rev().find(|event| {
            matches!(
                event.data,
                SessionEventData::TurnStart { .. } | SessionEventData::TurnEnd { .. }
            )
        });
        if let Some(event) = last_turn_boundary {
            if let SessionEventData::TurnStart { turn } = event.data {
                return Err(SessionForkError {
                    message: format!(
                        "fork boundary {boundary} in session \"{}\" ends inside open turn {turn}",
                        session.header.id.as_str()
                    ),
                    code: SessionForkErrorCode::OpenTurn,
                });
            }
        }
        Ok(events[..=(boundary as usize)].to_vec())
    })
}
