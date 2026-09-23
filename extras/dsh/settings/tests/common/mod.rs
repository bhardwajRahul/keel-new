//! Shared test fixture: the in-memory settings backend (upstream
//! `tests/memory.ts`) plus boot and settling helpers.
#![allow(dead_code)] // each test crate uses a different subset of the fixture

use dsh_cordis::{App, Context, Fiber, Inject, plugin_fn};
use dsh_settings::{
    SettingsBackend, SettingsNamespace, SettingsService, SettingsUpdateSource, SettingsUpdated,
};
use futures::future::LocalBoxFuture;
use serde_json::{Map, Value};
use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};
use std::time::Duration;

fn as_map(doc: Value) -> Map<String, Value> {
    match doc {
        Value::Null => Map::new(),
        Value::Object(map) => map,
        other => panic!("document fixture must be an object, got {other}"),
    }
}

/// In-memory provider backend exposing the storage hooks to tests.
pub struct MemoryBackend {
    /// Raw document the provider "storage" currently holds.
    pub doc: Rc<RefCell<Map<String, Value>>>,
    /// Every persist call observed, in order: (namespace, section).
    pub persisted: Rc<RefCell<Vec<(String, Map<String, Value>)>>>,
    /// When false, writes must reject before reaching persist.
    pub writable: Cell<bool>,
    /// Artificial persist latency so tests can interleave concurrent writes.
    pub persist_delay_ms: Cell<u64>,
    service: RefCell<Weak<SettingsService>>,
}

impl MemoryBackend {
    pub fn new(doc: Value) -> Rc<MemoryBackend> {
        Rc::new(MemoryBackend {
            doc: Rc::new(RefCell::new(as_map(doc))),
            persisted: Rc::default(),
            writable: Cell::new(true),
            persist_delay_ms: Cell::new(0),
            service: RefCell::new(Weak::new()),
        })
    }

    /// Simulate an external storage change reaching the provider.
    pub fn push_external(&self, doc: Value) {
        let map = as_map(doc);
        *self.doc.borrow_mut() = map.clone();
        if let Some(service) = self.service.borrow().upgrade() {
            service.publish(map);
        }
    }
}

impl SettingsBackend for MemoryBackend {
    fn writable(&self) -> bool {
        self.writable.get()
    }

    fn attach(&self, service: Weak<SettingsService>) {
        *self.service.borrow_mut() = service;
    }

    fn load(&self) -> LocalBoxFuture<'static, anyhow::Result<Map<String, Value>>> {
        let doc = self.doc.borrow().clone();
        Box::pin(async move { Ok(doc) })
    }

    fn persist(
        &self,
        ns: &SettingsNamespace,
        section: &Map<String, Value>,
    ) -> LocalBoxFuture<'static, anyhow::Result<()>> {
        let delay = self.persist_delay_ms.get();
        let ns = ns.to_string();
        let section = section.clone();
        let doc = self.doc.clone();
        let persisted = self.persisted.clone();
        Box::pin(async move {
            if delay > 0 {
                tokio::time::sleep(Duration::from_millis(delay)).await;
            }
            persisted.borrow_mut().push((ns.clone(), section.clone()));
            doc.borrow_mut().insert(ns, Value::Object(section));
            Ok(())
        })
    }
}

pub struct Booted {
    #[allow(dead_code)]
    pub app: App,
    pub ctx: Context,
    pub backend: Rc<MemoryBackend>,
    pub service: Rc<SettingsService>,
    pub fiber: Fiber,
}

pub async fn boot(doc: Value) -> Booted {
    boot_with(doc, true, 0).await
}

pub async fn boot_with(doc: Value, writable: bool, persist_delay_ms: u64) -> Booted {
    let app = App::new();
    let ctx = app.root();
    let backend = MemoryBackend::new(doc);
    backend.writable.set(writable);
    backend.persist_delay_ms.set(persist_delay_ms);
    let plugin_backend = backend.clone();
    let plugin = plugin_fn("memory-settings", Inject::default(), move |ctx, _| {
        let backend = plugin_backend.clone();
        async move { dsh_settings::mount(&ctx, backend).await.map(|_| ()) }
    });
    let fiber = ctx.plugin(Rc::new(plugin), Value::Null).unwrap();
    fiber.await_ready().await.unwrap();
    let service = ctx
        .try_service::<SettingsService>()
        .expect("settings service provided");
    Booted {
        app,
        ctx,
        backend,
        service,
        fiber,
    }
}

/// Let spawned tasks (emit listeners, watcher segments) run to completion.
pub async fn settle() {
    for _ in 0..25 {
        tokio::task::yield_now().await;
    }
}

/// Poll a condition with real-time backoff (upstream `vi.waitFor`).
pub async fn wait_until(mut cond: impl FnMut() -> bool) {
    for _ in 0..2000 {
        if cond() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    panic!("wait_until timed out");
}

pub type UpdateRecord = (String, Value, Value, SettingsUpdateSource);

/// Record every `settings/updated` emission.
pub fn record_updates(ctx: &Context) -> Rc<RefCell<Vec<UpdateRecord>>> {
    let events: Rc<RefCell<Vec<UpdateRecord>>> = Rc::default();
    let sink = events.clone();
    ctx.on::<SettingsUpdated, _, _>(Default::default(), move |_ctx, args| {
        sink.borrow_mut()
            .push((args.0.to_string(), args.1.clone(), args.2.clone(), args.3));
        std::future::ready(None::<()>)
    })
    .unwrap();
    events
}
