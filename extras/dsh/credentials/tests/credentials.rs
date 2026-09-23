//! Port of upstream `tests/credentials.spec.ts`: the reference brand and the
//! seam contract exercised through an in-memory provider (upstream
//! `tests/memory.ts`). The `tests/invariant.spec.ts` suite is not ported —
//! the `dsh-invariants` registry does not exist in this workspace.

use dsh_cordis::{App, Context, EventOptions, Plugin};
use dsh_credentials::{
    CredentialInfo, CredentialProvider, CredentialRef, Credentials, CredentialsUpdated,
    ResolvedCredential, credential_ref, notify_updated,
};
use futures::future::LocalBoxFuture;
use serde_json::Value;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// In-memory provider for seam tests: one always-writable `memory` source
/// seeded from plugin config.
struct MemoryCredentials {
    ctx: Context,
    store: RefCell<HashMap<String, String>>,
}

#[async_trait::async_trait(?Send)]
impl CredentialProvider for MemoryCredentials {
    async fn resolve(&self, r: &CredentialRef) -> anyhow::Result<Option<ResolvedCredential>> {
        Ok(self
            .store
            .borrow()
            .get(r.as_str())
            .filter(|v| !v.is_empty())
            .map(|v| ResolvedCredential {
                value: v.clone(),
                source: "memory".into(),
            }))
    }

    async fn describe(&self, r: &CredentialRef) -> anyhow::Result<CredentialInfo> {
        let configured = self
            .store
            .borrow()
            .get(r.as_str())
            .is_some_and(|v| !v.is_empty());
        Ok(CredentialInfo {
            configured,
            source: configured.then(|| "memory".to_string()),
            writable: true,
        })
    }

    async fn set(&self, r: &CredentialRef, value: &str) -> anyhow::Result<()> {
        if value.is_empty() {
            anyhow::bail!("memory credentials: an empty value cannot be stored; use unset");
        }
        self.store
            .borrow_mut()
            .insert(r.as_str().to_string(), value.to_string());
        notify_updated(&self.ctx, r);
        Ok(())
    }

    async fn unset(&self, r: &CredentialRef) -> anyhow::Result<()> {
        if self.store.borrow_mut().remove(r.as_str()).is_some() {
            notify_updated(&self.ctx, r);
        }
        Ok(())
    }
}

struct MemoryPlugin;

impl Plugin for MemoryPlugin {
    fn name(&self) -> Option<String> {
        Some("memory-credentials".into())
    }

    fn apply(&self, ctx: Context, config: Value) -> LocalBoxFuture<'static, anyhow::Result<()>> {
        Box::pin(async move {
            let seed: HashMap<String, String> = if config.is_null() {
                HashMap::new()
            } else {
                serde_json::from_value(config)?
            };
            let provider = Rc::new(MemoryCredentials {
                ctx: ctx.clone(),
                store: RefCell::new(seed),
            });
            ctx.provide_service(Rc::new(Credentials(provider)))?;
            Ok(())
        })
    }
}

fn key() -> CredentialRef {
    credential_ref("DEEPSEEK_API_KEY").unwrap()
}

async fn boot(app: &App, seed: Value) -> Context {
    let ctx = app.root();
    let fiber = ctx.plugin(Rc::new(MemoryPlugin), seed).unwrap();
    fiber.await_ready().await.unwrap();
    ctx
}

fn updates(ctx: &Context) -> Rc<RefCell<Vec<CredentialRef>>> {
    let seen: Rc<RefCell<Vec<CredentialRef>>> = Rc::default();
    let sink = seen.clone();
    ctx.on::<CredentialsUpdated, _, _>(EventOptions::default(), move |_, r| {
        sink.borrow_mut().push(r.clone());
        std::future::ready(None)
    })
    .unwrap();
    seen
}

#[test]
fn brands_posix_shell_identifiers() {
    for valid in ["DEEPSEEK_API_KEY", "_private", "lower_case9"] {
        assert_eq!(credential_ref(valid).unwrap().as_str(), valid);
    }
}

#[test]
fn rejects_every_other_shape() {
    for invalid in ["", "9LEADING", "WITH-DASH", "WITH SPACE", "ns:key"] {
        assert!(
            credential_ref(invalid).is_err(),
            "{invalid:?} should be rejected"
        );
    }
}

#[test]
fn mounts_as_credentials_and_resolves_a_seeded_reference() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = boot(&app, serde_json::json!({"DEEPSEEK_API_KEY": "sk-seeded"})).await;
        let creds = ctx.try_service::<Credentials>().unwrap();
        assert_eq!(
            creds.resolve(&key()).await.unwrap(),
            Some(ResolvedCredential {
                value: "sk-seeded".into(),
                source: "memory".into()
            })
        );
        assert_eq!(
            creds.describe(&key()).await.unwrap(),
            CredentialInfo {
                configured: true,
                source: Some("memory".into()),
                writable: true
            }
        );
    });
}

#[test]
fn treats_an_empty_stored_value_as_absent_everywhere() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = boot(&app, serde_json::json!({"DEEPSEEK_API_KEY": ""})).await;
        let creds = ctx.try_service::<Credentials>().unwrap();
        assert_eq!(creds.resolve(&key()).await.unwrap(), None);
        assert_eq!(
            creds.describe(&key()).await.unwrap(),
            CredentialInfo {
                configured: false,
                source: None,
                writable: true
            }
        );
    });
}

#[test]
fn stores_removes_and_emits_the_committed_change() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = boot(&app, Value::Null).await;
        let seen = updates(&ctx);
        let creds = ctx.try_service::<Credentials>().unwrap();

        creds.set(&key(), "sk-live").await.unwrap();
        assert_eq!(
            creds.resolve(&key()).await.unwrap(),
            Some(ResolvedCredential {
                value: "sk-live".into(),
                source: "memory".into()
            })
        );
        creds.unset(&key()).await.unwrap();
        assert_eq!(creds.resolve(&key()).await.unwrap(), None);
        assert_eq!(*seen.borrow(), vec![key(), key()]);
    });
}

#[test]
fn rejects_an_empty_set_and_keeps_an_absent_unset_silent() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = boot(&app, Value::Null).await;
        let seen = updates(&ctx);
        let creds = ctx.try_service::<Credentials>().unwrap();

        let error = creds.set(&key(), "").await.unwrap_err();
        assert!(error.to_string().contains("empty value"));
        creds.unset(&key()).await.unwrap();
        assert!(seen.borrow().is_empty());
    });
}

#[test]
fn removes_the_service_with_its_fiber() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let fiber = ctx.plugin(Rc::new(MemoryPlugin), Value::Null).unwrap();
        fiber.await_ready().await.unwrap();
        assert!(ctx.try_service::<Credentials>().is_some());
        fiber.dispose().await;
        assert!(ctx.try_service::<Credentials>().is_none());
    });
}
