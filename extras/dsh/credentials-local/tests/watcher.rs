//! Port of the hot-reload contract from upstream `tests/local.spec.ts` ("real
//! hot reload") and `tests/watcher.spec.ts`, driven through the real polling
//! change detector instead of a faked chokidar.
//!
//! Not ported: the injected-`readFile`-failure case (needs fs mocking; the
//! chmod-unreadable case covers the same warn-and-keep policy), the
//! `INVARIANT`-escape queue test (channel not ported), the still-absent-file
//! event no-op (cannot be triggered without a fakeable watcher), and the
//! watcher-option clamp (chokidar-specific).

use dsh_cordis::{App, Context, Fiber};
use dsh_credentials::{
    CredentialRef, Credentials, CredentialsUpdated, ResolvedCredential, credential_ref,
};
use dsh_credentials_local::LocalCredentials;
use serde_json::{Value, json};
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::time::Duration;

fn key() -> CredentialRef {
    credential_ref("DSH_CRED_PIPE").unwrap()
}

/// Credential documents are seeded owner-only, exactly as the provider
/// creates them.
fn write_credentials(file: &Path, text: &str) {
    use std::io::Write as _;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
        .open(file)
        .unwrap()
        .write_all(text.as_bytes())
        .unwrap();
}

async fn boot(ctx: &Context, config: Value) -> (Fiber, Rc<Credentials>) {
    let fiber = ctx.plugin(Rc::new(LocalCredentials), config).unwrap();
    fiber.await_ready().await.unwrap();
    (fiber, ctx.try_service::<Credentials>().unwrap())
}

fn updates(ctx: &Context) -> Rc<RefCell<Vec<CredentialRef>>> {
    let seen: Rc<RefCell<Vec<CredentialRef>>> = Rc::default();
    let sink = seen.clone();
    ctx.on::<CredentialsUpdated, _, _>(Default::default(), move |_, r| {
        sink.borrow_mut().push(r.clone());
        std::future::ready(None)
    })
    .unwrap();
    seen
}

fn file_source(value: &str) -> Option<ResolvedCredential> {
    Some(ResolvedCredential {
        value: value.into(),
        source: "file".into(),
    })
}

/// Wait (up to ~3s) for an asynchronously published state.
async fn wait_until(mut cond: impl AsyncFnMut() -> bool) {
    for _ in 0..300 {
        if cond().await {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("timed out waiting for the reload to publish");
}

#[test]
fn publishes_external_edits_replaces_wholesale_and_suppresses_self_writes() {
    dsh_cordis::run(async {
        let other = credential_ref("DSH_CRED_PIPE_OTHER").unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        // Watching starts on an existing document: creation racing detector
        // startup is a readiness gap, not the reload contract under test.
        write_credentials(&path, "DSH_CRED_PIPE: boot\n");
        let app = App::new();
        let ctx = app.root();
        let (_fiber, creds) = boot(
            &ctx,
            json!({"path": path.to_str().unwrap(), "debounceMs": 10}),
        )
        .await;
        let seen = updates(&ctx);

        write_credentials(&path, "DSH_CRED_PIPE: live\nDSH_CRED_PIPE_OTHER: extra\n");
        wait_until(async || creds.resolve(&key()).await.unwrap() == file_source("live")).await;

        // Wholesale replacement: an entry deleted on disk never lingers in
        // memory.
        write_credentials(&path, "DSH_CRED_PIPE: live\n");
        wait_until(async || creds.resolve(&other).await.unwrap().is_none()).await;

        let before = seen.borrow().len();
        creds.set(&key(), "self-written").await.unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;
        // Exactly the committed write's own event: the detector's echo of our
        // own content is recognized by the text cache and publishes nothing.
        assert_eq!(seen.borrow().len(), before + 1);
        assert_eq!(
            creds.resolve(&key()).await.unwrap(),
            file_source("self-written")
        );
    });
}

#[test]
fn empties_the_snapshot_when_the_document_is_deleted_and_emits_the_removals() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        write_credentials(&path, "DSH_CRED_PIPE: doomed\n");
        let app = App::new();
        let ctx = app.root();
        let (_fiber, creds) = boot(
            &ctx,
            json!({"path": path.to_str().unwrap(), "debounceMs": 10}),
        )
        .await;
        let seen = updates(&ctx);

        std::fs::remove_file(&path).unwrap();
        wait_until(async || creds.resolve(&key()).await.unwrap().is_none()).await;
        assert_eq!(*seen.borrow(), vec![key()]);
    });
}

#[test]
fn keeps_the_last_good_snapshot_when_an_external_edit_makes_the_document_invalid() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        write_credentials(&path, "DSH_CRED_PIPE: a\n");
        let app = App::new();
        let ctx = app.root();
        let (_fiber, creds) = boot(
            &ctx,
            json!({"path": path.to_str().unwrap(), "debounceMs": 10}),
        )
        .await;
        let seen = updates(&ctx);

        // A key the seam cannot address is a rejection, not preserved
        // content. A live reload must warn and keep serving the last good
        // snapshot rather than take the process down or drop the entry.
        write_credentials(&path, "BAD-KEY: 2\nDSH_CRED_PIPE: b\n");
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(creds.resolve(&key()).await.unwrap(), file_source("a"));
        assert!(seen.borrow().is_empty());

        // Repairing the document resumes publishing.
        write_credentials(&path, "DSH_CRED_PIPE: b\n");
        wait_until(async || creds.resolve(&key()).await.unwrap() == file_source("b")).await;
        assert_eq!(*seen.borrow(), vec![key()]);
    });
}

#[cfg(unix)]
#[test]
fn keeps_the_last_good_snapshot_when_the_file_turns_unreadable_at_runtime() {
    dsh_cordis::run(async {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        write_credentials(&path, "DSH_CRED_PIPE: good\n");
        let app = App::new();
        let ctx = app.root();
        let (_fiber, creds) = boot(
            &ctx,
            json!({"path": path.to_str().unwrap(), "debounceMs": 10}),
        )
        .await;

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).unwrap();
        // The warn-and-keep path needs a detector turn.
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(creds.resolve(&key()).await.unwrap(), file_source("good"));
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    });
}

#[test]
fn quiesces_the_refresh_pipeline_before_dispose_completes() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        write_credentials(&path, "DSH_CRED_PIPE: initial\n");
        let app = App::new();
        let ctx = app.root();
        let (fiber, _creds) = boot(
            &ctx,
            json!({"path": path.to_str().unwrap(), "debounceMs": 10}),
        )
        .await;

        let disposed = Rc::new(std::cell::Cell::new(false));
        let post_dispose_commits = Rc::new(std::cell::Cell::new(0u32));
        let disposed_flag = disposed.clone();
        let counter = post_dispose_commits.clone();
        ctx.on::<CredentialsUpdated, _, _>(Default::default(), move |_, _| {
            if disposed_flag.get() {
                counter.set(counter.get() + 1);
            }
            std::future::ready(None)
        })
        .unwrap();

        write_credentials(&path, "DSH_CRED_PIPE: changed\n");
        fiber.dispose().await;
        disposed.set(true);
        write_credentials(&path, "DSH_CRED_PIPE: after-dispose\n");
        tokio::time::sleep(Duration::from_millis(250)).await;
        assert_eq!(post_dispose_commits.get(), 0);
    });
}
