//! Port of upstream `tests/review-fixes.spec.ts`: read-modify-write under the
//! writer lock (external edits survive an API write) and the document
//! editor's isolation between entries.
//!
//! Not ported: the contained-fan-out cases (a listener throwing / rejecting,
//! and the `INVARIANT` rethrow) — dsh-cordis listeners are spawned futures
//! whose failures cannot reach the emitter, so containment is structural and
//! the invariant channel does not exist. The block-scalar sibling becomes a
//! quoted multi-line sibling (document subset). `tests/drain.spec.ts` is not
//! ported either: storage operations are synchronous here, so the
//! in-flight-versus-queued disposal race cannot arise.

use dsh_cordis::{App, Context, Fiber};
use dsh_credentials::{
    CredentialRef, Credentials, CredentialsUpdated, ResolvedCredential, credential_ref,
};
use dsh_credentials_local::LocalCredentials;
use serde_json::{Value, json};
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

fn alpha() -> CredentialRef {
    credential_ref("DSH_REVIEW_ALPHA").unwrap()
}

fn beta() -> CredentialRef {
    credential_ref("DSH_REVIEW_BETA").unwrap()
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

fn file_source(value: &str) -> Option<ResolvedCredential> {
    Some(ResolvedCredential {
        value: value.into(),
        source: "file".into(),
    })
}

#[test]
fn folds_an_unobserved_external_edit_into_a_write_instead_of_overwriting_it() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        let app = App::new();
        let ctx = app.root();
        let (_fiber, creds) = boot(
            &ctx,
            json!({"path": path.to_str().unwrap(), "watch": false}),
        )
        .await;
        let seen: Rc<RefCell<Vec<CredentialRef>>> = Rc::default();
        let sink = seen.clone();
        ctx.on::<CredentialsUpdated, _, _>(Default::default(), move |_, r| {
            sink.borrow_mut().push(r.clone());
            std::future::ready(None)
        })
        .unwrap();

        creds.set(&alpha(), "one").await.unwrap();
        // The external edit has landed on disk but no detector reported it
        // (watch is off — the same blind spot as a settle window or a missed
        // event).
        write_credentials(&path, "DSH_REVIEW_ALPHA: one\nDSH_REVIEW_BETA: external\n");
        creds.set(&alpha(), "two").await.unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("DSH_REVIEW_BETA: external"), "{text}");
        assert!(text.contains("DSH_REVIEW_ALPHA: two"), "{text}");
        // The fold published the unobserved entry before the write's own
        // commit.
        assert_eq!(*seen.borrow(), vec![alpha(), beta(), alpha()]);
        assert_eq!(
            creds.resolve(&beta()).await.unwrap(),
            file_source("external")
        );
    });
}

#[test]
fn keeps_both_refs_when_two_providers_write_the_same_document_concurrently() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        let config = json!({"path": path.to_str().unwrap(), "watch": false});
        let first_app = App::new();
        let second_app = App::new();
        let (_f1, first) = boot(&first_app.root(), config.clone()).await;
        let (_f2, second) = boot(&second_app.root(), config.clone()).await;

        let alpha_writes = async {
            for value in ["1", "2", "3"] {
                first.set(&alpha(), value).await.unwrap();
            }
        };
        let beta_writes = async {
            for value in ["1", "2", "3"] {
                second.set(&beta(), value).await.unwrap();
            }
        };
        futures::join!(alpha_writes, beta_writes);

        let third_app = App::new();
        let (_f3, third) = boot(&third_app.root(), config).await;
        assert_eq!(third.resolve(&alpha()).await.unwrap(), file_source("3"));
        assert_eq!(third.resolve(&beta()).await.unwrap(), file_source("3"));
    });
}

#[cfg(unix)]
#[test]
fn creates_the_credentials_directory_owner_only() {
    dsh_cordis::run(async {
        use std::os::unix::fs::MetadataExt;
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("home");
        let path = home.join(".credentials.yaml");
        let app = App::new();
        let (_fiber, creds) = boot(
            &app.root(),
            json!({"path": path.to_str().unwrap(), "watch": false}),
        )
        .await;
        creds.set(&alpha(), "one").await.unwrap();
        assert_eq!(std::fs::metadata(&home).unwrap().mode() & 0o777, 0o700);
    });
}

#[test]
fn leaves_a_sibling_multi_line_value_untouched_while_patching_one_entry() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        // Upstream seeds a `|-` block scalar; the subset's multi-line form is
        // a quoted scalar, which must survive a sibling edit byte-for-byte.
        let wrapped = "DSH_REVIEW_WRAPPED: \"line1\\nline2\"\nDSH_REVIEW_ALPHA: a\n";
        write_credentials(&path, wrapped);
        let app = App::new();
        let (_fiber, creds) = boot(
            &app.root(),
            json!({"path": path.to_str().unwrap(), "watch": false}),
        )
        .await;
        creds.set(&alpha(), "b").await.unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "DSH_REVIEW_WRAPPED: \"line1\\nline2\"\nDSH_REVIEW_ALPHA: b\n",
        );
        assert_eq!(
            creds
                .resolve(&credential_ref("DSH_REVIEW_WRAPPED").unwrap())
                .await
                .unwrap(),
            file_source("line1\nline2"),
        );
    });
}

#[test]
fn stores_a_value_that_looks_like_another_entry_without_creating_one() {
    dsh_cordis::run(async {
        let inner = credential_ref("DSH_REVIEW_INNER").unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        let config = json!({"path": path.to_str().unwrap(), "watch": false});
        {
            let app = App::new();
            let (_fiber, creds) = boot(&app.root(), config.clone()).await;
            // The stored text must stay a value: a write that leaked its own
            // structure would silently mint a credential nobody stored.
            creds
                .set(&alpha(), "DSH_REVIEW_INNER: injected")
                .await
                .unwrap();
        }
        let app = App::new();
        let (_fiber, reread) = boot(&app.root(), config).await;
        assert_eq!(
            reread.resolve(&alpha()).await.unwrap(),
            file_source("DSH_REVIEW_INNER: injected"),
        );
        assert_eq!(reread.resolve(&inner).await.unwrap(), None);
    });
}
