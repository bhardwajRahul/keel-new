//! Port of upstream `tests/local.spec.ts`: spec resolution, layering and
//! reads, the layer ladder, document validation, and document writes.
//!
//! Adaptations to the Rust port's document subset and environment handling:
//! - Quoted values are JSON-style double-quoted scalars; unsetting the last
//!   entry leaves an empty file instead of `{}`.
//! - `vi.stubEnv` becomes real process-environment mutation with per-test
//!   unique variable names (tests run on parallel threads).
//! - OS error assertions match Rust's `io::Error` display ("Not a
//!   directory", "Is a directory") instead of Node codes.

use dsh_cordis::{App, Context, Fiber};
use dsh_credentials::{
    CredentialInfo, CredentialRef, Credentials, CredentialsUpdated, ResolvedCredential,
    credential_ref,
};
use dsh_credentials_local::{Config, DSH_LAUNCH_ENVIRONMENT_KEY, LocalCredentials, resolve_spec};
use dsh_launch_environment::{
    LaunchEnvironmentLayerInput, LaunchEnvironmentSnapshot, LaunchEnvironmentSource,
};
use serde_json::{Value, json};
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

fn key() -> CredentialRef {
    credential_ref("DSH_CRED_TEST").unwrap()
}

fn other() -> CredentialRef {
    credential_ref("DSH_CRED_OTHER").unwrap()
}

/// Credential documents are seeded owner-only, exactly as the provider
/// creates them.
fn write_credentials(file: &Path, text: &str) {
    write_with_mode(file, text, 0o600);
}

fn write_with_mode(file: &Path, text: &str, _mode: u32) {
    use std::io::Write as _;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(_mode);
    }
    let mut handle = options.open(file).unwrap();
    handle.write_all(text.as_bytes()).unwrap();
}

async fn boot(ctx: &Context, config: Value) -> (Fiber, Rc<Credentials>) {
    let fiber = ctx.plugin(Rc::new(LocalCredentials), config).unwrap();
    fiber.await_ready().await.unwrap();
    let creds = ctx.try_service::<Credentials>().unwrap();
    (fiber, creds)
}

async fn boot_failure(config: Value) -> String {
    let app = App::new();
    let fiber = app
        .root()
        .plugin(Rc::new(LocalCredentials), config)
        .unwrap();
    let error = fiber.await_ready().await.unwrap_err();
    error.to_string()
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

fn layers(
    entries: &[(LaunchEnvironmentSource, Option<&str>, &[(&str, &str)])],
) -> Rc<LaunchEnvironmentSnapshot> {
    Rc::new(LaunchEnvironmentSnapshot::new(entries.iter().map(
        |(source, path, values)| {
            LaunchEnvironmentLayerInput {
                source: *source,
                path: path.map(PathBuf::from),
                values: values
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
            }
        },
    )))
}

fn file_source(value: &str) -> Option<ResolvedCredential> {
    Some(ResolvedCredential {
        value: value.into(),
        source: "file".into(),
    })
}

// --- resolveSpec -----------------------------------------------------------

#[test]
fn resolve_spec_defaults_to_credentials_yaml_under_the_home_with_watching_on() {
    let spec = resolve_spec(&Config {
        dsh_home: Some("/custom/home".into()),
        ..Config::default()
    });
    assert_eq!(
        spec.filename,
        PathBuf::from("/custom/home/.credentials.yaml")
    );
    assert!(spec.watch);
    assert_eq!(spec.debounce_ms, 100);
}

#[test]
fn resolve_spec_lets_an_explicit_path_win_over_the_home() {
    let spec = resolve_spec(&Config {
        path: Some("/etc/dsh/creds.yaml".into()),
        dsh_home: Some("/ignored".into()),
        watch: Some(false),
        debounce_ms: Some(5),
    });
    assert_eq!(spec.filename, PathBuf::from("/etc/dsh/creds.yaml"));
    assert!(!spec.watch);
    assert_eq!(spec.debounce_ms, 5);
}

// --- layering and reads ----------------------------------------------------

#[test]
fn treats_an_absent_file_as_an_empty_writable_store() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        let app = App::new();
        let (_fiber, creds) = boot(
            &app.root(),
            json!({"path": path.to_str().unwrap(), "watch": false}),
        )
        .await;
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
fn serves_file_entries_alongside_comments_and_quoted_values() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        write_credentials(
            &path,
            "# notes\nDSH_CRED_TEST: plain\nDSH_CRED_OTHER: \"with space\"\n",
        );
        let app = App::new();
        let (_fiber, creds) = boot(
            &app.root(),
            json!({"path": path.to_str().unwrap(), "watch": false}),
        )
        .await;
        assert_eq!(creds.resolve(&key()).await.unwrap(), file_source("plain"));
        assert_eq!(
            creds.resolve(&other()).await.unwrap(),
            file_source("with space")
        );
        assert_eq!(
            creds.describe(&key()).await.unwrap(),
            CredentialInfo {
                configured: true,
                source: Some("file".into()),
                writable: true
            }
        );
    });
}

#[test]
fn lets_a_non_empty_process_environment_win_read_only_over_the_file() {
    dsh_cordis::run(async {
        let r = credential_ref("DSH_CRED_LOCAL_ENVWIN").unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        write_credentials(&path, "DSH_CRED_LOCAL_ENVWIN: from-file\n");
        let app = App::new();
        let (_fiber, creds) = boot(
            &app.root(),
            json!({"path": path.to_str().unwrap(), "watch": false}),
        )
        .await;
        // Stubbed after boot: the fallback environment lookup is live.
        // SAFETY: the variable name is unique to this test.
        unsafe { std::env::set_var("DSH_CRED_LOCAL_ENVWIN", "from-env") };
        assert_eq!(
            creds.resolve(&r).await.unwrap(),
            Some(ResolvedCredential {
                value: "from-env".into(),
                source: "env".into()
            })
        );
        assert_eq!(
            creds.describe(&r).await.unwrap(),
            CredentialInfo {
                configured: true,
                source: Some("env".into()),
                writable: false
            }
        );
        unsafe { std::env::remove_var("DSH_CRED_LOCAL_ENVWIN") };
    });
}

#[test]
fn treats_an_empty_environment_value_as_absent_falling_through_to_the_file() {
    dsh_cordis::run(async {
        let r = credential_ref("DSH_CRED_LOCAL_EMPTYENV").unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        write_credentials(&path, "DSH_CRED_LOCAL_EMPTYENV: stored\n");
        let app = App::new();
        let (_fiber, creds) = boot(
            &app.root(),
            json!({"path": path.to_str().unwrap(), "watch": false}),
        )
        .await;
        // SAFETY: the variable name is unique to this test.
        unsafe { std::env::set_var("DSH_CRED_LOCAL_EMPTYENV", "") };
        assert_eq!(creds.resolve(&r).await.unwrap(), file_source("stored"));
        assert_eq!(
            creds.describe(&r).await.unwrap(),
            CredentialInfo {
                configured: true,
                source: Some("file".into()),
                writable: true
            }
        );
        unsafe { std::env::remove_var("DSH_CRED_LOCAL_EMPTYENV") };
    });
}

#[test]
fn fails_boot_loud_when_the_document_exists_but_cannot_be_read() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("occupied");
        std::fs::create_dir(&path).unwrap();
        let message = boot_failure(json!({"path": path.to_str().unwrap(), "watch": false})).await;
        assert!(
            message.contains("plugin startup failed"),
            "unexpected error: {message}"
        );
    });
}

// --- layer ladder ----------------------------------------------------------

async fn boot_layered(
    app: &App,
    path: &Path,
    snapshot: Rc<LaunchEnvironmentSnapshot>,
) -> (Fiber, Rc<Credentials>) {
    let ctx = app.root();
    ctx.provide(DSH_LAUNCH_ENVIRONMENT_KEY, snapshot, None)
        .unwrap();
    boot(
        &ctx,
        json!({"path": path.to_str().unwrap(), "watch": false}),
    )
    .await
}

#[test]
fn lets_the_stored_value_beat_the_user_env_so_a_ui_write_takes_effect() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        write_credentials(&path, "DSH_CRED_TEST: stored\n");
        let app = App::new();
        let (_fiber, creds) = boot_layered(
            &app,
            &path,
            layers(&[
                (LaunchEnvironmentSource::Process, None, &[]),
                (
                    LaunchEnvironmentSource::UserEnv,
                    Some("/home/.dsh/.env"),
                    &[("DSH_CRED_TEST", "older-user-env")],
                ),
            ]),
        )
        .await;
        assert_eq!(creds.resolve(&key()).await.unwrap(), file_source("stored"));
        // A key sitting in the user's .env does not make the stored one
        // unwritable.
        assert_eq!(
            creds.describe(&key()).await.unwrap(),
            CredentialInfo {
                configured: true,
                source: Some("file".into()),
                writable: true
            }
        );
        creds.set(&key(), "rotated").await.unwrap();
        assert_eq!(creds.resolve(&key()).await.unwrap(), file_source("rotated"));
    });
}

#[test]
fn serves_the_user_env_only_when_nothing_is_stored() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        let app = App::new();
        let (_fiber, creds) = boot_layered(
            &app,
            &path,
            layers(&[
                (LaunchEnvironmentSource::Process, None, &[]),
                (
                    LaunchEnvironmentSource::UserEnv,
                    Some("/home/.dsh/.env"),
                    &[("DSH_CRED_TEST", "from-user-env")],
                ),
            ]),
        )
        .await;
        assert_eq!(
            creds.resolve(&key()).await.unwrap(),
            Some(ResolvedCredential {
                value: "from-user-env".into(),
                source: "user-env".into()
            })
        );
        // Writable: storing a key replaces it as the effective one.
        assert_eq!(
            creds.describe(&key()).await.unwrap(),
            CredentialInfo {
                configured: true,
                source: Some("user-env".into()),
                writable: true
            }
        );
    });
}

#[test]
fn serves_the_project_env_over_the_user_one_but_never_over_the_store() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        let ladder: &[(LaunchEnvironmentSource, Option<&str>, &[(&str, &str)])] = &[
            (LaunchEnvironmentSource::Process, None, &[]),
            (
                LaunchEnvironmentSource::ProjectEnv,
                Some("/work/.env"),
                &[("DSH_CRED_TEST", "from-project")],
            ),
            (
                LaunchEnvironmentSource::UserEnv,
                Some("/home/.dsh/.env"),
                &[("DSH_CRED_TEST", "from-user")],
            ),
        ];
        let bare = App::new();
        let (_f1, creds) = boot_layered(&bare, &path, layers(ladder)).await;
        assert_eq!(
            creds.resolve(&key()).await.unwrap(),
            Some(ResolvedCredential {
                value: "from-project".into(),
                source: "project-env".into()
            })
        );
        assert_eq!(
            creds.describe(&key()).await.unwrap(),
            CredentialInfo {
                configured: true,
                source: Some("project-env".into()),
                writable: true
            }
        );

        write_credentials(&path, "DSH_CRED_TEST: stored\n");
        let stored = App::new();
        let (_f2, creds) = boot_layered(&stored, &path, layers(ladder)).await;
        assert_eq!(creds.resolve(&key()).await.unwrap(), file_source("stored"));
    });
}

#[cfg(unix)]
#[test]
fn refuses_a_document_other_os_users_can_read() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        write_with_mode(&path, "DSH_CRED_TEST: leaked\n", 0o644);
        // Before the contents are read at all: serving secrets out of a
        // world-readable file would make the provider's 0600 meaningless.
        let message = boot_failure(json!({"path": path.to_str().unwrap(), "watch": false})).await;
        assert!(
            message.contains("readable beyond its owner (mode 644)"),
            "unexpected error: {message}"
        );
    });
}

#[test]
fn propagates_a_permission_check_that_fails_for_a_reason_other_than_absence() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let not_a_directory = dir.path().join("occupied");
        std::fs::write(&not_a_directory, "a regular file\n").unwrap();
        // An absent document is an empty store, but a path that cannot be
        // reached at all is a misconfiguration: the parent is a file, so the
        // check fails with ENOTDIR rather than concluding "no credentials".
        let path = not_a_directory.join(".credentials.yaml");
        let message = boot_failure(json!({"path": path.to_str().unwrap(), "watch": false})).await;
        assert!(
            message.contains("ot a directory"),
            "unexpected error: {message}"
        );
    });
}

#[test]
fn propagates_a_permission_check_rejected_before_the_os_lookup() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials\0.yaml");
        let message = boot_failure(json!({"path": path.to_str().unwrap(), "watch": false})).await;
        assert!(
            message.contains("plugin startup failed"),
            "unexpected error: {message}"
        );
    });
}

#[cfg(unix)]
#[test]
fn propagates_a_read_that_fails_for_a_reason_other_than_absence() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        // Owner-only, so the permission check passes, and unreadable as a
        // file: present but unparsable must fail the launch rather than
        // silently serve nothing.
        let mut builder = std::fs::DirBuilder::new();
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(&path).unwrap();
        let message = boot_failure(json!({"path": path.to_str().unwrap(), "watch": false})).await;
        assert!(message.contains("directory"), "unexpected error: {message}");
    });
}

#[test]
fn lets_only_the_inherited_environment_shadow_the_store_read_only() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        write_credentials(&path, "DSH_CRED_TEST: stored\n");
        let app = App::new();
        let (_fiber, creds) = boot_layered(
            &app,
            &path,
            layers(&[
                (
                    LaunchEnvironmentSource::Process,
                    None,
                    &[("DSH_CRED_TEST", "from-shell")],
                ),
                (
                    LaunchEnvironmentSource::UserEnv,
                    Some("/home/.dsh/.env"),
                    &[("DSH_CRED_TEST", "from-user-env")],
                ),
            ]),
        )
        .await;
        assert_eq!(
            creds.resolve(&key()).await.unwrap(),
            Some(ResolvedCredential {
                value: "from-shell".into(),
                source: "env".into()
            })
        );
        assert_eq!(
            creds.describe(&key()).await.unwrap(),
            CredentialInfo {
                configured: true,
                source: Some("env".into()),
                writable: false
            }
        );
        let error = creds.set(&key(), "next").await.unwrap_err();
        assert!(
            error.to_string().contains("launching environment"),
            "{error}"
        );
    });
}

// --- document validation ---------------------------------------------------

#[test]
fn fails_boot_on_documents_the_seam_cannot_address() {
    // Every rejection is a boot failure rather than a skipped entry: this
    // document holds nothing but credentials, so an ignored key would read
    // as "the secret I stored has no effect".
    let cases: &[(&str, &str, &str)] = &[
        ("a non-mapping root", "just a string\n", "must be a mapping"),
        ("a sequence root", "- DSH_CRED_TEST\n", "must be a mapping"),
        (
            "a key that is not a POSIX identifier",
            "not-a-ref: value\n",
            "credential ref",
        ),
        (
            "a non-string value",
            "DSH_CRED_TEST: 123\n",
            "must be a string",
        ),
        ("an empty value", "DSH_CRED_TEST: \"\"\n", "is empty"),
        (
            "duplicate keys",
            "DSH_CRED_TEST: one\nDSH_CRED_TEST: two\n",
            "invalid document",
        ),
        (
            "malformed quoting",
            "DSH_CRED_TEST: \"unterminated\n",
            "invalid document",
        ),
    ];
    for (name, text, expected) in cases {
        dsh_cordis::run(async {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join(".credentials.yaml");
            write_credentials(&path, text);
            let message =
                boot_failure(json!({"path": path.to_str().unwrap(), "watch": false})).await;
            assert!(
                message.contains(expected),
                "case {name:?}: unexpected error: {message}"
            );
        });
    }
}

#[test]
fn never_puts_a_credential_value_in_a_diagnostic() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        let secret = "sk-live-DO-NOT-LOG-abcdef123456";
        // The malformed line holds the secret itself; boot stderr and the
        // watcher's logger both receive whatever this produces.
        write_credentials(&path, &format!("DSH_CRED_TEST: \"{secret}\n"));
        let message = boot_failure(json!({"path": path.to_str().unwrap(), "watch": false})).await;
        assert!(
            message.contains("invalid document"),
            "unexpected error: {message}"
        );
        // The position survives; the line's contents do not.
        assert!(message.contains("line 1"), "unexpected error: {message}");
        assert!(
            !message.contains(secret),
            "secret leaked into the diagnostic: {message}"
        );
    });
}

#[test]
fn reads_an_empty_document_as_an_empty_store() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        write_credentials(&path, "# nothing stored yet\n");
        let app = App::new();
        let (_fiber, creds) = boot(
            &app.root(),
            json!({"path": path.to_str().unwrap(), "watch": false}),
        )
        .await;
        assert_eq!(creds.resolve(&key()).await.unwrap(), None);
    });
}

// --- document writes -------------------------------------------------------

#[test]
fn adds_a_missing_key_to_a_fresh_0600_document_and_emits_the_commit() {
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
        let seen = updates(&ctx);
        creds.set(&key(), "sk-fresh").await.unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "DSH_CRED_TEST: sk-fresh\n"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            assert_eq!(std::fs::metadata(&path).unwrap().mode() & 0o777, 0o600);
        }
        assert_eq!(
            creds.resolve(&key()).await.unwrap(),
            file_source("sk-fresh")
        );
        assert_eq!(*seen.borrow(), vec![key()]);
    });
}

#[test]
fn patches_one_entry_preserving_comments_and_every_untouched_entry() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        write_credentials(
            &path,
            "# deployment notes\nDSH_CRED_OTHER: keep\n\n# the one under edit\nDSH_CRED_TEST: old\n",
        );
        let app = App::new();
        let (_fiber, creds) = boot(
            &app.root(),
            json!({"path": path.to_str().unwrap(), "watch": false}),
        )
        .await;
        creds.set(&key(), "new value!").await.unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "# deployment notes\nDSH_CRED_OTHER: keep\n\n# the one under edit\nDSH_CRED_TEST: new value!\n",
        );
    });
}

#[test]
fn round_trips_values_no_dotenv_line_could_represent() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        let multi_line = "line one\nline two";
        let mixed_quotes = "both ' and \"";
        {
            let app = App::new();
            let (_fiber, creds) = boot(
                &app.root(),
                json!({"path": path.to_str().unwrap(), "watch": false}),
            )
            .await;
            creds.set(&key(), multi_line).await.unwrap();
            creds.set(&other(), mixed_quotes).await.unwrap();
        }
        let app = App::new();
        let (_fiber, reread) = boot(
            &app.root(),
            json!({"path": path.to_str().unwrap(), "watch": false}),
        )
        .await;
        assert_eq!(
            reread.resolve(&key()).await.unwrap(),
            file_source(multi_line)
        );
        assert_eq!(
            reread.resolve(&other()).await.unwrap(),
            file_source(mixed_quotes)
        );
        assert_eq!(
            reread.describe(&key()).await.unwrap(),
            CredentialInfo {
                configured: true,
                source: Some("file".into()),
                writable: true
            }
        );
    });
}

#[test]
fn unsets_only_the_owning_entry_with_its_annotation_and_keeps_absent_unset_silent() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        // Comments above an entry are that entry's annotation and go with it
        // when it is removed — including anything above the document's first
        // entry. Every other entry keeps its own comments.
        write_credentials(
            &path,
            "# about the doomed one\nDSH_CRED_TEST: gone\n# about the survivor\nDSH_CRED_OTHER: stays\n",
        );
        let app = App::new();
        let ctx = app.root();
        let (_fiber, creds) = boot(
            &ctx,
            json!({"path": path.to_str().unwrap(), "watch": false}),
        )
        .await;
        let seen = updates(&ctx);
        creds.unset(&key()).await.unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "# about the survivor\nDSH_CRED_OTHER: stays\n",
        );
        creds.unset(&key()).await.unwrap();
        assert_eq!(*seen.borrow(), vec![key()]);
    });
}

#[test]
fn rejects_empty_values_and_writes_the_environment_would_shadow() {
    dsh_cordis::run(async {
        let r = credential_ref("DSH_CRED_LOCAL_SHADOWED").unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        write_credentials(&path, "DSH_CRED_LOCAL_SHADOWED: stored\n");
        let app = App::new();
        let (_fiber, creds) = boot(
            &app.root(),
            json!({"path": path.to_str().unwrap(), "watch": false}),
        )
        .await;

        let error = creds.set(&r, "").await.unwrap_err();
        assert!(error.to_string().contains("empty value"), "{error}");

        // SAFETY: the variable name is unique to this test.
        unsafe { std::env::set_var("DSH_CRED_LOCAL_SHADOWED", "shadowing") };
        let error = creds.set(&r, "next").await.unwrap_err();
        assert!(error.to_string().contains("shadowed"), "{error}");
        let error = creds.unset(&r).await.unwrap_err();
        assert!(error.to_string().contains("shadowed"), "{error}");
        unsafe { std::env::remove_var("DSH_CRED_LOCAL_SHADOWED") };
    });
}

#[test]
fn leaves_an_empty_document_after_unsetting_the_only_entry() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        write_credentials(&path, "DSH_CRED_TEST: only\n");
        {
            let app = App::new();
            let (_fiber, creds) = boot(
                &app.root(),
                json!({"path": path.to_str().unwrap(), "watch": false}),
            )
            .await;
            creds.unset(&key()).await.unwrap();
        }
        // Upstream leaves `{}\n` (YAML's empty flow map); the subset's empty
        // document is an empty file. Either way it reloads as an empty store.
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "");
        let app = App::new();
        let (_fiber, reread) = boot(
            &app.root(),
            json!({"path": path.to_str().unwrap(), "watch": false}),
        )
        .await;
        assert_eq!(reread.resolve(&key()).await.unwrap(), None);
    });
}

#[test]
fn fails_a_write_loud_when_the_on_disk_document_became_invalid() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        let app = App::new();
        let (_fiber, creds) = boot(
            &app.root(),
            json!({"path": path.to_str().unwrap(), "watch": false}),
        )
        .await;
        // An external editor left the document unparsable: the
        // read-modify-write must refuse rather than overwrite content it
        // cannot understand.
        write_credentials(&path, "DSH_CRED_TEST: \"unterminated\n");
        let error = creds.set(&other(), "lands").await.unwrap_err();
        assert!(error.to_string().contains("invalid document"), "{error}");
    });
}

#[test]
fn continues_past_a_rejected_write_so_one_bad_value_cannot_poison_later_ones() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        let app = App::new();
        let (_fiber, creds) = boot(
            &app.root(),
            json!({"path": path.to_str().unwrap(), "watch": false}),
        )
        .await;
        let bad = creds.set(&key(), "").await;
        assert!(bad.unwrap_err().to_string().contains("empty value"));
        creds.set(&other(), "lands").await.unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "DSH_CRED_OTHER: lands\n"
        );
    });
}

#[test]
fn serializes_concurrent_writes_so_both_land_in_the_one_document() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        let app = App::new();
        let (_fiber, creds) = boot(
            &app.root(),
            json!({"path": path.to_str().unwrap(), "watch": false}),
        )
        .await;
        let (key_ref, other_ref) = (key(), other());
        let (first, second) =
            futures::join!(creds.set(&key_ref, "one"), creds.set(&other_ref, "two"));
        first.unwrap();
        second.unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "DSH_CRED_TEST: one\nDSH_CRED_OTHER: two\n",
        );
    });
}

#[test]
fn refuses_writes_after_disposal() {
    dsh_cordis::run(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".credentials.yaml");
        let app = App::new();
        let ctx = app.root();
        let (fiber, creds) = boot(
            &ctx,
            json!({"path": path.to_str().unwrap(), "watch": false}),
        )
        .await;
        // The handle is captured first: disposal also removes the service.
        fiber.dispose().await;
        assert!(ctx.try_service::<Credentials>().is_none());
        let error = creds.set(&key(), "late").await.unwrap_err();
        assert!(error.to_string().contains("disposed"), "{error}");
    });
}
