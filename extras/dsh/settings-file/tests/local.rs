//! Provider behavior ported from upstream `tests/local.spec.ts`,
//! `tests/watcher.spec.ts` (the mock-free subset), and
//! `tests/loader-composition.spec.ts` (re-expressed over plugin composition —
//! the TS Loader/Include stack has no Rust counterpart). Not ported:
//! comment-preservation cases (serde_yaml keeps no comments; see the crate
//! module doc) and the chokidar-mock / fs-fault-injection cases.

use dsh_cordis::{App, Context, Fiber, Inject, plugin_fn};
use dsh_settings::{
    Schema, SettingsDescribeOptions, SettingsNamespace, SettingsRegisterOptions,
    SettingsSectionHooks, SettingsService, SettingsUpdateSource, SettingsUpdated,
    install_settings_section, settings_namespace,
};
use dsh_settings_file::{Config, FileSettingsProvider, resolve_spec};
use serde_json::{Value, json};
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Duration;
use tempfile::TempDir;

fn ns(value: &str) -> SettingsNamespace {
    settings_namespace(value).unwrap()
}

fn theme_schema() -> Schema {
    Schema::object([
        (
            "theme",
            Schema::union([json!("dark"), json!("light")]).default(json!("dark")),
        ),
        ("fontSize", Schema::number().default(json!(14))),
    ])
}

struct Booted {
    #[allow(dead_code)]
    app: App,
    ctx: Context,
    service: Rc<SettingsService>,
    fiber: Fiber,
}

async fn boot(config: Value) -> anyhow::Result<Booted> {
    let app = App::new();
    let ctx = app.root();
    let fiber = ctx
        .plugin(Rc::new(FileSettingsProvider), config)
        .map_err(anyhow::Error::from)?;
    fiber.await_ready().await.map_err(anyhow::Error::from)?;
    let service = ctx
        .try_service::<SettingsService>()
        .expect("settings service provided");
    Ok(Booted {
        app,
        ctx,
        service,
        fiber,
    })
}

async fn wait_until(mut cond: impl FnMut() -> bool) {
    for _ in 0..2000 {
        if cond() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("wait_until timed out");
}

fn write(path: &Path, contents: &str) {
    std::fs::write(path, contents).unwrap();
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap()
}

fn entries(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

#[cfg(unix)]
fn mode_of(path: &Path) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path).unwrap().permissions().mode() & 0o777
}

fn temp() -> (TempDir, PathBuf) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("settings.yaml");
    (dir, path)
}

fn err_text(error: anyhow::Error) -> String {
    format!("{error:#}")
}

// ------------------------------------------------------------- resolve_spec

#[test]
fn defaults_watch_and_debounce_when_construction_bypasses_schema_normalization() {
    let spec = resolve_spec(&Config {
        path: Some("/tmp/anywhere/settings.yaml".into()),
        ..Default::default()
    })
    .unwrap();
    assert!(spec.watch);
    assert_eq!(spec.debounce_ms, 100);
}

// ----------------------------------------------------------- boot and reads

#[test]
fn resolves_defaults_over_an_absent_file_and_reports_writable() {
    dsh_cordis::run(async {
        let (_dir, path) = temp();
        let booted = boot(json!({"path": path, "watch": false})).await.unwrap();
        let scope = booted
            .service
            .register(
                &booted.ctx,
                ns("ui-theme"),
                theme_schema(),
                SettingsRegisterOptions {
                    base: Some(json!({"fontSize": 16})),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(scope.get(), json!({"theme": "dark", "fontSize": 16}));
        assert!(booted.service.writable());
        assert_eq!(booted.service.document_path(), Some(path));
    });
}

#[test]
fn prepares_an_absent_owner_only_document_without_changing_resolved_settings() {
    dsh_cordis::run(async {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("nested").join("settings.yaml");
        let booted = boot(json!({"path": path, "watch": false})).await.unwrap();
        let scope = booted
            .service
            .register(
                &booted.ctx,
                ns("ui-theme"),
                theme_schema(),
                Default::default(),
            )
            .unwrap();

        assert_eq!(
            booted.service.prepare_document().await.unwrap(),
            Some(path.clone())
        );
        assert_eq!(read(&path), "");
        #[cfg(unix)]
        assert_eq!(mode_of(&path), 0o600);
        assert_eq!(scope.get(), json!({"theme": "dark", "fontSize": 14}));
    });
}

#[test]
fn preparing_an_existing_document_preserves_its_contents() {
    dsh_cordis::run(async {
        let (_dir, path) = temp();
        let contents = "ui-theme:\n  theme: light\n";
        write(&path, contents);
        let booted = boot(json!({"path": path, "watch": false})).await.unwrap();

        assert_eq!(
            booted.service.prepare_document().await.unwrap(),
            Some(path.clone())
        );
        assert_eq!(read(&path), contents);
    });
}

#[test]
fn reads_sections_from_an_existing_yaml_document() {
    dsh_cordis::run(async {
        let (_dir, path) = temp();
        write(&path, "ui-theme:\n  theme: light\n");
        let booted = boot(json!({"path": path, "watch": false})).await.unwrap();
        let scope = booted
            .service
            .register(
                &booted.ctx,
                ns("ui-theme"),
                theme_schema(),
                Default::default(),
            )
            .unwrap();
        assert_eq!(scope.get(), json!({"theme": "light", "fontSize": 14}));
    });
}

#[test]
fn reads_sections_from_a_json_document() {
    dsh_cordis::run(async {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("settings.json");
        write(&path, &json!({"ui-theme": {"fontSize": 18}}).to_string());
        let booted = boot(json!({"path": path, "watch": false})).await.unwrap();
        let scope = booted
            .service
            .register(
                &booted.ctx,
                ns("ui-theme"),
                theme_schema(),
                Default::default(),
            )
            .unwrap();
        assert_eq!(scope.get(), json!({"theme": "dark", "fontSize": 18}));
    });
}

#[test]
fn defaults_the_file_location_under_the_configured_harness_home() {
    dsh_cordis::run(async {
        let dir = TempDir::new().unwrap();
        let booted = boot(json!({"dshHome": dir.path(), "watch": false}))
            .await
            .unwrap();
        let expected = dir.path().join("settings.yaml");
        assert_eq!(booted.service.document_path(), Some(expected.clone()));
        let scope = booted
            .service
            .register(
                &booted.ctx,
                ns("ui-theme"),
                theme_schema(),
                Default::default(),
            )
            .unwrap();
        scope.update(json!({"theme": "light"})).await.unwrap();
        assert!(read(&expected).contains("theme: light"));
    });
}

#[test]
fn reads_an_empty_yaml_document_as_no_sections() {
    dsh_cordis::run(async {
        let (_dir, path) = temp();
        write(&path, "");
        let booted = boot(json!({"path": path, "watch": false})).await.unwrap();
        let scope = booted
            .service
            .register(
                &booted.ctx,
                ns("ui-theme"),
                theme_schema(),
                Default::default(),
            )
            .unwrap();
        assert_eq!(scope.get(), json!({"theme": "dark", "fontSize": 14}));
    });
}

#[test]
fn reads_an_empty_json_document_as_no_sections() {
    dsh_cordis::run(async {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("settings.json");
        write(&path, "");
        let booted = boot(json!({"path": path, "watch": false})).await.unwrap();
        let scope = booted
            .service
            .register(
                &booted.ctx,
                ns("ui-theme"),
                theme_schema(),
                Default::default(),
            )
            .unwrap();
        assert_eq!(scope.get(), json!({"theme": "dark", "fontSize": 14}));
    });
}

#[cfg(unix)]
#[test]
fn fails_loud_at_boot_when_the_document_exists_but_is_unreadable() {
    use std::os::unix::fs::PermissionsExt;
    dsh_cordis::run(async {
        let (dir, path) = temp();
        write(&path, "ui-theme:\n  theme: light\n");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).unwrap();
        let error = boot(json!({"path": path, "watch": false}))
            .await
            .err()
            .unwrap();
        let text = err_text(error).to_lowercase();
        assert!(text.contains("permission"), "{text}");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        drop(dir);
    });
}

#[test]
fn fails_loud_when_the_document_path_names_a_directory() {
    dsh_cordis::run(async {
        let (_dir, path) = temp();
        std::fs::create_dir(&path).unwrap();
        assert!(boot(json!({"path": path, "watch": false})).await.is_err());
    });
}

#[test]
fn fails_loud_on_an_unsupported_extension() {
    dsh_cordis::run(async {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("settings.toml");
        let error = boot(json!({"path": path, "watch": false}))
            .await
            .err()
            .unwrap();
        assert!(err_text(error).contains("not supported"));
    });
}

#[test]
fn fails_loud_at_boot_on_unparsable_yaml() {
    dsh_cordis::run(async {
        let (_dir, path) = temp();
        write(&path, "ui-theme: [unclosed\n");
        assert!(boot(json!({"path": path, "watch": false})).await.is_err());
    });
}

#[test]
fn fails_loud_at_boot_when_the_root_is_not_a_map_of_sections() {
    dsh_cordis::run(async {
        let (_dir, path) = temp();
        write(&path, "- just\n- a list\n");
        let error = boot(json!({"path": path, "watch": false}))
            .await
            .err()
            .unwrap();
        assert!(err_text(error).contains("map of namespace sections"));
    });
}

// ----------------------------------------------------------------- persist

#[test]
fn writes_the_merged_section_creating_the_file_with_owner_only_permissions() {
    dsh_cordis::run(async {
        let (dir, path) = temp();
        let booted = boot(json!({"path": path, "watch": false})).await.unwrap();
        let scope = booted
            .service
            .register(
                &booted.ctx,
                ns("ui-theme"),
                theme_schema(),
                Default::default(),
            )
            .unwrap();
        scope.update(json!({"theme": "light"})).await.unwrap();

        assert!(read(&path).contains("theme: light"));
        #[cfg(unix)]
        assert_eq!(mode_of(&path), 0o600);
        // Atomic replace leaves no temp artifact (and no lock) behind.
        assert_eq!(entries(dir.path()), vec!["settings.yaml".to_string()]);
    });
}

#[test]
fn serializes_cross_namespace_writes_into_one_on_disk_document() {
    dsh_cordis::run(async {
        let (_dir, path) = temp();
        let booted = boot(json!({"path": path, "watch": false})).await.unwrap();
        let alpha = booted
            .service
            .register(&booted.ctx, ns("alpha"), theme_schema(), Default::default())
            .unwrap();
        let beta = booted
            .service
            .register(&booted.ctx, ns("beta"), theme_schema(), Default::default())
            .unwrap();
        let (first, second) = futures::join!(
            alpha.update(json!({"theme": "light"})),
            beta.update(json!({"fontSize": 20})),
        );
        first.unwrap();
        second.unwrap();
        let text = read(&path);
        assert!(text.contains("alpha:"));
        assert!(text.contains("beta:"));
        assert_eq!(alpha.get()["theme"], json!("light"));
        assert_eq!(beta.get()["fontSize"], json!(20));
    });
}

#[cfg(unix)]
#[test]
fn never_follows_a_planted_symlink_and_never_leaves_the_document_a_symlink() {
    dsh_cordis::run(async {
        let (dir, path) = temp();
        let victim = dir.path().join("victim.txt");
        write(&victim, "precious");
        // A hostile sibling plants the historic fixed temp name as a symlink.
        std::os::unix::fs::symlink(&victim, dir.path().join("settings.yaml.tmp")).unwrap();
        let booted = boot(json!({"path": path, "watch": false})).await.unwrap();
        let scope = booted
            .service
            .register(
                &booted.ctx,
                ns("ui-theme"),
                theme_schema(),
                Default::default(),
            )
            .unwrap();
        scope.update(json!({"theme": "light"})).await.unwrap();

        assert_eq!(read(&victim), "precious");
        assert!(
            !std::fs::symlink_metadata(&path)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(mode_of(&path), 0o600);
        assert!(read(&path).contains("theme: light"));
    });
}

#[test]
fn preserves_unregistered_sections_across_updates() {
    // Divergence: upstream also proves comments survive; serde_yaml keeps
    // none, so only the unregistered-section half of the contract holds.
    dsh_cordis::run(async {
        let (_dir, path) = temp();
        write(
            &path,
            "ui-theme:\n  theme: light\nfuture-plugin:\n  keep: me\n",
        );
        let booted = boot(json!({"path": path, "watch": false})).await.unwrap();
        let scope = booted
            .service
            .register(
                &booted.ctx,
                ns("ui-theme"),
                theme_schema(),
                Default::default(),
            )
            .unwrap();
        scope.update(json!({"fontSize": 18})).await.unwrap();

        let written = read(&path);
        assert!(written.contains("keep: me"));
        assert!(written.contains("fontSize: 18"));
        assert!(written.contains("theme: light"));
    });
}

#[test]
fn creates_a_json_document_from_scratch() {
    dsh_cordis::run(async {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("settings.json");
        let booted = boot(json!({"path": path, "watch": false})).await.unwrap();
        let scope = booted
            .service
            .register(
                &booted.ctx,
                ns("ui-theme"),
                theme_schema(),
                Default::default(),
            )
            .unwrap();
        scope.update(json!({"theme": "light"})).await.unwrap();
        let written: Value = serde_json::from_str(&read(&path)).unwrap();
        assert_eq!(written, json!({"ui-theme": {"theme": "light"}}));
    });
}

#[test]
fn rejects_and_recovers_when_the_document_path_becomes_a_directory() {
    dsh_cordis::run(async {
        let (dir, path) = temp();
        let backup = dir.path().join("settings.committed.yaml");
        write(&path, "ui-theme:\n  theme: light\n");
        let booted = boot(json!({"path": path, "watch": false})).await.unwrap();
        let scope = booted
            .service
            .register(
                &booted.ctx,
                ns("ui-theme"),
                theme_schema(),
                Default::default(),
            )
            .unwrap();
        std::fs::rename(&path, &backup).unwrap();
        std::fs::create_dir(&path).unwrap();
        assert!(scope.update(json!({"theme": "dark"})).await.is_err());
        std::fs::remove_dir_all(&path).unwrap();
        std::fs::rename(&backup, &path).unwrap();
        assert_eq!(entries(dir.path()), vec!["settings.yaml".to_string()]);
        assert_eq!(scope.get()["theme"], json!("light"));
        // The failed persist must not poison the document write chain.
        scope.update(json!({"theme": "dark"})).await.unwrap();
        assert_eq!(scope.get()["theme"], json!("dark"));
    });
}

#[test]
fn round_trips_a_json_document() {
    dsh_cordis::run(async {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("settings.json");
        write(
            &path,
            &serde_json::to_string_pretty(&json!({"other": {"keep": true}})).unwrap(),
        );
        let booted = boot(json!({"path": path, "watch": false})).await.unwrap();
        let scope = booted
            .service
            .register(
                &booted.ctx,
                ns("ui-theme"),
                theme_schema(),
                Default::default(),
            )
            .unwrap();
        scope.update(json!({"theme": "light"})).await.unwrap();
        let written: Value = serde_json::from_str(&read(&path)).unwrap();
        assert_eq!(
            written,
            json!({"other": {"keep": true}, "ui-theme": {"theme": "light"}})
        );
    });
}

// ------------------------------------------------------------------- watch

#[test]
fn publishes_an_external_edit_to_registered_scopes() {
    dsh_cordis::run(async {
        let (_dir, path) = temp();
        write(&path, "ui-theme:\n  theme: light\n");
        let booted = boot(json!({"path": path, "debounceMs": 10})).await.unwrap();
        let scope = booted
            .service
            .register(
                &booted.ctx,
                ns("ui-theme"),
                theme_schema(),
                Default::default(),
            )
            .unwrap();
        assert_eq!(scope.get()["theme"], json!("light"));

        write(&path, "ui-theme:\n  theme: dark\n  fontSize: 20\n");
        wait_until(|| scope.get() == json!({"theme": "dark", "fontSize": 20})).await;
        booted.fiber.dispose().await;
    });
}

#[test]
fn keeps_the_last_good_document_over_an_invalid_edit_then_recovers() {
    dsh_cordis::run(async {
        let (_dir, path) = temp();
        write(&path, "ui-theme:\n  theme: light\n");
        let booted = boot(json!({"path": path, "debounceMs": 10})).await.unwrap();
        let scope = booted
            .service
            .register(
                &booted.ctx,
                ns("ui-theme"),
                theme_schema(),
                Default::default(),
            )
            .unwrap();

        // Replace atomically so the watcher observes one complete invalid
        // document instead of a transient empty file during truncation.
        dsh_atomic_write::write_file_atomic(
            &path,
            "ui-theme: [unclosed\n",
            dsh_atomic_write::WriteFileAtomicOptions {
                mode: 0o600,
                dir_mode: None,
            },
        )
        .unwrap();
        // The bad edit must never take the live tree down or reset the value.
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(scope.get(), json!({"theme": "light", "fontSize": 14}));

        dsh_atomic_write::write_file_atomic(
            &path,
            "ui-theme:\n  theme: dark\n",
            dsh_atomic_write::WriteFileAtomicOptions {
                mode: 0o600,
                dir_mode: None,
            },
        )
        .unwrap();
        wait_until(|| scope.get()["theme"] == json!("dark")).await;
        booted.fiber.dispose().await;
    });
}

#[test]
fn treats_file_removal_as_an_empty_document() {
    dsh_cordis::run(async {
        let (_dir, path) = temp();
        write(&path, "ui-theme:\n  theme: light\n");
        let booted = boot(json!({"path": path, "debounceMs": 10})).await.unwrap();
        let scope = booted
            .service
            .register(
                &booted.ctx,
                ns("ui-theme"),
                theme_schema(),
                Default::default(),
            )
            .unwrap();

        std::fs::remove_file(&path).unwrap();
        wait_until(|| scope.get() == json!({"theme": "dark", "fontSize": 14})).await;
        booted.fiber.dispose().await;
    });
}

#[test]
fn does_not_republish_its_own_persisted_write() {
    dsh_cordis::run(async {
        let (_dir, path) = temp();
        let booted = boot(json!({"path": path, "debounceMs": 10})).await.unwrap();
        let events: Rc<RefCell<Vec<(String, SettingsUpdateSource)>>> = Rc::default();
        let sink = events.clone();
        booted
            .ctx
            .on::<SettingsUpdated, _, _>(Default::default(), move |_ctx, args| {
                sink.borrow_mut().push((args.0.to_string(), args.3));
                std::future::ready(None::<()>)
            })
            .unwrap();
        let scope = booted
            .service
            .register(
                &booted.ctx,
                ns("ui-theme"),
                theme_schema(),
                Default::default(),
            )
            .unwrap();
        scope.update(json!({"theme": "light"})).await.unwrap();
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(
            *events.borrow(),
            vec![("ui-theme".to_string(), SettingsUpdateSource::Update)]
        );
        booted.fiber.dispose().await;
    });
}

#[test]
fn quiesces_the_refresh_pipeline_before_dispose_completes() {
    dsh_cordis::run(async {
        let (_dir, path) = temp();
        write(&path, "ui-theme:\n  theme: light\n");
        let booted = boot(json!({"path": path, "debounceMs": 10})).await.unwrap();
        let scope = booted
            .service
            .register(
                &booted.ctx,
                ns("ui-theme"),
                theme_schema(),
                Default::default(),
            )
            .unwrap();
        let disposed = Rc::new(std::cell::Cell::new(false));
        let post_dispose_commits = Rc::new(std::cell::Cell::new(0u32));
        let disposed_probe = disposed.clone();
        let commits = post_dispose_commits.clone();
        booted
            .ctx
            .on::<SettingsUpdated, _, _>(Default::default(), move |_ctx, _args| {
                if disposed_probe.get() {
                    commits.set(commits.get() + 1);
                }
                std::future::ready(None::<()>)
            })
            .unwrap();

        write(&path, "ui-theme:\n  theme: dark\n");
        wait_until(|| scope.get()["theme"] == json!("dark")).await;
        booted.fiber.dispose().await;
        disposed.set(true);
        write(&path, "ui-theme:\n  theme: light\n");
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(post_dispose_commits.get(), 0);
    });
}

#[test]
fn folds_an_unobserved_external_edit_into_a_write_instead_of_overwriting_it() {
    dsh_cordis::run(async {
        // The external edit has landed on disk but no watcher event has fired
        // for it (a debounce window, or a missed event): the write must fold
        // it in, not resurrect the stale document. Watch is off so the test
        // exercises exactly the read-modify-write fold.
        let (_dir, path) = temp();
        write(&path, "ui-theme:\n  theme: light\n");
        let booted = boot(json!({"path": path, "watch": false})).await.unwrap();
        let theme = booted
            .service
            .register(
                &booted.ctx,
                ns("ui-theme"),
                theme_schema(),
                Default::default(),
            )
            .unwrap();
        let editor = booted
            .service
            .register(
                &booted.ctx,
                ns("editor"),
                Schema::object([("tabWidth", Schema::number().default(json!(2)))]),
                Default::default(),
            )
            .unwrap();
        write(&path, "ui-theme:\n  theme: light\neditor:\n  tabWidth: 8\n");
        theme.update(json!({"theme": "dark"})).await.unwrap();
        let text = read(&path);
        assert!(text.contains("tabWidth: 8"));
        assert!(text.contains("theme: dark"));
        // The fold published the unobserved section before the write
        // committed.
        assert_eq!(editor.get(), json!({"tabWidth": 8}));
    });
}

#[test]
fn fails_a_write_loud_when_the_on_disk_document_turned_invalid_unobserved() {
    dsh_cordis::run(async {
        let (_dir, path) = temp();
        write(&path, "ui-theme:\n  theme: light\n");
        let booted = boot(json!({"path": path, "watch": false})).await.unwrap();
        let scope = booted
            .service
            .register(
                &booted.ctx,
                ns("ui-theme"),
                theme_schema(),
                Default::default(),
            )
            .unwrap();
        let broken = "ui-theme: [unclosed\n  flow: {\n";
        write(&path, broken);
        let error = scope.update(json!({"theme": "dark"})).await.unwrap_err();
        assert!(err_text(error).contains("invalid document"));
        // The user's manual edit stays on disk untouched and the cache keeps
        // the last good value.
        assert_eq!(read(&path), broken);
        assert_eq!(scope.get(), json!({"theme": "light", "fontSize": 14}));
    });
}

// ------------------------------------------------------- real composition

struct ConsumerState {
    /// What the consumer is actually running with, settings or not.
    applied: Rc<RefCell<Value>>,
    changes: Rc<std::cell::Cell<u32>>,
}

/// The documented consumer shape: no hard dependency — entry config alone is
/// the running state, and the scoped inject overlays the user layer only
/// while a settings service exists.
fn spawn_consumer(ctx: &Context, entry: Value) -> ConsumerState {
    let schema = theme_schema();
    let applied = Rc::new(RefCell::new(schema.resolve(Some(&entry)).unwrap().unwrap()));
    let changes = Rc::new(std::cell::Cell::new(0u32));
    let applied_plugin = applied.clone();
    let changes_plugin = changes.clone();
    let consumer = plugin_fn("settings-consumer", Inject::default(), move |ctx, _| {
        let applied = applied_plugin.clone();
        let changes = changes_plugin.clone();
        let entry = entry.clone();
        let schema = schema.clone();
        async move {
            let current: Rc<RefCell<Rc<dyn Fn() -> Value>>> = {
                let entry = entry.clone();
                Rc::new(RefCell::new(
                    Rc::new(move || entry.clone()) as Rc<dyn Fn() -> Value>
                ))
            };
            let source_cell = current.clone();
            install_settings_section(
                &ctx,
                ns("ui-theme"),
                schema,
                entry,
                SettingsSectionHooks {
                    set_source: Rc::new(move |source| {
                        *source_cell.borrow_mut() = source;
                    }),
                    on_change: Rc::new(move || {
                        *applied.borrow_mut() = (current.borrow().clone())();
                        changes.set(changes.get() + 1);
                    }),
                    validate: None,
                },
            )
            .map_err(anyhow::Error::from)?;
            Ok(())
        }
    });
    let fiber = ctx.plugin(Rc::new(consumer), Value::Null).unwrap();
    tokio::task::spawn_local(async move {
        let _ = fiber.await_ready().await;
    });
    ConsumerState { applied, changes }
}

#[test]
fn boots_a_composition_and_hot_publishes_an_external_settings_edit() {
    dsh_cordis::run(async {
        let (_dir, path) = temp();
        write(&path, "ui-theme:\n  theme: light\n");
        let app = App::new();
        let ctx = app.root();
        let state = spawn_consumer(&ctx, json!({"fontSize": 16}));
        let provider = ctx
            .plugin(
                Rc::new(FileSettingsProvider),
                json!({"path": path, "debounceMs": 10}),
            )
            .unwrap();
        provider.await_ready().await.unwrap();

        // Composition resolution: user layer over the consumer's base.
        wait_until(|| *state.applied.borrow() == json!({"theme": "light", "fontSize": 16})).await;
        let service = ctx.try_service::<SettingsService>().unwrap();
        let namespaces: Vec<String> = service
            .describe(SettingsDescribeOptions::default())
            .iter()
            .map(|d| d.ns.to_string())
            .collect();
        assert_eq!(namespaces, vec!["ui-theme".to_string()]);

        write(&path, "ui-theme:\n  theme: dark\n  fontSize: 20\n");
        wait_until(|| *state.applied.borrow() == json!({"theme": "dark", "fontSize": 20})).await;
        provider.dispose().await;
    });
}

#[test]
fn boots_the_same_consumer_without_a_settings_entry_and_keeps_entry_config_resolution() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let state = spawn_consumer(&ctx, json!({"fontSize": 16}));
        for _ in 0..25 {
            tokio::task::yield_now().await;
        }
        // No settings service anywhere in the composition, so the consumer
        // runs on schema defaults plus its composition base and never
        // receives a scope.
        assert!(ctx.try_service::<SettingsService>().is_none());
        assert_eq!(
            *state.applied.borrow(),
            json!({"theme": "dark", "fontSize": 16})
        );
        assert_eq!(state.changes.get(), 0);
    });
}
