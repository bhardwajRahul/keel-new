//! Cross-instance and writer-lock behavior, ported from upstream
//! `tests/concurrency.spec.ts`. Two providers on one document are the
//! in-process equivalent of two dsh processes sharing a harness home —
//! neither knows the other's cache, so only the read-modify-write cycle
//! under the `<file>.lock` sibling keeps both namespaces alive on disk.
//! The fs-fault-injection cases of `tests/lock-race.spec.ts` are not ported
//! (no counterpart to mocking `node:fs/promises`).

use dsh_cordis::{App, Context, Fiber};
use dsh_settings::{Schema, SettingsNamespace, SettingsService, settings_namespace};
use dsh_settings_file::FileSettingsProvider;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use tempfile::TempDir;

fn ns(value: &str) -> SettingsNamespace {
    settings_namespace(value).unwrap()
}

fn value_schema() -> Schema {
    Schema::object([("value", Schema::number().default(json!(0)))])
}

struct Booted {
    #[allow(dead_code)]
    app: App,
    ctx: Context,
    service: Rc<SettingsService>,
    #[allow(dead_code)]
    fiber: Fiber,
}

async fn boot(path: &Path) -> Booted {
    let app = App::new();
    let ctx = app.root();
    let fiber = ctx
        .plugin(
            Rc::new(FileSettingsProvider),
            json!({"path": path, "watch": false}),
        )
        .unwrap();
    fiber.await_ready().await.unwrap();
    let service = ctx.try_service::<SettingsService>().unwrap();
    Booted {
        app,
        ctx,
        service,
        fiber,
    }
}

fn temp() -> (TempDir, PathBuf) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("settings.yaml");
    (dir, path)
}

#[test]
fn keeps_both_namespaces_when_two_providers_write_the_same_document_concurrently() {
    dsh_cordis::run(async {
        let (_dir, path) = temp();
        let first = boot(&path).await;
        let second = boot(&path).await;
        let alpha = first
            .service
            .register(&first.ctx, ns("alpha"), value_schema(), Default::default())
            .unwrap();
        let beta = second
            .service
            .register(&second.ctx, ns("beta"), value_schema(), Default::default())
            .unwrap();
        let alpha_rounds = async {
            for value in 1..=5 {
                alpha.update(json!({"value": value})).await.unwrap();
            }
        };
        let beta_rounds = async {
            for value in 1..=5 {
                beta.update(json!({"value": value})).await.unwrap();
            }
        };
        futures::join!(alpha_rounds, beta_rounds);
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("alpha:"));
        assert!(text.contains("beta:"));
        // A third instance resolves both final values from the shared
        // document.
        let third = boot(&path).await;
        let alpha3 = third
            .service
            .register(&third.ctx, ns("alpha"), value_schema(), Default::default())
            .unwrap();
        let beta3 = third
            .service
            .register(&third.ctx, ns("beta"), value_schema(), Default::default())
            .unwrap();
        assert_eq!(alpha3.get(), json!({"value": 5}));
        assert_eq!(beta3.get(), json!({"value": 5}));
    });
}

#[test]
fn waits_for_a_busy_writer_lock_instead_of_failing() {
    dsh_cordis::run(async {
        let (_dir, path) = temp();
        let booted = boot(&path).await;
        let scope = booted
            .service
            .register(&booted.ctx, ns("alpha"), value_schema(), Default::default())
            .unwrap();
        let lock_path = PathBuf::from(format!("{}.lock", path.display()));
        std::fs::write(&lock_path, "holder\n").unwrap();
        // The lock backoff blocks this (single) runtime thread, so the
        // release has to come from a real OS thread.
        let release_target = lock_path.clone();
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(120));
            let _ = std::fs::remove_file(&release_target);
        });
        scope.update(json!({"value": 7})).await.unwrap();
        releaser.join().unwrap();
        assert!(std::fs::read_to_string(&path).unwrap().contains("value: 7"));
    });
}

#[test]
fn does_not_steal_an_old_writer_lock() {
    dsh_cordis::run(async {
        let (_dir, path) = temp();
        std::fs::write(&path, "alpha:\n  value: 4\n").unwrap();
        let booted = boot(&path).await;
        let scope = booted
            .service
            .register(&booted.ctx, ns("alpha"), value_schema(), Default::default())
            .unwrap();
        let lock_path = PathBuf::from(format!("{}.lock", path.display()));
        std::fs::write(&lock_path, "slow-holder\n").unwrap();

        let error = scope.update(json!({"value": 9})).await.unwrap_err();
        assert!(format!("{error:#}").contains("timed out waiting for the writer lock"));
        assert!(std::fs::read_to_string(&path).unwrap().contains("value: 4"));
        assert_eq!(
            std::fs::read_to_string(&lock_path).unwrap(),
            "slow-holder\n"
        );
    });
}

#[cfg(unix)]
#[test]
fn surfaces_a_non_contention_lock_failure_as_the_write_error() {
    use std::os::unix::fs::PermissionsExt;
    dsh_cordis::run(async {
        let (dir, path) = temp();
        let booted = boot(&path).await;
        let scope = booted
            .service
            .register(&booted.ctx, ns("alpha"), value_schema(), Default::default())
            .unwrap();
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o500)).unwrap();
        let error = scope.update(json!({"value": 1})).await.unwrap_err();
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let text = format!("{error:#}").to_lowercase();
        assert!(text.contains("permission"), "{text}");
    });
}
