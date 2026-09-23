//! Port of `packages/util/home-paths/tests/home-paths.spec.ts`.

use std::path::{Path, PathBuf};

use dsh_home_paths::{
    DEFAULT_DSH_HOME_DISPLAY, DSH_HOME_DIR_NAME, DSH_HOME_ENV, canonicalize_watch_path,
    default_dsh_home, dsh_home_display, dsh_home_path, expand_home_path, resolve_dsh_home_from,
};

fn home() -> PathBuf {
    #[allow(deprecated)]
    std::env::home_dir().unwrap()
}

#[test]
fn owns_the_shared_default_dsh_home_directory_name() {
    assert_eq!(DSH_HOME_DIR_NAME, ".dsh");
    assert_eq!(DEFAULT_DSH_HOME_DISPLAY, "~/.dsh");
    assert_eq!(DSH_HOME_ENV, "DSH_HOME");
    assert_eq!(default_dsh_home(), home().join(".dsh"));
}

#[test]
fn expands_tilde_paths_without_changing_non_tilde_paths() {
    assert_eq!(expand_home_path("~"), home());
    assert_eq!(expand_home_path("~/.dsh"), home().join(".dsh"));
    assert_eq!(expand_home_path("~\\.dsh"), home().join(".dsh"));
    assert_eq!(expand_home_path("/tmp/.dsh"), PathBuf::from("/tmp/.dsh"));
    assert_eq!(
        expand_home_path("~other/.dsh"),
        PathBuf::from("~other/.dsh")
    );
}

#[test]
fn resolves_explicit_path_before_dsh_home_and_the_default() {
    let env_home = home().join("env-dsh");
    assert_eq!(
        resolve_dsh_home_from(Some("/tmp/explicit-dsh"), Some("~/env-dsh")),
        std::path::absolute("/tmp/explicit-dsh").unwrap()
    );
    assert_eq!(resolve_dsh_home_from(None, Some("~/env-dsh")), env_home);
    assert_eq!(resolve_dsh_home_from(None, None), default_dsh_home());
}

#[test]
fn treats_an_empty_or_whitespace_only_dsh_home_as_unset() {
    assert_eq!(resolve_dsh_home_from(None, Some("")), default_dsh_home());
    assert_eq!(resolve_dsh_home_from(None, Some("   ")), default_dsh_home());
}

#[test]
fn joins_child_segments_onto_the_resolved_dsh_home() {
    // The only test that touches the process environment; no other test in
    // this binary reads DSH_HOME through the process-env entry point.
    unsafe { std::env::set_var(DSH_HOME_ENV, "~/env-dsh") };
    assert_eq!(dsh_home_path(&[]), home().join("env-dsh"));
    assert_eq!(
        dsh_home_path(&["storages", "cache"]),
        home().join("env-dsh").join("storages").join("cache")
    );
    unsafe { std::env::remove_var(DSH_HOME_ENV) };
}

#[test]
fn labels_a_resolved_home_by_whether_it_is_the_default_root() {
    assert_eq!(dsh_home_display(&default_dsh_home()), "~/.dsh");
    assert_eq!(dsh_home_display(Path::new("/some/other/root")), "$DSH_HOME");
}

#[test]
fn canonicalizes_a_watcher_ancestor_while_preserving_a_missing_suffix() {
    let root = tempfile::Builder::new()
        .prefix("dsh-watch-path-")
        .tempdir()
        .unwrap();
    let target = root.path().join("target");
    let alias = root.path().join("alias");
    std::fs::create_dir(&target).unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(&target, &alias).unwrap();
    #[cfg(windows)]
    std::os::windows::fs::symlink_dir(&target, &alias).unwrap();

    let canonical = canonicalize_watch_path(&alias.join("later").join("config.yml")).unwrap();
    assert_eq!(
        canonical,
        std::fs::canonicalize(&target)
            .unwrap()
            .join("later")
            .join("config.yml")
    );

    let file = root.path().join("file");
    std::fs::write(&file, "not a directory").unwrap();
    let error = canonicalize_watch_path(&file.join("child")).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::NotADirectory);
}
