//! Port of `packages/util/atomic-write/tests/atomic-write.spec.ts`.

use std::fs;
use std::path::Path;

use dsh_atomic_write::{WriteFileAtomicOptions, with_file_lock, write_file_atomic};

fn scratch() -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix("dsh-atomic-write-")
        .tempdir()
        .unwrap()
}

fn opts(mode: u32) -> WriteFileAtomicOptions {
    WriteFileAtomicOptions {
        mode,
        dir_mode: None,
    }
}

#[cfg(unix)]
fn mode_bits(path: &Path) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    fs::metadata(path).unwrap().permissions().mode() & 0o777
}

#[test]
fn creates_the_file_and_its_parents_with_exactly_the_stated_mode() {
    let dir = scratch();
    let target = dir.path().join("nested").join("deep").join("doc.yaml");
    write_file_atomic(&target, "a: 1\n", opts(0o600)).unwrap();
    assert_eq!(fs::read_to_string(&target).unwrap(), "a: 1\n");
    #[cfg(unix)]
    assert_eq!(mode_bits(&target), 0o600);
}

#[test]
fn replaces_existing_content_and_narrows_a_wider_permission_file() {
    let dir = scratch();
    let target = dir.path().join("doc.yaml");
    fs::write(&target, "old").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&target, fs::Permissions::from_mode(0o644)).unwrap();
    }
    write_file_atomic(&target, "new", opts(0o600)).unwrap();
    assert_eq!(fs::read_to_string(&target).unwrap(), "new");
    #[cfg(unix)]
    assert_eq!(mode_bits(&target), 0o600);
}

#[cfg(unix)]
#[test]
fn replaces_a_symlinked_target_itself_without_writing_through_to_the_referent() {
    let dir = scratch();
    let victim = dir.path().join("victim");
    fs::write(&victim, "victim-content").unwrap();
    let target = dir.path().join("doc.yaml");
    std::os::unix::fs::symlink(&victim, &target).unwrap();
    write_file_atomic(&target, "replaced", opts(0o600)).unwrap();
    assert!(
        !fs::symlink_metadata(&target)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert_eq!(fs::read_to_string(&target).unwrap(), "replaced");
    assert_eq!(fs::read_to_string(&victim).unwrap(), "victim-content");
}

#[test]
fn leaves_no_temp_sibling_and_returns_the_error_when_the_rename_fails() {
    let dir = scratch();
    let target = dir.path().join("occupied");
    fs::create_dir(&target).unwrap();
    write_file_atomic(&target, "content", opts(0o600)).unwrap_err();
    let leftovers: Vec<_> = fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.contains(".tmp"))
        .collect();
    assert_eq!(leftovers, Vec::<String>::new());
}

#[test]
fn with_file_lock_rejects_an_invalid_parent_hierarchy_before_running_the_operation() {
    let dir = scratch();
    let parent = dir.path().join("not-a-directory");
    fs::write(&parent, "occupied").unwrap();
    let mut called = false;
    let error = with_file_lock(&parent.join("document"), || called = true).unwrap_err();
    assert!(
        matches!(
            error.kind(),
            std::io::ErrorKind::NotADirectory | std::io::ErrorKind::NotFound
        ),
        "unexpected error: {error:?}"
    );
    assert!(!called);
}

#[test]
fn with_file_lock_runs_the_operation_removes_the_lock_and_returns_the_result() {
    let dir = scratch();
    let target = dir.path().join("document");
    let lock = dir.path().join("document.lock");
    let result = with_file_lock(&target, || {
        assert!(lock.exists(), "lock sibling held during the operation");
        42
    })
    .unwrap();
    assert_eq!(result, 42);
    assert!(!lock.exists(), "lock released after the operation");
}
