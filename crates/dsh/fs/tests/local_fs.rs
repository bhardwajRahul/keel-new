//! Behavior tests for the local provider, mirroring the portable parts of
//! upstream `fs-local/tests/filesystem.spec.ts` + `fsio.spec.ts` against
//! real tempdirs. Not ported: Win32 DACL suites, fault-injection internals
//! (staged-temp inspection hooks), and mid-flight signal races — the Rust
//! provider is synchronous, so only entry cancellation is observable.

mod common;

use common::*;
use dsh_fs::{
    FsEditRequest, FsEntryType, FsErrorCode, FsPathEntryType, FsVersion, FsWriteIntent,
    FsWriteOperation, LineEndings, LocalFileSystem, LocalFileSystemConfig, apply_literal_edit,
    normalize_line_endings, restore_line_endings,
};
use dsh_timeout::AbortController;
use std::os::unix::fs::PermissionsExt;

fn local(root: &std::path::Path) -> LocalFileSystem {
    LocalFileSystem::new(LocalFileSystemConfig {
        cwd: Some(root.to_path_buf()),
        diff_basis_max_bytes: None,
    })
    .unwrap()
}

fn root_dir() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(dir.path()).unwrap();
    (dir, root)
}

#[test]
fn resolve_uses_the_call_cwd_over_the_config_cwd() {
    let (_dir, root) = root_dir();
    let (_other_dir, other) = root_dir();
    write(&root, "a.txt", "root");
    write(&other, "a.txt", "other");
    let fs = local(&root);
    let default = fs.resolve("a.txt", None, None).unwrap();
    assert_eq!(default.display_path, root.join("a.txt").to_string_lossy());
    let overridden = fs.resolve("a.txt", Some(&other), None).unwrap();
    assert_eq!(
        overridden.display_path,
        other.join("a.txt").to_string_lossy()
    );
    // An absolute path ignores both cwds.
    let absolute = fs
        .resolve(&root.join("a.txt").to_string_lossy(), Some(&other), None)
        .unwrap();
    assert_eq!(absolute.target_key, default.target_key);
}

#[test]
fn resolve_keeps_a_missing_target_key_stable_across_creation() {
    let (_dir, root) = root_dir();
    let fs = local(&root);
    let before = fs.resolve("sub/deep/new.txt", None, None).unwrap();
    write(&root, "sub/deep/new.txt", "created");
    let after = fs.resolve("sub/deep/new.txt", None, None).unwrap();
    assert_eq!(before.target_key, after.target_key);
}

#[test]
fn resolve_shares_identity_across_symlink_aliases() {
    let (_dir, root) = root_dir();
    write(&root, "real.txt", "content");
    std::os::unix::fs::symlink(root.join("real.txt"), root.join("link.txt")).unwrap();
    let fs = local(&root);
    let real = fs.resolve("real.txt", None, None).unwrap();
    let link = fs.resolve("link.txt", None, None).unwrap();
    assert_eq!(real.target_key, link.target_key);
    assert_ne!(real.display_path, link.display_path);
}

#[test]
fn resolve_rejects_blank_paths_and_file_parents() {
    let (_dir, root) = root_dir();
    write(&root, "afile", "not a dir");
    let fs = local(&root);
    assert_eq!(
        fs.resolve("  ", None, None).unwrap_err().code,
        FsErrorCode::NotFound
    );
    let error = fs.resolve("afile/child.txt", None, None).unwrap_err();
    assert_eq!(error.code, FsErrorCode::NotFound);
    assert!(
        error.message.contains("not a directory"),
        "{}",
        error.message
    );
}

#[test]
fn stat_reports_type_size_and_absence() {
    let (_dir, root) = root_dir();
    write(&root, "a.txt", "12345");
    std::fs::create_dir(root.join("sub")).unwrap();
    let fs = local(&root);
    let file = fs
        .stat(&fs.resolve("a.txt", None, None).unwrap(), None)
        .unwrap()
        .unwrap();
    assert_eq!(file.entry_type, FsEntryType::File);
    assert_eq!(file.size, Some(5));
    let dir = fs
        .stat(&fs.resolve("sub", None, None).unwrap(), None)
        .unwrap()
        .unwrap();
    assert_eq!(dir.entry_type, FsEntryType::Directory);
    assert_eq!(dir.size, None);
    assert!(
        fs.stat(&fs.resolve("missing", None, None).unwrap(), None)
            .unwrap()
            .is_none()
    );
}

#[test]
fn versions_change_when_content_is_rewritten() {
    let (_dir, root) = root_dir();
    write(&root, "a.txt", "one");
    let fs = local(&root);
    let target = fs.resolve("a.txt", None, None).unwrap();
    let first = fs.stat(&target, None).unwrap().unwrap().version;
    fs.write_text(&target, "two", None, None).unwrap();
    let second = fs.stat(&target, None).unwrap().unwrap().version;
    assert_ne!(first, second);
}

#[test]
fn lstat_does_not_follow_the_final_symlink() {
    let (_dir, root) = root_dir();
    write(&root, "real.txt", "content");
    std::os::unix::fs::symlink(root.join("real.txt"), root.join("link.txt")).unwrap();
    let fs = local(&root);
    assert_eq!(
        fs.lstat("link.txt", None, None)
            .unwrap()
            .unwrap()
            .entry_type,
        FsPathEntryType::Symlink
    );
    assert_eq!(
        fs.lstat("real.txt", None, None)
            .unwrap()
            .unwrap()
            .entry_type,
        FsPathEntryType::File
    );
    assert!(fs.lstat("missing", None, None).unwrap().is_none());
}

#[test]
fn read_text_decodes_and_rejects_non_text_targets() {
    let (_dir, root) = root_dir();
    write(&root, "ok.txt", "hello\nworld");
    std::fs::create_dir(root.join("sub")).unwrap();
    std::fs::write(root.join("bin.dat"), [0x68u8, 0x00, 0x69]).unwrap();
    std::fs::write(root.join("bad.txt"), [0xffu8, 0xfe, 0x41]).unwrap();
    let fs = local(&root);
    let read = |name: &str| fs.read_text(&fs.resolve(name, None, None).unwrap(), None);
    assert_eq!(read("ok.txt").unwrap(), "hello\nworld");
    assert_eq!(read("missing.txt").unwrap_err().code, FsErrorCode::NotFound);
    assert_eq!(read("sub").unwrap_err().code, FsErrorCode::NotRegularFile);
    assert_eq!(read("bin.dat").unwrap_err().code, FsErrorCode::NotText);
    assert_eq!(read("bad.txt").unwrap_err().code, FsErrorCode::NotText);
}

#[test]
fn list_dir_returns_sorted_children_with_resolved_targets() {
    let (_dir, root) = root_dir();
    write(&root, "b.txt", "bb");
    write(&root, "a.txt", "a");
    std::fs::create_dir(root.join("sub")).unwrap();
    let fs = local(&root);
    let entries = fs
        .list_dir(&fs.resolve(".", None, None).unwrap(), None)
        .unwrap();
    let names: Vec<&str> = entries.iter().map(|entry| entry.name.as_str()).collect();
    assert_eq!(names, vec!["a.txt", "b.txt", "sub"]);
    assert_eq!(entries[0].entry_type, FsEntryType::File);
    assert_eq!(entries[0].size, Some(1));
    assert_eq!(entries[2].entry_type, FsEntryType::Directory);
    assert_eq!(entries[2].size, None);
    assert_eq!(
        entries[0].target.display_path,
        root.join("a.txt").to_string_lossy()
    );

    let missing = fs.resolve("nope", None, None).unwrap();
    assert_eq!(
        fs.list_dir(&missing, None).unwrap_err().code,
        FsErrorCode::NotFound
    );
    let file = fs.resolve("a.txt", None, None).unwrap();
    assert_eq!(
        fs.list_dir(&file, None).unwrap_err().code,
        FsErrorCode::NotDirectory
    );
}

#[test]
fn contains_is_a_canonical_containment_check() {
    let (_dir, root) = root_dir();
    std::fs::create_dir(root.join("inside")).unwrap();
    write(&root, "inside/file.txt", "x");
    let (_other_dir, other) = root_dir();
    let fs = local(&root);
    let parent = fs.resolve(".", None, None).unwrap();
    let child = fs.resolve("inside/file.txt", None, None).unwrap();
    let outside = fs.resolve(&other.to_string_lossy(), None, None).unwrap();
    assert!(fs.contains(&parent, &child));
    assert!(fs.contains(&parent, &parent));
    assert!(!fs.contains(&parent, &outside));
    assert!(!fs.contains(&child, &parent));
    // Traversal through a symlink cannot escape: identity is the realpath.
    std::os::unix::fs::symlink(&other, root.join("escape")).unwrap();
    let via_link = fs.resolve("escape", None, None).unwrap();
    assert!(!fs.contains(&parent, &via_link));
}

#[test]
fn create_if_absent_creates_and_refuses_to_clobber() {
    let (_dir, root) = root_dir();
    let fs = local(&root);
    let target = fs.resolve("new.txt", None, None).unwrap();
    let outcome = fs
        .write_text(&target, "fresh", Some(&FsWriteIntent::CreateIfAbsent), None)
        .unwrap();
    assert_eq!(outcome.operation, FsWriteOperation::Create);
    assert_eq!(outcome.before, None);
    assert_eq!(outcome.after, "fresh");
    assert_eq!(
        std::fs::read_to_string(root.join("new.txt")).unwrap(),
        "fresh"
    );

    let error = fs
        .write_text(&target, "blind", Some(&FsWriteIntent::CreateIfAbsent), None)
        .unwrap_err();
    assert_eq!(error.code, FsErrorCode::NotObserved);
    // The existing file was preserved.
    assert_eq!(
        std::fs::read_to_string(root.join("new.txt")).unwrap(),
        "fresh"
    );
}

#[test]
fn replace_if_version_guards_freshness() {
    let (_dir, root) = root_dir();
    write(&root, "a.txt", "old");
    let fs = local(&root);
    let target = fs.resolve("a.txt", None, None).unwrap();
    let version = fs.stat(&target, None).unwrap().unwrap().version;

    let outcome = fs
        .write_text(
            &target,
            "new",
            Some(&FsWriteIntent::ReplaceIfVersion(version.clone())),
            None,
        )
        .unwrap();
    assert_eq!(outcome.operation, FsWriteOperation::Update);
    assert_eq!(outcome.before.as_deref(), Some("old"));
    assert_eq!(outcome.after, "new");

    // The recorded version is now stale.
    let stale = fs
        .write_text(
            &target,
            "again",
            Some(&FsWriteIntent::ReplaceIfVersion(version)),
            None,
        )
        .unwrap_err();
    assert_eq!(stale.code, FsErrorCode::StaleVersion);

    // A deleted target reports stale too, without being recreated.
    let current = fs.stat(&target, None).unwrap().unwrap().version;
    std::fs::remove_file(root.join("a.txt")).unwrap();
    let gone = fs
        .write_text(
            &target,
            "zombie",
            Some(&FsWriteIntent::ReplaceIfVersion(current)),
            None,
        )
        .unwrap_err();
    assert_eq!(gone.code, FsErrorCode::StaleVersion);
    assert!(!root.join("a.txt").exists());
}

#[test]
fn unconditional_write_overwrites_and_rejects_directories() {
    let (_dir, root) = root_dir();
    write(&root, "a.txt", "old");
    std::fs::create_dir(root.join("sub")).unwrap();
    let fs = local(&root);
    let target = fs.resolve("a.txt", None, None).unwrap();
    let outcome = fs.write_text(&target, "new", None, None).unwrap();
    assert_eq!(outcome.operation, FsWriteOperation::Update);
    assert_eq!(std::fs::read_to_string(root.join("a.txt")).unwrap(), "new");

    let dir_target = fs.resolve("sub", None, None).unwrap();
    assert_eq!(
        fs.write_text(&dir_target, "nope", None, None)
            .unwrap_err()
            .code,
        FsErrorCode::NotRegularFile
    );
}

#[test]
fn overwrite_reports_an_lf_normalized_diff_basis() {
    let (_dir, root) = root_dir();
    write(&root, "a.txt", "one\r\ntwo\r\n");
    let fs = local(&root);
    let target = fs.resolve("a.txt", None, None).unwrap();
    let outcome = fs
        .write_text(&target, "one\r\nthree\r\n", None, None)
        .unwrap();
    // A CRLF rewrite must not read as every line changed.
    assert_eq!(outcome.before.as_deref(), Some("one\ntwo\n"));
    assert_eq!(outcome.after, "one\nthree\n");
}

#[test]
fn undiffable_prior_content_reports_no_basis_but_still_writes() {
    let (_dir, root) = root_dir();
    std::fs::write(root.join("bin.dat"), [0x41u8, 0x00, 0x42]).unwrap();
    let fs = local(&root);
    let target = fs.resolve("bin.dat", None, None).unwrap();
    let outcome = fs.write_text(&target, "text now", None, None).unwrap();
    assert_eq!(outcome.before, None);
    assert_eq!(
        std::fs::read_to_string(root.join("bin.dat")).unwrap(),
        "text now"
    );
}

#[test]
fn diff_basis_byte_limit_gates_both_sides() {
    let (_dir, root) = root_dir();
    write(&root, "a.txt", "1234");
    let fs = LocalFileSystem::new(LocalFileSystemConfig {
        cwd: Some(root.clone()),
        diff_basis_max_bytes: Some(8),
    })
    .unwrap();
    let target = fs.resolve("a.txt", None, None).unwrap();
    // New content at the limit: no contextual basis.
    let big = fs.write_text(&target, "12345678", None, None).unwrap();
    assert_eq!(big.before, None);
    // Both sides below the limit keep the basis.
    let small = fs.write_text(&target, "abc", None, None).unwrap();
    assert_eq!(small.before, None); // prior file is 8 bytes — at the limit
    let smaller = fs.write_text(&target, "ab", None, None).unwrap();
    assert_eq!(smaller.before.as_deref(), Some("abc"));
}

#[test]
fn write_preserves_the_existing_mode() {
    let (_dir, root) = root_dir();
    let path = write(&root, "script.sh", "#!/bin/sh\n");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    let fs = local(&root);
    let target = fs.resolve("script.sh", None, None).unwrap();
    fs.write_text(&target, "#!/bin/sh\necho hi\n", None, None)
        .unwrap();
    let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o755);
}

#[test]
fn edit_applies_at_the_observed_version() {
    let (_dir, root) = root_dir();
    write(&root, "a.txt", "alpha beta alpha");
    let fs = local(&root);
    let target = fs.resolve("a.txt", None, None).unwrap();
    let version = fs.stat(&target, None).unwrap().unwrap().version;
    let outcome = fs
        .edit_text(
            &target,
            &FsEditRequest {
                old_string: "beta".into(),
                new_string: "gamma".into(),
                replace_all: false,
            },
            Some(&version),
            None,
        )
        .unwrap();
    assert_eq!(outcome.before, "alpha beta alpha");
    assert_eq!(outcome.after, "alpha gamma alpha");
    assert_eq!(
        std::fs::read_to_string(root.join("a.txt")).unwrap(),
        "alpha gamma alpha"
    );
    // The returned version matches a fresh stat, so a follow-up edit works.
    assert_eq!(
        outcome.version,
        fs.stat(&target, None).unwrap().unwrap().version
    );
}

#[test]
fn edit_checks_staleness_before_literal_matching() {
    let (_dir, root) = root_dir();
    write(&root, "a.txt", "original text");
    let fs = local(&root);
    let target = fs.resolve("a.txt", None, None).unwrap();
    let version = fs.stat(&target, None).unwrap().unwrap().version;
    write(&root, "a.txt", "replaced externally");
    // old_string no longer matches — but the STALE code must win.
    let error = fs
        .edit_text(
            &target,
            &FsEditRequest {
                old_string: "original".into(),
                new_string: "x".into(),
                replace_all: false,
            },
            Some(&version),
            None,
        )
        .unwrap_err();
    assert_eq!(error.code, FsErrorCode::StaleVersion);
}

#[test]
fn unconditional_edit_works_and_missing_targets_are_stale() {
    let (_dir, root) = root_dir();
    write(&root, "a.txt", "one two");
    let fs = local(&root);
    let target = fs.resolve("a.txt", None, None).unwrap();
    let outcome = fs
        .edit_text(
            &target,
            &FsEditRequest {
                old_string: "two".into(),
                new_string: "three".into(),
                replace_all: false,
            },
            None,
            None,
        )
        .unwrap();
    assert_eq!(outcome.after, "one three");

    let missing = fs.resolve("missing.txt", None, None).unwrap();
    let error = fs
        .edit_text(
            &missing,
            &FsEditRequest {
                old_string: "x".into(),
                new_string: "y".into(),
                replace_all: false,
            },
            None,
            None,
        )
        .unwrap_err();
    assert_eq!(error.code, FsErrorCode::StaleVersion);
}

#[test]
fn edit_reports_match_failures_at_the_right_version() {
    let (_dir, root) = root_dir();
    write(&root, "a.txt", "dup dup");
    let fs = local(&root);
    let target = fs.resolve("a.txt", None, None).unwrap();
    let none = fs
        .edit_text(
            &target,
            &FsEditRequest {
                old_string: "absent".into(),
                new_string: "x".into(),
                replace_all: false,
            },
            None,
            None,
        )
        .unwrap_err();
    assert_eq!(none.code, FsErrorCode::EditNotFound);
    let ambiguous = fs
        .edit_text(
            &target,
            &FsEditRequest {
                old_string: "dup".into(),
                new_string: "x".into(),
                replace_all: false,
            },
            None,
            None,
        )
        .unwrap_err();
    assert_eq!(ambiguous.code, FsErrorCode::AmbiguousEdit);
    let all = fs
        .edit_text(
            &target,
            &FsEditRequest {
                old_string: "dup".into(),
                new_string: "x".into(),
                replace_all: true,
            },
            None,
            None,
        )
        .unwrap();
    assert_eq!(all.after, "x x");
}

#[test]
fn edit_round_trips_crlf_storage() {
    let (_dir, root) = root_dir();
    write(&root, "a.txt", "one\r\ntwo\r\n");
    let fs = local(&root);
    let target = fs.resolve("a.txt", None, None).unwrap();
    let outcome = fs
        .edit_text(
            &target,
            &FsEditRequest {
                old_string: "two".into(),
                new_string: "three".into(),
                replace_all: false,
            },
            None,
            None,
        )
        .unwrap();
    // The diff basis is LF; the stored file keeps its CRLF style.
    assert_eq!(outcome.before, "one\ntwo\n");
    assert_eq!(outcome.after, "one\nthree\n");
    assert_eq!(
        std::fs::read_to_string(root.join("a.txt")).unwrap(),
        "one\r\nthree\r\n"
    );
}

#[test]
fn pre_aborted_signals_reject_without_touching_the_filesystem() {
    dsh_cordis::run(async {
        let (_dir, root) = root_dir();
        write(&root, "a.txt", "content");
        let fs = local(&root);
        let controller = AbortController::new();
        controller.abort("test cancels");
        let signal = controller.signal();
        assert_eq!(
            fs.resolve("a.txt", None, Some(&signal)).unwrap_err().code,
            FsErrorCode::Aborted
        );
        let target = fs.resolve("a.txt", None, None).unwrap();
        assert_eq!(
            fs.read_text(&target, Some(&signal)).unwrap_err().code,
            FsErrorCode::Aborted
        );
        assert_eq!(
            fs.write_text(&target, "x", None, Some(&signal))
                .unwrap_err()
                .code,
            FsErrorCode::Aborted
        );
        assert_eq!(
            fs.edit_text(
                &target,
                &FsEditRequest {
                    old_string: "content".into(),
                    new_string: "x".into(),
                    replace_all: false
                },
                None,
                Some(&signal),
            )
            .unwrap_err()
            .code,
            FsErrorCode::Aborted
        );
        assert_eq!(
            std::fs::read_to_string(root.join("a.txt")).unwrap(),
            "content"
        );
    });
}

#[test]
fn line_ending_helpers_round_trip() {
    assert_eq!(normalize_line_endings("a\r\nb\rc\n"), "a\nb\rc\n");
    assert_eq!(
        restore_line_endings("a\nb\n", LineEndings::Crlf),
        "a\r\nb\r\n"
    );
    assert_eq!(restore_line_endings("a\nb\n", LineEndings::Lf), "a\nb\n");
    // Already-CRLF input is never doubled to \r\r\n.
    assert_eq!(
        restore_line_endings("a\r\nb\n", LineEndings::Crlf),
        "a\r\nb\r\n"
    );
}

#[test]
fn apply_literal_edit_counts_and_replaces() {
    let (content, replacements) = apply_literal_edit("a b a", "a", "z", true, "f").unwrap();
    assert_eq!((content.as_str(), replacements), ("z b z", 2));
    assert_eq!(
        apply_literal_edit("abc", "", "z", false, "f")
            .unwrap_err()
            .code,
        FsErrorCode::EditNotFound
    );
    // CRLF inside the needle is normalized before matching.
    let (content, _) = apply_literal_edit("one\ntwo", "one\r\ntwo", "joined", false, "f").unwrap();
    assert_eq!(content, "joined");
}

#[test]
fn version_tokens_are_opaque_but_comparable() {
    let (_dir, root) = root_dir();
    write(&root, "a.txt", "x");
    let fs = local(&root);
    let target = fs.resolve("a.txt", None, None).unwrap();
    let version = fs.stat(&target, None).unwrap().unwrap().version;
    assert_eq!(version, FsVersion::new(version.as_str()));
}
