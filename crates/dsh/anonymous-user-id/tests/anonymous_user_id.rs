//! Port of upstream `tests/anonymous-user-id.spec.ts`. The
//! `tests/invariant.spec.ts` suite is not ported (`dsh-invariants` does not
//! exist in this workspace).

use dsh_anonymous_user_id::{
    ANONYMOUS_USER_ID_FILE_NAME, AnonymousUserIdOptions, get_or_create_anonymous_user_id,
};
use std::path::Path;

fn is_uuid(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 36
        && bytes.iter().enumerate().all(|(i, byte)| match i {
            8 | 13 | 18 | 23 => *byte == b'-',
            _ => byte.is_ascii_hexdigit(),
        })
}

fn with_home(home: &Path) -> AnonymousUserIdOptions<'_> {
    AnonymousUserIdOptions {
        dsh_home: Some(home.to_str().unwrap()),
        ..Default::default()
    }
}

fn stored(home: &Path) -> String {
    std::fs::read_to_string(home.join(ANONYMOUS_USER_ID_FILE_NAME)).unwrap()
}

#[test]
fn creates_persists_and_returns_a_bare_uuid_line_on_first_use() {
    let home = tempfile::tempdir().unwrap();
    let id = get_or_create_anonymous_user_id(with_home(home.path()));
    assert!(is_uuid(id.as_str()));
    assert_eq!(stored(home.path()), format!("{id}\n"));
}

#[test]
fn creates_the_home_directory_when_missing() {
    let base = tempfile::tempdir().unwrap();
    let home = base.path().join("nested").join("home");
    let id = get_or_create_anonymous_user_id(with_home(&home));
    assert_eq!(stored(&home), format!("{id}\n"));
}

#[test]
fn returns_the_persisted_id_tolerating_surrounding_whitespace() {
    let home = tempfile::tempdir().unwrap();
    let existing = "01234567-89ab-4cde-8f01-23456789abcd";
    std::fs::write(
        home.path().join(ANONYMOUS_USER_ID_FILE_NAME),
        format!("  {existing}\n\n"),
    )
    .unwrap();
    let id = get_or_create_anonymous_user_id(with_home(home.path()));
    assert_eq!(id.as_str(), existing);
}

#[test]
fn overwrites_a_corrupt_file_with_a_fresh_id() {
    let home = tempfile::tempdir().unwrap();
    std::fs::write(
        home.path().join(ANONYMOUS_USER_ID_FILE_NAME),
        "not-a-uuid\n",
    )
    .unwrap();
    let id = get_or_create_anonymous_user_id(with_home(home.path()));
    assert!(is_uuid(id.as_str()));
    assert_eq!(stored(home.path()), format!("{id}\n"));
}

#[test]
fn adopts_a_concurrent_winner_written_after_the_initial_read() {
    let home = tempfile::tempdir().unwrap();
    let winner = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee";
    let file = home.path().join(ANONYMOUS_USER_ID_FILE_NAME);
    // The generator hook runs between the initial read (absent) and the
    // exclusive-create write, so planting the winner here simulates the
    // concurrent first launch.
    let plant = || {
        std::fs::write(&file, format!("{winner}\n")).unwrap();
        "ffffffff-0000-4000-8000-000000000000".to_string()
    };
    let id = get_or_create_anonymous_user_id(AnonymousUserIdOptions {
        dsh_home: Some(home.path().to_str().unwrap()),
        random_uuid: Some(&plant),
    });
    assert_eq!(id.as_str(), winner);
}

#[test]
fn returns_a_usable_id_when_the_home_cannot_contain_files() {
    let base = tempfile::tempdir().unwrap();
    let blocked = base.path().join("blocked");
    std::fs::write(&blocked, "occupied\n").unwrap();
    let id = get_or_create_anonymous_user_id(with_home(&blocked));
    assert!(is_uuid(id.as_str()));
    assert!(!blocked.join(ANONYMOUS_USER_ID_FILE_NAME).exists());
}

#[test]
fn memoizes_per_resolved_home_for_the_process_lifetime() {
    let home = tempfile::tempdir().unwrap();
    let first = get_or_create_anonymous_user_id(with_home(home.path()));
    std::fs::remove_file(home.path().join(ANONYMOUS_USER_ID_FILE_NAME)).unwrap();
    let second = get_or_create_anonymous_user_id(with_home(home.path()));
    assert_eq!(first, second);
}

#[test]
fn keeps_distinct_homes_on_distinct_ids() {
    let a_home = tempfile::tempdir().unwrap();
    let b_home = tempfile::tempdir().unwrap();
    let a = get_or_create_anonymous_user_id(with_home(a_home.path()));
    let b = get_or_create_anonymous_user_id(with_home(b_home.path()));
    assert_ne!(a, b);
}

#[test]
fn reads_the_process_environment_by_default() {
    let home = tempfile::tempdir().unwrap();
    let previous = std::env::var_os("DSH_HOME");
    // SAFETY: this is the only test in the binary that touches the process
    // environment, and it restores the prior value before finishing.
    unsafe { std::env::set_var("DSH_HOME", home.path()) };
    let id = get_or_create_anonymous_user_id(AnonymousUserIdOptions::default());
    let text = stored(home.path());
    unsafe {
        match previous {
            Some(value) => std::env::set_var("DSH_HOME", value),
            None => std::env::remove_var("DSH_HOME"),
        }
    }
    assert_eq!(text, format!("{id}\n"));
}
