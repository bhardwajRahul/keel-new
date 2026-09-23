//! Upstream `dsh-brand` ships no runtime tests (the brand is type-only there).
//! These tests pin the Rust-side contract: branded ids behave as ordinary
//! strings for comparison, hashing, logging, and serialization.

use std::collections::HashSet;

use dsh_brand::Branded;

enum SessionIdMark {}
type SessionId = Branded<SessionIdMark>;

#[test]
fn behaves_as_an_ordinary_string() {
    let a = SessionId::new("abc");
    let b = SessionId::new("abc");
    let c = SessionId::new("abd");
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert!(a < c);
    assert_eq!(a.as_str(), "abc");
    assert_eq!(a.to_string(), "abc");
    assert_eq!(format!("{a:?}"), "\"abc\"");
    assert_eq!(a.clone().into_inner(), "abc");

    let mut set = HashSet::new();
    set.insert(a);
    set.insert(b); // duplicate value collapses like a plain string
    set.insert(c);
    assert_eq!(set.len(), 2);
    assert!(set.contains("abc")); // Borrow<str> lookup, no allocation
}

#[test]
fn serializes_transparently_as_the_string() {
    let id = SessionId::new("s-123");
    assert_eq!(serde_json::to_string(&id).unwrap(), "\"s-123\"");
    let back: SessionId = serde_json::from_str("\"s-123\"").unwrap();
    assert_eq!(back, id);
}

#[test]
fn marker_never_constrains_auto_traits() {
    fn assert_send_sync<T: Send + Sync>() {}
    struct NotSync(#[allow(dead_code)] std::cell::Cell<u8>);
    assert_send_sync::<Branded<NotSync>>();
}
