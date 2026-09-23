//! Port of `packages/util/brand` (`@deepseek-ai/dsh-brand`): the nominal-typing
//! primitive shared by every package that owns a cross-boundary id.
//!
//! A brand makes structurally-identical strings non-interchangeable in the type
//! system: a `SessionId` cannot be handed to a function expecting a `CallId`,
//! even though both wrap a plain string at runtime. Comparison, logging,
//! hashing, ordering, and serialization all behave exactly like the underlying
//! string.
//!
//! Policy carried over from upstream: a package brands the ids it owns, and
//! branding is reserved for ids that cross package boundaries and could
//! plausibly be confused — not every string needs a brand. This crate owns only
//! the primitive, so the brand vocabulary stays dependency-free.
//!
//! Divergences from the TS original:
//! - Upstream `Branded<B>` is a compile-time-only intersection type with a
//!   per-id cast factory in the owning package. Rust has no erased nominal
//!   types, so the primitive is a zero-cost newtype [`Branded<M>`] generic
//!   over an (usually uninhabited) marker type; the owning package declares
//!   the marker and a `pub type` alias instead of a cast factory.
//! - Serde support is added here (transparent string encoding) because Rust
//!   serialization cannot be erased the way a TS brand is.

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A string carrying a compile-time-only brand `M`.
///
/// `M` is a marker type (typically an uninhabited `enum`) that never exists at
/// runtime; the wrapper stores only the string. Two `Branded` types with
/// different markers are distinct types:
///
/// ```
/// use dsh_brand::Branded;
/// enum SessionIdMark {}
/// type SessionId = Branded<SessionIdMark>;
/// let id = SessionId::new("s-123");
/// assert_eq!(id.as_str(), "s-123");
/// ```
///
/// ```compile_fail
/// use dsh_brand::Branded;
/// enum SessionIdMark {}
/// enum CallIdMark {}
/// fn takes_call(_: Branded<CallIdMark>) {}
/// takes_call(Branded::<SessionIdMark>::new("s-123")); // brand mismatch
/// ```
pub struct Branded<M: ?Sized> {
    value: String,
    // fn() -> M keeps the marker covariant and irrelevant to Send/Sync/auto traits.
    _brand: PhantomData<fn() -> M>,
}

impl<M: ?Sized> Branded<M> {
    /// Brand a string. Owning packages expose this through their id alias;
    /// construction is where the "never a bare string past this point"
    /// decision is made.
    pub fn new(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            _brand: PhantomData,
        }
    }

    /// The branded string as a borrowed `&str`.
    pub fn as_str(&self) -> &str {
        &self.value
    }

    /// Drop the brand and recover the owned string.
    pub fn into_inner(self) -> String {
        self.value
    }
}

// Manual impls: derives would demand bounds on the phantom marker `M`.

impl<M: ?Sized> Clone for Branded<M> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            _brand: PhantomData,
        }
    }
}

impl<M: ?Sized> PartialEq for Branded<M> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<M: ?Sized> Eq for Branded<M> {}

impl<M: ?Sized> PartialOrd for Branded<M> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<M: ?Sized> Ord for Branded<M> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

impl<M: ?Sized> Hash for Branded<M> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl<M: ?Sized> fmt::Debug for Branded<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.value, f)
    }
}

impl<M: ?Sized> fmt::Display for Branded<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.value, f)
    }
}

impl<M: ?Sized> AsRef<str> for Branded<M> {
    fn as_ref(&self) -> &str {
        &self.value
    }
}

impl<M: ?Sized> Borrow<str> for Branded<M> {
    fn borrow(&self) -> &str {
        &self.value
    }
}

impl<M: ?Sized> Serialize for Branded<M> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.value.serialize(serializer)
    }
}

impl<'de, M: ?Sized> Deserialize<'de> for Branded<M> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::new(String::deserialize(deserializer)?))
    }
}
