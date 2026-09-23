//! Port of `packages/credentials/credentials` (`@deepseek-ai/dsh-credentials`):
//! the credential-reference capability seam (`ctx.credentials`).
//!
//! Settings and composition files carry *references* to secrets — environment
//! variable names — while a provider owns the values and their storage.
//! Consumers resolve a reference once per operation, so a rotated credential
//! reaches the next operation without a plugin restart, and configuration
//! surfaces can describe a reference without ever seeing its value.
//!
//! Divergences from the TS original:
//! - The abstract `CredentialProvider` service class becomes the
//!   [`CredentialProvider`] trait plus the [`Credentials`] service newtype a
//!   provider plugin registers under the `"credentials"` name; consumers read
//!   it with `ctx.service::<Credentials>()` / `ctx.try_service::<Credentials>()`.
//! - `credentialRef` threw a `TypeError`; [`credential_ref`] returns
//!   `Result<_, InvalidCredentialRef>`.
//! - The contained `credentials/updated` fan-out: dsh-cordis dispatches every
//!   notification listener as a spawned future, so a listener failure can
//!   never reach the emitter — the upstream containment (run every listener,
//!   log failures, never fail the committed operation) is inherent here. The
//!   `INVARIANT`-coded rethrow channel is not ported: upstream restricts it to
//!   synchronous listeners and this port has none, and the `dsh-invariants`
//!   registry it serves is not ported either.
//! - The `./invariant` companion is not ported (`dsh-invariants` does not
//!   exist in this workspace).

use dsh_cordis::{Context, Event, Service};
use std::rc::Rc;

/// Compile-time marker for [`CredentialRef`].
pub enum CredentialRefMark {}

/// Nominal reference to one credential: a POSIX-style environment-variable
/// name. Construct through [`credential_ref`] so an unaddressable name never
/// enters the seam.
pub type CredentialRef = dsh_brand::Branded<CredentialRefMark>;

/// A candidate reference that is not a POSIX shell identifier.
#[derive(Debug, Clone, thiserror::Error)]
#[error("credential ref \"{0}\" must match /^[A-Za-z_][A-Za-z0-9_]*$/")]
pub struct InvalidCredentialRef(pub String);

/// Brand a raw string as a [`CredentialRef`]. Accepts exactly the POSIX shell
/// identifiers (`[A-Za-z_][A-Za-z0-9_]*`), such as `DEEPSEEK_API_KEY`.
pub fn credential_ref(value: &str) -> Result<CredentialRef, InvalidCredentialRef> {
    let mut chars = value.chars();
    let head_ok = matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_');
    if head_ok && chars.all(|c| c.is_ascii_alphanumeric() || c == '_') {
        Ok(CredentialRef::new(value))
    } else {
        Err(InvalidCredentialRef(value.to_string()))
    }
}

/// One resolved credential value and the source layer that supplied it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedCredential {
    /// The non-empty secret value.
    pub value: String,
    /// Provider-defined source layer id (the local provider uses `env`,
    /// `file`, `project-env`, and `user-env`).
    pub source: String,
}

/// Source and writability facts for one reference, safe for configuration
/// surfaces — never the value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CredentialInfo {
    /// Whether [`CredentialProvider::resolve`] would currently return a value.
    pub configured: bool,
    /// Source layer currently supplying the value; `None` while unconfigured.
    pub source: Option<String>,
    /// Whether [`CredentialProvider::set`] would currently succeed for this
    /// reference.
    pub writable: bool,
}

/// Abstract credential operations a provider implements over its source
/// layers. One seam-wide rule binds every implementation: an empty stored
/// value is absent everywhere — `resolve` skips it and `describe` reports it
/// unconfigured — so a blank can never pass for a configured secret.
#[async_trait::async_trait(?Send)]
pub trait CredentialProvider: 'static {
    /// Resolve one reference to its current value, or `None` while
    /// unconfigured. Resolution is per call: consumers re-resolve at each
    /// operation and must not cache across operations, which is what lets a
    /// changed credential reach the next operation without a restart.
    async fn resolve(&self, r: &CredentialRef) -> anyhow::Result<Option<ResolvedCredential>>;

    /// Describe one reference for configuration surfaces without exposing the
    /// value: configured state, supplying source, and writability.
    async fn describe(&self, r: &CredentialRef) -> anyhow::Result<CredentialInfo>;

    /// Durably store one non-empty value in the provider-managed writable
    /// source. Fails while a read-only source shadows the reference — the
    /// write would look successful while resolution kept returning the
    /// shadowing value — and fails on an empty value (use
    /// [`CredentialProvider::unset`]).
    async fn set(&self, r: &CredentialRef, value: &str) -> anyhow::Result<()>;

    /// Remove one reference from the provider-managed writable source;
    /// removing an absent reference is a no-op. Fails while a read-only
    /// source shadows the reference, like [`CredentialProvider::set`].
    async fn unset(&self, r: &CredentialRef) -> anyhow::Result<()>;
}

/// The `ctx.credentials` service: a provider registered under the seam name.
/// Derefs to the provider so consumers call the operations directly.
pub struct Credentials(pub Rc<dyn CredentialProvider>);

impl Service for Credentials {
    const NAME: &'static str = "credentials";
}

impl std::ops::Deref for Credentials {
    type Target = dyn CredentialProvider;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

/// Committed change to a provider-managed credential source: a `set`, an
/// `unset`, or an external edit observed in storage. Ambient
/// process-environment changes are not observable and never emit. Dispatched
/// with `emit`, so listeners are spawned observers whose failures cannot
/// reach the emitter.
pub struct CredentialsUpdated;

impl Event for CredentialsUpdated {
    const NAME: &'static str = "credentials/updated";
    type Args = CredentialRef;
    type Ret = ();
}

/// Fan `credentials/updated` out to every listener. Providers call this only
/// after the write or reload actually committed, so a broken observer can
/// never make a durable change look failed.
pub fn notify_updated(ctx: &Context, r: &CredentialRef) {
    ctx.emit::<CredentialsUpdated>(r);
}
