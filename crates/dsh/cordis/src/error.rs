//! Framework errors with stable machine-readable codes, ported from
//! `vendor/cordis/src/fiber.ts` (`CordisError`, `ValidationError`).

use std::fmt;

/// Framework error with a stable code, mirroring upstream `CordisError` plus
/// the error shapes the TS implementation raises as plain `Error`s.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CordisError {
    /// `INACTIVE_EFFECT` — effect created on a disposed or unloading fiber.
    #[error("cannot create effect on inactive context")]
    InactiveEffect,
    /// Config failed schema validation before the plugin started.
    #[error("invalid config:\n{0}")]
    InvalidConfig(String),
    /// A service name was provided twice within one isolation scope.
    #[error("service \"{name}\" has been registered at <{provider}>")]
    ServiceConflict { name: String, provider: String },
    /// A required service was read while its providing fiber was inactive.
    #[error("cannot get required service \"{0}\" in inactive context")]
    ServiceInactive(String),
    /// A service was read without a matching `inject` declaration.
    #[error("cannot get property \"{0}\" without inject")]
    ServiceMissing(String),
    /// Plugin startup failed; carries the formatted error chain (the fiber
    /// re-throws on every `await()`, so the variant must stay `Clone + Send`).
    #[error("plugin startup failed: {0}")]
    Plugin(String),
}

impl CordisError {
    /// Wrap a plugin-body error for storage on the fiber.
    pub fn plugin(error: anyhow::Error) -> Self {
        CordisError::Plugin(format!("{error:#}"))
    }
}

/// Result alias used across the framework surface.
pub type Result<T, E = CordisError> = std::result::Result<T, E>;

/// Assert-style helper mirroring upstream `ValidationError` message layout:
/// one `  - message (at path)` line per issue.
pub struct ValidationIssues(pub Vec<(Option<String>, String)>);

impl fmt::Display for ValidationIssues {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, (path, message)) in self.0.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            match path {
                Some(path) => write!(f, "  - {message} (at {path})")?,
                None => write!(f, "  - {message}")?,
            }
        }
        Ok(())
    }
}

impl From<ValidationIssues> for CordisError {
    fn from(issues: ValidationIssues) -> Self {
        CordisError::InvalidConfig(issues.to_string())
    }
}
