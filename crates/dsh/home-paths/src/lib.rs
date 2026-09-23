//! Port of `packages/util/home-paths` (`@deepseek-ai/dsh-home-paths`): shared
//! filesystem path helpers for DeepSeek Harness user data. The harness keeps
//! all user data under one root, resolved from an explicit override, then
//! `$DSH_HOME`, then `~/.dsh`.
//!
//! Divergences from the TS original:
//! - [`canonicalize_watch_path`] is blocking (`std::fs`) instead of
//!   promise-based.
//! - Upstream `resolveDshHome` takes an environment record defaulting to
//!   `process.env`; here [`resolve_dsh_home`] reads the process environment
//!   and [`resolve_dsh_home_from`] is the pure variant taking the `$DSH_HOME`
//!   value directly.
//! - The Cordis `./invariant` companion is not ported: the package declares no
//!   runtime invariant.

use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};

/// Directory name for the default DeepSeek Harness home under the OS home.
pub const DSH_HOME_DIR_NAME: &str = ".dsh";

/// Stable user-facing display form for the default DeepSeek Harness home.
pub const DEFAULT_DSH_HOME_DISPLAY: &str = "~/.dsh";

/// Environment variable that overrides the default DeepSeek Harness home.
pub const DSH_HOME_ENV: &str = "DSH_HOME";

/// The operating-system home directory.
fn home_dir() -> PathBuf {
    #[allow(deprecated)] // un-deprecated on current stable; kept for older toolchains
    std::env::home_dir().expect("the OS home directory must be resolvable")
}

/// Give a native filesystem watcher one canonical spelling of a path, even
/// when its final components do not exist yet.
///
/// The deepest existing ancestor is resolved through `canonicalize`; when a
/// suffix is missing, that ancestor is additionally proved to be an
/// enumerable directory before the suffix is restored. This keeps a
/// regular-file ancestor from being reported as ordinary absence (Windows
/// probes it that way) and keeps short-name aliases out of paths a native
/// watcher backend later emits in long form.
///
/// # Errors
/// Fails when ancestor traversal hits an error other than absence, or when
/// the existing ancestor of a missing suffix is not an enumerable directory.
pub fn canonicalize_watch_path(path: &Path) -> io::Result<PathBuf> {
    let mut current = std::path::absolute(path)?;
    let mut missing: Vec<OsString> = Vec::new();
    loop {
        match std::fs::canonicalize(&current) {
            Ok(canonical) => {
                if !missing.is_empty() {
                    // Opening the resolved ancestor preserves the
                    // cross-platform directory requirement.
                    std::fs::read_dir(&canonical)?;
                }
                let mut result = canonical;
                for component in missing.iter().rev() {
                    result.push(component);
                }
                return Ok(result);
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                let (Some(name), Some(parent)) = (current.file_name(), current.parent()) else {
                    // A filesystem root exists, so traversal normally resolves
                    // before running out of components.
                    return Err(error);
                };
                missing.push(name.to_owned());
                current = parent.to_path_buf();
            }
            Err(error) => return Err(error),
        }
    }
}

/// The absolute default DeepSeek Harness home (`~/.dsh` under the OS home).
pub fn default_dsh_home() -> PathBuf {
    home_dir().join(DSH_HOME_DIR_NAME)
}

/// Expand a supported tilde prefix (`~`, `~/`, `~\`) against the OS home;
/// any other value passes through unchanged.
pub fn expand_home_path(path: &str) -> PathBuf {
    if path == "~" {
        return home_dir();
    }
    if let Some(rest) = path.strip_prefix("~/").or_else(|| path.strip_prefix("~\\")) {
        return home_dir().join(rest);
    }
    PathBuf::from(path)
}

/// Resolve the single-root DeepSeek Harness home from the process
/// environment. See [`resolve_dsh_home_from`] for the precedence contract.
pub fn resolve_dsh_home(configured: Option<&str>) -> PathBuf {
    let env = std::env::var(DSH_HOME_ENV).ok();
    resolve_dsh_home_from(configured, env.as_deref())
}

/// Resolve the single-root DeepSeek Harness home.
///
/// Precedence, highest first: the explicit `configured` path, the `$DSH_HOME`
/// value, then `~/.dsh`. An empty or whitespace-only `$DSH_HOME` counts as
/// unset, so a blank override never resolves the home to the current working
/// directory. The result is absolute and tilde-expanded.
pub fn resolve_dsh_home_from(configured: Option<&str>, dsh_home_env: Option<&str>) -> PathBuf {
    let selected = match configured {
        Some(path) => expand_home_path(path),
        None => match dsh_home_env {
            Some(value) if !value.trim().is_empty() => expand_home_path(value),
            _ => default_dsh_home(),
        },
    };
    std::path::absolute(&selected).unwrap_or(selected)
}

/// Join path segments onto the resolved DeepSeek Harness home; an empty list
/// returns the home itself.
pub fn dsh_home_path(segments: &[&str]) -> PathBuf {
    let mut path = resolve_dsh_home(None);
    for segment in segments {
        path.push(segment);
    }
    path
}

/// Describe a resolved harness home symbolically for user-facing display.
///
/// Never returns an absolute machine path: the default home is labelled
/// `~/.dsh`, and any configured home is labelled `$DSH_HOME`.
pub fn dsh_home_display(resolved_home: &Path) -> String {
    let default = default_dsh_home();
    let default = std::path::absolute(&default).unwrap_or(default);
    if resolved_home == default {
        DEFAULT_DSH_HOME_DISPLAY.to_owned()
    } else {
        format!("${DSH_HOME_ENV}")
    }
}
