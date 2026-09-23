//! The user-selected decision backend. An absent setting defaults to the
//! bundled local model; malformed or unreadable settings fail closed to the
//! ordinary harness instead of choosing an unexpected external service.

use std::path::{Path, PathBuf};

pub use laya_local::InstallProgress as LayaInstallProgress;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionMode {
    Laya,
    Jev,
    Normal,
}

pub fn read(data_dir: &Path) -> DecisionMode {
    let path = data_dir.join("decisions/mode");
    match std::fs::read_to_string(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => DecisionMode::Laya,
        Err(_) => DecisionMode::Normal,
        Ok(value) => match value.trim() {
            "laya" => DecisionMode::Laya,
            "jev" => DecisionMode::Jev,
            "normal" => DecisionMode::Normal,
            _ => DecisionMode::Normal,
        },
    }
}

/// The existing local TypeSafe credential. No environment variable or app
/// onboarding field is read. A missing, linked, or exposed file disables Jev.
pub fn protected_typesafe_key_path() -> Option<PathBuf> {
    jev_core::protected_local_key_path()
}

/// The released app carries a standalone Laya worker and a pinned local
/// checkpoint. Environment overrides support an explicit development install;
/// neither path is resolved through a network model ID.
pub fn laya_assets_in(data_dir: &Path) -> Option<(PathBuf, PathBuf)> {
    let worker = laya_worker()?;
    let executable = std::env::current_exe().ok()?;
    let resources = executable.parent()?.parent()?.join("Resources");
    let bundled = resources.join("laya-model");
    let downloaded = data_dir.join("laya/model");
    let model = std::env::var_os("KEEL_LAYA_MODEL")
        .map(PathBuf::from)
        .or_else(|| {
            bundled
                .join("coreml_config.json")
                .is_file()
                .then_some(bundled)
        })
        .or_else(|| laya_local::model_is_installed(&downloaded).then_some(downloaded))?;
    if !worker.is_file() || !model.join("coreml_config.json").is_file() {
        return None;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if std::fs::metadata(&worker).ok()?.permissions().mode() & 0o111 == 0 {
            return None;
        }
    }
    Some((worker, model))
}

/// Locate the packaged worker separately so onboarding can offer the explicit
/// model download before a model directory exists.
pub fn laya_worker() -> Option<PathBuf> {
    let worker = match std::env::var_os("KEEL_LAYA_WORKER") {
        Some(worker) => PathBuf::from(worker),
        None => {
            let executable = std::env::current_exe().ok()?;
            executable.parent()?.parent()?.join("Resources/laya-worker")
        }
    };
    if !worker.is_file() {
        return None;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if std::fs::metadata(&worker).ok()?.permissions().mode() & 0o111 == 0 {
            return None;
        }
    }
    Some(worker)
}

pub fn laya_assets() -> Option<(PathBuf, PathBuf)> {
    laya_assets_in(Path::new(""))
}

/// Install the pinned local checkpoint using the worker shipped with Keel.
/// Callers must tie this to an explicit user action.
pub async fn install_laya_model(data_dir: &Path) -> Result<PathBuf, laya_local::InstallError> {
    let worker = laya_worker().ok_or(laya_local::InstallError::WorkerUnavailable)?;
    laya_local::install_model(data_dir, &worker).await
}

pub async fn install_laya_model_with_progress(
    data_dir: &Path,
    progress: tokio::sync::mpsc::UnboundedSender<LayaInstallProgress>,
) -> Result<PathBuf, laya_local::InstallError> {
    let worker = laya_worker().ok_or(laya_local::InstallError::WorkerUnavailable)?;
    laya_local::install_model_with_progress(data_dir, &worker, progress).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn laya_is_default_and_jev_requires_an_explicit_mode() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read(dir.path()), DecisionMode::Laya);
        std::fs::create_dir_all(dir.path().join("decisions")).unwrap();
        let path = dir.path().join("decisions/mode");
        std::fs::write(&path, "jev\n").unwrap();
        assert_eq!(read(dir.path()), DecisionMode::Jev);
        std::fs::write(&path, "normal\n").unwrap();
        assert_eq!(read(dir.path()), DecisionMode::Normal);
        std::fs::write(&path, "unexpected").unwrap();
        assert_eq!(read(dir.path()), DecisionMode::Normal);
    }
}
