//! First-run decision mode. Laya runs locally; optional Jev uses an existing
//! protected credential. The app does not collect provider API keys.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionMode {
    Laya,
    Jev,
    Normal,
}

impl DecisionMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Laya => "laya",
            Self::Jev => "jev",
            Self::Normal => "normal",
        }
    }
}

pub fn mode_path(data_dir: &Path) -> PathBuf {
    data_dir.join("decisions").join("mode")
}

/// A fresh installation uses Laya. An invalid persisted value is an error,
/// rather than silently enabling a different decision service.
pub fn load_mode(data_dir: &Path) -> io::Result<DecisionMode> {
    let value = match fs::read_to_string(mode_path(data_dir)) {
        Ok(value) => value,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(DecisionMode::Laya),
        Err(err) => return Err(err),
    };
    match value.trim() {
        "laya" => Ok(DecisionMode::Laya),
        "jev" => Ok(DecisionMode::Jev),
        "normal" => Ok(DecisionMode::Normal),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid decision mode",
        )),
    }
}

pub fn save_mode(data_dir: &Path, mode: DecisionMode) -> io::Result<()> {
    let path = mode_path(data_dir);
    let dir = path.parent().expect("decision mode path has parent");
    fs::create_dir_all(dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if fs::symlink_metadata(dir)?.file_type().is_symlink() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "decision directory is a link",
            ));
        }
        fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
    }
    let tmp = dir.join(format!(".mode-{}.tmp", uuid::Uuid::new_v4()));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let result = (|| {
        let mut file = options.open(&tmp)?;
        file.write_all(mode.as_str().as_bytes())?;
        file.sync_all()?;
        fs::rename(&tmp, &path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

/// Checks only that the files needed for local decisions are present. The
/// worker validates its model when the first decision runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayaStatus {
    pub installed: bool,
    pub downloadable: bool,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayaInstallState {
    #[default]
    Idle,
    Starting,
    Downloading {
        downloaded_bytes: u64,
        total_bytes: u64,
    },
    Verifying,
}

impl LayaInstallState {
    pub fn is_running(self) -> bool {
        self != Self::Idle
    }

    pub fn fraction(self) -> Option<f32> {
        match self {
            Self::Downloading {
                downloaded_bytes,
                total_bytes,
            } if total_bytes > 0 => {
                Some((downloaded_bytes as f32 / total_bytes as f32).clamp(0.0, 1.0))
            }
            _ => None,
        }
    }

    pub fn label(self) -> String {
        match self {
            Self::Idle => String::new(),
            Self::Starting => "Preparing Laya download…".into(),
            Self::Downloading {
                downloaded_bytes,
                total_bytes,
            } if total_bytes > 0 => format!(
                "Downloading Laya… {}% ({} of {} MB)",
                downloaded_bytes.saturating_mul(100) / total_bytes,
                downloaded_bytes / 1_000_000,
                total_bytes / 1_000_000
            ),
            Self::Downloading { .. } => "Downloading Laya…".into(),
            Self::Verifying => "Download complete · verifying model…".into(),
        }
    }
}

impl From<keel_engine::decision_mode::LayaInstallProgress> for LayaInstallState {
    fn from(progress: keel_engine::decision_mode::LayaInstallProgress) -> Self {
        match progress {
            keel_engine::decision_mode::LayaInstallProgress::Downloading {
                downloaded_bytes,
                total_bytes,
            } => Self::Downloading {
                downloaded_bytes,
                total_bytes,
            },
            keel_engine::decision_mode::LayaInstallProgress::Verifying => Self::Verifying,
        }
    }
}

fn laya_status_for(supported: bool, installed: bool, worker_ready: bool) -> LayaStatus {
    if !supported {
        return LayaStatus {
            installed: false,
            downloadable: false,
            message: "Laya requires macOS 15 or newer".into(),
        };
    }
    if installed {
        return LayaStatus {
            installed: true,
            downloadable: false,
            message: "Installed · ready for local decisions".into(),
        };
    }
    if worker_ready {
        return LayaStatus {
            installed: false,
            downloadable: true,
            message: "Local model not installed · download is about 680 MB".into(),
        };
    }
    LayaStatus {
        installed: false,
        downloadable: false,
        message: "The Laya worker is unavailable in this build".into(),
    }
}

/// Report both current availability and whether this build can install the
/// pinned local model after an explicit user action.
pub fn laya_status(data_dir: &Path) -> LayaStatus {
    if !cfg!(target_os = "macos") {
        return laya_status_for(false, false, false);
    }
    let version = std::process::Command::new("/usr/bin/sw_vers")
        .arg("-productVersion")
        .output();
    let supported = version
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|version| version.trim().split('.').next()?.parse::<u32>().ok())
        .is_some_and(|major| major >= 15);
    if !supported {
        return laya_status_for(false, false, false);
    }
    laya_status_for(
        true,
        keel_engine::decision_mode::laya_assets_in(data_dir).is_some(),
        keel_engine::decision_mode::laya_worker().is_some(),
    )
}

/// A saved backend can become unavailable after an app move or credential
/// change. This is for display only: persisting its Normal fallback would
/// discard a pending Laya preference before the model is installed.
pub fn available_mode(
    mode: DecisionMode,
    laya_available: bool,
    jev_available: bool,
) -> DecisionMode {
    match mode {
        DecisionMode::Laya if !laya_available => DecisionMode::Normal,
        DecisionMode::Jev if !jev_available => DecisionMode::Normal,
        _ => mode,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_backends_present_as_normal_harness() {
        assert_eq!(
            available_mode(DecisionMode::Laya, false, true),
            DecisionMode::Normal
        );
        assert_eq!(
            available_mode(DecisionMode::Jev, true, false),
            DecisionMode::Normal
        );
        assert_eq!(
            available_mode(DecisionMode::Laya, true, false),
            DecisionMode::Laya
        );
    }

    #[test]
    fn laya_download_is_only_offered_when_the_platform_and_worker_support_it() {
        assert_eq!(
            laya_status_for(true, false, true),
            LayaStatus {
                installed: false,
                downloadable: true,
                message: "Local model not installed · download is about 680 MB".into(),
            }
        );
        assert!(!laya_status_for(false, false, true).downloadable);
        assert!(!laya_status_for(true, false, false).downloadable);
        assert!(laya_status_for(true, true, true).installed);
    }

    #[test]
    fn laya_progress_uses_reported_bytes_and_names_verification() {
        let state = LayaInstallState::Downloading {
            downloaded_bytes: 250_000_000,
            total_bytes: 1_000_000_000,
        };
        assert_eq!(state.fraction(), Some(0.25));
        assert!(state.label().contains("25%"));
        assert!(LayaInstallState::Verifying.label().contains("verifying"));
        assert_eq!(LayaInstallState::Starting.fraction(), None);
    }

    #[test]
    fn laya_is_default_and_each_mode_is_private() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(load_mode(dir.path()).unwrap(), DecisionMode::Laya);
        for mode in [DecisionMode::Normal, DecisionMode::Laya, DecisionMode::Jev] {
            save_mode(dir.path(), mode).unwrap();
            assert_eq!(load_mode(dir.path()).unwrap(), mode);
        }
        assert_eq!(fs::read_to_string(mode_path(dir.path())).unwrap(), "jev");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(mode_path(dir.path()))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
            assert_eq!(
                fs::metadata(mode_path(dir.path()).parent().unwrap())
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o700
            );
        }
        fs::write(mode_path(dir.path()), "unknown").unwrap();
        assert!(load_mode(dir.path()).is_err());
        fs::write(mode_path(dir.path()), "jev").unwrap();
        assert_eq!(load_mode(dir.path()).unwrap(), DecisionMode::Jev);
    }

    #[test]
    fn laya_preference_survives_fallback_until_model_is_available() {
        let dir = tempfile::tempdir().unwrap();
        let desired = load_mode(dir.path()).unwrap();
        assert_eq!(desired, DecisionMode::Laya);
        assert_eq!(available_mode(desired, false, false), DecisionMode::Normal);

        // Continuing without a downloaded model keeps the desired backend.
        save_mode(dir.path(), desired).unwrap();
        assert_eq!(load_mode(dir.path()).unwrap(), DecisionMode::Laya);
        assert_eq!(available_mode(desired, true, false), DecisionMode::Laya);

        // Only an explicit normal-harness choice changes the preference.
        save_mode(dir.path(), DecisionMode::Normal).unwrap();
        assert_eq!(load_mode(dir.path()).unwrap(), DecisionMode::Normal);
    }
}
