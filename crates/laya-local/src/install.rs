use sha2::{Digest, Sha256};
#[cfg(not(unix))]
use std::io::Write;
use std::{
    fmt,
    fs::{self, OpenOptions},
    io::{self, Read},
    path::{Path, PathBuf},
    process::Stdio,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    process::Command,
    sync::mpsc::UnboundedSender,
};

pub const MODEL_REVISION: &str = "8139e9089273319512c730218903784074133187";
const INSTALL_MARKER: &str = ".keel-laya-revision";
const PROTOCOL_PREFIX: &str = "KEEL_LAYA_JSON:";
const MODEL_BYTES: u64 = 679_871_580;
const MAX_PROGRESS_LINE_BYTES: usize = 1024;

// The snapshot is immutable at MODEL_REVISION. Checking every runtime input,
// including the model weights, makes a completed install independent of the
// downloader's cache metadata.
const MODEL_FILES: &[(&str, &str)] = &[
    (
        "model.mlpackage/Manifest.json",
        "9f5ae62247c9be221c7ced17157b1b5ec25ef70bdd3cb809d45ba698da579418",
    ),
    (
        "model.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
        "6b906ab7b6b8bbc0f11608425f6091b508bbe968788f27d8e81df25399c281ba",
    ),
    (
        "model.mlpackage/Data/com.apple.CoreML/model.mlmodel",
        "d6ea5688f45c6afa3c041698d33337ac9c62de7b3f1077c9cb39be7c965fe0de",
    ),
    (
        "encoder/config.json",
        "83f6916d13ef0f556ac461f28308dc2bffa7ebeadee8ec9e2db5812020ea5bb4",
    ),
    (
        "tokenizer/tokenizer_config.json",
        "6c6b2d8e3c84ce0e671c129cd6b374b235d6f9863042a5836358d00a89bbb5a1",
    ),
    (
        "tokenizer/tokenizer.json",
        "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
    ),
    (
        "coreml_config.json",
        "8131967fdb403243f2817d7eeb09721d1e822bd7301c244d25afb44bcc258352",
    ),
    (
        "rl_agent_config.json",
        "25061739243b617ad88d1219ba6f8a9c86c5881ca28df024fa2d9b3b2fcc30c6",
    ),
];

#[derive(Debug)]
pub enum InstallError {
    Busy,
    WorkerUnavailable,
    DownloadFailed,
    InvalidModel,
    Io(io::Error),
    TaskFailed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InstallProgress {
    Downloading {
        downloaded_bytes: u64,
        total_bytes: u64,
    },
    Verifying,
}

impl fmt::Display for InstallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Busy => "a Laya model install is already running",
            Self::WorkerUnavailable => "the bundled Laya worker is unavailable",
            Self::DownloadFailed => "the Laya model download failed",
            Self::InvalidModel => "the downloaded Laya model failed integrity validation",
            Self::Io(_) => "the Laya model could not be installed",
            Self::TaskFailed => "the Laya model validation task failed",
        };
        f.write_str(message)
    }
}

impl std::error::Error for InstallError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for InstallError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Download and atomically activate the pinned Laya checkpoint.
///
/// This must only be called after an explicit user action. The standalone
/// worker performs the network transfer; inference never accepts a remote ID.
pub async fn install_model(
    data_dir: &Path,
    worker_executable: &Path,
) -> Result<PathBuf, InstallError> {
    install_model_inner(data_dir, worker_executable, None).await
}

pub async fn install_model_with_progress(
    data_dir: &Path,
    worker_executable: &Path,
    progress: UnboundedSender<InstallProgress>,
) -> Result<PathBuf, InstallError> {
    install_model_inner(data_dir, worker_executable, Some(progress)).await
}

async fn install_model_inner(
    data_dir: &Path,
    worker_executable: &Path,
    progress: Option<UnboundedSender<InstallProgress>>,
) -> Result<PathBuf, InstallError> {
    if !is_executable_file(worker_executable) {
        return Err(InstallError::WorkerUnavailable);
    }

    let laya_dir = data_dir.join("laya");
    fs::create_dir_all(&laya_dir)?;
    let target = laya_dir.join("model");
    if model_is_installed(&target) {
        return Ok(target);
    }

    let lock_path = laya_dir.join(".install.lock");
    let _lock = InstallLock::acquire(&lock_path)?;
    if model_is_installed(&target) {
        return Ok(target);
    }

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let staging = laya_dir.join(format!(".model.download-{}-{nonce}", std::process::id()));
    let mut cleanup = RemoveOnDrop(staging.clone());

    let mut child = Command::new(worker_executable)
        .arg("--download-model")
        .arg("--output")
        .arg(&staging)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .map_err(|_| InstallError::WorkerUnavailable)?;
    let stdout = child.stdout.take().ok_or(InstallError::WorkerUnavailable)?;
    let mut lines = BufReader::new(stdout).lines();
    let mut last_downloaded = 0;
    while let Some(line) = lines.next_line().await.map_err(InstallError::Io)? {
        if line.len() > MAX_PROGRESS_LINE_BYTES {
            return Err(InstallError::DownloadFailed);
        }
        let Some(update) = parse_download_progress(&line, last_downloaded) else {
            continue;
        };
        last_downloaded = update;
        if let Some(progress) = &progress {
            let _ = progress.send(InstallProgress::Downloading {
                downloaded_bytes: update,
                total_bytes: MODEL_BYTES,
            });
        }
    }
    if !child.wait().await.map_err(InstallError::Io)?.success() {
        return Err(InstallError::DownloadFailed);
    }
    if last_downloaded != MODEL_BYTES {
        return Err(InstallError::DownloadFailed);
    }
    let cache = staging.join(".cache");
    if cache.exists() {
        fs::remove_dir_all(cache)?;
    }

    let validated = staging.clone();
    if let Some(progress) = &progress {
        let _ = progress.send(InstallProgress::Verifying);
    }
    tokio::task::spawn_blocking(move || validate_model(&validated))
        .await
        .map_err(|_| InstallError::TaskFailed)??;
    fs::write(staging.join(INSTALL_MARKER), format!("{MODEL_REVISION}\n"))?;

    let previous = laya_dir.join(format!(".model.previous-{}-{nonce}", std::process::id()));
    if target.exists() {
        fs::rename(&target, &previous)?;
    }
    if let Err(error) = fs::rename(&staging, &target) {
        if previous.exists() {
            let _ = fs::rename(&previous, &target);
        }
        return Err(InstallError::Io(error));
    }
    cleanup.0 = PathBuf::new();
    if previous.exists() {
        let _ = fs::remove_dir_all(previous);
    }
    Ok(target)
}

fn parse_download_progress(line: &str, previous: u64) -> Option<u64> {
    let payload = line.strip_prefix(PROTOCOL_PREFIX)?;
    let value: serde_json::Value = serde_json::from_str(payload).ok()?;
    let downloaded = value.get("downloaded_bytes")?.as_u64()?;
    let total = value.get("total_bytes")?.as_u64()?;
    (total == MODEL_BYTES && downloaded >= previous && downloaded <= total).then_some(downloaded)
}

/// Cheap startup check for a model that already passed full hash validation.
pub fn model_is_installed(model_dir: &Path) -> bool {
    fs::read_to_string(model_dir.join(INSTALL_MARKER))
        .is_ok_and(|value| value.trim() == MODEL_REVISION)
        && MODEL_FILES
            .iter()
            .all(|(relative, _)| regular_file(&model_dir.join(relative)))
}

fn validate_model(model_dir: &Path) -> Result<(), InstallError> {
    for (relative, expected) in MODEL_FILES {
        let path = model_dir.join(relative);
        let matches = sha256(&path).is_ok_and(|actual| actual == *expected);
        if !regular_file(&path) || !matches {
            return Err(InstallError::InvalidModel);
        }
    }
    Ok(())
}

fn regular_file(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok_and(|metadata| metadata.file_type().is_file())
}

fn is_executable_file(path: &Path) -> bool {
    if !regular_file(path) {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::metadata(path).is_ok_and(|metadata| metadata.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    {
        true
    }
}

fn sha256(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

struct InstallLock {
    _file: fs::File,
    #[cfg(not(unix))]
    path: PathBuf,
}

impl InstallLock {
    fn acquire(path: &Path) -> Result<Self, InstallError> {
        #[cfg(unix)]
        {
            use std::os::{fd::AsRawFd, unix::fs::OpenOptionsExt};

            // Keep the inode in place. An advisory lock is released by the OS
            // after a crash, and removing a live lock file could let another
            // process create a second inode and install concurrently.
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .custom_flags(libc::O_NOFOLLOW)
                .open(path)?;
            let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
            if result != 0 {
                let error = io::Error::last_os_error();
                return if error.kind() == io::ErrorKind::WouldBlock {
                    Err(InstallError::Busy)
                } else {
                    Err(InstallError::Io(error))
                };
            }
            return Ok(Self { _file: file });
        }

        #[cfg(not(unix))]
        match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(mut file) => {
                writeln!(file, "{}", std::process::id())?;
                Ok(Self {
                    _file: file,
                    path: path.to_owned(),
                })
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => Err(InstallError::Busy),
            Err(error) => Err(InstallError::Io(error)),
        }
    }
}

impl Drop for InstallLock {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            let _ = unsafe { libc::flock(self._file.as_raw_fd(), libc::LOCK_UN) };
        }
        #[cfg(not(unix))]
        let _ = fs::remove_file(&self.path);
    }
}

struct RemoveOnDrop(PathBuf);

impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        if !self.0.as_os_str().is_empty() {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn install_lock_recovers_after_crash_and_excludes_concurrent_installs() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join(".install.lock");
        fs::write(&path, b"stale pid from prior process\n").unwrap();
        let lock = InstallLock::acquire(&path).unwrap();
        assert!(matches!(
            InstallLock::acquire(&path),
            Err(InstallError::Busy)
        ));
        drop(lock);
        let recovered = InstallLock::acquire(&path).unwrap();
        drop(recovered);
    }

    #[test]
    fn installed_model_requires_pin_marker_and_regular_files() {
        let temp = tempfile::tempdir().unwrap();
        for (relative, _) in MODEL_FILES {
            let path = temp.path().join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, b"fixture").unwrap();
        }
        assert!(!model_is_installed(temp.path()));
        fs::write(temp.path().join(INSTALL_MARKER), MODEL_REVISION).unwrap();
        assert!(model_is_installed(temp.path()));
        fs::remove_file(temp.path().join(MODEL_FILES[0].0)).unwrap();
        assert!(!model_is_installed(temp.path()));
    }

    #[tokio::test]
    async fn missing_worker_fails_before_creating_an_install() {
        let temp = tempfile::tempdir().unwrap();
        let error = install_model(temp.path(), Path::new("/missing/laya-worker"))
            .await
            .unwrap_err();
        assert!(matches!(error, InstallError::WorkerUnavailable));
        assert!(!temp.path().join("laya/model").exists());
    }

    #[test]
    fn progress_requires_the_pinned_total_and_monotonic_bytes() {
        let first = format!(
            "{PROTOCOL_PREFIX}{{\"downloaded_bytes\":1048576,\"total_bytes\":{MODEL_BYTES}}}"
        );
        assert_eq!(parse_download_progress(&first, 0), Some(1_048_576));
        assert_eq!(parse_download_progress(&first, 2_000_000), None);
        assert_eq!(
            parse_download_progress(
                &format!("{PROTOCOL_PREFIX}{{\"downloaded_bytes\":1,\"total_bytes\":123}}"),
                0
            ),
            None
        );
        assert_eq!(parse_download_progress("unframed noise", 0), None);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn streams_download_then_verification_from_a_worker() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempfile::tempdir().unwrap();
        let worker = temp.path().join("worker");
        fs::write(
            &worker,
            format!(
                "#!/bin/sh\nmkdir -p \"$3\"\nprintf 'KEEL_LAYA_JSON:{{\"downloaded_bytes\":0,\"total_bytes\":{MODEL_BYTES}}}\\n'\nprintf 'KEEL_LAYA_JSON:{{\"downloaded_bytes\":{MODEL_BYTES},\"total_bytes\":{MODEL_BYTES}}}\\n'\n"
            ),
        )
        .unwrap();
        fs::set_permissions(&worker, fs::Permissions::from_mode(0o700)).unwrap();
        let (send, mut receive) = tokio::sync::mpsc::unbounded_channel();
        let error = install_model_with_progress(temp.path(), &worker, send)
            .await
            .unwrap_err();
        assert!(matches!(error, InstallError::InvalidModel));
        assert_eq!(
            receive.recv().await,
            Some(InstallProgress::Downloading {
                downloaded_bytes: 0,
                total_bytes: MODEL_BYTES
            })
        );
        assert_eq!(
            receive.recv().await,
            Some(InstallProgress::Downloading {
                downloaded_bytes: MODEL_BYTES,
                total_bytes: MODEL_BYTES
            })
        );
        assert_eq!(receive.recv().await, Some(InstallProgress::Verifying));
    }
}
