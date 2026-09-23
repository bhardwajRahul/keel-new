//! Narrow cloud adapters. Connecting a key performs read-only requests; only
//! `create_*` methods start paid work, so they must be wired to a user action.
//!
//! Cursor Cloud Agents API: https://cursor.com/docs/cloud-agent/api/endpoints
//! OpenAI Agents API: https://developers.openai.com/api/docs/guides/agents-api/quickstart
//! The OpenAI Agents API is a managed Codex harness, not the ChatGPT Codex
//! cloud task service. Neither provider exposes an OAuth flow for this app here.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use reqwest::{Client, Method, StatusCode};
use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;

#[derive(Debug, thiserror::Error)]
pub enum CloudError {
    #[error("invalid input")]
    InvalidInput,
    #[error("credential storage path is not a private app-owned directory")]
    UnsafeStorage,
    #[error("credential storage failed")]
    Storage,
    #[error("cloud request failed")]
    Transport,
    #[error("cloud response was invalid")]
    InvalidResponse,
    #[error("API key was rejected")]
    Unauthorized,
    #[error("API key lacks required access")]
    Forbidden,
    #[error("cloud API rate limit reached")]
    RateLimited,
    #[error("cloud API returned HTTP {0}")]
    Http(u16),
    #[error("{0}")]
    Unsupported(&'static str),
}

pub type Result<T> = std::result::Result<T, CloudError>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Provider {
    Cursor,
    OpenAiAgents,
}

impl Provider {
    fn file_name(self) -> &'static str {
        match self {
            Self::Cursor => "cursor.key",
            Self::OpenAiAgents => "openai-agents.key",
        }
    }
}

/// Stores each provider key in a separate 0600 file beneath a 0700 directory.
/// Callers should use their normal app-data directory, outside any repository.
pub struct KeyStore {
    dir: PathBuf,
}

impl KeyStore {
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        if !dir.is_absolute() {
            return Err(CloudError::UnsafeStorage);
        }
        check_ancestors(&dir)?;
        fs::create_dir_all(&dir).map_err(|_| CloudError::Storage)?;
        check_ancestors(&dir)?;
        let meta = fs::symlink_metadata(&dir).map_err(|_| CloudError::Storage)?;
        if !meta.is_dir() || meta.file_type().is_symlink() {
            return Err(CloudError::UnsafeStorage);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::{MetadataExt, PermissionsExt};
            if meta.uid() != unsafe { libc::geteuid() } {
                return Err(CloudError::UnsafeStorage);
            }
            fs::set_permissions(&dir, fs::Permissions::from_mode(0o700))
                .map_err(|_| CloudError::Storage)?;
        }
        Ok(Self { dir })
    }

    pub fn save(&self, provider: Provider, key: &str) -> Result<()> {
        if key.trim().is_empty() || key.chars().any(char::is_control) {
            return Err(CloudError::InvalidInput);
        }
        self.check_dir()?;
        let path = self.dir.join(provider.file_name());
        reject_existing_symlink(&path)?;
        let mut temp = NamedTempFile::new_in(&self.dir).map_err(|_| CloudError::Storage)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            temp.as_file()
                .set_permissions(fs::Permissions::from_mode(0o600))
                .map_err(|_| CloudError::Storage)?;
        }
        temp.write_all(key.as_bytes())
            .map_err(|_| CloudError::Storage)?;
        temp.as_file().sync_all().map_err(|_| CloudError::Storage)?;
        reject_existing_symlink(&path)?;
        temp.persist(&path).map_err(|_| CloudError::Storage)?;
        File::open(&self.dir)
            .and_then(|f| f.sync_all())
            .map_err(|_| CloudError::Storage)?;
        Ok(())
    }

    pub fn load(&self, provider: Provider) -> Result<Option<String>> {
        self.check_dir()?;
        let path = self.dir.join(provider.file_name());
        let mut opts = OpenOptions::new();
        opts.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.custom_flags(libc::O_NOFOLLOW);
        }
        let mut file = match opts.open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(_) => return Err(CloudError::UnsafeStorage),
        };
        let meta = file.metadata().map_err(|_| CloudError::Storage)?;
        if !meta.is_file() {
            return Err(CloudError::UnsafeStorage);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::{MetadataExt, PermissionsExt};
            if meta.uid() != unsafe { libc::geteuid() } || meta.permissions().mode() & 0o077 != 0 {
                return Err(CloudError::UnsafeStorage);
            }
        }
        let mut key = String::new();
        file.read_to_string(&mut key)
            .map_err(|_| CloudError::Storage)?;
        Ok(Some(key))
    }

    pub fn remove(&self, provider: Provider) -> Result<()> {
        self.check_dir()?;
        let path = self.dir.join(provider.file_name());
        reject_existing_symlink(&path)?;
        match fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(_) => Err(CloudError::Storage),
        }
    }

    fn check_dir(&self) -> Result<()> {
        check_ancestors(&self.dir)?;
        let meta = fs::symlink_metadata(&self.dir).map_err(|_| CloudError::Storage)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::{MetadataExt, PermissionsExt};
            if meta.uid() != unsafe { libc::geteuid() } || meta.permissions().mode() & 0o077 != 0 {
                return Err(CloudError::UnsafeStorage);
            }
        }
        if !meta.is_dir() || meta.file_type().is_symlink() {
            return Err(CloudError::UnsafeStorage);
        }
        Ok(())
    }
}

fn check_ancestors(path: &Path) -> Result<()> {
    for ancestor in path.ancestors() {
        match fs::symlink_metadata(ancestor) {
            Ok(meta) if meta.file_type().is_symlink() => {
                // macOS uses root-owned /var -> /private/var for temporary files.
                // User-owned links can redirect credentials to another location.
                #[cfg(unix)]
                {
                    use std::os::unix::fs::MetadataExt;
                    if ancestor != path && meta.uid() == 0 {
                        continue;
                    }
                }
                return Err(CloudError::UnsafeStorage);
            }
            Ok(meta) if !meta.is_dir() => return Err(CloudError::UnsafeStorage),
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => return Err(CloudError::Storage),
        }
    }
    Ok(())
}

fn reject_existing_symlink(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_symlink() || !meta.is_file() => {
            Err(CloudError::UnsafeStorage)
        }
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(_) => Err(CloudError::Storage),
    }
}

#[derive(Clone)]
pub struct CloudClient {
    http: Client,
    cursor_base: String,
    openai_base: String,
}

impl CloudClient {
    pub fn new() -> Result<Self> {
        let http = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|_| CloudError::Transport)?;
        Ok(Self {
            http,
            cursor_base: "https://api.cursor.com".into(),
            openai_base: "https://api.openai.com".into(),
        })
    }

    #[cfg(test)]
    fn for_test(base: String) -> Self {
        Self {
            http: Client::new(),
            cursor_base: base.clone(),
            openai_base: base,
        }
    }

    pub async fn cursor_validate(&self, key: &str) -> Result<CursorIdentity> {
        self.cursor_request(Method::GET, key, "/v1/me", None::<&()>)
            .await
    }

    pub async fn cursor_models(&self, key: &str) -> Result<Vec<CursorModel>> {
        let page: Items<CursorModel> = self
            .cursor_request(Method::GET, key, "/v1/models", None::<&()>)
            .await?;
        Ok(page.items)
    }

    /// Fetch only on a deliberate repository-picker open. Cursor documents a
    /// strict limit of 1 request per user per minute and 30 per hour.
    pub async fn cursor_repositories(&self, key: &str) -> Result<Vec<CursorRepository>> {
        let page: Items<CursorRepository> = self
            .cursor_request(Method::GET, key, "/v1/repositories", None::<&()>)
            .await?;
        Ok(page.items)
    }

    /// Starts a new Cursor Cloud Agent and its first run. Call only after an
    /// explicit user submit action; validation and onboarding never call this.
    pub async fn cursor_create_agent(
        &self,
        key: &str,
        request: &CursorCreateRequest,
    ) -> Result<CursorLaunch> {
        if request.prompt.trim().is_empty() || request.repository_url.trim().is_empty() {
            return Err(CloudError::InvalidInput);
        }
        let body = CursorCreateBody {
            prompt: Prompt {
                text: &request.prompt,
            },
            repos: [Repo {
                url: &request.repository_url,
                starting_ref: request.starting_ref.as_deref(),
            }],
            model: request.model_id.as_ref().map(|id| ModelChoice { id }),
            auto_create_pr: false,
        };
        let wire: CursorLaunchWire = self
            .cursor_request(Method::POST, key, "/v1/agents", Some(&body))
            .await?;
        Ok(CursorLaunch {
            agent_id: wire.agent.id,
            run_id: wire.run.id,
            status: wire.run.status,
        })
    }

    pub async fn cursor_get_run(
        &self,
        key: &str,
        agent_id: &str,
        run_id: &str,
    ) -> Result<CursorRun> {
        safe_id(agent_id)?;
        safe_id(run_id)?;
        self.cursor_request(
            Method::GET,
            key,
            &format!("/v1/agents/{agent_id}/runs/{run_id}"),
            None::<&()>,
        )
        .await
    }

    /// Read-only scope check for the OpenAI Agents API. Starting a session
    /// additionally requires `api.agents.write` and `api.responses.write`.
    pub async fn openai_validate_read_access(&self, key: &str) -> Result<()> {
        let _: serde_json::Value = self
            .openai_request(Method::GET, key, "/v1/agents/sessions?limit=1", None::<&()>)
            .await?;
        Ok(())
    }

    /// Starts an OpenAI-hosted managed Codex session in a fresh sandbox. This
    /// does not attach a Git repository or connect to ChatGPT Codex tasks.
    pub async fn openai_create_session(
        &self,
        key: &str,
        request: &OpenAiCreateRequest,
    ) -> Result<OpenAiSession> {
        if request.model.trim().is_empty() || request.input.trim().is_empty() {
            return Err(CloudError::InvalidInput);
        }
        let body = serde_json::json!({
            "agent": { "model": request.model, "instructions": request.instructions },
            "environment": { "type": "openai_hosted" },
            "input": request.input,
            "stream": false
        });
        self.openai_request(Method::POST, key, "/v1/agents/sessions", Some(&body))
            .await
    }

    pub async fn openai_get_session(&self, key: &str, session_id: &str) -> Result<OpenAiSession> {
        safe_id(session_id)?;
        self.openai_request(
            Method::GET,
            key,
            &format!("/v1/agents/sessions/{session_id}"),
            None::<&()>,
        )
        .await
    }

    async fn cursor_request<T: for<'de> Deserialize<'de>, B: Serialize + ?Sized>(
        &self,
        method: Method,
        key: &str,
        path: &str,
        body: Option<&B>,
    ) -> Result<T> {
        check_key(key)?;
        let mut req = self
            .http
            .request(method, format!("{}{path}", self.cursor_base))
            .basic_auth(key, Some(""));
        if let Some(body) = body {
            req = req.json(body);
        }
        let response = req.send().await.map_err(|_| CloudError::Transport)?;
        decode(response).await
    }

    async fn openai_request<T: for<'de> Deserialize<'de>, B: Serialize + ?Sized>(
        &self,
        method: Method,
        key: &str,
        path: &str,
        body: Option<&B>,
    ) -> Result<T> {
        check_key(key)?;
        let mut req = self
            .http
            .request(method, format!("{}{path}", self.openai_base))
            .bearer_auth(key)
            .header("OpenAI-Beta", "agents=v1");
        if let Some(body) = body {
            req = req.json(body);
        }
        let response = req.send().await.map_err(|_| CloudError::Transport)?;
        decode(response).await
    }
}

fn check_key(key: &str) -> Result<()> {
    if key.trim().is_empty() || key.chars().any(char::is_control) {
        Err(CloudError::InvalidInput)
    } else {
        Ok(())
    }
}

fn safe_id(id: &str) -> Result<()> {
    if id.is_empty()
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
    {
        Err(CloudError::InvalidInput)
    } else {
        Ok(())
    }
}

async fn decode<T: for<'de> Deserialize<'de>>(response: reqwest::Response) -> Result<T> {
    match response.status() {
        StatusCode::UNAUTHORIZED => return Err(CloudError::Unauthorized),
        StatusCode::FORBIDDEN => return Err(CloudError::Forbidden),
        StatusCode::TOO_MANY_REQUESTS => return Err(CloudError::RateLimited),
        status if !status.is_success() => return Err(CloudError::Http(status.as_u16())),
        _ => {}
    }
    response
        .json()
        .await
        .map_err(|_| CloudError::InvalidResponse)
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CursorIdentity {
    pub api_key_name: String,
    pub user_email: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CursorModel {
    pub id: String,
    pub display_name: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CursorRepository {
    pub url: String,
}

#[derive(Deserialize)]
struct Items<T> {
    items: Vec<T>,
}

#[derive(Clone)]
pub struct CursorCreateRequest {
    pub prompt: String,
    pub repository_url: String,
    pub starting_ref: Option<String>,
    pub model_id: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CursorCreateBody<'a> {
    prompt: Prompt<'a>,
    repos: [Repo<'a>; 1],
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<ModelChoice<'a>>,
    auto_create_pr: bool,
}

#[derive(Serialize)]
struct Prompt<'a> {
    text: &'a str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Repo<'a> {
    url: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    starting_ref: Option<&'a str>,
}

#[derive(Serialize)]
struct ModelChoice<'a> {
    id: &'a str,
}

#[derive(Deserialize)]
struct CursorLaunchWire {
    agent: AgentId,
    run: CursorRun,
}

#[derive(Deserialize)]
struct AgentId {
    id: String,
}

#[derive(Clone, Debug)]
pub struct CursorLaunch {
    pub agent_id: String,
    pub run_id: String,
    pub status: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CursorRun {
    pub id: String,
    pub agent_id: String,
    pub status: String,
}

#[derive(Clone)]
pub struct OpenAiCreateRequest {
    pub model: String,
    pub instructions: String,
    pub input: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct OpenAiSession {
    pub id: String,
    pub status: String,
}

/// ChatGPT Codex Cloud task linking is a separate product surface; there is no
/// documented third-party OAuth/API-key task bridge in the APIs above.
pub fn chatgpt_codex_task_link_capability() -> Result<()> {
    Err(CloudError::Unsupported(
        "ChatGPT Codex Cloud task linking is not available through the documented Agents API",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn private_key_files_and_symlink_rejection() {
        let parent = tempfile::tempdir().unwrap();
        let store = KeyStore::new(parent.path().join("keys")).unwrap();
        store.save(Provider::Cursor, "private-cursor-key").unwrap();
        assert_eq!(
            store.load(Provider::Cursor).unwrap().as_deref(),
            Some("private-cursor-key")
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::{PermissionsExt, symlink};
            let path = parent.path().join("keys/cursor.key");
            assert_eq!(
                fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
            fs::remove_file(&path).unwrap();
            symlink(parent.path().join("elsewhere"), &path).unwrap();
            assert!(matches!(
                store.load(Provider::Cursor),
                Err(CloudError::UnsafeStorage)
            ));
            assert!(matches!(
                store.save(Provider::Cursor, "replacement"),
                Err(CloudError::UnsafeStorage)
            ));
            symlink(
                parent.path().join("keys"),
                parent.path().join("linked-keys"),
            )
            .unwrap();
            assert!(matches!(
                KeyStore::new(parent.path().join("linked-keys")),
                Err(CloudError::UnsafeStorage)
            ));
        }
    }

    #[tokio::test]
    async fn cursor_validation_is_read_only_and_creation_is_explicit() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            for (expected, body) in [
                ("GET /v1/me", r#"{"apiKeyName":"test"}"#),
                (
                    "GET /v1/models",
                    r#"{"items":[{"id":"composer-2","displayName":"Composer 2"}]}"#,
                ),
                (
                    "POST /v1/agents",
                    r#"{"agent":{"id":"bc-1"},"run":{"id":"run-1","agentId":"bc-1","status":"CREATING"}}"#,
                ),
                (
                    "GET /v1/agents/bc-1/runs/run-1",
                    r#"{"id":"run-1","agentId":"bc-1","status":"FINISHED"}"#,
                ),
            ] {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut buf = [0u8; 8192];
                let size = socket.read(&mut buf).await.unwrap();
                let request = String::from_utf8_lossy(&buf[..size]);
                assert!(request.starts_with(expected));
                assert!(
                    request
                        .to_ascii_lowercase()
                        .contains("authorization: basic ")
                );
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            }
        });
        let client = CloudClient::for_test(base);
        assert_eq!(
            client.cursor_validate("secret").await.unwrap().api_key_name,
            "test"
        );
        assert_eq!(
            client.cursor_models("secret").await.unwrap()[0].id,
            "composer-2"
        );
        let launch = client
            .cursor_create_agent(
                "secret",
                &CursorCreateRequest {
                    prompt: "Fix tests".into(),
                    repository_url: "https://github.com/example/repo".into(),
                    starting_ref: None,
                    model_id: Some("composer-2".into()),
                },
            )
            .await
            .unwrap();
        assert_eq!(launch.agent_id, "bc-1");
        assert_eq!(
            client
                .cursor_get_run("secret", &launch.agent_id, &launch.run_id)
                .await
                .unwrap()
                .status,
            "FINISHED"
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn openai_scope_probe_does_not_start_a_session() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 8192];
            let size = socket.read(&mut buf).await.unwrap();
            let request = String::from_utf8_lossy(&buf[..size]);
            assert!(request.starts_with("GET /v1/agents/sessions?limit=1"));
            assert!(
                request
                    .to_ascii_lowercase()
                    .contains("openai-beta: agents=v1")
            );
            assert!(
                request
                    .to_ascii_lowercase()
                    .contains("authorization: bearer ")
            );
            let body = "{}";
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });
        CloudClient::for_test(base)
            .openai_validate_read_access("secret")
            .await
            .unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn openai_session_creation_requires_a_separate_call() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 8192];
            let size = socket.read(&mut buf).await.unwrap();
            let request = String::from_utf8_lossy(&buf[..size]);
            assert!(request.starts_with("POST /v1/agents/sessions"));
            assert!(request.contains("\"openai_hosted\""));
            let body = r#"{"id":"sess_1","status":"in_progress"}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });
        let session = CloudClient::for_test(base)
            .openai_create_session(
                "secret",
                &OpenAiCreateRequest {
                    model: "gpt-6-astra".into(),
                    instructions: "Write clean code".into(),
                    input: "Create a script".into(),
                },
            )
            .await
            .unwrap();
        assert_eq!(session.id, "sess_1");
        server.await.unwrap();
    }
}
