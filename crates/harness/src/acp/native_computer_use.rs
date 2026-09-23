//! Reuse the locally installed Codex computer-use MCP server in ACP sessions.
//! Keel does not ship or replace the Mac capture service; the user's Codex
//! installation owns its permissions, process, and update lifecycle.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};

pub(super) fn installed_server() -> Option<Value> {
    let home = std::env::var_os("CODEX_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".codex")))?;
    installed_server_in(&home)
}

fn installed_server_in(codex_home: &Path) -> Option<Value> {
    let config = std::fs::read_to_string(codex_home.join("config.toml")).ok()?;
    let section = config
        .split("[plugins.\"unified-computer-use@openai-bundled\"]")
        .nth(1)?
        .split('\n')
        .take_while(|line| !line.trim_start().starts_with('['))
        .collect::<Vec<_>>()
        .join("\n");
    if !section.lines().any(|line| line.trim() == "enabled = true") {
        return None;
    }

    let versions = codex_home.join("plugins/cache/openai-bundled/unified-computer-use");
    let mut manifests = std::fs::read_dir(versions)
        .ok()?
        .flatten()
        .map(|entry| entry.path().join(".mcp.json"))
        .filter(|path| path.is_file())
        .collect::<Vec<_>>();
    manifests.sort();
    for manifest in manifests.into_iter().rev() {
        let Ok(bytes) = std::fs::read(&manifest) else {
            continue;
        };
        let Ok(root) = serde_json::from_slice::<Value>(&bytes) else {
            continue;
        };
        let Some(server) = root.pointer("/mcpServers/cua_repl") else {
            continue;
        };
        if server.get("enabled").and_then(Value::as_bool) != Some(true) {
            continue;
        }
        let command = server.get("command")?.as_str()?;
        if !Path::new(command).is_absolute() || !Path::new(command).is_file() {
            continue;
        }
        let args = server.get("args")?.as_array()?;
        let args = args.iter().map(Value::as_str).collect::<Option<Vec<_>>>()?;
        if args.first().is_none_or(|path| !Path::new(path).is_file()) {
            continue;
        }
        let env = server.get("env")?.as_object()?;
        let env = env
            .iter()
            .filter_map(|(name, value)| {
                let value = value.as_str()?;
                let allowed = name == "CODEX_HOME"
                    || name == "SKY_CUA_SERVICE_PATH"
                    || name.starts_with("NODE_REPL_")
                    || name.starts_with("BROWSER_USE_")
                    || name.starts_with("CUA_REPL_");
                allowed.then(|| json!({"name": name, "value": value}))
            })
            .collect::<Vec<_>>();
        return Some(json!({
            "name": "cua_repl",
            "command": command,
            "args": args,
            "env": env,
        }));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_enabled_installed_plugin_is_shared() {
        let home = tempfile::tempdir().unwrap();
        let root = home.path();
        let plugin = root.join("plugins/cache/openai-bundled/unified-computer-use/1");
        std::fs::create_dir_all(&plugin).unwrap();
        let command = root.join("node");
        let script = root.join("server.mjs");
        std::fs::write(&command, "").unwrap();
        std::fs::write(&script, "").unwrap();
        std::fs::write(
            plugin.join(".mcp.json"),
            json!({"mcpServers":{"cua_repl":{
                "enabled":true,"command":command,"args":[script],
                "env":{"CODEX_HOME":root,"NODE_REPL_TRUSTED_SERVICES":"{}","SECRET":"no"}
            }}})
            .to_string(),
        )
        .unwrap();
        assert!(installed_server_in(root).is_none());
        std::fs::write(
            root.join("config.toml"),
            "[plugins.\"unified-computer-use@openai-bundled\"]\nenabled = true\n",
        )
        .unwrap();
        let server = installed_server_in(root).unwrap();
        assert_eq!(server["name"], "cua_repl");
        assert_eq!(server["env"].as_array().unwrap().len(), 2);
        std::fs::write(
            root.join("config.toml"),
            "[plugins.\"unified-computer-use@openai-bundled\"]\nenabled = false\n",
        )
        .unwrap();
        assert!(installed_server_in(root).is_none());
    }
}
