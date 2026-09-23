# Local connections

| Tool | Detection and account state | Session |
| --- | --- | --- |
| Codex | Installed CLI and its local authentication are checked when Accounts opens. | Keel starts the installed Codex ACP adapter. The Codex CLI keeps its own model and credentials. |
| Cursor | The installed Cursor CLI is detected locally. Its own CLI manages authentication. | Keel starts Cursor's ACP adapter when selected. |
| Claude Code | The installed CLI and locally confirmed account status are shown. | Keel starts the installed Claude ACP adapter when selected. |
| Other ACP agents | Keel probes the local executable; unconfirmed account state is shown as CLI managed. | Each agent retains its own model and tool loop. |

Keel does not collect provider keys, exchange OAuth tokens, or sync coding sessions to a Keel cloud. The legacy cloud-connect crate remains source-only for a later release; its hosted API controls are absent from this build.

When Codex's local Computer Use plugin is enabled, its MCP server is supplied to compatible ACP sessions. The plugin service and permissions belong to the installed Codex app. [ACP session setup](https://github.com/agentclientprotocol/agent-client-protocol/blob/main/docs/protocol/v1/session-setup.mdx) describes the stdio MCP transport. Keel confirms plugin files and starts a session with the server; an agent's ability to use a particular desktop control still depends on that agent and model.
