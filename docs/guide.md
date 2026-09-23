# user guide

[Documentation](README.md) / guide

Keel is a macOS coding workspace. A decision selector can choose among the options the host prepares; your coding agent performs the task.

## before you start

Use Apple Silicon with macOS 15 or later for the packaged Laya model. [Build the app](build.md), or use a local development package you already have.

Install and authenticate the coding CLI you want to use. Keel reuses that provider's setup.

## 1. choose a decision mode

![Decision modes: Laya locally, Jev through TypeSafe, or Normal mode without a selector.](diagrams/decision-modes.svg)

- **Local Laya:** saved default. Requires the worker and verified model files.
- **Jev:** optional hosted selector. Requires an existing protected TypeSafe credential; this build has no key-entry screen.
- **Normal harness:** skips selector calls and uses the ordinary route.

Only one backend runs per decision. A selected preference and an available backend can differ while a model download is pending.

## 2. install Laya if needed

In the light package, choose **Download and activate Laya**. The interface shows downloaded bytes, percentage, and verification. The recorded checkpoint download is about 680 MB.

The full package already includes the model. It avoids this first download.

**Continuing setup does not clear the saved Laya preference.** The ordinary harness stays usable until Laya is ready. Choosing **Use normal harness** explicitly turns selection off.

You can reopen the flow in **Settings → Onboarding**, or download the model from **Settings → Agents**.

## 3. connect a coding provider

1. Open **Settings → Accounts** to refresh installed CLI status.
2. Open **Settings → Agents** to choose available adapters.
3. Select the provider and model for your task.
4. Use reasoning, plan, or fast controls where the connected agent advertises them.

Each provider keeps its own credentials and model configuration. An installed adapter can still need authentication when the session starts. See [local connections](connections.md).

## 4. start a task

Automatic route selection applies to **fresh, unpinned tasks**. Pinned routes and live or resumed sessions keep their route.

The chat shows a Laya or Jev activity label only while that selector runs. The decision record shows the result, host checks, fallback, and observed outcome.

Inside the embedded DeepSeek loop, the selector can choose a bounded tool focus. External ACP agents keep their own internal loops.

## terminal status and installation

Use the actual path to your app. For an installation under `~/Applications`:

```sh
"$HOME/Applications/Keel.app/Contents/MacOS/keel" laya status
"$HOME/Applications/Keel.app/Contents/MacOS/keel" laya install
"$HOME/Applications/Keel.app/Contents/MacOS/keel" status
```

`laya install` downloads, verifies, and selects the pinned model. Status reports local availability; it does not prove the selector ran on a task.

## common setup issues

| What you see | What to check |
| --- | --- |
| Laya is selected, but the ordinary harness is active | Model installation may still be pending. Check download and verification status. |
| Jev cannot be selected | The required protected local credential is unavailable. Keel has no key-entry screen. |
| A coding provider is missing | Confirm its CLI is installed and its adapter is enabled. |
| A provider asks you to sign in | Complete that provider's normal authentication flow. Keel does not supply a shared login. |
| A full package fails model verification | Use the pinned files expected by the packaging script; do not bypass the hash checks. |
| No selector label appears on an existing task | Pinned, live, and resumed routes are preserved. The activity label also appears only during a call. |

## computer use and voice

Compatible ACP sessions can receive tools from an installed, enabled Codex Computer Use MCP plugin. Keel does not bundle that service. Provider, plugin, and macOS permissions still apply.

`keel computer-use decide` chooses a host-prepared, low-risk action ID or abstains. **It does not execute desktop actions.**

Use macOS Dictation in the composer for voice text. Native Codex realtime voice and hosted cloud sync are not part of this build.

## know what was checked

The [build report](archive/build-report-0.2.0.md) records the September 2026 local checks, provider setup results, and packaging limits. It is historical evidence, not a guarantee about your machine.
