# Keel 0.2.0 build and verification

23 September 2026. Apple Silicon macOS 15+ local build. The installed app is `~/Applications/Keel.app`. Both app ZIPs contain the same 0.2.0 code: the full ZIP bundles the pinned Laya model; the light ZIP offers the model download in onboarding and Settings.

## Verified

- `cargo fmt --all -- --check` passed. The serial workspace suite passed **1,477 tests across 153 targets**, with 18 live-provider tests intentionally ignored. That run began before the final review fixes; the final `keel-ui` library suite then passed **444 tests**. The focused stale-engine state suite passed **40 tests**. The release build completed after these fixes.
- The local-only sync prompt was caused by a synthetic signed-in state being mistaken for cloud sign-in. Both UI sync-state paths now require a configured WorkOS client. Startup also rejects an IPC engine with a different persisted device ID, and local-only startup rejects a synced engine on the same ID. Same-installation reconnect and rejection cases have focused tests.
- Settings → Onboarding reopens the Laya, Jev, and normal harness choices. Settings → Agents refreshes the saved choice when onboarding changes it. Laya download progress and completed ownership synchronize between the two pages. Focused UI tests pass.
- Both app ZIPs and the source ZIP passed archive integrity checks through the links in this task's outputs. Both app ZIPs were extracted; the full `Info.plist` passed `plutil -lint`, and `codesign --verify --deep --strict` passed for both extracted apps and the installed app. The installed bundle and workspace report **0.2.0**; the app launched and the Devices page showed `v0.2.0`. Both packages contain a worker advertising `download-model-v1 download-progress-v1`.
- The full package contains the pinned Core ML checkpoint. Packaging checks SHA-256 for all eight runtime inputs. A previous full-package worker probe returned typed decisions for two prompts; this final run checked the worker capabilities and package hashes, and did not repeat model inference. A previous live light-package download emitted monotonic byte progress and matched the same pinned hashes.
- Old Keel QA app bundles, obsolete source trees, and the duplicate source archive were removed. The original Avid app and the user's app data were not changed. A source scan found no legacy brand references or credential-shaped files.

## Local integration status

| Integration | Observed result |
| --- | --- |
| Claude Code, Codex, Grok ACP | Local initialize, session creation, and a safe prompt completed in earlier disposable probes. |
| Cursor ACP | CLI initialized; session creation requested local authentication. |
| Pi ACP | Version-aware adapter reaches session creation; local Pi authentication is absent. |
| Hermes ACP | Installed launcher points to a missing executable; Keel marks it unavailable. |
| Embedded DeepSeek Harness | In-process bridge and resume tests pass; picker stays unavailable without a local DeepSeek credential. |
| TypeSafe Jev | Optional direct route uses an existing protected local credential; no key-entry UI. This release check did not make a paid live TypeSafe call. |
| Computer use | Installed Codex Computer Use MCP is attached where an ACP adapter accepts it. External agents still own their internal tool loops. |
| Voice | macOS Dictation enters text into the composer. Native Codex realtime voice is not in this build. |

## Release limits

- This is an ad hoc signed local development build. Public distribution still requires a Developer ID signature and Apple notarization; Gatekeeper will reject it as a public installer.
- Cloud sync is disabled in this local-only build. The false sync-restart wizard is now hidden. Local coding-tool discovery and the same-device engine reconnect remain available.
- The final installed start screen, Settings → Onboarding, and Devices `v0.2.0` row were visually inspected after the Mac was unlocked. A shell launch left the window on its loading view while it was in the background; activating the window advanced to the intro and workspace.
- Documents immediately evicted newly created large ZIPs into data placeholders on this Mac. To keep the deliverables readable, the three ZIP links in `outputs` point to verified local files in `~/Downloads/Keel-0.2.0`. The installed app remains in `~/Applications/Keel.app`.
