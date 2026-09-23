# developer guide

[Documentation](README.md) / development

## repository layout

| Path | Responsibility |
| --- | --- |
| `apps/keel/` | Desktop app entry point and app CLI. Default Cargo member. |
| `crates/` | Libraries needed by the app: UI, engine, document state, protocols, adapters, selectors, and updates. |
| `crates/dsh/` | The embedded DeepSeek loop and its required support libraries. |
| `extras/dsh/` | Nine imported support libraries outside Keel's dependency graph. |
| `tools/dsh-cli/` | Optional standalone `dsh` executable. |
| `tools/diagnostics/` | Manually invoked probes, fixture generators, and recovery tools. |
| `tools/design/` | Icon source images and the Swift drawing script. |
| `tools/docs/` | Documentation diagram generator. |
| `tools/upstream/` | Optional upstream reference checkout helper. |
| `scripts/` | App and worker build/package entry points. |
| `assets/` | The compiled macOS app icon used by the packager. UI runtime assets stay with `crates/ui`. |
| `licenses/` | Retained third-party licenses and notices included in app packages. |
| `docs/` | Current guides; `archive/` holds dated records and `proposals/` holds future designs. |

Cargo's local dependency graph contains 31 packages reachable from `keel`.
The nine extra libraries and the standalone CLI remain workspace members so
their tests can still run. They are not dependencies of the desktop app.
Tests and fixtures stay beside the libraries they verify.

The reorganization preserves all Rust source and retained license texts. The eight Rust
diagnostics moved out of crate `examples/` directories, with explicit Cargo
example paths preserving their existing command names.

## build and verify

Run from the repository root:

```sh
cargo build --locked
cargo fmt --all -- --check
cargo check --workspace --all-targets --locked
cargo test --workspace --locked -- --test-threads=1
```

Plain `cargo build` and `cargo run` select Keel. `-p <package>` selects an
optional component, and `--workspace` includes every maintained package.
Provider tests marked ignored need their documented live environment; ordinary
workspace tests do not enable them.

See [tools](../tools/README.md) for diagnostic commands and
[extras](../extras/README.md) for the optional library list.

## local and generated files

`target/`, `build/`, `dist/`, and `.refs/` are ignored. They contain compiler
outputs, downloaded models and worker builds, packaged apps, and optional
upstream checkouts. Keep those files out of source commits. The lockfile and
the app's source assets remain tracked.

## runtime and distribution

Laya-CoreML is the default local decision layer when its worker and checkpoint
are available. The ordinary coding harness remains available when they are
not. The selected coding CLI owns its own model, tools, and authentication.
See [decision architecture](decision-architecture.md) for the supported scope.

Follow [build from source](build.md) to supply the Laya worker and optional
checkpoint to `scripts/package-macos.sh`. The app bundle retains `LICENSE`,
`THIRD_PARTY_NOTICES.md`, and the complete `licenses/` directory.
