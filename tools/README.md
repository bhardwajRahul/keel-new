# developer tools

These programs support development and investigation. They are not included
in the Keel application bundle. Run the commands below from the repository root.

## standalone DeepSeek CLI

```sh
cargo run -p dsh-cli -- --help
```

The package lives in `dsh-cli/` and keeps the `dsh` binary name. Running a coding
task requires its provider configuration; displaying help does not.

## Rust diagnostics

The source files live in `diagnostics/<owning-crate>/`. Explicit `[[example]]`
entries in the owning manifests preserve existing Cargo commands and keep
these programs covered by `cargo check --workspace --all-targets`.

| Package | Example | Purpose |
| --- | --- | --- |
| `keel-doc` | `gen_fixture` | Write a document fixture to a supplied path. |
| `keel-doc` | `rebuild_whale` | Inspect a supplied snapshot and report rebuild accounting. |
| `keel-engine` | `sidecar_probe` | Exercise sidecar upload against a local HTTP listener. |
| `keel-rpc` | `rpc_probe` | Call or subscribe to a running engine's RPC socket. |
| `keel-rpc` | `e2e_driver` | Manually exercise a configured two-device cloud environment. |
| `keel-sync` | `chat2_live` | Manually exercise a configured deployed chat room. |
| `keel-sync` | `s2_reader` | Inspect a configured Loro room. |
| `keel-sync` | `doc_surgery` | Inspect or modify a supplied document store. |

For example:

```sh
cargo run -p keel-doc --example gen_fixture -- /tmp/keel-fixture.bin
cargo run -p keel-doc --example rebuild_whale -- /tmp/keel-fixture.bin
```

Read a diagnostic's source for its arguments. The cloud probes depend on
external infrastructure not supplied by this repository. `doc_surgery` has
write modes; use a disposable copy of a document store for investigation.

## assets and documentation

- `design/make-icon.swift` draws a PNG at the supplied output path.
- `design/Keel.iconset/` contains the source sizes for the packaged app icon.
- `docs/generate-diagrams.py` writes current diagrams to `docs/diagrams/` and
  the proposed improvement diagram to `docs/proposals/` at the repository root.

```sh
iconutil -c icns tools/design/Keel.iconset -o assets/Keel.icns
python3 tools/docs/generate-diagrams.py
```

## upstream reference

`sh tools/upstream/fetch-reference.sh` downloads the upstream TypeScript
DeepSeek Harness into the ignored `.refs/` directory. This is optional reference
material; Rust builds use the tracked ports in `crates/dsh/` and `extras/dsh/`.
