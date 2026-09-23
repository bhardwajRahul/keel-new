# build from source

[Documentation](README.md) / build

These commands reproduce the build setup. They are not a claim that the commands were rerun for this documentation update.

## prerequisites

- Apple Silicon and macOS 15 or later for the packaged Core ML model.
- Rust with the toolchain in [`rust-toolchain.toml`](../rust-toolchain.toml).
- Xcode command line tools.
- Python 3.11–3.13. The asset script defaults to `python3.12`.
- Network access for Rust dependencies, the Python runtime packages, and the pinned Laya checkpoint.

Coding providers also need their own installed CLIs and authentication. Compiling Keel does not sign you into those services.

## 1. clone the source

```sh
git clone https://github.com/codejunkie99/keel.git
cd keel
```

## 2. build the local decision worker

```sh
bash scripts/build-laya-assets.sh
```

This installs the worker's Python dependencies, downloads the pinned checkpoint, and creates the standalone worker bundle. Unlike the app's first-run screen, this build command downloads the model immediately.

To use a supported Python executable with another name:

```sh
PYTHON_FOR_LAYA=python3.11 bash scripts/build-laya-assets.sh
```

| Output | Purpose |
| --- | --- |
| `build/laya/dist/laya-worker` | Standalone worker bundle. |
| `build/laya/model` | Pinned Core ML checkpoint. |
| `build/laya/venv` | Python environment used to build the worker. |

## 3. compile the app

```sh
cargo build -p keel --bin keel --release
```

The binary is written to `target/release/keel`.
Keel is the default workspace member, so `cargo build --release` selects the
same application. Optional crates in `extras/` and `tools/dsh-cli` remain
available through `-p` or `--workspace`.

## 4. package the app

### light package

Includes the worker. The user installs the model from onboarding or Settings → Agents.

```sh
KEEL_LAYA_WORKER_BUNDLE=build/laya/dist/laya-worker \
  bash scripts/package-macos.sh dist/Keel-macOS-light.zip target/release/keel
```

### full package

Includes the worker and pinned model. Packaging checks the model files against the expected hashes.

```sh
KEEL_LAYA_WORKER_BUNDLE=build/laya/dist/laya-worker \
KEEL_LAYA_MODEL=build/laya/model \
  bash scripts/package-macos.sh dist/Keel-macOS-full.zip target/release/keel
```

The full package supports local selector use without a model download. Coding providers may still require a network connection.

## run outside an app bundle

Point Keel at the worker executable and checkpoint directory:

```sh
KEEL_LAYA_WORKER="$PWD/build/laya/dist/laya-worker/laya-worker" \
KEEL_LAYA_MODEL="$PWD/build/laya/model" \
  ./target/release/keel
```

Keel stores app state in `~/.keel`. Set `KEEL_DATA_DIR` to use another directory.

## optional checks

Run these when you want to check a local build:

```sh
cargo fmt --all -- --check
cargo check --workspace --all-targets --locked
cargo test -p laya-local -p jev-core -p keel-engine -p keel-harness
```

For the complete suite, use `cargo test --workspace --locked -- --test-threads=1`.
The [developer guide](development.md) explains the optional packages and tools.

The historical results and their limits are in the [build report](archive/build-report-0.2.0.md). These commands do not measure coding-quality gains.

## pinned assets

| Asset | Pin |
| --- | --- |
| Laya Core ML runtime | `laya-coreml==0.1.0` |
| Packaging tool | `pyinstaller==6.22.3` |
| Model repository | `aac6fef/laya-multilingual-coreml` |
| Model revision | `8139e9089273319512c730218903784074133187` |

The worker uses the multilingual 1024-token checkpoint. Download verification checks file integrity; it does not establish decision quality.

## distribution status

The recorded app packages are ad hoc signed for local evaluation. A notarized public installer requires Developer ID signing and Apple notarization.

The repository currently publishes source, not downloadable app releases. Do not treat the historical package names as release links.
