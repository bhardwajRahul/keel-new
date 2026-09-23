# Third-party notices

Keel includes components under the licenses retained in this repository.

## deepseek-harness

- Source: https://github.com/deepseek-ai/deepseek-harness (v0.1.0-rc.5)
- Copyright (c) 2026 DeepSeek
- License: MIT — see [retained license](licenses/LICENSE.deepseek-harness)
- Scope: the coding loop, plugin contracts, sessions, and CLI behavior are ported from the upstream TypeScript implementation.

## Syntax and UI dependencies

- The retained MIT notice for the UI base is in [LICENSE.ui-base](licenses/LICENSE.ui-base).
- Tree-sitter parsers and query licenses are listed in [syntax notices](licenses/THIRD_PARTY_NOTICES.syntax.md) and pinned in `Cargo.lock`.
- The bundled Laya-CoreML worker and checkpoint retain their upstream license and model metadata in the app bundle.

All retained license and notice texts are collected in [`licenses/`](licenses/README.md).
The macOS packager includes this directory alongside this index and Keel's license.
