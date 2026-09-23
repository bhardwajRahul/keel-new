# Keel developer handoff

Keel is a local macOS coding workspace. The UI and Rust engine are in
`crates/{ui,engine,harness,proto,doc,syntax,rpc}`; the app entry point is
`apps/keel`. The in-process coding loop is in `crates/dsh`.

Laya-CoreML is the default local decision layer when its packaged worker and
checkpoint are available. The ordinary coding harness remains available when
they are not. The selected coding CLI still owns its own model, tools, and
authentication. See `README.md` and `docs/decision-architecture.md` for the
current supported scope and build steps.

Run `cargo test --workspace` for the Rust workspace. The packaged macOS app
also needs the Laya worker and checkpoint supplied to `scripts/package-macos.sh`.
Keep the third-party license texts in the source and app distribution.
