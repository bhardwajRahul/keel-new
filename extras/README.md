# optional imported libraries

These Rust ports are retained for reference and reuse. Neither the Keel
application nor the standalone `dsh` CLI depends on them. They remain Cargo
workspace members, with their original tests and package names.

| Package | Purpose |
| --- | --- |
| `dsh-anonymous-user-id` | Anonymous installation identity. |
| `dsh-atomic-write` | Atomic file writes. |
| `dsh-attachment` | Attachment service contracts. |
| `dsh-cmdline` | Command-line argument parsing. |
| `dsh-credentials` | Credential service contracts. |
| `dsh-credentials-local` | Local credential storage. |
| `dsh-launch-environment` | Shell launch environment discovery. |
| `dsh-settings` | Settings contracts and schema. |
| `dsh-settings-file` | File-backed settings. |

Run a package's tests explicitly, for example:

```sh
cargo test -p dsh-atomic-write --locked
```

`cargo test --workspace` also covers these libraries. If Keel begins using one,
move it back into `crates/dsh/` and update its workspace dependency path.
The [DeepSeek license](../licenses/LICENSE.deepseek-harness) is retained.
