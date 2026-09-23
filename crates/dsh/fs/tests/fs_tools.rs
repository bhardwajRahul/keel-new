//! Behavior tests for the read/write/edit tool suite over a real tempdir,
//! mirroring the portable parts of upstream `tool-fs/tests/tools.spec.ts`,
//! `diff.spec.ts`, and `error.spec.ts`. The mock-provider fixtures
//! (size-less backends, streaming routes, sandbox escalation) do not
//! translate and are not ported.

mod common;

use common::*;
use dsh_fs::{
    FsToolsConfig, compute_hunk_diffs, diffs_from_meta, install_observation_policy, lang_from_path,
    read_meta_from_meta, register_fs_tools,
};
use dsh_tools::{ToolCallView, ToolExecutionMode, ToolResult, ToolResultView};
use serde_json::{Value, json};

fn setup_tools(fixture: &Fixture, config: FsToolsConfig) {
    register_fs_tools(&fixture.ctx, &fixture.tools, &fixture.fs, config).unwrap();
    install_observation_policy(&fixture.ctx).unwrap();
}

#[test]
fn registers_read_write_edit_with_expected_visibility() {
    dsh_cordis::run(async {
        let fixture = fixture();
        let handles = register_fs_tools(
            &fixture.ctx,
            &fixture.tools,
            &fixture.fs,
            FsToolsConfig::default(),
        )
        .unwrap();
        let names: Vec<String> = fixture
            .tools
            .schemas(None)
            .into_iter()
            .map(|schema| schema.name)
            .collect();
        assert_eq!(names, vec!["read", "write", "edit"]);
        // Only read opts into parallel dispatch; mutations stay exclusive.
        assert_eq!(
            fixture
                .tools
                .execution_mode(&input("read", json!({ "file_path": "a" }), None)),
            ToolExecutionMode::Parallel
        );
        assert_eq!(
            fixture.tools.execution_mode(&input(
                "write",
                json!({ "file_path": "a", "content": "" }),
                None
            )),
            ToolExecutionMode::Exclusive
        );
        // Disposal withdraws every registration (HMR safety).
        for handle in handles {
            handle.dispose().await;
        }
        assert!(fixture.tools.schemas(None).is_empty());
    });
}

#[test]
fn read_formats_numbered_lines_with_the_envelope_and_footer() {
    dsh_cordis::run(async {
        let fixture = fixture();
        setup_tools(&fixture, FsToolsConfig::default());
        write(&fixture.root, "a.txt", "alpha\nbeta\n");
        let result = fixture
            .tools
            .execute(input("read", json!({ "file_path": "a.txt" }), None))
            .await;
        assert!(!result.is_error());
        let text = text_of(&result);
        let display = fixture.root.join("a.txt").to_string_lossy().into_owned();
        assert!(text.starts_with(&format!(
            "<path>{display}</path>\n<type>file</type>\n<content>\n"
        )));
        assert!(text.contains("1: alpha\n2: beta"), "{text}");
        assert!(text.contains("(End of file - total 2 lines)"), "{text}");
    });
}

#[test]
fn read_renders_an_empty_file_as_just_the_footer() {
    dsh_cordis::run(async {
        let fixture = fixture();
        setup_tools(&fixture, FsToolsConfig::default());
        write(&fixture.root, "empty.txt", "");
        let result = fixture
            .tools
            .execute(input("read", json!({ "file_path": "empty.txt" }), None))
            .await;
        assert!(!result.is_error());
        assert!(text_of(&result).contains("(End of file - total 0 lines)"));
    });
}

#[test]
fn read_windows_by_offset_and_limit_with_a_continuation_footer() {
    dsh_cordis::run(async {
        let fixture = fixture();
        setup_tools(&fixture, FsToolsConfig::default());
        write(&fixture.root, "a.txt", "one\ntwo\nthree\n");
        let result = fixture
            .tools
            .execute(input(
                "read",
                json!({ "file_path": "a.txt", "offset": 2, "limit": 1 }),
                None,
            ))
            .await;
        let text = text_of(&result);
        assert!(text.contains("2: two"), "{text}");
        assert!(!text.contains("1: one"));
        assert!(
            text.contains("(Showing lines 2-2 of 3. Use offset=3 to continue.)"),
            "{text}"
        );
    });
}

#[test]
fn read_rejects_bad_windows_and_missing_targets() {
    dsh_cordis::run(async {
        let fixture = fixture();
        setup_tools(
            &fixture,
            FsToolsConfig {
                read_limit: 10,
                ..FsToolsConfig::default()
            },
        );
        write(&fixture.root, "a.txt", "one\n");
        let run = |args: Value| fixture.tools.execute(input("read", args, None));

        let offset = run(json!({ "file_path": "a.txt", "offset": 0 })).await;
        assert!(text_of(&offset).contains("offset must be a positive integer"));
        let fractional = run(json!({ "file_path": "a.txt", "offset": 1.5 })).await;
        assert!(fractional.is_error());
        let over_cap = run(json!({ "file_path": "a.txt", "limit": 11 })).await;
        assert!(text_of(&over_cap).contains("limit must be less than or equal to 10"));
        let blank = run(json!({ "file_path": "  " })).await;
        assert!(text_of(&blank).contains("file_path must be a non-empty string"));
        let past_eof = run(json!({ "file_path": "a.txt", "offset": 5 })).await;
        assert_eq!(code_of(&past_eof), "FS_NOT_FOUND");
        assert!(text_of(&past_eof).contains("offset 5 is out of range"));
        let missing = run(json!({ "file_path": "missing.txt" })).await;
        assert_eq!(code_of(&missing), "FS_NOT_FOUND");
        let dir = run(json!({ "file_path": "." })).await;
        assert_eq!(code_of(&dir), "FS_NOT_REGULAR_FILE");
    });
}

#[test]
fn read_caps_lines_and_bytes() {
    dsh_cordis::run(async {
        let fixture = fixture();
        setup_tools(
            &fixture,
            FsToolsConfig {
                read_max_line_length: 5,
                read_max_bytes: 45,
                ..FsToolsConfig::default()
            },
        );
        write(&fixture.root, "long.txt", "0123456789\nshort\n");
        let result = fixture
            .tools
            .execute(input("read", json!({ "file_path": "long.txt" }), None))
            .await;
        let text = text_of(&result);
        assert!(
            text.contains("01234... (line truncated to 5 chars)"),
            "{text}"
        );

        // A byte-capped window reports the capped footer.
        let big = "x".repeat(20);
        write(&fixture.root, "big.txt", &format!("{big}\n{big}\n{big}\n"));
        let result = fixture
            .tools
            .execute(input("read", json!({ "file_path": "big.txt" }), None))
            .await;
        let text = text_of(&result);
        assert!(
            text.contains("(Output capped. Showing lines 1-1. Use offset=2 to continue.)"),
            "{text}"
        );
    });
}

#[test]
fn write_creates_and_updates_through_the_observation_gate() {
    dsh_cordis::run(async {
        let fixture = fixture();
        setup_tools(&fixture, FsToolsConfig::default());
        let agent = agent_with_cwd(&fixture.ctx, "writer", &fixture.root);

        let created = fixture
            .tools
            .execute(input(
                "write",
                json!({ "file_path": "new.txt", "content": "hello\n" }),
                Some(agent.clone()),
            ))
            .await;
        assert!(!created.is_error());
        assert!(text_of(&created).contains("Created file"));
        assert_eq!(
            std::fs::read_to_string(fixture.root.join("new.txt")).unwrap(),
            "hello\n"
        );
        // A create has no prior content: no applied hunks in meta.
        assert_eq!(created.meta().unwrap()["diffs"], json!([]));

        // The write observed its own result, so an immediate overwrite by
        // the same session is a guarded update.
        let updated = fixture
            .tools
            .execute(input(
                "write",
                json!({ "file_path": "new.txt", "content": "hello\nworld\n" }),
                Some(agent),
            ))
            .await;
        assert!(text_of(&updated).contains("Updated file"));
        let diffs = diffs_from_meta(updated.meta().unwrap()).unwrap();
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].old_text.as_deref(), Some("hello"));
        assert_eq!(diffs[0].new_text, "hello\nworld");
    });
}

#[test]
fn writing_an_unread_existing_file_is_denied_with_the_remedy() {
    dsh_cordis::run(async {
        let fixture = fixture();
        setup_tools(&fixture, FsToolsConfig::default());
        write(&fixture.root, "existing.txt", "precious");
        let agent = agent_with_cwd(&fixture.ctx, "blind", &fixture.root);
        let result = fixture
            .tools
            .execute(input(
                "write",
                json!({ "file_path": "existing.txt", "content": "clobber" }),
                Some(agent),
            ))
            .await;
        assert_eq!(code_of(&result), "FS_NOT_OBSERVED");
        let text = text_of(&result);
        assert!(text.contains("without reading it first"), "{text}");
        assert!(text.contains("— read the file, then retry"), "{text}");
        assert_eq!(
            std::fs::read_to_string(fixture.root.join("existing.txt")).unwrap(),
            "precious"
        );
    });
}

#[test]
fn edit_requires_a_prior_read_and_succeeds_after_one() {
    dsh_cordis::run(async {
        let fixture = fixture();
        setup_tools(&fixture, FsToolsConfig::default());
        write(&fixture.root, "a.txt", "alpha beta\n");
        let agent = agent_with_cwd(&fixture.ctx, "editor", &fixture.root);

        let denied = fixture
            .tools
            .execute(input(
                "edit",
                json!({ "file_path": "a.txt", "old_string": "beta", "new_string": "gamma" }),
                Some(agent.clone()),
            ))
            .await;
        assert_eq!(code_of(&denied), "FS_NOT_OBSERVED");
        assert!(
            text_of(&denied).contains("edit requires reading"),
            "{}",
            text_of(&denied)
        );

        let read = fixture
            .tools
            .execute(input(
                "read",
                json!({ "file_path": "a.txt" }),
                Some(agent.clone()),
            ))
            .await;
        assert!(!read.is_error());

        let edited = fixture
            .tools
            .execute(input(
                "edit",
                json!({ "file_path": "a.txt", "old_string": "beta", "new_string": "gamma" }),
                Some(agent),
            ))
            .await;
        assert!(!edited.is_error(), "{:?}", edited.error());
        assert!(text_of(&edited).ends_with("has been updated successfully."));
        assert_eq!(
            std::fs::read_to_string(fixture.root.join("a.txt")).unwrap(),
            "alpha gamma\n"
        );
    });
}

#[test]
fn edit_after_an_external_change_reports_stale_with_the_remedy() {
    dsh_cordis::run(async {
        let fixture = fixture();
        setup_tools(&fixture, FsToolsConfig::default());
        write(&fixture.root, "a.txt", "alpha\n");
        let agent = agent_with_cwd(&fixture.ctx, "editor", &fixture.root);
        fixture
            .tools
            .execute(input(
                "read",
                json!({ "file_path": "a.txt" }),
                Some(agent.clone()),
            ))
            .await;
        // The file changes underneath the session's observation.
        write(&fixture.root, "a.txt", "alpha changed\n");
        let result = fixture
            .tools
            .execute(input(
                "edit",
                json!({ "file_path": "a.txt", "old_string": "alpha", "new_string": "omega" }),
                Some(agent),
            ))
            .await;
        assert_eq!(code_of(&result), "FS_STALE_VERSION");
        assert!(
            text_of(&result).contains("— re-read the file, then retry"),
            "{}",
            text_of(&result)
        );
    });
}

#[test]
fn edit_validates_arguments_and_reports_match_failures() {
    dsh_cordis::run(async {
        let fixture = fixture();
        setup_tools(&fixture, FsToolsConfig::default());
        write(&fixture.root, "a.txt", "dup dup\n");
        let agent = agent_with_cwd(&fixture.ctx, "editor", &fixture.root);
        fixture
            .tools
            .execute(input(
                "read",
                json!({ "file_path": "a.txt" }),
                Some(agent.clone()),
            ))
            .await;

        let same = fixture
            .tools
            .execute(input(
                "edit",
                json!({ "file_path": "a.txt", "old_string": "x", "new_string": "x" }),
                Some(agent.clone()),
            ))
            .await;
        assert!(text_of(&same).contains("old_string and new_string must differ"));
        let empty = fixture
            .tools
            .execute(input(
                "edit",
                json!({ "file_path": "a.txt", "old_string": "", "new_string": "x" }),
                Some(agent.clone()),
            ))
            .await;
        assert!(text_of(&empty).contains("old_string must be a non-empty string"));
        let ambiguous = fixture
            .tools
            .execute(input(
                "edit",
                json!({ "file_path": "a.txt", "old_string": "dup", "new_string": "x" }),
                Some(agent.clone()),
            ))
            .await;
        assert_eq!(code_of(&ambiguous), "FS_AMBIGUOUS_EDIT");
        assert!(text_of(&ambiguous).contains("set replace_all to true"));
        let all = fixture
            .tools
            .execute(input(
                "edit",
                json!({ "file_path": "a.txt", "old_string": "dup", "new_string": "x", "replace_all": true }),
                Some(agent),
            ))
            .await;
        assert!(text_of(&all).contains("All occurrences were successfully replaced."));
        assert_eq!(
            std::fs::read_to_string(fixture.root.join("a.txt")).unwrap(),
            "x x\n"
        );
    });
}

#[test]
fn read_meta_narrows_into_a_read_card_and_declines_errors() {
    dsh_cordis::run(async {
        let fixture = fixture();
        setup_tools(&fixture, FsToolsConfig::default());
        write(&fixture.root, "code.rs", "fn main() {}\n");
        let result = fixture
            .tools
            .execute(input("read", json!({ "file_path": "code.rs" }), None))
            .await;
        let meta = result.meta().unwrap().clone();
        let narrowed = read_meta_from_meta(&meta).unwrap();
        assert_eq!(narrowed.total_lines, 1);
        assert_eq!(narrowed.lang.as_deref(), Some("rs"));
        assert_eq!(narrowed.lines[0].text, "fn main() {}");

        let definition = fixture.tools.get("read", None).unwrap();
        let present = definition.present_result.unwrap();
        let args = json!({ "file_path": "code.rs" });
        let view = present(
            &args,
            &ToolResult {
                content: result.content().to_vec(),
                is_error: false,
                meta: Some(meta.clone()),
            },
        )
        .unwrap();
        match view {
            ToolResultView::Read(read) => {
                assert_eq!(read.total_lines, 1);
                assert_eq!(read.lang.as_deref(), Some("rs"));
                // The envelope is stripped from the carried content.
                match read.content.unwrap().as_slice() {
                    [dsh_llm::ContentBlock::Text { text }] => {
                        assert!(text.starts_with("1: fn main()"), "{text}")
                    }
                    other => panic!("expected text content, got {other:?}"),
                }
            }
            other => panic!("expected a read card, got {other:?}"),
        }
        // Errors and malformed meta decline to the generic fallback.
        let error_result = ToolResult {
            content: vec![dsh_llm::ContentBlock::Text {
                text: "Error: nope".into(),
            }],
            is_error: true,
            meta: Some(meta),
        };
        assert!(present(&args, &error_result).is_none());
        assert!(read_meta_from_meta(&json!({ "offset": 0 })).is_none());
        assert!(
            read_meta_from_meta(&json!({
                "path": "p", "offset": 1, "totalLines": 1,
                "lines": [{ "number": 5, "text": "past the total" }],
            }))
            .is_none()
        );
    });
}

#[test]
fn presenters_render_call_cards_purely_from_args() {
    dsh_cordis::run(async {
        let fixture = fixture();
        setup_tools(&fixture, FsToolsConfig::default());

        let read = fixture.tools.get("read", None).unwrap();
        let call =
            (read.present_call.unwrap())(&json!({ "file_path": "f.txt", "offset": 5, "limit": 4 }));
        match call.unwrap() {
            ToolCallView::Generic(view) => {
                assert_eq!(view.title, "Read f.txt (5 - 8)");
                assert_eq!(view.locations.unwrap()[0].line, Some(5));
            }
            other => panic!("expected generic, got {other:?}"),
        }

        let write_tool = fixture.tools.get("write", None).unwrap();
        let call =
            (write_tool.present_call.unwrap())(&json!({ "file_path": "f.txt", "content": "body" }));
        match call.unwrap() {
            ToolCallView::Diff(view) => {
                assert_eq!(view.title, "Write f.txt");
                assert_eq!(view.diffs[0].old_text, None);
                assert_eq!(view.diffs[0].new_text, "body");
            }
            other => panic!("expected diff, got {other:?}"),
        }

        let edit = fixture.tools.get("edit", None).unwrap();
        let call = (edit.present_call.unwrap())(
            &json!({ "file_path": "f.txt", "old_string": "a", "new_string": "b" }),
        );
        match call.unwrap() {
            ToolCallView::Diff(view) => {
                assert_eq!(view.title, "Edit f.txt");
                assert_eq!(view.diffs[0].old_text.as_deref(), Some("a"));
                assert_eq!(view.diffs[0].new_text, "b");
            }
            other => panic!("expected diff, got {other:?}"),
        }
    });
}

#[test]
fn edit_meta_projects_the_applied_hunk_for_the_diff_card() {
    dsh_cordis::run(async {
        let fixture = fixture();
        setup_tools(&fixture, FsToolsConfig::default());
        write(
            &fixture.root,
            "a.txt",
            "one\ntwo\nthree\nfour\nfive\nsix\nseven\neight\n",
        );
        let agent = agent_with_cwd(&fixture.ctx, "editor", &fixture.root);
        fixture
            .tools
            .execute(input(
                "read",
                json!({ "file_path": "a.txt" }),
                Some(agent.clone()),
            ))
            .await;
        let result = fixture
            .tools
            .execute(input(
                "edit",
                json!({ "file_path": "a.txt", "old_string": "five", "new_string": "FIVE" }),
                Some(agent),
            ))
            .await;
        let diffs = diffs_from_meta(result.meta().unwrap()).unwrap();
        assert_eq!(diffs.len(), 1);
        // Three context lines each side of the applied change.
        assert_eq!(
            diffs[0].old_text.as_deref(),
            Some("two\nthree\nfour\nfive\nsix\nseven\neight")
        );
        assert_eq!(
            diffs[0].new_text,
            "two\nthree\nfour\nFIVE\nsix\nseven\neight"
        );

        let definition = fixture.tools.get("edit", None).unwrap();
        let view = (definition.present_result.unwrap())(
            &json!({ "file_path": "a.txt", "old_string": "five", "new_string": "FIVE" }),
            &ToolResult {
                content: result.content().to_vec(),
                is_error: false,
                meta: result.meta().cloned(),
            },
        );
        match view.unwrap() {
            ToolResultView::Diff(card) => assert_eq!(card.title.as_deref(), Some("Edit a.txt")),
            other => panic!("expected diff card, got {other:?}"),
        }
    });
}

#[test]
fn hunk_diffs_cover_insertions_deletions_and_scattered_changes() {
    // Pure insertion into empty content: nothing to diff against.
    let created = compute_hunk_diffs("f", "", "new\ncontent\n");
    assert_eq!(created.len(), 1);
    assert_eq!(created[0].old_text, None);
    assert_eq!(created[0].new_text, "new\ncontent");
    // Identical texts: no hunks.
    assert!(compute_hunk_diffs("f", "same\n", "same\n").is_empty());
    // Scattered replacements stay separate hunks.
    let mut before = String::new();
    let mut after = String::new();
    for index in 0..20 {
        before.push_str(&format!("line{index}\n"));
        after.push_str(&if index == 2 || index == 17 {
            format!("LINE{index}\n")
        } else {
            format!("line{index}\n")
        });
    }
    let scattered = compute_hunk_diffs("f", &before, &after);
    assert_eq!(scattered.len(), 2);
    // Defensive narrowing of the persisted payload.
    assert!(diffs_from_meta(&json!({ "diffs": [] })).is_none());
    assert!(
        diffs_from_meta(&json!({ "diffs": [{ "path": "p", "oldText": 3, "newText": "x" }] }))
            .is_none()
    );
    assert!(diffs_from_meta(&json!("not an object")).is_none());
    let ok =
        diffs_from_meta(&json!({ "diffs": [{ "path": "p", "oldText": null, "newText": "x" }] }));
    assert_eq!(ok.unwrap()[0].old_text, None);
}

#[test]
fn lang_hints_derive_from_extensions_only() {
    assert_eq!(lang_from_path("src/main.rs"), Some("rs"));
    assert_eq!(lang_from_path("A/B/T.TSX"), Some("tsx"));
    assert_eq!(lang_from_path(".gitignore"), None);
    assert_eq!(lang_from_path("no-extension"), None);
    assert_eq!(lang_from_path("weird.constructor"), None);
}
