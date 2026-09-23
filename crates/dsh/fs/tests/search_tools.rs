//! Behavior tests for the glob/grep discovery tools over real tempdir trees,
//! mirroring the portable parts of upstream
//! `tool-fs-search/tests/tools.spec.ts` + `presentation.spec.ts`. The
//! ripgrep spawn/transport fixtures (exit-code classification, --json
//! parsing, spill store, rg-path resolution) do not translate to the
//! in-process walker and are not ported.

mod common;

use common::*;
use dsh_fs::{
    SearchToolsConfig, preview_line, register_search_tools, search_view_from_meta,
    to_workdir_relative,
};
use dsh_tools::{SearchResultView, ToolCallView, ToolResult, ToolResultView};
use serde_json::json;
use std::path::Path;

fn setup_search(fixture: &Fixture, config: SearchToolsConfig) -> dsh_agent::AgentRef {
    register_search_tools(&fixture.ctx, &fixture.tools, config).unwrap();
    agent_with_cwd(&fixture.ctx, "searcher", &fixture.root)
}

#[test]
fn glob_matches_basenames_at_any_depth_relative_to_the_workdir() {
    dsh_cordis::run(async {
        let fixture = fixture();
        let agent = setup_search(&fixture, SearchToolsConfig::default());
        write(&fixture.root, "a.rs", "");
        write(&fixture.root, "sub/b.rs", "");
        write(&fixture.root, "c.txt", "");
        let result = fixture
            .tools
            .execute(input("glob", json!({ "pattern": "*.rs" }), Some(agent)))
            .await;
        assert!(!result.is_error(), "{:?}", result.error());
        let text = text_of(&result);
        assert!(text.contains("a.rs"), "{text}");
        assert!(text.contains("sub/b.rs"), "{text}");
        assert!(!text.contains("c.txt"), "{text}");
    });
}

#[test]
fn glob_respects_gitignore_and_prunes_vcs_metadata() {
    dsh_cordis::run(async {
        let fixture = fixture();
        let agent = setup_search(&fixture, SearchToolsConfig::default());
        write(&fixture.root, ".gitignore", "ignored.rs\n");
        write(&fixture.root, "kept.rs", "");
        write(&fixture.root, "ignored.rs", "");
        write(&fixture.root, ".git/config.rs", "");
        write(&fixture.root, ".hidden/visible.rs", "");
        let result = fixture
            .tools
            .execute(input("glob", json!({ "pattern": "*.rs" }), Some(agent)))
            .await;
        let text = text_of(&result);
        assert!(text.contains("kept.rs"), "{text}");
        // The .gitignore wins (the documented divergence from --no-ignore).
        assert!(!text.contains("ignored.rs"), "{text}");
        // VCS metadata is pruned even though hidden files are included.
        assert!(!text.contains(".git/config.rs"), "{text}");
        assert!(text.contains(".hidden/visible.rs"), "{text}");
    });
}

#[test]
fn glob_scopes_the_search_by_the_path_argument() {
    dsh_cordis::run(async {
        let fixture = fixture();
        let agent = setup_search(&fixture, SearchToolsConfig::default());
        write(&fixture.root, "top.rs", "");
        write(&fixture.root, "sub/inner.rs", "");
        let result = fixture
            .tools
            .execute(input(
                "glob",
                json!({ "pattern": "*.rs", "path": "sub" }),
                Some(agent),
            ))
            .await;
        let text = text_of(&result);
        assert!(text.contains("sub/inner.rs"), "{text}");
        assert!(!text.contains("top.rs"), "{text}");
        assert_eq!(result.value().unwrap()["root"], json!("sub"));
    });
}

#[test]
fn glob_caps_at_the_configured_head_and_reports_the_remainder() {
    dsh_cordis::run(async {
        let fixture = fixture();
        let agent = setup_search(
            &fixture,
            SearchToolsConfig {
                glob_max_results: 2,
                ..SearchToolsConfig::default()
            },
        );
        for name in ["one.rs", "two.rs", "three.rs"] {
            write(&fixture.root, name, "");
        }
        let result = fixture
            .tools
            .execute(input("glob", json!({ "pattern": "*.rs" }), Some(agent)))
            .await;
        let text = text_of(&result);
        assert!(
            text.contains("(Showing 2 of 3 paths. The complete result could not be saved; narrow pattern or path to see more.)"),
            "{text}"
        );
        let meta = result.meta().unwrap();
        assert_eq!(meta["truncated"], json!(true));
        assert_eq!(meta["total"], json!(3));
        assert_eq!(meta["paths"].as_array().unwrap().len(), 2);
    });
}

#[test]
fn glob_reports_no_files_and_validates_arguments() {
    dsh_cordis::run(async {
        let fixture = fixture();
        let agent = setup_search(&fixture, SearchToolsConfig::default());
        let none = fixture
            .tools
            .execute(input(
                "glob",
                json!({ "pattern": "*.nope" }),
                Some(agent.clone()),
            ))
            .await;
        assert_eq!(text_of(&none), "No files found");
        let blank = fixture
            .tools
            .execute(input(
                "glob",
                json!({ "pattern": "  " }),
                Some(agent.clone()),
            ))
            .await;
        assert!(text_of(&blank).contains("pattern must be a non-empty string"));
        let blank_path = fixture
            .tools
            .execute(input(
                "glob",
                json!({ "pattern": "*", "path": " " }),
                Some(agent.clone()),
            ))
            .await;
        assert!(text_of(&blank_path).contains("path must be a non-empty string when given"));
        let invalid = fixture
            .tools
            .execute(input("glob", json!({ "pattern": "[z-a]" }), Some(agent)))
            .await;
        assert_eq!(code_of(&invalid), "SEARCH_INVALID_PATTERN");
    });
}

#[test]
fn grep_groups_matches_by_file_with_line_numbers() {
    dsh_cordis::run(async {
        let fixture = fixture();
        let agent = setup_search(&fixture, SearchToolsConfig::default());
        write(&fixture.root, "a.txt", "alpha\nneedle here\nomega\n");
        write(&fixture.root, "b.txt", "needle one\nneedle two\n");
        let result = fixture
            .tools
            .execute(input("grep", json!({ "pattern": "needle" }), Some(agent)))
            .await;
        assert!(!result.is_error(), "{:?}", result.error());
        let text = text_of(&result);
        assert!(text.starts_with("Found 3 matches"), "{text}");
        assert!(text.contains("a.txt\nLine 2: needle here"), "{text}");
        assert!(
            text.contains("b.txt\nLine 1: needle one\nLine 2: needle two"),
            "{text}"
        );
    });
}

#[test]
fn grep_reports_a_single_match_in_the_singular() {
    dsh_cordis::run(async {
        let fixture = fixture();
        let agent = setup_search(&fixture, SearchToolsConfig::default());
        write(&fixture.root, "a.txt", "just one needle\n");
        let result = fixture
            .tools
            .execute(input("grep", json!({ "pattern": "needle" }), Some(agent)))
            .await;
        assert!(
            text_of(&result).starts_with("Found 1 match\n"),
            "{}",
            text_of(&result)
        );
    });
}

#[test]
fn grep_filters_by_one_include_glob() {
    dsh_cordis::run(async {
        let fixture = fixture();
        let agent = setup_search(&fixture, SearchToolsConfig::default());
        write(&fixture.root, "a.md", "needle\n");
        write(&fixture.root, "a.txt", "needle\n");
        let result = fixture
            .tools
            .execute(input(
                "grep",
                json!({ "pattern": "needle", "include": "*.md" }),
                Some(agent),
            ))
            .await;
        let text = text_of(&result);
        assert!(text.contains("a.md"), "{text}");
        assert!(!text.contains("a.txt"), "{text}");
    });
}

#[test]
fn grep_validates_pattern_path_and_include() {
    dsh_cordis::run(async {
        let fixture = fixture();
        let agent = setup_search(&fixture, SearchToolsConfig::default());
        write(&fixture.root, "a.txt", "  spaced\n");
        let empty = fixture
            .tools
            .execute(input("grep", json!({ "pattern": "" }), Some(agent.clone())))
            .await;
        assert!(text_of(&empty).contains("pattern must be a non-empty string"));
        // A whitespace-only pattern is a legitimate regex.
        let spaces = fixture
            .tools
            .execute(input(
                "grep",
                json!({ "pattern": "  " }),
                Some(agent.clone()),
            ))
            .await;
        assert!(
            text_of(&spaces).starts_with("Found 1 match"),
            "{}",
            text_of(&spaces)
        );
        let negated = fixture
            .tools
            .execute(input(
                "grep",
                json!({ "pattern": "x", "include": "!a" }),
                Some(agent.clone()),
            ))
            .await;
        assert!(text_of(&negated).contains("positive glob filter"));
        let list = fixture
            .tools
            .execute(input(
                "grep",
                json!({ "pattern": "x", "include": "a,b" }),
                Some(agent.clone()),
            ))
            .await;
        assert!(text_of(&list).contains("not a comma-separated list"));
        // Brace alternation is one glob, not a list.
        write(&fixture.root, "b.tsx", "needle\n");
        let braces = fixture
            .tools
            .execute(input(
                "grep",
                json!({ "pattern": "needle", "include": "*.{ts,tsx}" }),
                Some(agent.clone()),
            ))
            .await;
        assert!(text_of(&braces).contains("b.tsx"), "{}", text_of(&braces));
        let invalid = fixture
            .tools
            .execute(input("grep", json!({ "pattern": "(" }), Some(agent)))
            .await;
        assert_eq!(code_of(&invalid), "SEARCH_INVALID_PATTERN");
    });
}

#[test]
fn grep_caps_matches_and_previews_long_lines_on_utf8_boundaries() {
    dsh_cordis::run(async {
        let fixture = fixture();
        let agent = setup_search(
            &fixture,
            SearchToolsConfig {
                grep_max_matches: 2,
                grep_max_line_bytes: 10,
                ..SearchToolsConfig::default()
            },
        );
        // The long line has a multibyte char straddling the byte budget.
        write(
            &fixture.root,
            "a.txt",
            "needle 12é45678\nneedle two\nneedle three\n",
        );
        let result = fixture
            .tools
            .execute(input("grep", json!({ "pattern": "needle" }), Some(agent)))
            .await;
        let text = text_of(&result);
        assert!(text.starts_with("Found 2 of 3 matches"), "{text}");
        assert!(text.contains("(line truncated)"), "{text}");
        assert!(
            text.contains("(The complete result could not be saved; narrow pattern, path, or include to see more.)"),
            "{text}"
        );
        let meta = result.meta().unwrap();
        assert_eq!(meta["truncated"], json!(true));
        assert_eq!(meta["total"], json!(3));
    });
}

#[test]
fn grep_skips_gitignored_hidden_and_binary_files() {
    dsh_cordis::run(async {
        let fixture = fixture();
        let agent = setup_search(&fixture, SearchToolsConfig::default());
        write(&fixture.root, ".gitignore", "skipped.txt\n");
        write(&fixture.root, "kept.txt", "needle\n");
        write(&fixture.root, "skipped.txt", "needle\n");
        write(&fixture.root, ".hidden.txt", "needle\n");
        std::fs::write(fixture.root.join("bin.dat"), b"needle\x00binary").unwrap();
        let result = fixture
            .tools
            .execute(input("grep", json!({ "pattern": "needle" }), Some(agent)))
            .await;
        let text = text_of(&result);
        assert!(text.contains("kept.txt"), "{text}");
        assert!(!text.contains("skipped.txt"), "{text}");
        assert!(!text.contains(".hidden.txt"), "{text}");
        assert!(!text.contains("bin.dat"), "{text}");
    });
}

#[test]
fn grep_reports_no_matches_and_non_utf8_lines_as_placeholders() {
    dsh_cordis::run(async {
        let fixture = fixture();
        let agent = setup_search(&fixture, SearchToolsConfig::default());
        write(&fixture.root, "a.txt", "nothing here\n");
        let none = fixture
            .tools
            .execute(input(
                "grep",
                json!({ "pattern": "needle" }),
                Some(agent.clone()),
            ))
            .await;
        assert_eq!(text_of(&none), "No matches found");
        // An invalid-UTF-8 (non-NUL) matching line becomes a placeholder.
        std::fs::write(fixture.root.join("weird.txt"), b"needle \xff\xfe tail\n").unwrap();
        let placeholder = fixture
            .tools
            .execute(input("grep", json!({ "pattern": "needle" }), Some(agent)))
            .await;
        assert!(
            text_of(&placeholder).contains("Line 1: (line is not valid UTF-8)"),
            "{}",
            text_of(&placeholder)
        );
    });
}

#[test]
fn search_cards_narrow_from_meta_and_decline_mismatches() {
    dsh_cordis::run(async {
        let fixture = fixture();
        let agent = setup_search(&fixture, SearchToolsConfig::default());
        write(&fixture.root, "a.txt", "needle\n");
        let grep_result = fixture
            .tools
            .execute(input(
                "grep",
                json!({ "pattern": "needle" }),
                Some(agent.clone()),
            ))
            .await;
        let grep_tool = fixture.tools.get("grep", None).unwrap();
        let view = (grep_tool.present_result.unwrap())(
            &json!({ "pattern": "needle" }),
            &ToolResult {
                content: grep_result.content().to_vec(),
                is_error: false,
                meta: grep_result.meta().cloned(),
            },
        );
        match view.unwrap() {
            ToolResultView::Search(SearchResultView::Matches(card)) => {
                assert_eq!(card.total, 1);
                assert!(!card.truncated);
                assert_eq!(card.files[0].path, "a.txt");
                assert_eq!(card.files[0].matches[0].line_number, 1);
            }
            other => panic!("expected a matches card, got {other:?}"),
        }

        let glob_result = fixture
            .tools
            .execute(input("glob", json!({ "pattern": "*.txt" }), Some(agent)))
            .await;
        let glob_tool = fixture.tools.get("glob", None).unwrap();
        let present_glob = glob_tool.present_result.unwrap();
        let view = present_glob(
            &json!({ "pattern": "*.txt" }),
            &ToolResult {
                content: glob_result.content().to_vec(),
                is_error: false,
                meta: glob_result.meta().cloned(),
            },
        );
        match view.unwrap() {
            ToolResultView::Search(SearchResultView::Paths(card)) => {
                assert_eq!(card.paths, vec!["a.txt".to_string()]);
            }
            other => panic!("expected a paths card, got {other:?}"),
        }
        // The other tool's shape and malformed meta decline.
        let mismatched = present_glob(
            &json!({ "pattern": "*.txt" }),
            &ToolResult {
                content: glob_result.content().to_vec(),
                is_error: false,
                meta: grep_result.meta().cloned(),
            },
        );
        assert!(mismatched.is_none());
        assert!(
            search_view_from_meta(&json!({ "shape": "unknown", "truncated": false, "total": 0 }))
                .is_none()
        );
        assert!(search_view_from_meta(&json!(42)).is_none());
    });
}

#[test]
fn meta_byte_budget_drops_trailing_items_and_marks_truncation() {
    dsh_cordis::run(async {
        let fixture = fixture();
        let agent = setup_search(
            &fixture,
            SearchToolsConfig {
                search_meta_max_bytes: 120,
                ..SearchToolsConfig::default()
            },
        );
        for index in 0..10 {
            write(
                &fixture.root,
                &format!("file-with-a-long-name-{index}.rs"),
                "",
            );
        }
        let result = fixture
            .tools
            .execute(input("glob", json!({ "pattern": "*.rs" }), Some(agent)))
            .await;
        let meta = result.meta().unwrap();
        assert_eq!(meta["truncated"], json!(true));
        assert_eq!(meta["total"], json!(10));
        assert!(meta["paths"].as_array().unwrap().len() < 10);
        assert!(
            serde_json::to_string(meta).unwrap().len() <= 120 + 32,
            "meta stays near the budget"
        );
    });
}

#[test]
fn call_cards_title_by_pattern_and_scope() {
    dsh_cordis::run(async {
        let fixture = fixture();
        setup_search(&fixture, SearchToolsConfig::default());
        let glob_tool = fixture.tools.get("glob", None).unwrap();
        match (glob_tool.present_call.unwrap())(&json!({ "pattern": "*.ts", "path": "src" }))
            .unwrap()
        {
            ToolCallView::Generic(view) => assert_eq!(view.title, "Glob *.ts in src"),
            other => panic!("expected generic, got {other:?}"),
        }
        let grep_tool = fixture.tools.get("grep", None).unwrap();
        match (grep_tool.present_call.unwrap())(
            &json!({ "pattern": "fn main", "path": "src", "include": "*.rs" }),
        )
        .unwrap()
        {
            ToolCallView::Generic(view) => assert_eq!(view.title, "Grep fn main in src (*.rs)"),
            other => panic!("expected generic, got {other:?}"),
        }
    });
}

#[test]
fn timeout_budget_rides_the_tool_definitions() {
    dsh_cordis::run(async {
        let fixture = fixture();
        setup_search(
            &fixture,
            SearchToolsConfig {
                timeout_ms: 12_345.0,
                ..SearchToolsConfig::default()
            },
        );
        assert_eq!(
            fixture.tools.get("glob", None).unwrap().timeout_ms,
            Some(12_345.0)
        );
        assert_eq!(
            fixture.tools.get("grep", None).unwrap().timeout_ms,
            Some(12_345.0)
        );
    });
}

#[test]
fn helpers_relativize_and_preview() {
    let workdir = Path::new("/work/space");
    assert_eq!(
        to_workdir_relative(Path::new("/work/space/a/b.rs"), workdir),
        "a/b.rs"
    );
    assert_eq!(to_workdir_relative(Path::new("/work/space"), workdir), ".");
    assert_eq!(
        to_workdir_relative(Path::new("/elsewhere/x"), workdir),
        "/elsewhere/x"
    );
    assert_eq!(
        to_workdir_relative(Path::new("already/relative"), workdir),
        "already/relative"
    );
    assert_eq!(preview_line("short", 10), "short");
    // The cut backs off to a UTF-8 boundary instead of splitting the char.
    assert_eq!(preview_line("aé", 2), "a (line truncated)");
}
