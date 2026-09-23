//! Port of the handoff-structure contracts from
//! `packages/boot/cmdline/tests/cmdline.spec.ts`. The commander/Loader tests
//! there exercise `parseCmdline`, which is not ported (see the crate doc);
//! what remains is the immutable snapshot and the launcher exit request.

use std::sync::{Arc, Mutex};

use dsh_cmdline::{AppExit, CmdlineArgs, CmdlineHost};

#[test]
fn hands_the_app_a_snapshot_the_caller_cannot_mutate_afterwards() {
    let mut args = vec!["--resume".to_string(), "abc".to_string()];
    let snapshot = CmdlineArgs::new(args.clone());
    args.push("--tampered".to_string());
    assert_eq!(snapshot.get(), ["--resume", "abc"]);
}

#[test]
fn lets_multiple_readers_observe_the_same_immutable_snapshot() {
    let snapshot = CmdlineArgs::new(["--port", "8080"]);
    let reader = snapshot.clone();
    assert_eq!(snapshot.get(), reader.get());
    assert_eq!(reader.get(), ["--port", "8080"]);
}

#[test]
fn an_embedding_host_with_no_command_line_hands_over_an_empty_list() {
    assert_eq!(CmdlineArgs::default().get(), Vec::<String>::new());
}

#[test]
fn the_exit_request_carries_the_process_exit_code_to_the_launcher() {
    let exits: Arc<Mutex<Vec<i32>>> = Arc::new(Mutex::new(Vec::new()));
    let recorded = exits.clone();
    let exit: AppExit = Arc::new(move |code| recorded.lock().unwrap().push(code));
    let host = CmdlineHost {
        args: CmdlineArgs::new(["serve"]),
        exit,
    };
    (host.exit)(1);
    (host.exit)(0);
    assert_eq!(*exits.lock().unwrap(), [1, 0]);
    assert_eq!(host.args.get(), ["serve"]);
}
