//! Port of `packages/util/launch-environment/tests/launch-environment.spec.ts`.
//! The Cordis-backed `launchEnvironmentOf` tests map onto
//! `LaunchEnvironmentSnapshot::from_process_env`, the fallback that upstream
//! helper builds when no launcher snapshot exists.

use std::collections::HashMap;
use std::path::PathBuf;

use dsh_launch_environment::{
    LaunchEnvironmentEntry, LaunchEnvironmentLayerInput, LaunchEnvironmentSnapshot,
    LaunchEnvironmentSource,
};

fn values(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn layered() -> LaunchEnvironmentSnapshot {
    LaunchEnvironmentSnapshot::new([
        LaunchEnvironmentLayerInput {
            source: LaunchEnvironmentSource::Process,
            path: None,
            values: values(&[("SHARED", "from-process"), ("ONLY_PROCESS", "p")]),
        },
        LaunchEnvironmentLayerInput {
            source: LaunchEnvironmentSource::ProjectEnv,
            path: Some(PathBuf::from("/work/.env")),
            values: values(&[("SHARED", "from-project"), ("ONLY_PROJECT", "j")]),
        },
        LaunchEnvironmentLayerInput {
            source: LaunchEnvironmentSource::UserEnv,
            path: Some(PathBuf::from("/home/.dsh/.env")),
            values: values(&[("SHARED", "from-user"), ("ONLY_USER", "u")]),
        },
    ])
}

fn entry(
    value: &str,
    source: LaunchEnvironmentSource,
    path: Option<&str>,
) -> LaunchEnvironmentEntry {
    LaunchEnvironmentEntry {
        value: value.to_string(),
        source,
        path: path.map(PathBuf::from),
    }
}

#[test]
fn resolves_across_every_layer_most_trusted_first_and_reports_the_winning_source() {
    let snapshot = layered();
    assert_eq!(
        snapshot.get("SHARED"),
        Some(entry(
            "from-process",
            LaunchEnvironmentSource::Process,
            None
        ))
    );
    assert_eq!(
        snapshot.get("ONLY_PROJECT"),
        Some(entry(
            "j",
            LaunchEnvironmentSource::ProjectEnv,
            Some("/work/.env")
        ))
    );
    assert_eq!(
        snapshot.get("ONLY_USER"),
        Some(entry(
            "u",
            LaunchEnvironmentSource::UserEnv,
            Some("/home/.dsh/.env")
        ))
    );
    assert_eq!(snapshot.get("ABSENT"), None);
}

#[test]
fn filters_layers_without_changing_their_trust_order() {
    let snapshot = layered();
    // The point of get_from: a routing field that must never come from a
    // project directory cannot be reached by reordering, only by listing it.
    assert_eq!(
        snapshot.get_from(
            "ONLY_PROJECT",
            &[
                LaunchEnvironmentSource::Process,
                LaunchEnvironmentSource::UserEnv
            ]
        ),
        None
    );
    assert_eq!(
        snapshot.get_from(
            "SHARED",
            &[
                LaunchEnvironmentSource::UserEnv,
                LaunchEnvironmentSource::Process
            ]
        ),
        Some(entry(
            "from-process",
            LaunchEnvironmentSource::Process,
            None
        ))
    );
    assert_eq!(snapshot.get_from("SHARED", &[]), None);
}

#[test]
fn owns_each_layer_so_a_later_mutation_of_the_source_map_cannot_change_it() {
    let mut source = values(&[("KEY", "first")]);
    let snapshot = LaunchEnvironmentSnapshot::new([LaunchEnvironmentLayerInput {
        source: LaunchEnvironmentSource::Process,
        path: None,
        values: source.clone(),
    }]);
    source.insert("KEY".into(), "second".into());
    source.insert("LATE".into(), "added".into());
    assert_eq!(
        snapshot.get("KEY"),
        Some(entry("first", LaunchEnvironmentSource::Process, None))
    );
    assert_eq!(snapshot.get("LATE"), None);
}

#[test]
fn keeps_an_empty_value_as_a_present_value_for_its_owner_to_judge() {
    let snapshot = LaunchEnvironmentSnapshot::new([LaunchEnvironmentLayerInput {
        source: LaunchEnvironmentSource::Process,
        path: None,
        values: values(&[("EMPTY", "")]),
    }]);
    assert_eq!(
        snapshot.get("EMPTY"),
        Some(entry("", LaunchEnvironmentSource::Process, None))
    );
}

#[test]
fn orders_lookups_canonically_regardless_of_construction_order() {
    let reversed = LaunchEnvironmentSnapshot::new([
        LaunchEnvironmentLayerInput {
            source: LaunchEnvironmentSource::UserEnv,
            path: Some(PathBuf::from("/u")),
            values: values(&[("K", "u")]),
        },
        LaunchEnvironmentLayerInput {
            source: LaunchEnvironmentSource::Process,
            path: None,
            values: values(&[("K", "p")]),
        },
    ]);
    assert_eq!(
        reversed.get("K"),
        Some(entry("p", LaunchEnvironmentSource::Process, None))
    );
}

#[test]
fn from_process_env_exposes_the_inherited_environment_as_the_only_layer() {
    // The only test that touches the process environment in this binary.
    unsafe { std::env::set_var("DSH_ENV_SPEC_FALLBACK", "ambient") };
    let snapshot = LaunchEnvironmentSnapshot::from_process_env();
    unsafe { std::env::remove_var("DSH_ENV_SPEC_FALLBACK") };
    assert_eq!(
        snapshot.get("DSH_ENV_SPEC_FALLBACK"),
        Some(entry("ambient", LaunchEnvironmentSource::Process, None))
    );
}
