//! Behavior tests for the read-before-edit observation policy, mirroring
//! upstream `fs-observation-policy/tests/policy.spec.ts` over the Rust event
//! bus (owners are session-id keys instead of weak session objects).

use dsh_cordis::App;
use dsh_fs::{
    FsEditIntentSlot, FsError, FsErrorCode, FsObservation, FsObservedEvent, FsTarget, FsTargetKey,
    FsVersion, FsWriteIntent, FsWriteIntentSlot, install_observation_policy,
};

fn target(name: &str) -> FsTarget {
    FsTarget {
        target_key: FsTargetKey::new(format!("/real/{name}")),
        display_path: format!("/w/{name}"),
    }
}

fn present(version: &str) -> FsObservation {
    FsObservation::Present {
        version: FsVersion::new(version),
    }
}

async fn write_intent(
    ctx: &dsh_cordis::Context,
    target: FsTarget,
    owner: Option<String>,
) -> Option<FsWriteIntent> {
    ctx.waterfall::<FsWriteIntentSlot, _, _>((target, owner), |_args| async { Ok(None) })
        .await
        .unwrap()
}

async fn edit_intent(
    ctx: &dsh_cordis::Context,
    target: FsTarget,
    owner: Option<String>,
) -> anyhow::Result<Option<FsVersion>> {
    ctx.waterfall::<FsEditIntentSlot, _, _>((target, owner), |_args| async { Ok(None) })
        .await
}

#[test]
fn without_the_policy_both_slots_stay_unconditional() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        assert_eq!(
            write_intent(&ctx, target("a.txt"), Some("s1".into())).await,
            None
        );
        assert_eq!(
            edit_intent(&ctx, target("a.txt"), Some("s1".into()))
                .await
                .unwrap(),
            None
        );
    });
}

#[test]
fn an_unobserved_target_decides_create_if_absent() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        install_observation_policy(&ctx).unwrap();
        assert_eq!(
            write_intent(&ctx, target("a.txt"), Some("s1".into())).await,
            Some(FsWriteIntent::CreateIfAbsent)
        );
        // No owner: reads freely, but a write still means a guarded create.
        assert_eq!(
            write_intent(&ctx, target("a.txt"), None).await,
            Some(FsWriteIntent::CreateIfAbsent)
        );
    });
}

#[test]
fn an_observed_target_decides_replace_at_the_observed_version() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        install_observation_policy(&ctx).unwrap();
        ctx.parallel::<FsObservedEvent>(&(target("a.txt"), present("v7"), Some("s1".into())))
            .await;
        assert_eq!(
            write_intent(&ctx, target("a.txt"), Some("s1".into())).await,
            Some(FsWriteIntent::ReplaceIfVersion(FsVersion::new("v7")))
        );
        // A later observation supersedes the earlier one.
        ctx.parallel::<FsObservedEvent>(&(target("a.txt"), present("v8"), Some("s1".into())))
            .await;
        assert_eq!(
            write_intent(&ctx, target("a.txt"), Some("s1".into())).await,
            Some(FsWriteIntent::ReplaceIfVersion(FsVersion::new("v8")))
        );
    });
}

#[test]
fn edit_of_an_unseen_target_rejects_not_observed() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        install_observation_policy(&ctx).unwrap();
        let error = edit_intent(&ctx, target("a.txt"), Some("s1".into()))
            .await
            .unwrap_err();
        let fs_error = error.downcast_ref::<FsError>().unwrap();
        assert_eq!(fs_error.code, FsErrorCode::NotObserved);
        assert!(
            fs_error.message.contains("edit requires reading"),
            "{}",
            fs_error.message
        );
        // A call with no owner can never satisfy the policy either.
        let error = edit_intent(&ctx, target("a.txt"), None).await.unwrap_err();
        assert_eq!(
            error.downcast_ref::<FsError>().unwrap().code,
            FsErrorCode::NotObserved
        );
    });
}

#[test]
fn edit_of_a_confirmed_absent_target_rejects_not_found() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        install_observation_policy(&ctx).unwrap();
        ctx.parallel::<FsObservedEvent>(&(
            target("a.txt"),
            FsObservation::Absent,
            Some("s1".into()),
        ))
        .await;
        let error = edit_intent(&ctx, target("a.txt"), Some("s1".into()))
            .await
            .unwrap_err();
        assert_eq!(
            error.downcast_ref::<FsError>().unwrap().code,
            FsErrorCode::NotFound
        );
    });
}

#[test]
fn edit_of_an_observed_target_supplies_the_version_guard() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        install_observation_policy(&ctx).unwrap();
        ctx.parallel::<FsObservedEvent>(&(target("a.txt"), present("v3"), Some("s1".into())))
            .await;
        assert_eq!(
            edit_intent(&ctx, target("a.txt"), Some("s1".into()))
                .await
                .unwrap(),
            Some(FsVersion::new("v3"))
        );
    });
}

#[test]
fn observed_state_is_isolated_per_owner_and_per_target() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        install_observation_policy(&ctx).unwrap();
        ctx.parallel::<FsObservedEvent>(&(target("a.txt"), present("v1"), Some("alice".into())))
            .await;
        // Another owner has not read the file.
        let error = edit_intent(&ctx, target("a.txt"), Some("bob".into()))
            .await
            .unwrap_err();
        assert_eq!(
            error.downcast_ref::<FsError>().unwrap().code,
            FsErrorCode::NotObserved
        );
        // The same owner has not read another target.
        let error = edit_intent(&ctx, target("b.txt"), Some("alice".into()))
            .await
            .unwrap_err();
        assert_eq!(
            error.downcast_ref::<FsError>().unwrap().code,
            FsErrorCode::NotObserved
        );
        // The recorded pair still answers.
        assert_eq!(
            edit_intent(&ctx, target("a.txt"), Some("alice".into()))
                .await
                .unwrap(),
            Some(FsVersion::new("v1"))
        );
    });
}

#[test]
fn an_ownerless_observation_records_nothing() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        install_observation_policy(&ctx).unwrap();
        ctx.parallel::<FsObservedEvent>(&(target("a.txt"), present("v1"), None))
            .await;
        let error = edit_intent(&ctx, target("a.txt"), Some("s1".into()))
            .await
            .unwrap_err();
        assert_eq!(
            error.downcast_ref::<FsError>().unwrap().code,
            FsErrorCode::NotObserved
        );
    });
}
