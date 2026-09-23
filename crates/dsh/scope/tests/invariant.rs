//! Companion registration test. Upstream `tests/invariant.spec.ts` exercises
//! the `internal/dispatch` shape checks against the generated harness event
//! catalog; neither exists in the Rust port (scoped dispatch requires a
//! carrier at compile time), so only the registration contract remains.

use dsh_cordis::App;
use dsh_invariants::{InvariantInstaller, InvariantRegistry, InvariantsPlugin};
use serde_json::Value;
use std::rc::Rc;

#[test]
fn companion_reserves_this_packages_ownership() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        ctx.plugin(Rc::new(InvariantsPlugin), Value::Null)
            .unwrap()
            .await_ready()
            .await
            .unwrap();
        let companion = ctx
            .plugin(Rc::new(dsh_scope::invariant::plugin()), Value::Null)
            .unwrap();
        companion.await_ready().await.unwrap();

        let registry = ctx.service::<InvariantRegistry>().unwrap();
        let duplicate = registry
            .register(
                dsh_scope::invariant::PACKAGE_NAME,
                InvariantInstaller::new(|_ctx, _fail| async { Ok(()) }),
            )
            .await;
        match duplicate {
            Ok(_) => panic!("expected the companion's reservation to hold"),
            Err(error) => assert!(
                error.to_string().contains("already registered"),
                "{error:#}"
            ),
        }

        // Unloading the companion releases the reservation.
        companion.dispose().await;
        let handle = registry
            .register(
                dsh_scope::invariant::PACKAGE_NAME,
                InvariantInstaller::new(|_ctx, _fail| async { Ok(()) }),
            )
            .await
            .unwrap();
        handle.dispose().await;
    });
}
