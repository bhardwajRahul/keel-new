//! Redaction suite ported from upstream `tests/redact.spec.ts`. The
//! "tolerates structural nodes missing their relation maps" case is not
//! ported: this crate's schema nodes are total by construction. Dict entries
//! walk in key order (`serde_json::Map` is sorted), so secret positions from
//! dict containers list alphabetically rather than in insertion order.

mod common;

use common::boot;
use dsh_settings::{
    RedactedSecret, Schema, SettingsDescribeOptions, SettingsNamespace, SettingsRegisterOptions,
    redact_secrets, settings_namespace,
};
use serde_json::{Value, json};

fn ns(value: &str) -> SettingsNamespace {
    settings_namespace(value).unwrap()
}

fn profile() -> Schema {
    Schema::object([
        ("apiKey", Schema::string().role("secret")),
        ("apiKeyEnv", Schema::string().role("credential-ref")),
        ("baseURL", Schema::string()),
    ])
}

fn adapter() -> Schema {
    Schema::object([
        ("apiKey", Schema::string().role("secret")),
        ("providers", Schema::dict(profile())),
        ("fallbacks", Schema::array(profile())),
        (
            "nested",
            Schema::object([("token", Schema::string().role("secret"))]),
        ),
    ])
}

fn secret(path: &[&str], set: bool) -> RedactedSecret {
    RedactedSecret {
        path: path.iter().map(|s| s.to_string()).collect(),
        set,
    }
}

#[test]
fn strips_secrets_from_object_dict_and_array_containers_and_records_each_position() {
    let input = json!({
        "apiKey": "top-secret",
        "providers": {
            "openai": {"apiKey": "sk-live", "apiKeyEnv": "OPENAI_API_KEY", "baseURL": "https://x"},
            "anthropic": {"apiKeyEnv": "ANTHROPIC_API_KEY"},
        },
        "fallbacks": [{"apiKey": "fb", "baseURL": "https://y"}],
        "nested": {},
    });
    let redacted = redact_secrets(&adapter(), Some(&input));
    assert_eq!(
        redacted.value,
        Some(json!({
            "providers": {
                "openai": {"apiKeyEnv": "OPENAI_API_KEY", "baseURL": "https://x"},
                "anthropic": {"apiKeyEnv": "ANTHROPIC_API_KEY"},
            },
            "fallbacks": [{"baseURL": "https://y"}],
            "nested": {},
        }))
    );
    // Dict entries list in key order (anthropic before openai) — see the
    // module doc.
    assert_eq!(
        redacted.secrets,
        vec![
            secret(&["apiKey"], true),
            secret(&["providers", "anthropic", "apiKey"], false),
            secret(&["providers", "openai", "apiKey"], true),
            secret(&["fallbacks", "0", "apiKey"], true),
            secret(&["nested", "token"], false),
        ]
    );
}

#[test]
fn enumerates_unset_object_property_slots_without_inventing_containers() {
    let redacted = redact_secrets(&adapter(), None);
    assert_eq!(redacted.value, None);
    assert_eq!(
        redacted.secrets,
        vec![
            secret(&["apiKey"], false),
            secret(&["nested", "token"], false)
        ]
    );
}

#[test]
fn never_mutates_the_input_and_preserves_keys_outside_the_schema() {
    let input = json!({"apiKey": "frozen", "extra": {"keep": true}});
    let redacted = redact_secrets(&adapter(), Some(&input));
    assert_eq!(input["apiKey"], json!("frozen"));
    assert_eq!(redacted.value, Some(json!({"extra": {"keep": true}})));
}

#[test]
fn passes_malformed_container_values_through_untouched() {
    let input = json!({"providers": "not-a-dict", "fallbacks": "not-an-array"});
    let redacted = redact_secrets(&adapter(), Some(&input));
    assert_eq!(
        redacted.value,
        Some(json!({"providers": "not-a-dict", "fallbacks": "not-an-array"}))
    );
    assert_eq!(
        redacted.secrets,
        vec![
            secret(&["apiKey"], false),
            secret(&["nested", "token"], false)
        ]
    );
}

#[test]
fn treats_a_secret_role_container_as_one_opaque_secret_leaf() {
    let weird = Schema::object([(
        "blob",
        Schema::object([("inner", Schema::string())]).role("secret"),
    )]);
    let redacted = redact_secrets(&weird, Some(&json!({"blob": {"inner": "x"}})));
    assert_eq!(redacted.value, Some(json!({})));
    assert_eq!(redacted.secrets, vec![secret(&["blob"], true)]);
}

#[test]
fn drops_a_dict_entry_whose_entire_value_is_the_secret() {
    let tokens = Schema::object([("tokens", Schema::dict(Schema::string().role("secret")))]);
    let redacted = redact_secrets(&tokens, Some(&json!({"tokens": {"a": "x", "b": "y"}})));
    assert_eq!(redacted.value, Some(json!({"tokens": {}})));
    assert_eq!(
        redacted.secrets,
        vec![
            secret(&["tokens", "a"], true),
            secret(&["tokens", "b"], true)
        ]
    );
}

// -------------------------------------------- describe() layers and redaction

#[test]
fn exposes_detached_base_and_user_layers_beside_the_resolved_value() {
    dsh_cordis::run(async {
        let booted = boot(json!({"adapter": {"baseURL": "https://user"}})).await;
        let base = json!({"apiKey": "entry-key", "baseURL": "https://base"});
        booted
            .service
            .register(
                &booted.ctx,
                ns("adapter"),
                profile(),
                SettingsRegisterOptions {
                    base: Some(base.clone()),
                    ..Default::default()
                },
            )
            .unwrap();
        let descriptors = booted.service.describe(SettingsDescribeOptions::default());
        let descriptor = &descriptors[0];
        assert_eq!(descriptor.base, Some(base));
        assert_eq!(descriptor.user, Some(json!({"baseURL": "https://user"})));
        assert_eq!(
            descriptor.value,
            json!({"apiKey": "entry-key", "baseURL": "https://user"})
        );
        assert!(descriptor.secrets.is_none());
    });
}

#[test]
fn omits_the_layers_when_neither_a_base_nor_a_user_section_exists() {
    dsh_cordis::run(async {
        let booted = boot(Value::Null).await;
        booted
            .service
            .register(
                &booted.ctx,
                ns("adapter"),
                profile(),
                SettingsRegisterOptions::default(),
            )
            .unwrap();
        let descriptors = booted.service.describe(SettingsDescribeOptions::default());
        assert!(descriptors[0].base.is_none());
        assert!(descriptors[0].user.is_none());
    });
}

#[test]
fn describes_a_section_that_became_malformed_after_registration_as_having_no_user_layer() {
    dsh_cordis::run(async {
        let booted = boot(json!({"adapter": {"baseURL": "https://user"}})).await;
        booted
            .service
            .register(
                &booted.ctx,
                ns("adapter"),
                profile(),
                SettingsRegisterOptions {
                    base: Some(json!({"baseURL": "https://base"})),
                    ..Default::default()
                },
            )
            .unwrap();
        booted.backend.push_external(json!({"adapter": 5}));
        let descriptors = booted.service.describe(SettingsDescribeOptions::default());
        assert!(descriptors[0].user.is_none());
        // The malformed publish kept the last good resolved value.
        assert_eq!(descriptors[0].value, json!({"baseURL": "https://user"}));
    });
}

#[test]
fn redacts_a_descriptor_that_has_neither_base_nor_user_layer() {
    dsh_cordis::run(async {
        let booted = boot(Value::Null).await;
        booted
            .service
            .register(
                &booted.ctx,
                ns("adapter"),
                profile(),
                SettingsRegisterOptions::default(),
            )
            .unwrap();
        let descriptors = booted.service.describe(SettingsDescribeOptions {
            redact_secrets: true,
        });
        assert!(descriptors[0].base.is_none());
        assert!(descriptors[0].user.is_none());
        assert_eq!(
            descriptors[0].secrets,
            Some(vec![secret(&["apiKey"], false)])
        );
    });
}

#[test]
fn redacts_every_layer_and_enumerates_secret_slots_under_redact_secrets() {
    dsh_cordis::run(async {
        let booted =
            boot(json!({"adapter": {"apiKey": "user-key", "baseURL": "https://user"}})).await;
        booted
            .service
            .register(
                &booted.ctx,
                ns("adapter"),
                profile(),
                SettingsRegisterOptions {
                    base: Some(json!({"apiKey": "entry-key"})),
                    ..Default::default()
                },
            )
            .unwrap();
        let descriptors = booted.service.describe(SettingsDescribeOptions {
            redact_secrets: true,
        });
        let descriptor = &descriptors[0];
        assert_eq!(descriptor.value, json!({"baseURL": "https://user"}));
        assert_eq!(descriptor.base, Some(json!({})));
        assert_eq!(descriptor.user, Some(json!({"baseURL": "https://user"})));
        assert_eq!(descriptor.secrets, Some(vec![secret(&["apiKey"], true)]));
        let verbatim = booted.service.describe(SettingsDescribeOptions::default());
        assert_eq!(
            verbatim[0].value,
            json!({"apiKey": "user-key", "baseURL": "https://user"})
        );
    });
}
