//! Port of `packages/util/launch-environment`
//! (`@deepseek-ai/dsh-launch-environment`): an immutable launch-time
//! environment snapshot recording which layer supplied each value. Harness
//! consumers resolve through the snapshot instead of a flattened process
//! environment, so a later working-directory change, workspace switch, or
//! resumed session observes the same values a consumer resolved at boot.
//!
//! Divergences from the TS original:
//! - The Cordis integration (`DSH_LAUNCH_ENVIRONMENT_KEY`,
//!   `launchEnvironmentOf(ctx)`) is not ported; this crate is the pure data
//!   structure. The upstream fallback — the inherited environment as the sole
//!   layer — is available as
//!   [`LaunchEnvironmentSnapshot::from_process_env`].
//! - Upstream immutability-by-freeze becomes immutability-by-API: the
//!   constructor takes each layer's values by value and the snapshot exposes
//!   no mutating method.
//! - The Cordis `./invariant` companion is not ported: the package declares no
//!   runtime invariant.

use std::collections::HashMap;
use std::path::PathBuf;

/// Which layer supplied a value, from most to least trusted: the environment
/// this process inherited, the invoking directory's `.env`, the Harness
/// home's `.env`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LaunchEnvironmentSource {
    /// The environment the process inherited.
    Process,
    /// The invoking directory's `.env` file.
    ProjectEnv,
    /// The Harness home's `.env` file.
    UserEnv,
}

/// Layer order, most trusted first.
const SOURCE_ORDER: [LaunchEnvironmentSource; 3] = [
    LaunchEnvironmentSource::Process,
    LaunchEnvironmentSource::ProjectEnv,
    LaunchEnvironmentSource::UserEnv,
];

/// One resolved variable and the layer it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchEnvironmentEntry {
    /// The value as the layer supplied it; may be empty, which each owner
    /// judges for itself.
    pub value: String,
    /// The layer that supplied it.
    pub source: LaunchEnvironmentSource,
    /// Absolute path of the file that supplied it; `None` for
    /// [`LaunchEnvironmentSource::Process`].
    pub path: Option<PathBuf>,
}

/// One layer's raw contents, as [`LaunchEnvironmentSnapshot::new`] receives
/// them.
#[derive(Debug, Clone)]
pub struct LaunchEnvironmentLayerInput {
    /// The layer these values belong to.
    pub source: LaunchEnvironmentSource,
    /// Absolute path of the file behind this layer; `None` for the process
    /// layer.
    pub path: Option<PathBuf>,
    /// The layer's name-to-value mapping.
    pub values: HashMap<String, String>,
}

/// The map key one variable name resolves under. Windows treats environment
/// names case-insensitively; every other platform does not.
fn lookup_key(name: &str) -> String {
    if cfg!(windows) {
        name.to_uppercase()
    } else {
        name.to_owned()
    }
}

struct Layer {
    path: Option<PathBuf>,
    values: HashMap<String, String>,
}

/// The frozen environment of one launch. Construct through
/// [`LaunchEnvironmentSnapshot::new`]; nothing mutates it afterwards.
pub struct LaunchEnvironmentSnapshot {
    by_source: HashMap<LaunchEnvironmentSource, Layer>,
}

impl LaunchEnvironmentSnapshot {
    /// Build the snapshot from each layer's contents. Layers may arrive in
    /// any order; lookups always search them in canonical trust order. Names
    /// are folded on Windows so case variants cannot split precedence; other
    /// platforms stay exact.
    pub fn new(layers: impl IntoIterator<Item = LaunchEnvironmentLayerInput>) -> Self {
        let mut by_source = HashMap::new();
        for layer in layers {
            by_source.insert(
                layer.source,
                Layer {
                    path: layer.path,
                    values: layer
                        .values
                        .into_iter()
                        .map(|(name, value)| (lookup_key(&name), value))
                        .collect(),
                },
            );
        }
        Self { by_source }
    }

    /// The inherited process environment as the sole layer — the fallback a
    /// composition uses when no launcher supplied a snapshot.
    pub fn from_process_env() -> Self {
        Self::new([LaunchEnvironmentLayerInput {
            source: LaunchEnvironmentSource::Process,
            path: None,
            values: std::env::vars().collect(),
        }])
    }

    /// Resolve one name across every layer, most trusted first.
    ///
    /// Returns the winning entry, or `None` when no layer supplies it.
    pub fn get(&self, name: &str) -> Option<LaunchEnvironmentEntry> {
        self.get_from(name, &SOURCE_ORDER)
    }

    /// Resolve one name only from `sources`, retaining canonical trust order;
    /// omitted layers are unreachable regardless of the order `sources` lists
    /// them in.
    pub fn get_from(
        &self,
        name: &str,
        sources: &[LaunchEnvironmentSource],
    ) -> Option<LaunchEnvironmentEntry> {
        let key = lookup_key(name);
        for source in SOURCE_ORDER {
            if !sources.contains(&source) {
                continue;
            }
            let Some(layer) = self.by_source.get(&source) else {
                continue;
            };
            let Some(value) = layer.values.get(&key) else {
                continue;
            };
            return Some(LaunchEnvironmentEntry {
                value: value.clone(),
                source,
                path: layer.path.clone(),
            });
        }
        None
    }
}
