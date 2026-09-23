//! Port of `packages/attachment/attachment` (`@deepseek-ai/dsh-attachment`):
//! the durable immutable attachment storage seam (`ctx.attachments`) — the
//! identifier brand, the image vocabulary, the stable failure type, and the
//! abstract store contract. Implementations validate bytes before publishing
//! a reference and live in provider crates.
//!
//! Divergences from the TS original:
//! - The abstract `AttachmentStore` service class becomes the
//!   [`AttachmentStore`] trait plus the [`Attachments`] service newtype a
//!   provider registers under the `"attachments"` name.
//! - The upstream `readImage` `AbortSignal` parameter is dropped: Rust
//!   cancels by dropping the future.
//! - The abstract `imageLimits` readonly property becomes the
//!   [`AttachmentStore::image_limits`] method.
//! - `AttachmentError` keeps its message/`code` surface (consumers route on
//!   the code, never the type hierarchy); the upstream note about avoiding a
//!   `HarnessError` dependency cycle does not apply here.
//! - The `./invariant` companion is not ported (`dsh-invariants` does not
//!   exist in this workspace).

use dsh_cordis::Service;
use serde::{Deserialize, Serialize};
use std::rc::Rc;

/// Compile-time marker for [`AttachmentId`].
pub enum AttachmentIdMark {}

/// Opaque content-addressed identifier for one immutable attachment object;
/// never a filesystem path or bearer URL.
pub type AttachmentId = dsh_brand::Branded<AttachmentIdMark>;

/// Stable attachment failure suitable for host RPC error mapping. The
/// message never carries raw bytes or host paths.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct AttachmentError {
    /// Human-readable failure description.
    pub message: String,
    /// Stable machine-routing failure code.
    pub code: String,
    /// Optional chained cause.
    #[source]
    pub cause: Option<Box<dyn std::error::Error>>,
}

impl AttachmentError {
    /// A failure with a message and routing code.
    pub fn new(message: impl Into<String>, code: impl Into<String>) -> AttachmentError {
        AttachmentError {
            message: message.into(),
            code: code.into(),
            cause: None,
        }
    }

    /// A failure chained onto its underlying cause.
    pub fn with_cause(
        message: impl Into<String>,
        code: impl Into<String>,
        cause: impl std::error::Error + 'static,
    ) -> AttachmentError {
        AttachmentError {
            message: message.into(),
            code: code.into(),
            cause: Some(Box::new(cause)),
        }
    }
}

/// Raster image formats accepted by the version-one attachment path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageMediaType {
    #[serde(rename = "image/png")]
    Png,
    #[serde(rename = "image/jpeg")]
    Jpeg,
    #[serde(rename = "image/webp")]
    Webp,
    #[serde(rename = "image/gif")]
    Gif,
}

impl ImageMediaType {
    /// The MIME string this variant serializes as.
    pub fn as_str(&self) -> &'static str {
        match self {
            ImageMediaType::Png => "image/png",
            ImageMediaType::Jpeg => "image/jpeg",
            ImageMediaType::Webp => "image/webp",
            ImageMediaType::Gif => "image/gif",
        }
    }
}

impl std::fmt::Display for ImageMediaType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Durable, serializable metadata for one immutable image object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageAttachmentRef {
    /// Opaque storage identifier.
    pub attachment_id: AttachmentId,
    /// Media type verified from the stored bytes.
    pub media_type: ImageMediaType,
    /// Exact encoded byte length.
    pub bytes: u64,
    /// Intrinsic encoded width in pixels.
    pub width: u32,
    /// Intrinsic encoded height in pixels.
    pub height: u32,
    /// Optional display name stripped of local path information.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Deployment-resolved limits used by upload admission and request buffering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageAttachmentLimits {
    pub max_image_bytes: u64,
    pub max_images_per_message: u32,
    pub max_message_image_bytes: u64,
    pub max_image_pixels: u64,
    pub media_types: Vec<ImageMediaType>,
}

/// Request to validate and durably commit one image.
#[derive(Debug, Clone)]
pub struct SaveImageAttachment {
    /// Encoded bytes.
    pub data: Vec<u8>,
    /// Caller-declared media type, checked against fully decoded bytes.
    pub media_type: ImageMediaType,
    /// Optional browser/provider display name; never interpreted as a path.
    pub name: Option<String>,
}

/// Stored image bytes returned after reference and digest verification.
#[derive(Debug, Clone)]
pub struct StoredImageAttachment {
    /// The canonical reference (upstream field name: `ref`).
    pub reference: ImageAttachmentRef,
    pub data: Vec<u8>,
}

/// Immutable binary attachment operations. Implementations validate bytes
/// before publishing a reference.
#[async_trait::async_trait(?Send)]
pub trait AttachmentStore: 'static {
    /// Deployment-resolved image policy used by authoritative and fast-path
    /// validation.
    fn image_limits(&self) -> &ImageAttachmentLimits;

    /// Validate one image without persisting it; completes only after the
    /// encoded raster has been fully decoded. Batch callers validate every
    /// member before saving any member.
    async fn validate_image(&self, input: &SaveImageAttachment) -> Result<(), AttachmentError>;

    /// Validate and durably commit one image before its owning session event
    /// is appended; returns a durable content-addressed reference.
    async fn save_image(
        &self,
        input: &SaveImageAttachment,
    ) -> Result<ImageAttachmentRef, AttachmentError>;

    /// Read one image and verify the bytes still match the recorded
    /// reference; fails with a storage error when verification fails.
    async fn read_image(
        &self,
        reference: &ImageAttachmentRef,
    ) -> Result<StoredImageAttachment, AttachmentError>;
}

/// The `ctx.attachments` service: a store registered under the seam name.
/// Derefs to the store so consumers call the operations directly.
pub struct Attachments(pub Rc<dyn AttachmentStore>);

impl Service for Attachments {
    const NAME: &'static str = "attachments";
}

impl std::ops::Deref for Attachments {
    type Target = dyn AttachmentStore;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}
