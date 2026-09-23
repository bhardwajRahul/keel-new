//! Seam smoke tests. Upstream `packages/attachment/attachment` ships no test
//! suite (its behavior is pinned by provider suites); these checks pin the
//! Rust-visible surface instead: the durable reference's wire encoding and
//! the service mounting/removal lifecycle.

use dsh_attachment::{
    AttachmentError, AttachmentId, AttachmentStore, Attachments, ImageAttachmentLimits,
    ImageAttachmentRef, ImageMediaType, SaveImageAttachment, StoredImageAttachment,
};
use dsh_cordis::{App, Context, Plugin};
use futures::future::LocalBoxFuture;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::rc::Rc;

/// Content-addressed in-memory store: ids are the SHA-256 of the bytes.
struct MemoryStore {
    limits: ImageAttachmentLimits,
}

fn digest_id(data: &[u8]) -> AttachmentId {
    let mut hasher = Sha256::new();
    hasher.update(data);
    AttachmentId::new(format!("{:x}", hasher.finalize()))
}

#[async_trait::async_trait(?Send)]
impl AttachmentStore for MemoryStore {
    fn image_limits(&self) -> &ImageAttachmentLimits {
        &self.limits
    }

    async fn validate_image(&self, input: &SaveImageAttachment) -> Result<(), AttachmentError> {
        if input.data.len() as u64 > self.limits.max_image_bytes {
            return Err(AttachmentError::new("image too large", "IMAGE_TOO_LARGE"));
        }
        Ok(())
    }

    async fn save_image(
        &self,
        input: &SaveImageAttachment,
    ) -> Result<ImageAttachmentRef, AttachmentError> {
        self.validate_image(input).await?;
        Ok(ImageAttachmentRef {
            attachment_id: digest_id(&input.data),
            media_type: input.media_type,
            bytes: input.data.len() as u64,
            width: 1,
            height: 1,
            name: input.name.clone(),
        })
    }

    async fn read_image(
        &self,
        reference: &ImageAttachmentRef,
    ) -> Result<StoredImageAttachment, AttachmentError> {
        Err(AttachmentError::new(
            format!("attachment {} is not stored", reference.attachment_id),
            "ATTACHMENT_MISSING",
        ))
    }
}

struct MemoryStorePlugin;

impl Plugin for MemoryStorePlugin {
    fn name(&self) -> Option<String> {
        Some("memory-attachments".into())
    }

    fn apply(&self, ctx: Context, _config: Value) -> LocalBoxFuture<'static, anyhow::Result<()>> {
        Box::pin(async move {
            let store = Rc::new(MemoryStore {
                limits: ImageAttachmentLimits {
                    max_image_bytes: 8,
                    max_images_per_message: 4,
                    max_message_image_bytes: 32,
                    max_image_pixels: 1_000_000,
                    media_types: vec![ImageMediaType::Png, ImageMediaType::Jpeg],
                },
            });
            ctx.provide_service(Rc::new(Attachments(store)))?;
            Ok(())
        })
    }
}

#[test]
fn mounts_as_attachments_and_round_trips_the_store_contract() {
    dsh_cordis::run(async {
        let app = App::new();
        let ctx = app.root();
        let fiber = ctx.plugin(Rc::new(MemoryStorePlugin), Value::Null).unwrap();
        fiber.await_ready().await.unwrap();

        let store = ctx.try_service::<Attachments>().unwrap();
        assert_eq!(store.image_limits().max_image_bytes, 8);

        let input = SaveImageAttachment {
            data: vec![1, 2, 3],
            media_type: ImageMediaType::Png,
            name: Some("shot.png".into()),
        };
        store.validate_image(&input).await.unwrap();
        let reference = store.save_image(&input).await.unwrap();
        // Content-addressed: the same bytes yield the same identifier.
        assert_eq!(
            reference.attachment_id,
            store.save_image(&input).await.unwrap().attachment_id
        );
        assert_eq!(reference.bytes, 3);

        let oversized = SaveImageAttachment {
            data: vec![0; 9],
            media_type: ImageMediaType::Png,
            name: None,
        };
        let error = store.validate_image(&oversized).await.unwrap_err();
        assert_eq!(error.code, "IMAGE_TOO_LARGE");

        let missing = store.read_image(&reference).await.unwrap_err();
        assert_eq!(missing.code, "ATTACHMENT_MISSING");

        fiber.dispose().await;
        assert!(ctx.try_service::<Attachments>().is_none());
    });
}

#[test]
fn image_ref_serializes_with_camel_case_keys_and_mime_media_types() {
    let reference = ImageAttachmentRef {
        attachment_id: AttachmentId::new("abc123"),
        media_type: ImageMediaType::Webp,
        bytes: 42,
        width: 640,
        height: 480,
        name: None,
    };
    let encoded = serde_json::to_value(&reference).unwrap();
    assert_eq!(
        encoded,
        json!({
            "attachmentId": "abc123",
            "mediaType": "image/webp",
            "bytes": 42,
            "width": 640,
            "height": 480,
        })
    );
    let decoded: ImageAttachmentRef = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, reference);
    assert_eq!(ImageMediaType::Jpeg.to_string(), "image/jpeg");
}
