# tongflow-modal-paddle

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Document-to-text (OCR) extraction with **PaddleOCR**, running on [Modal](https://modal.com). An alternative to `tongflow-modal-docling` on the same `parse-document` slot.

## Capabilities

- **Document → text** (`parse-document`) — OCR a document or scanned page into plain text.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |

On first use the plugin deploys to your Modal account automatically and caches the build. No Hugging Face token required.
