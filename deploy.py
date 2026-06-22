"""Modal deploy entry for paddle.

Document-to-text (OCR) via PaddleOCR (PP-OCRv5).

Deploy:
  modal deploy deploy.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import modal
from tongflow import deploy

from tongflow.models.parse_document import ParseDocumentInput, ParseDocumentOutput
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset_as_path
from tongflow.slots import node_slot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

image = (
    modal.Image.from_registry("paddlepaddle/paddle:3.2.0")
    .pip_install(
        "tongflow==0.1.0",
        "paddleocr[all]",
    )
)

app = modal.App(Path(__file__).resolve().parent.name, image=image)
secrets = modal.Secret.from_dict({})

# Plugin-internal defaults; not exposed via ABI.
OCR_LANG = "ch"


def run_ocr(image_path: str) -> str:
    """Run PP-OCRv5 OCR and return the recognized text as newline-joined lines."""
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=True, lang=OCR_LANG, use_gpu=False)
    result = ocr.ocr(image_path, cls=True)
    if not result:
        return ""

    lines: list[str] = []
    for page_result in result:
        if not page_result:
            continue
        for line in page_result:
            if line and len(line) >= 2:
                text_info = line[1]  # (text, confidence)
                lines.append(text_info[0])
    return "\n".join(lines)


@deploy
@app.cls(cpu=2.0, memory=4096, timeout=600, secrets=[secrets], scaledown_window=5)
class Inference:
    @modal.method()
    @node_slot(NodeSlots.PARSE_DOCUMENT)
    def parse_document(self, input: ParseDocumentInput) -> ParseDocumentOutput:
        if input.document is None:
            return ParseDocumentOutput(success=False, error="Missing `document` Asset")
        try:
            with asset_as_path(input.document, suffix=".bin") as image_path:
                text = run_ocr(str(image_path))
            return ParseDocumentOutput(success=True, text=text)
        except Exception as e:
            logger.error(f"OCR inference error: {e}", exc_info=True)
            return ParseDocumentOutput(success=False, error=f"infer error: {e}")
