"""
PaddlePaddle OCR module - Modal Function

Deploy:
  modal deploy deploy.py
"""

from __future__ import annotations

import logging
import modal
from tongflow import deploy
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Optional, cast





from tongflow.models.parse_document import ParseDocumentInput, ParseDocumentOutput
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset_as_path
from tongflow.slots import node_slot

image = (
    modal.Image.from_registry("paddlepaddle/paddle:3.2.0")
    .pip_install(
        "tongflow==0.1.0",
        "paddleocr[all]",
    )
)

app = modal.App(Path(__file__).resolve().parent.name, image=image)
secrets = modal.Secret.from_dict({})

# Plugin-internal default; not exposed via ABI.
DEFAULT_INFERENCE_MODE = "ocr"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =========================
# Helpers
# =========================
def download_image(url: str, save_dir: str = None) -> str:
    """Download an image to a local temp file."""
    if save_dir is None:
        save_dir = tempfile.gettempdir()

    # Derive file extension from URL
    ext = os.path.splitext(url.split("?")[0])[-1] or ".png"
    local_path = os.path.join(save_dir, f"input_image_{int(time.time())}{ext}")

    logger.info(f"download image: {url} -> {local_path}")
    urllib.request.urlretrieve(url, local_path)

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"download failed: {local_path}")

    file_size = os.path.getsize(local_path)
    logger.info(f"image downloaded: {local_path}, size: {file_size} bytes")

    return local_path


# =========================
# PaddleOCR inference functions
# =========================
def run_paddleocr_ocr(image_path: str, **kwargs) -> dict:
    """Run PP-OCRv5 inference via Python API."""
    try:
        from paddleocr import PaddleOCR

        # Initialize PP-OCRv5
        # Newer PaddleOCR versions changed parameters; unsupported ones removed.
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang='ch',
            use_gpu=False
        )

        # Run OCR
        result = ocr.ocr(image_path, cls=True)
        
        if result is None or len(result) == 0:
            return {
                "success": True,
                "model": "PP-OCRv5",
                "output": "",
                "texts": []
            }
        
        # Extract text results
        texts = []
        full_text_lines = []

        for page_result in result:
            if page_result is None:
                continue
            for line in page_result:
                if line and len(line) >= 2:
                    box = line[0]  # bounding box
                    text_info = line[1]  # (text, confidence)
                    text = text_info[0]
                    confidence = text_info[1]
                    texts.append({
                        "text": text,
                        "confidence": round(confidence, 4),
                        "box": box
                    })
                    full_text_lines.append(text)
        
        full_text = "\n".join(full_text_lines)
        
        return {
            "success": True,
            "model": "PP-OCRv5",
            "output": full_text,
            "texts": texts
        }
    except Exception as e:
        logger.error(f"OCR inference error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def run_paddleocr_structure(image_path: str, **kwargs) -> dict:
    """Run PP-StructureV3 inference via CLI."""
    try:
        cmd = [
            "paddleocr", "pp_structurev3",
            "-i", image_path,
            "--use_doc_orientation_classify", "False",
            "--use_doc_unwarping", "False"
        ]
        
        logger.info(f"running command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )

        # Merge stdout and stderr
        output = result.stdout + result.stderr
        logger.info(f"Structure command output length: stdout={len(result.stdout)}, stderr={len(result.stderr)}")
        
        if result.returncode != 0:
            return {
                "success": False,
                "error": f"Structure recognition failed: {output}"
            }
        
        return {
            "success": True,
            "model": "PP-StructureV3",
            "output": output
        }
    except Exception as e:
        logger.error(f"Structure inference error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def run_paddleocr_chatocrv4(image_path: str, query_key: str, qianfan_api_key: str, **kwargs) -> dict:
    """Run PP-ChatOCRv4 inference via CLI."""
    if not qianfan_api_key:
        return {
            "success": False,
            "error": "qianfan_api_key is required for PP-ChatOCRv4"
        }
    
    try:
        cmd = [
            "paddleocr", "pp_chatocrv4_doc",
            "-i", image_path,
            "-k", query_key or "",
            "--qianfan_api_key", qianfan_api_key,
            "--use_doc_orientation_classify", "False",
            "--use_doc_unwarping", "False"
        ]
        
        logger.info("running ChatOCRv4 command")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )

        output = result.stdout + result.stderr
        logger.info(f"ChatOCRv4 command output length: stdout={len(result.stdout)}, stderr={len(result.stderr)}")
        
        if result.returncode != 0:
            return {
                "success": False,
                "error": f"ChatOCRv4 failed: {output}"
            }
        
        return {
            "success": True,
            "model": "PP-ChatOCRv4",
            "output": output
        }
    except Exception as e:
        logger.error(f"ChatOCRv4 inference error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def run_paddleocr_vl(image_path: str, **kwargs) -> dict:
    """Run PaddleOCR-VL document parsing via CLI."""
    try:
        cmd = [
            "paddleocr", "doc_parser",
            "-i", image_path
        ]
        
        logger.info(f"running command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )

        output = result.stdout + result.stderr
        logger.info(f"VL command output length: stdout={len(result.stdout)}, stderr={len(result.stderr)}")
        
        if result.returncode != 0:
            return {
                "success": False,
                "error": f"Document parsing failed: {output}"
            }
        
        return {
            "success": True,
            "model": "PaddleOCR-VL",
            "output": output
        }
    except Exception as e:
        logger.error(f"VL inference error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# Inference-mode dispatch table
INFERENCE_MODES = {
    "ocr": run_paddleocr_ocr,
    "structure": run_paddleocr_structure,
    "chatocrv4": run_paddleocr_chatocrv4,
    "vl": run_paddleocr_vl,
}


# =========================
# Modal entry
# =========================
def _paddle_infer_core(task: dict) -> dict:
    task_id = task.get("taskId")
    prompt = task.get("prompt", {})
    image_url = prompt.get("image_url")
    mode = prompt.get("mode", "ocr")
    local_image_path = None

    try:
        logger.info(f"[{task_id}] start inference: mode={mode}, image={image_url}")

        # Check mode is supported
        if mode not in INFERENCE_MODES:
            error_msg = f"unsupported inference mode: {mode}"
            logger.error(f"[{task_id}] {error_msg}")
            return {"success": False, "error": error_msg}

        # Download image locally
        logger.info(f"[{task_id}] downloading image...")
        local_image_path = download_image(image_url)
        logger.info(f"[{task_id}] image downloaded: {local_image_path}")

        # Pick the inference function
        infer_func = INFERENCE_MODES[mode]

        # Run inference using the local path
        infer_kwargs = {
            "image_path": local_image_path,
            "query_key": prompt.get("query_key"),
            "qianfan_api_key": prompt.get("qianfan_api_key"),
        }

        inference_result = infer_func(**infer_kwargs)

        if not inference_result.get("success"):
            error_msg = inference_result.get("error", "unknown error")
            logger.error(f"[{task_id}] inference failed: {error_msg}")
            return {"success": False, "error": error_msg}

        # Build response — primitive Python types only
        response_data = {
            "success": True,
            "taskId": task_id,
            "mode": mode,
            "image_url": image_url,
            "model": inference_result.get("model"),
            "output": inference_result.get("output"),
            "output_length": len(inference_result.get("output", "")),
        }

        # Append per-line text results if available
        if "texts" in inference_result:
            response_data["texts"] = inference_result["texts"]
            response_data["text_count"] = len(inference_result["texts"])

        logger.info(f"[{task_id}] inference ok: mode={mode} - output length: {response_data['output_length']}")

        return response_data

    except Exception as e:
        error_msg = f"inference error: {str(e)}"
        logger.error(f"[{task_id}] {error_msg}", exc_info=True)
        return {"success": False, "error": error_msg}
    finally:
        # Clean up the temp file
        if local_image_path and os.path.exists(local_image_path):
            try:
                os.remove(local_image_path)
                logger.info(f"[{task_id}] cleaned up temp file: {local_image_path}")
            except Exception as e:
                logger.warning(f"failed to clean temp file: {e}")


@app.function(cpu=2.0, memory=4096, timeout=600, secrets=[secrets], scaledown_window=5)
def paddle_infer(task: dict) -> dict:
    return _paddle_infer_core(task)


@deploy
@app.cls(cpu=2.0, memory=4096, timeout=600, secrets=[secrets], scaledown_window=5)
class Inference:
    @modal.method()
    @node_slot(NodeSlots.PARSE_DOCUMENT)
    def parse_document(self, input: ParseDocumentInput) -> ParseDocumentOutput:
        if input.document is None:
            return ParseDocumentOutput(
                success=False, error="Missing `document` Asset"
            )
        with asset_as_path(input.document, suffix=".bin") as image_path:
            result = INFERENCE_MODES[DEFAULT_INFERENCE_MODE](
                image_path=str(image_path),
                query_key=None,
                qianfan_api_key=None,
            )
            if not result.get("success"):
                return ParseDocumentOutput(
                    success=False,
                    error=str(result.get("error") or "infer failed"),
                )
            return ParseDocumentOutput(
                success=True,
                text=str(result.get("output") or ""),
            )
