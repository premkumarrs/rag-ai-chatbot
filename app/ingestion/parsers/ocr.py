"""OCR helpers for scanned PDF pages and image files."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from app.config import OCR_ENABLED

logger = logging.getLogger(__name__)


class OCREngine:
    """Local OCR wrapper with replaceable backend."""

    def __init__(self) -> None:
        self._available = False
        self._pytesseract = None
        self._Image = None
        if not OCR_ENABLED:
            return
        try:
            import pytesseract
            from PIL import Image

            if shutil.which("tesseract") is None:
                logger.warning(
                    "OCR enabled but Tesseract executable not found on PATH. "
                    "Install Tesseract OCR to use scanned-document ingestion."
                )
                return
            self._pytesseract = pytesseract
            self._Image = Image
            self._available = True
        except ImportError:
            logger.warning(
                "OCR dependencies not installed. Install pytesseract and Pillow."
            )

    @property
    def available(self) -> bool:
        return self._available

    def extract_text_from_image(self, image_path: Path) -> str:
        if not self._available or self._pytesseract is None or self._Image is None:
            raise RuntimeError("OCR is not available.")
        image = self._Image.open(image_path)
        return self._pytesseract.image_to_string(image).strip()

    def extract_text_from_pdf_page_image(self, image_bytes: bytes) -> str:
        if not self._available or self._pytesseract is None or self._Image is None:
            raise RuntimeError("OCR is not available.")
        from io import BytesIO

        image = self._Image.open(BytesIO(image_bytes))
        return self._pytesseract.image_to_string(image).strip()


_ocr_engine: OCREngine | None = None


def get_ocr_engine() -> OCREngine:
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = OCREngine()
    return _ocr_engine
