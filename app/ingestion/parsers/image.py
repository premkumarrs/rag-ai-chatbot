"""Image parser using OCR."""

from __future__ import annotations

import logging
from pathlib import Path

from app.ingestion.parsers.base import BaseParser, build_parsed_document
from app.ingestion.parsers.ocr import get_ocr_engine

logger = logging.getLogger(__name__)


class ImageParser(BaseParser):
    file_type = "image"
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    def parse(self, file_path: Path, source_path: str):
        ocr_engine = get_ocr_engine()
        if not ocr_engine.available:
            raise RuntimeError(
                "OCR is required for image ingestion but is not available."
            )
        text = ocr_engine.extract_text_from_image(file_path)
        sections = [
            (
                text,
                {
                    "source_path": source_path,
                    "filename": file_path.name,
                    "file_type": self.file_type,
                    "ocr_used": True,
                },
            )
        ]
        return build_parsed_document(file_path, source_path, self.file_type, sections)
