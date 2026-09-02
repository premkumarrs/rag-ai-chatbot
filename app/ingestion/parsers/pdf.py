"""PDF parser with optional OCR fallback."""

from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader

from app.config import OCR_MIN_TEXT_CHARS
from app.ingestion.models import ParsedDocument
from app.ingestion.parsers.base import BaseParser, build_parsed_document
from app.ingestion.parsers.ocr import get_ocr_engine

logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    file_type = "pdf"
    extensions = {".pdf"}

    def parse(self, file_path: Path, source_path: str) -> ParsedDocument:
        reader = PdfReader(str(file_path))
        sections: list[tuple[str, dict[str, object]]] = []
        ocr_engine = get_ocr_engine()
        ocr_pages = 0
        ocr_failures = 0

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            used_ocr = False

            if len(text) < OCR_MIN_TEXT_CHARS and ocr_engine.available:
                try:
                    ocr_text = self._ocr_page(page, ocr_engine)
                    if len(ocr_text.strip()) >= OCR_MIN_TEXT_CHARS:
                        text = ocr_text.strip()
                        used_ocr = True
                        ocr_pages += 1
                except Exception as exc:
                    ocr_failures += 1
                    logger.warning(
                        "OCR failed for %s page %s: %s",
                        source_path,
                        page_number,
                        exc,
                    )

            if not text:
                continue

            sections.append(
                (
                    text,
                    {
                        "source_path": source_path,
                        "filename": file_path.name,
                        "file_type": self.file_type,
                        "page_number": page_number,
                        "ocr_used": used_ocr,
                    },
                )
            )

        parsed = build_parsed_document(file_path, source_path, self.file_type, sections)
        parsed.sections  # noqa: B018 - ensure built
        if ocr_pages or ocr_failures:
            logger.info(
                "PDF OCR summary for %s: ocr_pages=%s ocr_failures=%s",
                source_path,
                ocr_pages,
                ocr_failures,
            )
        return parsed

    def _ocr_page(self, page, ocr_engine) -> str:
        images = page.images if hasattr(page, "images") else []
        if images:
            return ocr_engine.extract_text_from_pdf_page_image(images[0].data)
        return ""
