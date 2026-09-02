"""Parser base types and helpers."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.ingestion.models import ParsedDocument


def file_content_hash(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parsed_document(
    file_path: Path,
    source_path: str,
    file_type: str,
    sections: list[tuple[str, dict[str, Any]]],
) -> ParsedDocument:
    from app.ingestion.models import DocumentSection

    parsed_sections = [
        DocumentSection(content=content, metadata=metadata)
        for content, metadata in sections
        if content.strip()
    ]
    return ParsedDocument(
        source_path=source_path,
        filename=file_path.name,
        file_type=file_type,
        title=file_path.stem,
        content_hash=file_content_hash(file_path),
        sections=parsed_sections,
    )


class BaseParser(ABC):
    file_type: str
    extensions: set[str]

    @abstractmethod
    def parse(self, file_path: Path, source_path: str) -> ParsedDocument:
        """Parse a file into normalized sections."""
