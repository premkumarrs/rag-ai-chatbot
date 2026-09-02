"""Normalized ingestion document models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentSection:
    """A logical section of a source document before chunking."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """Normalized parser output for a single source file."""

    source_path: str
    filename: str
    file_type: str
    title: str
    content_hash: str
    sections: list[DocumentSection] = field(default_factory=list)

    @property
    def has_content(self) -> bool:
        return any(section.content.strip() for section in self.sections)
