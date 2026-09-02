"""Plain text parser."""

from __future__ import annotations

from pathlib import Path

from app.ingestion.parsers.base import BaseParser, build_parsed_document


class TXTParser(BaseParser):
    file_type = "txt"
    extensions = {".txt"}

    def parse(self, file_path: Path, source_path: str):
        raw = file_path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
        sections = [
            (
                paragraph,
                {
                    "source_path": source_path,
                    "filename": file_path.name,
                    "file_type": self.file_type,
                },
            )
            for paragraph in paragraphs
        ]
        return build_parsed_document(file_path, source_path, self.file_type, sections)
