"""Markdown parser."""

from __future__ import annotations

from pathlib import Path

from app.ingestion.parsers.base import BaseParser, build_parsed_document


class MarkdownParser(BaseParser):
    file_type = "md"
    extensions = {".md", ".markdown"}

    def parse(self, file_path: Path, source_path: str):
        text = file_path.read_text(encoding="utf-8", errors="replace")
        sections: list[tuple[str, dict[str, object]]] = []
        current_heading: str | None = None
        buffer: list[str] = []

        def flush_buffer() -> None:
            nonlocal buffer
            if not buffer:
                return
            sections.append(
                (
                    "\n".join(buffer).strip(),
                    {
                        "source_path": source_path,
                        "filename": file_path.name,
                        "file_type": self.file_type,
                        "section": current_heading,
                    },
                )
            )
            buffer = []

        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                flush_buffer()
                current_heading = stripped.lstrip("#").strip()
                sections.append(
                    (
                        stripped,
                        {
                            "source_path": source_path,
                            "filename": file_path.name,
                            "file_type": self.file_type,
                            "section": current_heading,
                            "content_type": "heading",
                        },
                    )
                )
                continue
            if not stripped:
                flush_buffer()
                continue
            buffer.append(line)

        flush_buffer()
        return build_parsed_document(file_path, source_path, self.file_type, sections)
