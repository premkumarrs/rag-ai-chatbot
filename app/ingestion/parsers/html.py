"""HTML parser."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from app.ingestion.parsers.base import BaseParser, build_parsed_document


class HTMLParser(BaseParser):
    file_type = "html"
    extensions = {".html", ".htm"}

    def parse(self, file_path: Path, source_path: str):
        html = file_path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
            tag.decompose()

        sections: list[tuple[str, dict[str, object]]] = []
        current_heading: str | None = None

        for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table"]):
            if element.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                current_heading = element.get_text(" ", strip=True)
                if current_heading:
                    sections.append(
                        (
                            current_heading,
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

            if element.name == "table":
                rows = []
                for row in element.find_all("tr"):
                    cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
                    row_text = " | ".join(cell for cell in cells if cell)
                    if row_text:
                        rows.append(row_text)
                if rows:
                    sections.append(
                        (
                            "\n".join(rows),
                            {
                                "source_path": source_path,
                                "filename": file_path.name,
                                "file_type": self.file_type,
                                "section": current_heading,
                                "content_type": "table",
                            },
                        )
                    )
                continue

            text = element.get_text(" ", strip=True)
            if text:
                sections.append(
                    (
                        text,
                        {
                            "source_path": source_path,
                            "filename": file_path.name,
                            "file_type": self.file_type,
                            "section": current_heading,
                        },
                    )
                )

        return build_parsed_document(file_path, source_path, self.file_type, sections)
