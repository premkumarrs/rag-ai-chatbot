"""CSV parser."""

from __future__ import annotations

import csv
from pathlib import Path

from app.config import CSV_ROWS_PER_CHUNK
from app.ingestion.parsers.base import BaseParser, build_parsed_document


class CSVParser(BaseParser):
    file_type = "csv"
    extensions = {".csv"}

    def parse(self, file_path: Path, source_path: str):
        sections: list[tuple[str, dict[str, object]]] = []
        with file_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
            except csv.Error:
                dialect = csv.excel
            reader = csv.reader(handle, dialect)
            rows = list(reader)

        if not rows:
            return build_parsed_document(file_path, source_path, self.file_type, sections)

        headers = rows[0]
        data_rows = rows[1:] if len(rows) > 1 else []
        header_line = " | ".join(headers)

        for start in range(0, len(data_rows), CSV_ROWS_PER_CHUNK):
            batch = data_rows[start : start + CSV_ROWS_PER_CHUNK]
            lines = [header_line]
            for row in batch:
                padded = row + [""] * max(0, len(headers) - len(row))
                pairs = [
                    f"{headers[index]}: {padded[index]}"
                    for index in range(len(headers))
                    if padded[index].strip()
                ]
                if pairs:
                    lines.append(" | ".join(pairs))
            sections.append(
                (
                    "\n".join(lines),
                    {
                        "source_path": source_path,
                        "filename": file_path.name,
                        "file_type": self.file_type,
                        "row_start": start + 1,
                        "row_end": start + len(batch),
                    },
                )
            )

        return build_parsed_document(file_path, source_path, self.file_type, sections)
