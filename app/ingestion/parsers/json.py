"""JSON parser."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.ingestion.parsers.base import BaseParser, build_parsed_document


def _json_to_text(value: Any, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_prefix = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            if isinstance(item, (dict, list)):
                lines.extend(_json_to_text(item, key_prefix))
            else:
                lines.append(f"{key_prefix}: {item}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            item_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            if isinstance(item, (dict, list)):
                lines.extend(_json_to_text(item, item_prefix))
            else:
                lines.append(f"{item_prefix}: {item}")
    else:
        lines.append(f"{prefix}: {value}" if prefix else str(value))
    return lines


class JSONParser(BaseParser):
    file_type = "json"
    extensions = {".json"}

    def parse(self, file_path: Path, source_path: str):
        raw = file_path.read_text(encoding="utf-8", errors="replace")
        data = json.loads(raw)
        text = "\n".join(_json_to_text(data))
        sections = [
            (
                text,
                {
                    "source_path": source_path,
                    "filename": file_path.name,
                    "file_type": self.file_type,
                },
            )
        ]
        return build_parsed_document(file_path, source_path, self.file_type, sections)
