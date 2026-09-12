"""File discovery and parser routing."""

from __future__ import annotations

import logging
from pathlib import Path

from app.ingestion.models import ParsedDocument
from app.ingestion.parsers import EXTENSION_TO_PARSER, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

IGNORED_FILE_NAMES = {".env", ".gitignore"}
IGNORED_EXTENSIONS = {
    ".exe",
    ".dll",
    ".bin",
    ".zip",
    ".rar",
    ".7z",
    ".pyc",
    ".pyo",
    ".pem",
    ".key",
}


def to_source_path(file_path: Path, source_root: Path) -> str:
    try:
        return file_path.relative_to(source_root).as_posix()
    except ValueError:
        return file_path.name


def discover_files(data_dir: Path) -> list[Path]:
    """Discover supported files only in data_dir itself (no subfolders)."""
    if not data_dir.is_dir():
        return []

    discovered: list[Path] = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name in IGNORED_FILE_NAMES:
            continue
        extension = path.suffix.lower()
        if extension in IGNORED_EXTENSIONS:
            continue
        discovered.append(path)
    return discovered


def parse_file(file_path: Path, project_root: Path) -> ParsedDocument | None:
    extension = file_path.suffix.lower()
    parser = EXTENSION_TO_PARSER.get(extension)
    source_path = to_source_path(file_path, project_root)

    if parser is None:
        logger.info("Unsupported file skipped: %s", source_path)
        return None

    return parser.parse(file_path, source_path)


def is_supported_extension(extension: str) -> bool:
    return extension.lower() in SUPPORTED_EXTENSIONS
