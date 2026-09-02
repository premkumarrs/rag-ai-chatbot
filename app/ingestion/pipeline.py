"""Knowledge-base ingestion pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from app.config import DATA_DIR
from app.database import get_connection
from app.ingestion.chunking import chunk_parsed_document
from app.ingestion.loader import discover_files, is_supported_extension, parse_file, to_source_path
from app.ingestion.parsers.base import file_content_hash
from app.ingestion.report import IngestionReport
from app.ingestion.storage import (
    clear_document_chunks,
    embed_documents,
    get_stored_content_hash,
    replace_document_chunks,
    upsert_document,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / DATA_DIR


def resolve_source_root(data_path: Path) -> Path:
    try:
        data_path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return data_path.resolve()
    return PROJECT_ROOT


def ingest_file(
    conn,
    file_path: Path,
    report: IngestionReport,
    source_root: Path = PROJECT_ROOT,
) -> None:
    source_path = to_source_path(file_path, source_root)
    print(f"Processing: {source_path}")

    current_hash = file_content_hash(file_path)
    existing_hash = get_stored_content_hash(conn, source_path)
    if existing_hash == current_hash:
        report.files_unchanged += 1
        report.record_type(file_path.suffix.lstrip(".").lower())
        print("  Unchanged: skipped re-embedding")
        return

    parsed = parse_file(file_path, source_root)
    if parsed is None:
        report.files_skipped += 1
        return

    report.record_type(parsed.file_type)

    if not parsed.has_content:
        print("  Skipped: no extractable text")
        clear_document_chunks(conn, source_path)
        conn.commit()
        report.files_skipped += 1
        return

    chunks = chunk_parsed_document(parsed)
    if not chunks:
        print("  Skipped: no chunks produced")
        clear_document_chunks(conn, source_path)
        conn.commit()
        report.files_skipped += 1
        return

    for chunk in chunks:
        if chunk.metadata.get("ocr_used"):
            report.ocr_pages += 1

    print(f"  Chunks: {len(chunks)}")
    embeddings = embed_documents(chunks)
    document_id = upsert_document(conn, parsed)
    replace_document_chunks(conn, document_id, chunks, embeddings)
    conn.commit()

    report.files_processed += 1
    report.chunks_created += len(chunks)
    report.embeddings_created += len(embeddings)
    print("  Done")


def run_ingestion(data_path: Path | None = None) -> IngestionReport:
    report = IngestionReport()
    target = data_path or DATA_PATH
    started = time.perf_counter()

    if not target.is_dir():
        print(f"Data directory not found: {target}")
        return report

    all_files = discover_files(target)
    report.files_discovered = len(all_files)

    supported_files = [path for path in all_files if is_supported_extension(path.suffix)]
    unsupported_count = len(all_files) - len(supported_files)
    report.files_skipped += unsupported_count

    if unsupported_count:
        logger.info("Skipped %s unsupported files", unsupported_count)

    if not supported_files:
        print(f"No supported files found under {target}")
        report.duration_seconds = time.perf_counter() - started
        report.print_summary()
        return report

    print(f"Found {len(supported_files)} supported file(s) in {target}")

    source_root = resolve_source_root(target)

    with get_connection() as conn:
        for file_path in supported_files:
            try:
                ingest_file(conn, file_path, report, source_root)
            except Exception as exc:
                conn.rollback()
                report.files_failed += 1
                source_path = to_source_path(file_path, source_root)
                message = f"{source_path}: {exc}"
                report.failures.append(message)
                logger.exception("Ingestion failed for %s", source_path)
                print(f"  ERROR: {exc}")

    report.duration_seconds = time.perf_counter() - started
    report.print_summary()
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    report = run_ingestion()
    if report.files_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
