"""Structure-aware chunking."""

from __future__ import annotations

from datetime import UTC, datetime

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.ingestion.models import ParsedDocument


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_parsed_document(parsed: ParsedDocument) -> list[Document]:
    splitter = get_text_splitter()
    chunks: list[Document] = []
    ingested_at = datetime.now(UTC).isoformat()

    for section in parsed.sections:
        if not section.content.strip():
            continue
        metadata = {
            **section.metadata,
            "source_path": parsed.source_path,
            "filename": parsed.filename,
            "file_type": parsed.file_type,
            "ingested_at": ingested_at,
        }
        section_document = Document(page_content=section.content, metadata=metadata)
        split_chunks = splitter.split_documents([section_document])
        chunks.extend(split_chunks)

    return chunks
