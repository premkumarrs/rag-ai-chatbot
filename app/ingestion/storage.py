"""PostgreSQL storage for ingested documents."""

from __future__ import annotations

import json

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from pgvector import Vector
from psycopg import Connection

from app.config import EMBEDDING_MODEL, OLLAMA_HOST
from app.ingestion.models import ParsedDocument


_embeddings: OllamaEmbeddings | None = None


def get_embeddings() -> OllamaEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
    return _embeddings


def get_stored_content_hash(conn: Connection, source_path: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT content_hash FROM documents WHERE source_path = %s;",
            (source_path,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return row[0]


def upsert_document(
    conn: Connection,
    parsed: ParsedDocument,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (source_path, title, file_type, content_hash)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (source_path) DO UPDATE
            SET title = EXCLUDED.title,
                file_type = EXCLUDED.file_type,
                content_hash = EXCLUDED.content_hash,
                updated_at = NOW()
            RETURNING id;
            """,
            (parsed.source_path, parsed.title, parsed.file_type, parsed.content_hash),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"Failed to upsert document: {parsed.source_path}")
        return row[0]


def replace_document_chunks(
    conn: Connection,
    document_id: int,
    chunks: list[Document],
    embeddings: list[list[float]],
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("Chunk and embedding counts must match")

    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM document_chunks WHERE document_id = %s;",
            (document_id,),
        )
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                """
                INSERT INTO document_chunks (
                    document_id,
                    chunk_index,
                    content,
                    embedding,
                    chunk_metadata
                )
                VALUES (%s, %s, %s, %s, %s::jsonb);
                """,
                (
                    document_id,
                    index,
                    chunk.page_content,
                    Vector(embedding),
                    json.dumps(chunk.metadata),
                ),
            )


def clear_document_chunks(conn: Connection, source_path: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM documents WHERE source_path = %s;",
            (source_path,),
        )
        row = cur.fetchone()
        if row is None:
            return
        cur.execute(
            "DELETE FROM document_chunks WHERE document_id = %s;",
            (row[0],),
        )
        cur.execute(
            """
            UPDATE documents
            SET updated_at = NOW()
            WHERE id = %s;
            """,
            (row[0],),
        )


def embed_documents(documents: list[Document]) -> list[list[float]]:
    if not documents:
        return []
    texts = [document.page_content for document in documents]
    return get_embeddings().embed_documents(texts)
