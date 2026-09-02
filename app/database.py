"""PostgreSQL + pgvector database access."""

from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector
from psycopg import Connection

from app.config import DATABASE_URL

# nomic-embed-text produces 768-dimensional embeddings.
EMBEDDING_DIMENSION = 768


def get_connection() -> Connection:
    """Return a psycopg connection with pgvector types registered."""
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def init_db() -> None:
    """Create the pgvector extension and document/chunk tables if they do not exist."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id BIGSERIAL PRIMARY KEY,
                    source_path TEXT NOT NULL UNIQUE,
                    title TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    document_id BIGINT NOT NULL
                        REFERENCES documents (id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({EMBEDDING_DIMENSION}) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (document_id, chunk_index)
                );
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
                    ON document_chunks
                    USING hnsw (embedding vector_cosine_ops);
                """
            )

            cur.execute(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_type TEXT;"
            )
            cur.execute(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash TEXT;"
            )
            cur.execute(
                """
                ALTER TABLE document_chunks
                ADD COLUMN IF NOT EXISTS chunk_metadata JSONB
                DEFAULT '{}'::jsonb;
                """
            )

        conn.commit()
