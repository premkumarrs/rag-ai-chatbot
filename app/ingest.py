"""Ingest knowledge-base documents into PostgreSQL + pgvector."""

from __future__ import annotations

from app.ingestion.pipeline import main

if __name__ == "__main__":
    main()
