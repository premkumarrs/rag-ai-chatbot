"""pgvector semantic retrieval."""

from __future__ import annotations

import json
from typing import Any

from langchain_ollama import OllamaEmbeddings
from pgvector import Vector

from app.config import EMBEDDING_MODEL, OLLAMA_HOST, RETRIEVAL_MIN_SIMILARITY, VECTOR_TOP_K
from app.database import get_connection
from app.retrieval.models import RetrievalCandidate

_embeddings: OllamaEmbeddings | None = None


def get_embeddings() -> OllamaEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
    return _embeddings


def _parse_metadata(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def vector_search(
    query: str,
    top_k: int | None = None,
    min_similarity: float | None = None,
) -> list[RetrievalCandidate]:
    """Return vector-similarity candidates for a query."""
    limit = VECTOR_TOP_K if top_k is None else top_k
    threshold = RETRIEVAL_MIN_SIMILARITY if min_similarity is None else min_similarity
    if not query.strip() or limit <= 0:
        return []

    query_embedding = Vector(get_embeddings().embed_query(query))
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    dc.id,
                    dc.document_id,
                    dc.content,
                    d.source_path,
                    dc.embedding <=> %s AS distance,
                    coalesce(dc.chunk_metadata, '{}'::jsonb) AS chunk_metadata
                FROM document_chunks AS dc
                INNER JOIN documents AS d ON d.id = dc.document_id
                ORDER BY dc.embedding <=> %s
                LIMIT %s;
                """,
                (query_embedding, query_embedding, limit),
            )
            rows = cur.fetchall()

    candidates: list[RetrievalCandidate] = []
    for rank, row in enumerate(rows, start=1):
        chunk_id, document_id, content, source_path, distance, metadata = row
        distance_value = float(distance)
        similarity = 1.0 - distance_value
        if similarity < threshold:
            continue
        candidates.append(
            RetrievalCandidate(
                chunk_id=int(chunk_id),
                document_id=int(document_id),
                content=content,
                source_path=source_path,
                similarity=similarity,
                distance=distance_value,
                vector_rank=rank,
                retrieval_methods={"vector"},
                chunk_metadata=_parse_metadata(metadata),
            )
        )
    return candidates
