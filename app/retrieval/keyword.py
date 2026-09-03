"""PostgreSQL full-text / keyword retrieval."""

from __future__ import annotations

import json
import re
from typing import Any

from app.config import KEYWORD_TOP_K
from app.database import get_connection
from app.retrieval.models import RetrievalCandidate
from app.retrieval.normalize import extract_technical_terms

_SAFE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_/]*")


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


def _build_tsquery(query: str) -> str | None:
    """Build a plaintsquery-compatible string from normalized query text."""
    tokens = _SAFE_TOKEN.findall(query)
    if not tokens:
        return None
    stop = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "to",
        "of",
        "in",
        "on",
        "for",
        "and",
        "or",
        "what",
        "which",
        "who",
        "how",
        "why",
        "when",
        "where",
    }
    kept = [token for token in tokens if token.lower() not in stop]
    if not kept:
        kept = tokens
    return " ".join(kept)


def keyword_search(query: str, top_k: int | None = None) -> list[RetrievalCandidate]:
    """Return keyword/FTS candidates, with exact technical-term boosting."""
    limit = KEYWORD_TOP_K if top_k is None else top_k
    if not query.strip() or limit <= 0:
        return []

    ts_input = _build_tsquery(query)
    tech_terms = extract_technical_terms(query)
    if ts_input is None and not tech_terms:
        return []

    # Use empty string / empty list instead of NULL for stable parameter typing.
    fts_query = ts_input or ""
    terms = tech_terms or []

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH scored AS (
                    SELECT
                        dc.id,
                        dc.document_id,
                        dc.content,
                        d.source_path,
                        coalesce(dc.chunk_metadata, '{}'::jsonb) AS chunk_metadata,
                        CASE
                            WHEN %s = '' THEN 0.0
                            ELSE ts_rank_cd(
                                dc.content_tsv,
                                plainto_tsquery('english', %s)
                            )
                        END AS fts_score,
                        CASE
                            WHEN cardinality(%s::text[]) = 0 THEN 0.0
                            ELSE (
                                SELECT coalesce(sum(
                                    CASE
                                        WHEN dc.content ILIKE '%%' || term || '%%'
                                        THEN 1.0
                                        ELSE 0.0
                                    END
                                ), 0.0)
                                FROM unnest(%s::text[]) AS term
                            )
                        END AS tech_hits
                    FROM document_chunks AS dc
                    INNER JOIN documents AS d ON d.id = dc.document_id
                    WHERE
                        (
                            %s <> ''
                            AND dc.content_tsv @@ plainto_tsquery('english', %s)
                        )
                        OR (
                            cardinality(%s::text[]) > 0
                            AND EXISTS (
                                SELECT 1
                                FROM unnest(%s::text[]) AS term
                                WHERE dc.content ILIKE '%%' || term || '%%'
                            )
                        )
                )
                SELECT
                    id,
                    document_id,
                    content,
                    source_path,
                    chunk_metadata,
                    (fts_score + tech_hits) AS keyword_score
                FROM scored
                WHERE (fts_score + tech_hits) > 0
                ORDER BY keyword_score DESC, id ASC
                LIMIT %s;
                """,
                (
                    fts_query,
                    fts_query,
                    terms,
                    terms,
                    fts_query,
                    fts_query,
                    terms,
                    terms,
                    limit,
                ),
            )
            rows = cur.fetchall()

    candidates: list[RetrievalCandidate] = []
    for rank, row in enumerate(rows, start=1):
        chunk_id, document_id, content, source_path, metadata, keyword_score = row
        candidates.append(
            RetrievalCandidate(
                chunk_id=int(chunk_id),
                document_id=int(document_id),
                content=content,
                source_path=source_path,
                keyword_score=float(keyword_score),
                keyword_rank=rank,
                retrieval_methods={"keyword"},
                chunk_metadata=_parse_metadata(metadata),
            )
        )
    return candidates
