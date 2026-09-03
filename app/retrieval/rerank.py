"""Lightweight deterministic reranking of fused candidates."""

from __future__ import annotations

from app.config import RERANKING_ENABLED
from app.retrieval.models import RetrievalCandidate
from app.retrieval.normalize import extract_technical_terms


def _technical_overlap_bonus(query: str, content: str) -> float:
    query_terms = {term.lower() for term in extract_technical_terms(query)}
    if not query_terms:
        return 0.0
    content_lower = content.lower()
    hits = sum(1 for term in query_terms if term in content_lower)
    return 0.08 * hits


def _token_overlap_bonus(query: str, content: str) -> float:
    query_tokens = {token.lower() for token in query.split() if len(token) > 2}
    if not query_tokens:
        return 0.0
    content_tokens = {token.lower() for token in content.split() if len(token) > 2}
    if not content_tokens:
        return 0.0
    overlap = len(query_tokens & content_tokens) / len(query_tokens)
    return 0.1 * overlap


def rerank_candidates(
    query: str,
    candidates: list[RetrievalCandidate],
    *,
    enabled: bool | None = None,
) -> list[RetrievalCandidate]:
    """
    Rerank a small candidate set using deterministic relevance signals.

    Modular entry point: a model-based reranker can replace this later.
    """
    use_rerank = RERANKING_ENABLED if enabled is None else enabled
    if not candidates:
        return []

    if not use_rerank:
        for candidate in candidates:
            candidate.rerank_score = candidate.fused_score
        return candidates

    for candidate in candidates:
        score = candidate.fused_score
        if candidate.similarity is not None:
            score += 0.35 * candidate.similarity
        if candidate.keyword_score is not None:
            # Normalize unbounded FTS/tech scores gently.
            score += 0.15 * min(candidate.keyword_score, 3.0) / 3.0
        if candidate.found_by_both:
            score += 0.12
        score += _technical_overlap_bonus(query, candidate.content)
        score += _token_overlap_bonus(query, candidate.content)
        candidate.rerank_score = score

    candidates.sort(
        key=lambda item: (
            -item.rerank_score,
            -item.fused_score,
            -(item.similarity or 0.0),
            item.chunk_id,
        )
    )
    return candidates
