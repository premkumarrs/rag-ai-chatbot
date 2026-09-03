"""Deterministic hybrid candidate fusion (Reciprocal Rank Fusion)."""

from __future__ import annotations

from app.config import KEYWORD_WEIGHT, RRF_K, VECTOR_WEIGHT
from app.retrieval.models import RetrievalCandidate


def fuse_candidates(
    vector_candidates: list[RetrievalCandidate],
    keyword_candidates: list[RetrievalCandidate],
    *,
    vector_weight: float | None = None,
    keyword_weight: float | None = None,
    rrf_k: int | None = None,
) -> list[RetrievalCandidate]:
    """
    Merge vector and keyword candidates by chunk_id using weighted RRF.

    Candidates found by both methods receive a stronger combined signal.
    """
    v_weight = VECTOR_WEIGHT if vector_weight is None else vector_weight
    k_weight = KEYWORD_WEIGHT if keyword_weight is None else keyword_weight
    k = RRF_K if rrf_k is None else rrf_k

    merged: dict[int, RetrievalCandidate] = {}

    for candidate in vector_candidates:
        rank = candidate.vector_rank or 1
        existing = merged.get(candidate.chunk_id)
        if existing is None:
            candidate.fused_score = v_weight / (k + rank)
            merged[candidate.chunk_id] = candidate
        else:
            existing.fused_score += v_weight / (k + rank)
            existing.retrieval_methods.update(candidate.retrieval_methods)
            existing.similarity = candidate.similarity
            existing.distance = candidate.distance
            existing.vector_rank = candidate.vector_rank

    for candidate in keyword_candidates:
        rank = candidate.keyword_rank or 1
        existing = merged.get(candidate.chunk_id)
        if existing is None:
            candidate.fused_score = k_weight / (k + rank)
            merged[candidate.chunk_id] = candidate
        else:
            existing.fused_score += k_weight / (k + rank)
            existing.retrieval_methods.update(candidate.retrieval_methods)
            existing.keyword_score = candidate.keyword_score
            existing.keyword_rank = candidate.keyword_rank
            if (
                "vector" in existing.retrieval_methods
                and "keyword" in existing.retrieval_methods
            ):
                existing.fused_score += 0.15 * min(v_weight, k_weight)

    fused = list(merged.values())
    fused.sort(
        key=lambda item: (
            -item.fused_score,
            -(item.similarity or 0.0),
            -(item.keyword_score or 0.0),
            item.chunk_id,
        )
    )
    return fused
