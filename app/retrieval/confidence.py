"""Retrieval confidence evaluation (not answer correctness)."""

from __future__ import annotations

from app.config import (
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    RETRIEVAL_MIN_SIMILARITY,
)
from app.retrieval.models import ConfidenceLevel, RetrievalCandidate


def _support_score(candidate: RetrievalCandidate) -> float:
    """Map candidate signals onto an approximately 0–1 support scale."""
    scores: list[float] = []
    if candidate.similarity is not None:
        scores.append(candidate.similarity)
    if candidate.keyword_score is not None and candidate.keyword_score > 0:
        scores.append(min(1.0, 0.45 + 0.2 * min(candidate.keyword_score, 3.0)))
    if not scores and candidate.fused_score > 0:
        scores.append(min(1.0, candidate.fused_score * 4.0))

    base = max(scores) if scores else 0.0
    if candidate.found_by_both:
        base = min(1.0, base + 0.08)
    return base


def evaluate_confidence(
    candidates: list[RetrievalCandidate],
    *,
    high_threshold: float | None = None,
    medium_threshold: float | None = None,
) -> ConfidenceLevel:
    """
    Classify whether retrieved context looks sufficiently relevant.

    HIGH / MEDIUM / LOW reflect retrieval support quality only.
    """
    if not candidates:
        return ConfidenceLevel.LOW

    high = HIGH_CONFIDENCE_THRESHOLD if high_threshold is None else high_threshold
    medium = (
        MEDIUM_CONFIDENCE_THRESHOLD if medium_threshold is None else medium_threshold
    )

    scores = [_support_score(candidate) for candidate in candidates]
    best = scores[0]
    second = scores[1] if len(scores) > 1 else 0.0
    gap = best - second

    strong_similarity = any(
        (candidate.similarity or 0.0) >= RETRIEVAL_MIN_SIMILARITY
        for candidate in candidates[:3]
    )
    agreement = any(candidate.found_by_both for candidate in candidates[:3])
    enough_support = sum(1 for score in scores if score >= medium) >= 2
    strong_keyword = any(
        (candidate.keyword_score or 0.0) >= 1.0 for candidate in candidates[:3]
    )

    if best >= high and (
        agreement or enough_support or strong_keyword or (gap >= 0.05 and strong_similarity)
    ):
        return ConfidenceLevel.HIGH

    if best >= medium and (strong_similarity or agreement or strong_keyword):
        return ConfidenceLevel.MEDIUM

    if best >= medium:
        return ConfidenceLevel.MEDIUM

    return ConfidenceLevel.LOW
