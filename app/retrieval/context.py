"""Assemble a bounded, de-duplicated context pack for generation."""

from __future__ import annotations

from app.config import MAX_CONTEXT_CHARS, MAX_CONTEXT_CHUNKS, NEAR_DUPLICATE_OVERLAP
from app.retrieval.models import ConfidenceLevel, RetrievalCandidate


def _token_set(text: str) -> set[str]:
    return {token.lower() for token in text.split() if len(token) > 2}


def _is_near_duplicate(a: str, b: str, threshold: float) -> bool:
    tokens_a = _token_set(a)
    tokens_b = _token_set(b)
    if not tokens_a or not tokens_b:
        return False
    overlap = len(tokens_a & tokens_b) / max(len(tokens_a), len(tokens_b))
    return overlap >= threshold


def assemble_context(
    candidates: list[RetrievalCandidate],
    confidence: ConfidenceLevel,
    *,
    max_chunks: int | None = None,
    max_chars: int | None = None,
    near_duplicate_overlap: float | None = None,
) -> list[RetrievalCandidate]:
    """
    Select the strongest non-duplicate chunks within configured limits.

    LOW confidence yields an empty context so the generator falls back.
    """
    if confidence == ConfidenceLevel.LOW or not candidates:
        return []

    limit_chunks = MAX_CONTEXT_CHUNKS if max_chunks is None else max_chunks
    limit_chars = MAX_CONTEXT_CHARS if max_chars is None else max_chars
    dup_threshold = (
        NEAR_DUPLICATE_OVERLAP if near_duplicate_overlap is None else near_duplicate_overlap
    )

    selected: list[RetrievalCandidate] = []
    total_chars = 0

    for candidate in candidates:
        if len(selected) >= limit_chunks:
            break

        if any(
            _is_near_duplicate(candidate.content, existing.content, dup_threshold)
            for existing in selected
        ):
            continue

        content_len = len(candidate.content)
        if selected and total_chars + content_len > limit_chars:
            continue

        selected.append(candidate)
        total_chars += content_len

    # MEDIUM confidence: require at least one reasonably scored chunk.
    if confidence == ConfidenceLevel.MEDIUM and not selected:
        return []

    return selected
