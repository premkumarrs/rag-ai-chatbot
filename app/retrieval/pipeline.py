"""Intelligent hybrid retrieval pipeline orchestration."""

from __future__ import annotations

import logging
from typing import Any

from app.config import FINAL_TOP_K
from app.request_context import RequestMetrics, StageTimer
from app.retrieval.confidence import evaluate_confidence
from app.retrieval.context import assemble_context
from app.retrieval.fusion import fuse_candidates
from app.retrieval.keyword import keyword_search
from app.retrieval.models import AssembledContext, ConfidenceLevel, RetrievalCandidate
from app.retrieval.normalize import normalize_query
from app.retrieval.rerank import rerank_candidates
from app.retrieval.vector import vector_search

logger = logging.getLogger("rag_ai_chatbot")


def run_retrieval(
    question: str,
    *,
    metrics: RequestMetrics | None = None,
    final_top_k: int | None = None,
) -> AssembledContext:
    """
    Execute the full intelligent retrieval pipeline.

    Query → normalize → vector + keyword → fuse → rerank → confidence → context
    """
    limit = FINAL_TOP_K if final_top_k is None else final_top_k
    original = question.strip()
    normalized = normalize_query(original)
    query_for_search = normalized or original

    diagnostics: dict[str, Any] = {
        "original_query_length": len(original),
        "normalized_query": normalized,
    }

    embed_and_vector_timer = StageTimer()
    vector_candidates = vector_search(query_for_search)
    vector_ms = embed_and_vector_timer.elapsed_ms()

    keyword_timer = StageTimer()
    keyword_candidates = keyword_search(query_for_search)
    keyword_ms = keyword_timer.elapsed_ms()

    fusion_timer = StageTimer()
    fused = fuse_candidates(vector_candidates, keyword_candidates)
    reranked = rerank_candidates(query_for_search, fused)
    if limit > 0:
        reranked = reranked[: max(limit * 3, limit)]
    fusion_ms = fusion_timer.elapsed_ms()

    confidence = evaluate_confidence(reranked)
    selected = assemble_context(reranked, confidence)
    if limit > 0:
        selected = selected[:limit]

    diagnostics.update(
        {
            "vector_candidates": len(vector_candidates),
            "keyword_candidates": len(keyword_candidates),
            "fused_candidates": len(fused),
            "selected_chunks": len(selected),
            "confidence": confidence.value,
            "vector_ms": round(vector_ms, 1),
            "keyword_ms": round(keyword_ms, 1),
            "fusion_rerank_ms": round(fusion_ms, 1),
        }
    )

    if metrics is not None:
        # embedding_ms approximated inside vector stage for compatibility.
        metrics.embedding_ms = vector_ms
        metrics.retrieval_ms = keyword_ms + fusion_ms
        metrics.extra.update(diagnostics)

    logger.info(
        "retrieval request_id=%s confidence=%s vector=%s keyword=%s fused=%s selected=%s "
        "vector_ms=%.1f keyword_ms=%.1f fusion_ms=%.1f normalized_len=%s",
        getattr(metrics, "request_id", "-") if metrics else "-",
        confidence.value,
        len(vector_candidates),
        len(keyword_candidates),
        len(fused),
        len(selected),
        vector_ms,
        keyword_ms,
        fusion_ms,
        len(normalized),
    )

    return AssembledContext(
        chunks=selected,
        total_chars=sum(len(chunk.content) for chunk in selected),
        confidence=confidence,
        normalized_query=normalized,
        candidate_count=len(fused),
        diagnostics=diagnostics,
    )


def candidates_to_public_chunks(
    candidates: list[RetrievalCandidate],
) -> list[dict[str, object]]:
    """Map internal candidates to the public RetrievedChunk-compatible shape."""
    results: list[dict[str, object]] = []
    for candidate in candidates:
        metadata = candidate.chunk_metadata or {}
        results.append(
            {
                "content": candidate.content,
                "source_path": candidate.source_path,
                "distance": float(candidate.distance)
                if candidate.distance is not None
                else 1.0 - float(candidate.similarity or 0.0),
                "similarity": float(candidate.similarity or 0.0),
                "chunk_id": candidate.chunk_id,
                "document_id": candidate.document_id,
                "keyword_score": candidate.keyword_score,
                "combined_score": candidate.rerank_score or candidate.fused_score,
                "retrieval_methods": sorted(candidate.retrieval_methods),
                "filename": metadata.get("filename"),
                "file_type": metadata.get("file_type"),
                "page_number": metadata.get("page_number"),
                "sheet_name": metadata.get("sheet_name"),
                "slide_number": metadata.get("slide_number"),
                "section": metadata.get("section") or metadata.get("title"),
            }
        )
    return results
