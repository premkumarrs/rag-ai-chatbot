"""Orchestrate retrieval and grounded answer generation."""

from __future__ import annotations

from collections.abc import Iterator

from app.generator import GenerationResult, generate_answer, stream_answer_tokens
from app.request_context import RequestMetrics, StageTimer, resolve_request_id
from app.retrieval.models import ConfidenceLevel
from app.retrieval.pipeline import candidates_to_public_chunks, run_retrieval
from app.retriever import RetrievedChunk


def answer_question(
    question: str,
    request_id: str | None = None,
) -> GenerationResult:
    """Answer a customer question using retrieved company documents."""
    metrics = RequestMetrics(request_id=resolve_request_id(request_id))
    total_timer = StageTimer()
    question = question.strip()

    if not question:
        result: GenerationResult = {
            "answer": "Please provide a question.",
            "sources": [],
            "fallback": True,
        }
        metrics.total_ms = total_timer.elapsed_ms()
        metrics.log_summary()
        return result

    assembled = run_retrieval(question, metrics=metrics)
    metrics.extra["confidence"] = assembled.confidence.value

    # LOW confidence → grounded fallback (no unsupported generation).
    if assembled.confidence == ConfidenceLevel.LOW or not assembled.chunks:
        result = generate_answer(question, [], metrics=metrics)
        metrics.postprocess_ms = StageTimer().elapsed_ms()
        metrics.total_ms = total_timer.elapsed_ms()
        metrics.log_summary()
        return result

    retrieved_chunks: list[RetrievedChunk] = candidates_to_public_chunks(
        assembled.chunks
    )  # type: ignore[assignment]
    result = generate_answer(question, retrieved_chunks, metrics=metrics)

    metrics.postprocess_ms = StageTimer().elapsed_ms()
    metrics.total_ms = total_timer.elapsed_ms()
    metrics.log_summary()
    return result


def stream_answer_question(
    question: str,
    request_id: str | None = None,
) -> Iterator[dict[str, object]]:
    """Stream an answer for a customer question."""
    metrics = RequestMetrics(request_id=resolve_request_id(request_id))
    total_timer = StageTimer()
    question = question.strip()

    if not question:
        yield {
            "type": "complete",
            "answer": "Please provide a question.",
            "sources": [],
            "fallback": True,
        }
        metrics.total_ms = total_timer.elapsed_ms()
        metrics.log_summary()
        return

    assembled = run_retrieval(question, metrics=metrics)
    metrics.extra["confidence"] = assembled.confidence.value

    if assembled.confidence == ConfidenceLevel.LOW or not assembled.chunks:
        for event in stream_answer_tokens(question, [], metrics=metrics):
            yield event
        metrics.postprocess_ms = StageTimer().elapsed_ms()
        metrics.total_ms = total_timer.elapsed_ms()
        metrics.log_summary()
        return

    retrieved_chunks: list[RetrievedChunk] = candidates_to_public_chunks(
        assembled.chunks
    )  # type: ignore[assignment]

    for event in stream_answer_tokens(question, retrieved_chunks, metrics=metrics):
        yield event

    metrics.postprocess_ms = StageTimer().elapsed_ms()
    metrics.total_ms = total_timer.elapsed_ms()
    metrics.log_summary()
