"""Orchestrate retrieval and grounded answer generation."""

from __future__ import annotations

from collections.abc import Iterator

from app.generator import GenerationResult, generate_answer, stream_answer_tokens
from app.request_context import RequestMetrics, StageTimer, resolve_request_id
from app.retriever import retrieve


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

    retrieved_chunks = retrieve(question, metrics=metrics)
    result = generate_answer(question, retrieved_chunks, metrics=metrics)

    postprocess_timer = StageTimer()
    metrics.postprocess_ms = postprocess_timer.elapsed_ms()
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

    retrieved_chunks = retrieve(question, metrics=metrics)

    for event in stream_answer_tokens(question, retrieved_chunks, metrics=metrics):
        yield event

    postprocess_timer = StageTimer()
    metrics.postprocess_ms = postprocess_timer.elapsed_ms()
    metrics.total_ms = total_timer.elapsed_ms()
    metrics.log_summary()
