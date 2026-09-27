"""Measure local RAG latency without changing retrieval or the model.

Requires Postgres, Ollama, and ingested documents. Not part of the unit suite.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm.factory import get_llm_provider
from app.rag import stream_answer_question
from app.retrieval.pipeline import run_retrieval
from app.retrieval.vector import get_embeddings

QUESTIONS = (
    ("supported", "What is this document about?"),
    ("unsupported", "What is the company refund phone number?"),
)


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def main() -> None:
    started = time.perf_counter()
    get_llm_provider()
    print(f"provider_init_ms={_ms(started):.1f}")

    for label, question in QUESTIONS:
        started = time.perf_counter()
        get_embeddings().embed_query(question)
        embed_ms = _ms(started)

        started = time.perf_counter()
        assembled = run_retrieval(question)
        retrieval_ms = _ms(started)
        diagnostics = assembled.diagnostics

        started = time.perf_counter()
        first_token_ms: float | None = None
        answer = ""
        sources: list[str] = []
        fallback = None
        for event in stream_answer_question(question):
            if event.get("type") == "token" and first_token_ms is None:
                first_token_ms = _ms(started)
            if event.get("type") == "complete":
                answer = str(event.get("answer") or "")
                sources = list(event.get("sources") or [])
                fallback = event.get("fallback")
        total_ms = _ms(started)

        print(f"--- {label} ---")
        print(f"embed_ms={embed_ms:.1f}")
        print(
            "retrieval_ms={retrieval:.1f} vector_ms={vector} keyword_ms={keyword} "
            "fusion_rerank_ms={fusion} confidence={confidence} chunks={chunks} chars={chars}".format(
                retrieval=retrieval_ms,
                vector=diagnostics.get("vector_ms"),
                keyword=diagnostics.get("keyword_ms"),
                fusion=diagnostics.get("fusion_rerank_ms"),
                confidence=assembled.confidence.value,
                chunks=len(assembled.chunks),
                chars=sum(len(chunk.content) for chunk in assembled.chunks),
            )
        )
        print(
            f"ttft_ms={first_token_ms if first_token_ms is not None else '-'} "
            f"request_ms={total_ms:.1f} fallback={fallback} sources={sources}"
        )
        print(f"answer={answer.replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
