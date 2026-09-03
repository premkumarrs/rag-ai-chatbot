"""Request ID and latency instrumentation helpers."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("rag_ai_chatbot")

REQUEST_ID_HEADER = "X-Request-ID"


@dataclass
class RequestMetrics:
    request_id: str
    provider: str | None = None
    model: str | None = None
    embedding_ms: float | None = None
    retrieval_ms: float | None = None
    prompt_ms: float | None = None
    generation_ms: float | None = None
    postprocess_ms: float | None = None
    total_ms: float | None = None
    time_to_first_token_ms: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def log_summary(self) -> None:
        extra = self.extra or {}
        logger.info(
            "request_id=%s provider=%s model=%s embedding_ms=%s retrieval_ms=%s "
            "prompt_ms=%s generation_ms=%s postprocess_ms=%s total_ms=%s "
            "time_to_first_token_ms=%s confidence=%s vector_ms=%s keyword_ms=%s "
            "fusion_rerank_ms=%s candidates=%s selected=%s normalized_len=%s",
            self.request_id,
            self.provider,
            self.model,
            _format_ms(self.embedding_ms),
            _format_ms(self.retrieval_ms),
            _format_ms(self.prompt_ms),
            _format_ms(self.generation_ms),
            _format_ms(self.postprocess_ms),
            _format_ms(self.total_ms),
            _format_ms(self.time_to_first_token_ms),
            extra.get("confidence", "-"),
            extra.get("vector_ms", "-"),
            extra.get("keyword_ms", "-"),
            extra.get("fusion_rerank_ms", "-"),
            extra.get("fused_candidates", "-"),
            extra.get("selected_chunks", "-"),
            len(str(extra.get("normalized_query", "")))
            if extra.get("normalized_query") is not None
            else "-",
        )


def resolve_request_id(header_value: str | None) -> str:
    if header_value and header_value.strip():
        return header_value.strip()
    return str(uuid.uuid4())


def _format_ms(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}"


class StageTimer:
    def __init__(self) -> None:
        self._started = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._started) * 1000
