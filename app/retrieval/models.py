"""Internal models for the intelligent retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class RetrievalCandidate:
    chunk_id: int
    document_id: int
    content: str
    source_path: str
    similarity: float | None = None
    distance: float | None = None
    keyword_score: float | None = None
    vector_rank: int | None = None
    keyword_rank: int | None = None
    fused_score: float = 0.0
    rerank_score: float = 0.0
    retrieval_methods: set[str] = field(default_factory=set)
    chunk_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def found_by_both(self) -> bool:
        return "vector" in self.retrieval_methods and "keyword" in self.retrieval_methods


@dataclass
class AssembledContext:
    chunks: list[RetrievalCandidate]
    total_chars: int
    confidence: ConfidenceLevel
    normalized_query: str
    candidate_count: int
    diagnostics: dict[str, Any] = field(default_factory=dict)
