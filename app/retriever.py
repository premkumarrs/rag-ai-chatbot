"""Retrieve relevant document chunks for a user question."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import OllamaEmbeddings
from pgvector import Vector
from pydantic import ConfigDict

from app.config import TOP_K
from app.request_context import RequestMetrics
from app.retrieval.pipeline import candidates_to_public_chunks, run_retrieval
from app.retrieval.vector import get_embeddings, vector_search


class RetrievedChunk(TypedDict):
    content: str
    source_path: str
    distance: float
    similarity: float
    chunk_id: NotRequired[int]
    document_id: NotRequired[int]
    keyword_score: NotRequired[float | None]
    combined_score: NotRequired[float]
    retrieval_methods: NotRequired[list[str]]
    filename: NotRequired[str | None]
    file_type: NotRequired[str | None]
    page_number: NotRequired[int | None]
    sheet_name: NotRequired[str | None]
    slide_number: NotRequired[int | None]
    section: NotRequired[str | None]


class PgVectorRetriever(BaseRetriever):
    """LangChain retriever backed by PostgreSQL + pgvector (vector path)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    embeddings: OllamaEmbeddings
    top_k: int = TOP_K

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        candidates = vector_search(query, top_k=self.top_k)
        documents: list[Document] = []
        for candidate in candidates:
            documents.append(
                Document(
                    page_content=candidate.content,
                    metadata={
                        "source_path": candidate.source_path,
                        "distance": candidate.distance,
                        "similarity": candidate.similarity,
                        "chunk_id": candidate.chunk_id,
                    },
                )
            )
        return documents


def retrieve(
    question: str,
    top_k: int | None = None,
    metrics: RequestMetrics | None = None,
) -> list[RetrievedChunk]:
    """
    Return the most relevant stored chunks for a question.

    Uses hybrid vector + keyword retrieval with fusion, lightweight reranking,
    confidence gating, and context assembly.
    """
    assembled = run_retrieval(question, metrics=metrics, final_top_k=top_k)
    public_chunks = candidates_to_public_chunks(assembled.chunks)
    return public_chunks  # type: ignore[return-value]


# Re-export for callers that still construct embeddings directly.
__all__ = [
    "PgVectorRetriever",
    "RetrievedChunk",
    "get_embeddings",
    "retrieve",
    "Vector",
]
