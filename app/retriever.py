"""Retrieve relevant document chunks for a user question."""

from __future__ import annotations

from typing import TypedDict

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import OllamaEmbeddings
from pgvector import Vector
from pydantic import ConfigDict

from app.config import EMBEDDING_MODEL, OLLAMA_HOST, RETRIEVAL_MIN_SIMILARITY, TOP_K
from app.database import get_connection
from app.request_context import RequestMetrics, StageTimer


class RetrievedChunk(TypedDict):
    content: str
    source_path: str
    distance: float
    similarity: float


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)


def _search_documents(query_embedding: Vector, top_k: int) -> list[Document]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    dc.content,
                    d.source_path,
                    dc.embedding <=> %s AS distance
                FROM document_chunks AS dc
                INNER JOIN documents AS d ON d.id = dc.document_id
                ORDER BY dc.embedding <=> %s
                LIMIT %s;
                """,
                (query_embedding, query_embedding, top_k),
            )
            rows = cur.fetchall()

    documents: list[Document] = []
    for content, source_path, distance in rows:
        distance_value = float(distance)
        similarity = 1.0 - distance_value
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source_path": source_path,
                    "distance": distance_value,
                    "similarity": similarity,
                },
            )
        )
    return documents


class PgVectorRetriever(BaseRetriever):
    """LangChain retriever backed by the existing PostgreSQL + pgvector schema."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    embeddings: OllamaEmbeddings
    top_k: int = TOP_K

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        query_embedding = Vector(self.embeddings.embed_query(query))
        return _search_documents(query_embedding, self.top_k)


def _document_to_chunk(document: Document) -> RetrievedChunk:
    metadata = document.metadata
    return {
        "content": document.page_content,
        "source_path": str(metadata["source_path"]),
        "distance": float(metadata["distance"]),
        "similarity": float(metadata["similarity"]),
    }


def retrieve(
    question: str,
    top_k: int | None = None,
    metrics: RequestMetrics | None = None,
) -> list[RetrievedChunk]:
    """Return the most relevant stored chunks for a question."""
    limit = TOP_K if top_k is None else top_k
    embeddings = get_embeddings()

    embed_timer = StageTimer()
    query_embedding = Vector(embeddings.embed_query(question))
    if metrics is not None:
        metrics.embedding_ms = embed_timer.elapsed_ms()

    retrieval_timer = StageTimer()
    documents = _search_documents(query_embedding, limit)
    if metrics is not None:
        metrics.retrieval_ms = retrieval_timer.elapsed_ms()

    results: list[RetrievedChunk] = []
    for document in documents:
        similarity = float(document.metadata["similarity"])
        if similarity < RETRIEVAL_MIN_SIMILARITY:
            continue
        results.append(_document_to_chunk(document))

    return results
