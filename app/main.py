"""FastAPI entry point for the RAG customer-support chatbot."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import LLM_STREAMING_ENABLED
from app.llm.base import LLMMessage, LLMProviderError
from app.rag import answer_question, stream_answer_question
from app.request_context import REQUEST_ID_HEADER, resolve_request_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_ai_chatbot")


def _warm_local_models() -> None:
    """Load Ollama models before the first user question.

    A cold load of qwen3.5:4b was measured at about 20 seconds. Doing it here
    keeps that cost off the request path. Skipped during unit tests.
    """
    if "unittest" in sys.modules:
        return
    if os.getenv("LLM_WARMUP", "1").strip().lower() in {"0", "false", "no", "off"}:
        return
    try:
        from app.llm.factory import get_llm_provider
        from app.retrieval.vector import get_embeddings

        get_embeddings().embed_query("warmup")
        get_llm_provider().generate([LLMMessage(role="human", content="Reply with OK.")])
        logger.info("Local Ollama models are loaded.")
    except Exception:
        logger.warning("Local model warmup failed", exc_info=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_warm_local_models, daemon=True).start()
    yield


app = FastAPI(title="RAG Customer Support Chatbot", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    fallback: bool


class HealthResponse(BaseModel):
    status: str


def _resolve_request_id(request: Request) -> str:
    return resolve_request_id(request.headers.get(REQUEST_ID_HEADER))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request_body: ChatRequest, request: Request) -> ChatResponse:
    request_id = _resolve_request_id(request)
    try:
        return answer_question(request_body.question, request_id=request_id)
    except LLMProviderError:
        raise HTTPException(
            status_code=503,
            detail="The assistant is temporarily unavailable.",
        ) from None
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your question.",
        ) from exc


@app.post("/chat/stream")
def chat_stream(request_body: ChatRequest, request: Request) -> StreamingResponse:
    if not LLM_STREAMING_ENABLED:
        raise HTTPException(status_code=404, detail="Streaming is disabled.")

    request_id = _resolve_request_id(request)

    def event_generator():
        try:
            for event in stream_answer_question(
                request_body.question,
                request_id=request_id,
            ):
                payload = {"request_id": request_id, **event}
                yield f"data: {json.dumps(payload)}\n\n"
        except LLMProviderError:
            payload = {
                "request_id": request_id,
                "type": "error",
                "detail": "The assistant is temporarily unavailable.",
            }
            yield f"data: {json.dumps(payload)}\n\n"
        except Exception:
            payload = {
                "request_id": request_id,
                "type": "error",
                "detail": "An error occurred while processing your question.",
            }
            yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
