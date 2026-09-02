"""FastAPI entry point for the RAG customer-support chatbot."""

from __future__ import annotations

import json
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import LLM_STREAMING_ENABLED
from app.llm.base import LLMProviderError
from app.rag import answer_question, stream_answer_question
from app.request_context import REQUEST_ID_HEADER, resolve_request_id

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="RAG Customer Support Chatbot")


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
