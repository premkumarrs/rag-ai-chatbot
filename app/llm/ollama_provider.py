"""Ollama LLM provider using LangChain ChatOllama."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.config import (
    LLM_MAX_OUTPUT_TOKENS,
    LLM_REASONING_ENABLED,
    LLM_REQUEST_TIMEOUT,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from app.llm.base import (
    LLMMessage,
    LLMProvider,
    LLMProviderError,
    LLMResponse,
    LLMStreamChunk,
    LLMUsage,
)


def _to_langchain_messages(messages: list[LLMMessage]) -> list[BaseMessage]:
    converted: list[BaseMessage] = []
    for message in messages:
        if message.role == "system":
            converted.append(SystemMessage(content=message.content))
        elif message.role == "human":
            converted.append(HumanMessage(content=message.content))
        else:
            converted.append(HumanMessage(content=message.content))
    return converted


def _extract_usage(response: AIMessage | AIMessageChunk) -> LLMUsage | None:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return None
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    return LLMUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _extract_finish_reason(metadata: dict[str, Any]) -> str | None:
    finish_reason = metadata.get("done_reason")
    return str(finish_reason) if finish_reason else None


class OllamaProvider(LLMProvider):
    def __init__(self) -> None:
        reasoning: bool | None
        if LLM_REASONING_ENABLED:
            reasoning = True
        else:
            reasoning = False

        self._client = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURE,
            num_predict=LLM_MAX_OUTPUT_TOKENS,
            reasoning=reasoning,
        )

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return OLLAMA_MODEL

    def generate(self, messages: list[LLMMessage]) -> LLMResponse:
        started = time.perf_counter()
        try:
            response = self._client.invoke(_to_langchain_messages(messages))
        except Exception as exc:
            raise LLMProviderError("Local Ollama generation failed.") from exc

        if not isinstance(response, AIMessage):
            raise LLMProviderError("Ollama returned an unexpected response type.")

        content = response.content
        if not isinstance(content, str) or not content.strip():
            raise LLMProviderError("Ollama returned an empty chat response.")

        metadata = dict(response.response_metadata or {})
        latency_ms = (time.perf_counter() - started) * 1000
        return LLMResponse(
            answer=content.strip(),
            model=str(metadata.get("model", self.model_name)),
            provider=self.provider_name,
            latency_ms=latency_ms,
            finish_reason=_extract_finish_reason(metadata),
            usage=_extract_usage(response),
            metadata=metadata,
        )

    def stream(self, messages: list[LLMMessage]) -> Iterator[LLMStreamChunk]:
        started = time.perf_counter()
        try:
            chunks = self._client.stream(_to_langchain_messages(messages))
            for chunk in chunks:
                if not isinstance(chunk, AIMessageChunk):
                    continue
                text = chunk.content if isinstance(chunk.content, str) else ""
                metadata = dict(chunk.response_metadata or {})
                is_final = bool(metadata.get("done"))
                if text:
                    yield LLMStreamChunk(text=text, is_final=is_final, metadata=metadata)
                elif is_final:
                    metadata["latency_ms"] = (time.perf_counter() - started) * 1000
                    yield LLMStreamChunk(text="", is_final=True, metadata=metadata)
        except Exception as exc:
            raise LLMProviderError("Local Ollama streaming failed.") from exc

    @property
    def request_timeout(self) -> float:
        return LLM_REQUEST_TIMEOUT
