"""Placeholder provider for future company GPU / OpenAI-compatible endpoints."""

from __future__ import annotations

from collections.abc import Iterator

from app.config import OPENAI_COMPAT_BASE_URL, OPENAI_COMPAT_MODEL
from app.llm.base import (
    LLMMessage,
    LLMProvider,
    LLMProviderError,
    LLMResponse,
    LLMStreamChunk,
)


class OpenAICompatibleProvider(LLMProvider):
    """Future integration point for internal GPU-served models."""

    def __init__(self) -> None:
        if not OPENAI_COMPAT_BASE_URL or not OPENAI_COMPAT_MODEL:
            raise LLMProviderError(
                "OpenAI-compatible provider is not configured. "
                "Set OPENAI_COMPAT_BASE_URL and OPENAI_COMPAT_MODEL."
            )

    @property
    def provider_name(self) -> str:
        return "openai_compatible"

    @property
    def model_name(self) -> str:
        return OPENAI_COMPAT_MODEL

    def generate(self, messages: list[LLMMessage]) -> LLMResponse:
        raise LLMProviderError(
            "OpenAI-compatible provider is not implemented yet."
        )

    def stream(self, messages: list[LLMMessage]) -> Iterator[LLMStreamChunk]:
        raise LLMProviderError(
            "OpenAI-compatible provider streaming is not implemented yet."
        )
        yield LLMStreamChunk(text="")
