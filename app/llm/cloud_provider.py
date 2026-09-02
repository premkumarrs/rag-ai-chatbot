"""Placeholder cloud LLM provider for future approved external APIs."""

from __future__ import annotations

from collections.abc import Iterator

from app.config import ALLOW_CLOUD_FALLBACK, CLOUD_BASE_URL, CLOUD_MODEL
from app.llm.base import (
    LLMMessage,
    LLMProvider,
    LLMProviderError,
    LLMResponse,
    LLMStreamChunk,
)


class CloudProvider(LLMProvider):
    """Future cloud API integration point.

    This provider is intentionally inactive until ALLOW_CLOUD_FALLBACK is enabled
    and the required credentials/endpoints are configured.
    """

    def __init__(self) -> None:
        if not ALLOW_CLOUD_FALLBACK:
            raise LLMProviderError(
                "Cloud fallback is disabled. Set ALLOW_CLOUD_FALLBACK=true to enable."
            )
        if not CLOUD_MODEL or not CLOUD_BASE_URL:
            raise LLMProviderError(
                "Cloud provider is not configured. Set CLOUD_MODEL and CLOUD_BASE_URL."
            )

    @property
    def provider_name(self) -> str:
        return "cloud"

    @property
    def model_name(self) -> str:
        return CLOUD_MODEL

    def generate(self, messages: list[LLMMessage]) -> LLMResponse:
        raise LLMProviderError("Cloud provider is not implemented yet.")

    def stream(self, messages: list[LLMMessage]) -> Iterator[LLMStreamChunk]:
        raise LLMProviderError("Cloud provider streaming is not implemented yet.")
        yield LLMStreamChunk(text="")
