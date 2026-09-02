"""LLM provider factory."""

from __future__ import annotations

from app.config import ALLOW_CLOUD_FALLBACK, LLM_PROVIDER
from app.llm.base import LLMProvider, LLMProviderError
from app.llm.cloud_provider import CloudProvider
from app.llm.ollama_provider import OllamaProvider
from app.llm.openai_compatible_provider import OpenAICompatibleProvider


def get_llm_provider() -> LLMProvider:
    provider = LLM_PROVIDER.strip().lower()

    if provider == "ollama":
        return OllamaProvider()

    if provider == "cloud":
        if not ALLOW_CLOUD_FALLBACK:
            raise LLMProviderError(
                "Cloud provider requested but ALLOW_CLOUD_FALLBACK is disabled."
            )
        return CloudProvider()

    if provider in {"openai_compatible", "company_gpu"}:
        return OpenAICompatibleProvider()

    raise LLMProviderError(f"Unsupported LLM provider: {LLM_PROVIDER}")
