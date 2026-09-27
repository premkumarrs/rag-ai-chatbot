"""LLM provider factory."""

from __future__ import annotations

import threading

from app.config import ALLOW_CLOUD_FALLBACK, LLM_PROVIDER
from app.llm.base import LLMProvider, LLMProviderError
from app.llm.cloud_provider import CloudProvider
from app.llm.ollama_provider import OllamaProvider
from app.llm.openai_compatible_provider import OpenAICompatibleProvider

_lock = threading.Lock()
_providers: dict[str, LLMProvider] = {}


def get_llm_provider() -> LLMProvider:
    """Return the configured provider, reusing the local client within this process."""
    provider = LLM_PROVIDER.strip().lower()
    with _lock:
        cached = _providers.get(provider)
    if cached is not None:
        return cached

    if provider == "ollama":
        created: LLMProvider = OllamaProvider()
    elif provider == "cloud":
        if not ALLOW_CLOUD_FALLBACK:
            raise LLMProviderError(
                "Cloud provider requested but ALLOW_CLOUD_FALLBACK is disabled."
            )
        created = CloudProvider()
    elif provider in {"openai_compatible", "company_gpu"}:
        created = OpenAICompatibleProvider()
    else:
        raise LLMProviderError(f"Unsupported LLM provider: {LLM_PROVIDER}")

    with _lock:
        cached = _providers.get(provider)
        if cached is not None:
            return cached
        _providers[provider] = created
        return created
