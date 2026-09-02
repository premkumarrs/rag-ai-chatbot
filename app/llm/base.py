"""LLM provider abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class LLMResponse:
    answer: str
    model: str
    provider: str
    latency_ms: float
    finish_reason: str | None = None
    usage: LLMUsage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMStreamChunk:
    text: str
    is_final: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMProviderError(Exception):
    """Raised when an LLM provider fails in a controlled way."""


class LLMProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier such as ollama or cloud."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Active model name."""

    @abstractmethod
    def generate(self, messages: list[LLMMessage]) -> LLMResponse:
        """Generate a complete response."""

    @abstractmethod
    def stream(self, messages: list[LLMMessage]) -> Iterator[LLMStreamChunk]:
        """Stream response text chunks."""
