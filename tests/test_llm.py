"""Tests for LLM provider integration and RAG behavior."""

from __future__ import annotations

import logging
import unittest
from unittest.mock import MagicMock, patch

from app.config import ALLOW_CLOUD_FALLBACK, LLM_PROVIDER
from app.generator import SYSTEM_PROMPT, generate_answer
from app.llm.base import LLMMessage, LLMProviderError, LLMResponse
from app.llm.factory import get_llm_provider
from app.llm.ollama_provider import OllamaProvider
from app.main import app
from app.rag import answer_question
from app.request_context import RequestMetrics, resolve_request_id
from app.retriever import RetrievedChunk


class ProviderFactoryTests(unittest.TestCase):
    def test_factory_selects_ollama_by_default(self) -> None:
        provider = get_llm_provider()
        self.assertEqual(provider.provider_name, "ollama")
        self.assertIsInstance(provider, OllamaProvider)

    def test_cloud_fallback_disabled_by_default(self) -> None:
        self.assertFalse(ALLOW_CLOUD_FALLBACK)

    def test_cloud_provider_rejected_when_disabled(self) -> None:
        with patch("app.llm.factory.LLM_PROVIDER", "cloud"):
            with self.assertRaises(LLMProviderError):
                get_llm_provider()


class GeneratorTests(unittest.TestCase):
    def test_empty_chunks_return_fallback_without_llm(self) -> None:
        with patch("app.generator.get_llm_provider") as mock_factory:
            result = generate_answer("What is the warranty period?", [])
            mock_factory.assert_not_called()
        self.assertTrue(result["fallback"])
        self.assertEqual(result["sources"], [])

    @patch("app.generator.get_llm_provider")
    def test_provider_error_returns_safe_fallback(self, mock_factory: MagicMock) -> None:
        provider = MagicMock()
        provider.generate.side_effect = LLMProviderError("failed")
        mock_factory.return_value = provider

        chunks: list[RetrievedChunk] = [
            {
                "content": "Example context",
                "source_path": "data/example.docx",
                "distance": 0.1,
                "similarity": 0.9,
            }
        ]
        result = generate_answer("Supported question?", chunks)
        self.assertTrue(result["fallback"])
        self.assertIn("temporarily unavailable", result["answer"])

    @patch("app.generator.get_llm_provider")
    def test_grounded_answer_uses_provider(self, mock_factory: MagicMock) -> None:
        provider = MagicMock()
        provider.generate.return_value = LLMResponse(
            answer="Grounded answer",
            model="qwen3.5:4b",
            provider="ollama",
            latency_ms=10.0,
        )
        mock_factory.return_value = provider

        chunks: list[RetrievedChunk] = [
            {
                "content": "CWRU bearing dataset",
                "source_path": "data/Abstract.docx",
                "distance": 0.2,
                "similarity": 0.8,
            }
        ]
        result = generate_answer("What datasets were used?", chunks)
        self.assertFalse(result["fallback"])
        self.assertEqual(result["answer"], "Grounded answer")
        provider.generate.assert_called_once()
        messages = provider.generate.call_args.args[0]
        self.assertEqual(messages[0].role, "system")
        self.assertIn("Answer ONLY", messages[0].content)


class RagOrchestrationTests(unittest.TestCase):
    def test_empty_question_does_not_call_retriever(self) -> None:
        with patch("app.rag.run_retrieval") as mock_retrieve:
            result = answer_question("   ", request_id="test-empty")
            mock_retrieve.assert_not_called()
        self.assertTrue(result["fallback"])
        self.assertEqual(result["answer"], "Please provide a question.")

    @patch("app.rag.generate_answer")
    @patch("app.rag.run_retrieval")
    def test_supported_flow_calls_retrieval_then_generation(
        self,
        mock_retrieve: MagicMock,
        mock_generate: MagicMock,
    ) -> None:
        from app.retrieval.models import AssembledContext, ConfidenceLevel, RetrievalCandidate

        mock_retrieve.return_value = AssembledContext(
            chunks=[
                RetrievalCandidate(
                    chunk_id=1,
                    document_id=1,
                    content="x",
                    source_path="data/a.docx",
                    similarity=0.9,
                    distance=0.1,
                    retrieval_methods={"vector"},
                )
            ],
            total_chars=1,
            confidence=ConfidenceLevel.HIGH,
            normalized_query="Supported?",
            candidate_count=1,
        )
        mock_generate.return_value = {
            "answer": "Answer",
            "sources": ["data/a.docx"],
            "fallback": False,
        }

        result = answer_question("Supported?", request_id="test-supported")
        mock_retrieve.assert_called_once()
        mock_generate.assert_called_once()
        self.assertFalse(result["fallback"])

    @patch("app.rag.generate_answer")
    @patch("app.rag.run_retrieval")
    def test_unsupported_flow_still_reaches_generator_for_fallback(
        self,
        mock_retrieve: MagicMock,
        mock_generate: MagicMock,
    ) -> None:
        from app.retrieval.models import AssembledContext, ConfidenceLevel

        mock_retrieve.return_value = AssembledContext(
            chunks=[],
            total_chars=0,
            confidence=ConfidenceLevel.LOW,
            normalized_query="What is the capital of France?",
            candidate_count=0,
        )
        mock_generate.return_value = {
            "answer": "Fallback answer",
            "sources": [],
            "fallback": True,
        }

        result = answer_question("What is the capital of France?", request_id="test-unsupported")
        self.assertTrue(result["fallback"])
        self.assertEqual(result["sources"], [])
        mock_generate.assert_called_once()
        self.assertEqual(mock_generate.call_args.args[1], [])


class RequestContextTests(unittest.TestCase):
    def test_request_id_reused_from_header(self) -> None:
        self.assertEqual(resolve_request_id("abc-123"), "abc-123")

    def test_request_id_generated_when_missing(self) -> None:
        generated = resolve_request_id(None)
        self.assertTrue(generated)

    def test_metrics_logging(self) -> None:
        metrics = RequestMetrics(
            request_id="req-1",
            provider="ollama",
            model="qwen3.5:4b",
            embedding_ms=1.0,
            retrieval_ms=2.0,
            prompt_ms=3.0,
            generation_ms=4.0,
            total_ms=10.0,
        )
        with self.assertLogs("rag_ai_chatbot", level="INFO") as captured:
            metrics.log_summary()
        self.assertIn("request_id=req-1", captured.output[0])


class FastAPITests(unittest.TestCase):
    def test_health_route_exists(self) -> None:
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
