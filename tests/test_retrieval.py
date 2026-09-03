"""Tests for Phase 3 intelligent retrieval."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.retrieval.confidence import evaluate_confidence
from app.retrieval.context import assemble_context
from app.retrieval.fusion import fuse_candidates
from app.retrieval.models import ConfidenceLevel, RetrievalCandidate
from app.retrieval.normalize import extract_technical_terms, normalize_query
from app.retrieval.pipeline import candidates_to_public_chunks, run_retrieval
from app.retrieval.rerank import rerank_candidates


def _candidate(
    chunk_id: int,
    content: str,
    *,
    similarity: float | None = None,
    keyword_score: float | None = None,
    methods: set[str] | None = None,
    source_path: str = "data/doc.txt",
    metadata: dict | None = None,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id=1,
        content=content,
        source_path=source_path,
        similarity=similarity,
        distance=None if similarity is None else 1.0 - similarity,
        keyword_score=keyword_score,
        vector_rank=1 if methods and "vector" in methods else None,
        keyword_rank=1 if methods and "keyword" in methods else None,
        retrieval_methods=set(methods or set()),
        chunk_metadata=metadata or {},
    )


class NormalizeTests(unittest.TestCase):
    def test_clean_query_unchanged_meaning(self) -> None:
        query = "What sensors are used?"
        self.assertEqual(normalize_query(query), query)

    def test_informal_query_normalized(self) -> None:
        result = normalize_query("wat sensors r used?")
        self.assertIn("what", result.lower())
        self.assertIn("are", result.lower())
        self.assertIn("sensors", result.lower())

    def test_technical_terms_preserved(self) -> None:
        result = normalize_query("Does ADXL345 work with ESP32 and C-MAPSS FD001 RUL?")
        self.assertIn("ADXL345", result)
        self.assertIn("ESP32", result)
        self.assertIn("C-MAPSS", result)
        self.assertIn("FD001", result)
        self.assertIn("RUL", result)

    def test_extract_technical_terms(self) -> None:
        terms = extract_technical_terms("Use ADXL345 and DS18B20 on ESP32")
        self.assertIn("ADXL345", terms)
        self.assertIn("DS18B20", terms)
        self.assertIn("ESP32", terms)

    def test_whitespace_and_punctuation(self) -> None:
        self.assertEqual(normalize_query("  What   sensors  ?  "), "What sensors?")


class FusionTests(unittest.TestCase):
    def test_deduplicates_by_chunk_id(self) -> None:
        vector = [
            _candidate(1, "alpha", similarity=0.9, methods={"vector"}),
            _candidate(2, "beta", similarity=0.8, methods={"vector"}),
        ]
        keyword = [
            _candidate(1, "alpha", keyword_score=2.0, methods={"keyword"}),
            _candidate(3, "gamma", keyword_score=1.5, methods={"keyword"}),
        ]
        fused = fuse_candidates(vector, keyword)
        ids = [item.chunk_id for item in fused]
        self.assertEqual(len(ids), 3)
        self.assertEqual(len(set(ids)), 3)

    def test_dual_hit_outranks_single_hit(self) -> None:
        vector = [
            _candidate(1, "shared", similarity=0.7, methods={"vector"}),
            _candidate(2, "vector-only", similarity=0.85, methods={"vector"}),
        ]
        vector[0].vector_rank = 2
        vector[1].vector_rank = 1
        keyword = [
            _candidate(1, "shared", keyword_score=2.0, methods={"keyword"}),
        ]
        keyword[0].keyword_rank = 1
        fused = fuse_candidates(vector, keyword)
        dual = next(item for item in fused if item.chunk_id == 1)
        self.assertTrue(dual.found_by_both)
        self.assertGreater(dual.fused_score, fused[-1].fused_score)


class RerankTests(unittest.TestCase):
    def test_deterministic_ordering(self) -> None:
        candidates = [
            _candidate(1, "ADXL345 accelerometer sensor", similarity=0.6, methods={"vector"}),
            _candidate(2, "unrelated text about weather", similarity=0.65, methods={"vector"}),
        ]
        candidates[0].fused_score = 0.02
        candidates[1].fused_score = 0.03
        first = rerank_candidates("ADXL345 sensor", list(candidates))
        second = rerank_candidates("ADXL345 sensor", list(candidates))
        self.assertEqual([c.chunk_id for c in first], [c.chunk_id for c in second])
        self.assertEqual(first[0].chunk_id, 1)


class ConfidenceTests(unittest.TestCase):
    def test_high_confidence_for_strong_supported_query(self) -> None:
        candidates = [
            _candidate(
                1,
                "CWRU bearing dataset and NASA C-MAPSS FD001",
                similarity=0.88,
                keyword_score=2.0,
                methods={"vector", "keyword"},
            ),
            _candidate(
                2,
                "motor-bearing condition monitoring datasets",
                similarity=0.8,
                keyword_score=1.5,
                methods={"vector", "keyword"},
            ),
        ]
        for item in candidates:
            item.rerank_score = 0.9
        self.assertEqual(evaluate_confidence(candidates), ConfidenceLevel.HIGH)

    def test_low_confidence_for_unrelated(self) -> None:
        candidates = [
            _candidate(1, "bearing vibration analysis", similarity=0.2, methods={"vector"}),
        ]
        self.assertEqual(evaluate_confidence(candidates), ConfidenceLevel.LOW)

    def test_empty_candidates_are_low(self) -> None:
        self.assertEqual(evaluate_confidence([]), ConfidenceLevel.LOW)


class ContextAssemblyTests(unittest.TestCase):
    def test_respects_max_chunks(self) -> None:
        candidates = [
            _candidate(1, "Alpha bearing vibration analysis report section one", similarity=0.95, methods={"vector"}),
            _candidate(2, "Warranty return procedure for industrial controllers", similarity=0.9, methods={"vector"}),
            _candidate(3, "Shipping logistics schedule for spare part kits", similarity=0.88, methods={"vector"}),
        ]
        selected = assemble_context(
            candidates,
            ConfidenceLevel.HIGH,
            max_chunks=2,
            max_chars=10000,
        )
        self.assertEqual(len(selected), 2)

    def test_low_confidence_returns_empty(self) -> None:
        candidates = [
            _candidate(1, "something", similarity=0.9, methods={"vector"}),
        ]
        self.assertEqual(assemble_context(candidates, ConfidenceLevel.LOW), [])

    def test_near_duplicate_removal(self) -> None:
        candidates = [
            _candidate(1, "The ADXL345 sensor measures acceleration values carefully", similarity=0.9),
            _candidate(2, "The ADXL345 sensor measures acceleration values carefully today", similarity=0.88),
            _candidate(3, "Completely different warranty policy text about returns", similarity=0.85),
        ]
        selected = assemble_context(
            candidates,
            ConfidenceLevel.HIGH,
            max_chunks=5,
            max_chars=10000,
            near_duplicate_overlap=0.7,
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual({c.chunk_id for c in selected}, {1, 3})


class MetadataTests(unittest.TestCase):
    def test_public_chunk_preserves_metadata(self) -> None:
        candidate = _candidate(
            10,
            "Slide content",
            similarity=0.9,
            keyword_score=1.2,
            methods={"vector", "keyword"},
            metadata={
                "filename": "deck.pptx",
                "file_type": "pptx",
                "slide_number": 7,
                "title": "Troubleshooting",
            },
        )
        candidate.rerank_score = 1.1
        public = candidates_to_public_chunks([candidate])[0]
        self.assertEqual(public["chunk_id"], 10)
        self.assertEqual(public["filename"], "deck.pptx")
        self.assertEqual(public["slide_number"], 7)
        self.assertEqual(public["section"], "Troubleshooting")
        self.assertIn("vector", public["retrieval_methods"])
        self.assertIn("keyword", public["retrieval_methods"])


class PipelineTests(unittest.TestCase):
    @patch("app.retrieval.pipeline.keyword_search")
    @patch("app.retrieval.pipeline.vector_search")
    def test_hybrid_pipeline_combines_results(
        self,
        mock_vector: MagicMock,
        mock_keyword: MagicMock,
    ) -> None:
        mock_vector.return_value = [
            _candidate(1, "C-MAPSS FD001 dataset for RUL", similarity=0.86, methods={"vector"}),
        ]
        mock_vector.return_value[0].vector_rank = 1
        mock_keyword.return_value = [
            _candidate(1, "C-MAPSS FD001 dataset for RUL", keyword_score=2.5, methods={"keyword"}),
            _candidate(2, "ADXL345 sensor details", keyword_score=2.0, methods={"keyword"}),
        ]
        mock_keyword.return_value[0].keyword_rank = 1
        mock_keyword.return_value[1].keyword_rank = 2

        assembled = run_retrieval("C-MAPSS FD001")
        self.assertIn(assembled.confidence, {ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM})
        self.assertGreaterEqual(assembled.candidate_count, 2)
        self.assertTrue(assembled.chunks)

    @patch("app.retrieval.pipeline.keyword_search")
    @patch("app.retrieval.pipeline.vector_search")
    def test_unrelated_query_low_confidence(
        self,
        mock_vector: MagicMock,
        mock_keyword: MagicMock,
    ) -> None:
        mock_vector.return_value = [
            _candidate(1, "bearing monitoring text", similarity=0.2, methods={"vector"}),
        ]
        mock_keyword.return_value = []
        assembled = run_retrieval("What is the capital of France?")
        self.assertEqual(assembled.confidence, ConfidenceLevel.LOW)
        self.assertEqual(assembled.chunks, [])


class FastAPIChatTests(unittest.TestCase):
    @patch("app.main.answer_question")
    def test_chat_endpoint_contract(self, mock_answer: MagicMock) -> None:
        from fastapi.testclient import TestClient

        from app.main import app

        mock_answer.return_value = {
            "answer": "Grounded",
            "sources": ["data/a.docx"],
            "fallback": False,
        }
        client = TestClient(app)
        response = client.post("/chat", json={"question": "What sensors are used?"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer"], "Grounded")
        self.assertFalse(payload["fallback"])

    @patch("app.main.stream_answer_question")
    def test_chat_stream_endpoint_exists(self, mock_stream: MagicMock) -> None:
        from fastapi.testclient import TestClient

        from app.main import app

        mock_stream.return_value = iter(
            [
                {"type": "token", "text": "Hi"},
                {
                    "type": "complete",
                    "answer": "Hi",
                    "sources": [],
                    "fallback": False,
                },
            ]
        )
        client = TestClient(app)
        response = client.post("/chat/stream", json={"question": "Hello"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])


if __name__ == "__main__":
    unittest.main()
