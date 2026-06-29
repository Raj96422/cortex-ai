"""
Unit and Integration Tests for Cortex AI Provider-Abstracted Embedding Service.
Tests base provider ABCs, provider dependency injection, health checking,
enhanced metadata, caching, stats tracking, and exceptions.
"""

import sys
import unittest
import hashlib
from unittest.mock import MagicMock, patch
from pathlib import Path
from typing import Any, Dict, List

# Ensure workspace root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document
from core.exceptions import (
    EmbeddingTimeoutException,
    EmptyTextException,
    InvalidMetadataException,
    InvalidTextException,
    MissingAPIKeyException,
    RateLimitException,
)
from core.embeddings import EmbeddingService, EmbeddedChunk
EXPECTED_DIMENSION = 768
from core.providers.base_embedding_provider import EmbeddingProvider
from core.providers.gemini_embedding_provider import GeminiEmbeddingProvider
from utils.constants import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_RETRY_DELAY,
    MAX_EMBEDDING_TEXT_LENGTH,
    EMBEDDING_CACHE_LIMIT,
)


class MockCustomProvider(EmbeddingProvider):
    """Custom mock provider to test dependency injection and future extensibility."""

    @property
    def provider_name(self) -> str:
        return "Mock Provider"

    @property
    def model_name(self) -> str:
        return "mock-model-v1"

    @property
    def dimension(self) -> int:
        return 128  # Custom dimension

    def initialize_model(self) -> None:
        pass

    def embed_text(self, text: str) -> List[float]:
        return [0.9] * self.dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.9] * self.dimension for _ in texts]

    def validate_embedding_dimension(self, vector: List[float]) -> bool:
        return len(vector) == self.dimension

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "provider": self.provider_name,
            "model": self.model_name,
            "checks": {
                "api_key_configured": True,
                "model_initialized": True,
                "connectivity": True
            },
            "details": "Mock provider is running."
        }


class TestEmbeddingServiceRefactored(unittest.TestCase):
    """Test suite for the refactored, provider-abstracted EmbeddingService."""

    def setUp(self):
        """Prepares a clean service instance with mock Gemini embeddings."""
        # Use standard Gemini provider but mock the underlying model client
        self.gemini_provider = GeminiEmbeddingProvider()
        self.mock_model = MagicMock()
        self.gemini_provider.model = self.mock_model
        
        self.service = EmbeddingService(provider=self.gemini_provider)
        self.dummy_vector = [0.5] * EXPECTED_DIMENSION

    def test_provider_dependency_injection(self):
        """Test that a custom provider can be successfully injected and used."""
        custom_provider = MockCustomProvider()
        service = EmbeddingService(provider=custom_provider)

        # Verify initialized details
        self.assertEqual(service.provider.provider_name, "Mock Provider")
        self.assertEqual(service.provider.model_name, "mock-model-v1")
        self.assertEqual(service.provider.dimension, 128)

        # Generate embedding and assert dimension match
        vector = service.embed_text("Verify injection works.")
        self.assertEqual(len(vector), 128)
        self.assertEqual(vector, [0.9] * 128)

    def test_default_provider_fallback(self):
        """Test that GeminiEmbeddingProvider is automatically loaded if no provider is injected."""
        default_service = EmbeddingService()
        self.assertIsInstance(default_service.provider, GeminiEmbeddingProvider)
        self.assertEqual(default_service.provider.provider_name, "Google Gemini")

    def test_hash_based_cache(self):
        """Test that cache keys are SHA-256 hashes of the text rather than raw text."""
        self.mock_model.embed_query.return_value = self.dummy_vector
        text = "Hello, this is a test text."
        expected_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        # Execute
        self.service.embed_text(text)

        # Assert key structure
        self.assertIn(expected_hash, self.service._cache)
        self.assertNotIn(text, self.service._cache)
        self.assertEqual(self.service._cache[expected_hash], self.dummy_vector)

    @patch("core.embeddings.EMBEDDING_CACHE_LIMIT", 2)
    def test_cache_limit_eviction(self):
        """Test that the cache evicts the oldest entry (FIFO/LRU) when reaching EMBEDDING_CACHE_LIMIT."""
        local_provider = GeminiEmbeddingProvider()
        local_provider.model = self.mock_model
        local_service = EmbeddingService(provider=local_provider)
        self.mock_model.embed_query.return_value = self.dummy_vector

        # Embed 3 distinct texts
        hash1 = local_service._generate_cache_key("Text 1")
        hash2 = local_service._generate_cache_key("Text 2")
        hash3 = local_service._generate_cache_key("Text 3")

        local_service.embed_text("Text 1")
        local_service.embed_text("Text 2")
        self.assertEqual(len(local_service._cache), 2)

        # Triggers eviction of Text 1
        local_service.embed_text("Text 3")
        self.assertEqual(len(local_service._cache), 2)
        self.assertNotIn(hash1, local_service._cache)  # Evicted!
        self.assertIn(hash2, local_service._cache)
        self.assertIn(hash3, local_service._cache)

    def test_validation_layer_text(self):
        """Test validation checks on None, empty, whitespace-only, and excessively long texts."""
        with self.assertRaises(InvalidTextException):
            self.service.embed_text(None)  # type: ignore

        with self.assertRaises(InvalidTextException):
            self.service.embed_text(42)  # type: ignore

        with self.assertRaises(EmptyTextException):
            self.service.embed_text("")

        with self.assertRaises(EmptyTextException):
            self.service.embed_text("    \n   ")

        too_long_text = "a" * (MAX_EMBEDDING_TEXT_LENGTH + 1)
        with self.assertRaises(InvalidTextException):
            self.service.embed_text(too_long_text)

    def test_validation_layer_metadata(self):
        """Test metadata validation on Document chunks."""
        self.mock_model.embed_documents.return_value = [self.dummy_vector]

        # 1. Missing metadata dict
        doc_no_meta = Document(page_content="Valid text content.")
        doc_no_meta.metadata = None  # type: ignore
        with self.assertRaises(InvalidMetadataException):
            self.service.embed_documents([doc_no_meta])

        # 2. Missing chunk_id
        doc_missing_id = Document(
            page_content="Valid content.",
            metadata={"document_id": "doc1"}  # lacks chunk_id
        )
        with self.assertRaises(InvalidMetadataException):
            self.service.embed_documents([doc_missing_id])

        # 3. Missing other required fields (e.g. source, page)
        doc_partial_meta = Document(
            page_content="Valid content.",
            metadata={
                "chunk_id": "chunk123",
                "document_id": "doc123",
                "chunk_index": 0,
            }
        )
        with self.assertRaises(InvalidMetadataException):
            self.service.embed_documents([doc_partial_meta])

    def test_enhanced_metadata_generation(self):
        """Test that vector embeddings return EmbeddedChunks containing the enhanced metadata schema."""
        custom_provider = MockCustomProvider()
        service = EmbeddingService(provider=custom_provider)

        doc = Document(
            page_content="Test data chunk for metadata.",
            metadata={
                "document_id": "doc_id_123",
                "chunk_id": "doc_id_123_c0",
                "chunk_index": 0,
                "source": "/path/to/doc.pdf",
                "page": 1,
                "total_pages": 5,
                "file_hash": "file_hash_abc",
                "created_at": "2026-06-29 17:00:00"
            }
        )

        chunks = service.embed_documents([doc])
        self.assertEqual(len(chunks), 1)

        meta = chunks[0].metadata
        # Check preserved document metadata
        self.assertEqual(meta["chunk_id"], "doc_id_123_c0")
        self.assertEqual(meta["page"], 1)

        # Check injected embedding provider metadata
        self.assertEqual(meta["embedding_provider"], "Mock Provider")
        self.assertEqual(meta["embedding_model"], "mock-model-v1")
        self.assertEqual(meta["embedding_dimension"], 128)
        self.assertEqual(meta["embedding_version"], "1.0.0")
        self.assertIn("generated_at", meta)

    def test_health_check_report(self):
        """Test that the health_check method returns structured diagnostics report."""
        # 1. Test custom mock provider
        custom_provider = MockCustomProvider()
        service = EmbeddingService(provider=custom_provider)
        report = service.health_check()
        
        self.assertEqual(report["status"], "healthy")
        self.assertEqual(report["provider"], "Mock Provider")
        self.assertEqual(report["model"], "mock-model-v1")
        self.assertTrue(report["checks"]["connectivity"])

        # 2. Test Gemini provider failing connection check (when GOOGLE_API_KEY is missing)
        with patch("core.providers.gemini_embedding_provider.GOOGLE_API_KEY", ""):
            unhealthy_provider = GeminiEmbeddingProvider()
            service_unhealthy = EmbeddingService(provider=unhealthy_provider)
            report_fail = service_unhealthy.health_check()
            self.assertEqual(report_fail["status"], "unhealthy")
            self.assertFalse(report_fail["checks"]["api_key_configured"])

    def test_enhanced_statistics_tracking(self):
        """Test that statistics are tracked accurately during operations, including token estimation and cache percentages."""
        custom_provider = MockCustomProvider()
        service = EmbeddingService(provider=custom_provider)

        # Verify initial state
        stats = service.get_statistics()
        self.assertEqual(stats["total_requests"], 0)
        self.assertEqual(stats["cache_efficiency_percentage"], 0.0)
        self.assertEqual(stats["total_tokens_processed"], 0)

        # 1. Process 16-character query (Cache Miss) -> Estimate: 16 // 4 = 4 tokens
        text1 = "Test Query Tokens"
        service.embed_text(text1)
        
        stats = service.get_statistics()
        self.assertEqual(stats["total_requests"], 1)
        self.assertEqual(stats["cache_misses"], 1)
        self.assertEqual(stats["api_requests"], 1)
        self.assertEqual(stats["total_tokens_processed"], 4)
        self.assertEqual(stats["cache_efficiency_percentage"], 0.0)

        # 2. Process same query (Cache Hit)
        service.embed_text(text1)
        
        stats = service.get_statistics()
        self.assertEqual(stats["total_requests"], 2)
        self.assertEqual(stats["cache_hits"], 1)
        self.assertEqual(stats["cache_efficiency_percentage"], 50.0)  # 1/2 is 50%
        self.assertEqual(stats["total_tokens_processed"], 8)  # 4 + 4 tokens

        # 3. Batch process containing 2 cached and 2 uncached texts
        # (Total tokens processed: sum length of all texts // 4)
        texts = ["Test Query Tokens", "New batch query", "Ping connect", "Test Query Tokens"]
        # Character lengths: 17, 15, 12, 17
        # Tokens: 4, 3, 3, 4 -> Total batch tokens = 14
        service.batch_embed(texts, batch_size=2)
        
        stats = service.get_statistics()
        self.assertEqual(stats["total_requests"], 6)  # 2 initial + 4 batch
        self.assertEqual(stats["cache_hits"], 3)      # 1 initial + 2 batch hits
        self.assertEqual(stats["cache_efficiency_percentage"], 50.0) # 3/6 is 50%
        self.assertEqual(stats["api_requests"], 2)      # 1 initial + 1 batch API call
        self.assertEqual(stats["average_batch_size"], 1.5)  # (1 chunk + 2 chunks) / 2 calls = 1.5 average
        self.assertGreater(stats["average_request_latency"], 0.0)

        # 4. Reset
        service.reset_statistics()
        reset_stats = service.get_statistics()
        self.assertEqual(reset_stats["total_requests"], 0)
        self.assertEqual(reset_stats["total_tokens_processed"], 0)

    @patch("time.sleep")
    def test_retry_mechanism_error_count(self, mock_sleep):
        """Test retry mechanisms update failed_embeddings stat on exception throw."""
        self.mock_model.embed_query.side_effect = Exception("Generic Gemini Error (500)")

        with self.assertRaises(Exception):
            self.service.embed_text("Test error tracing.")

        stats = self.service.get_statistics()
        self.assertEqual(stats["failed_embeddings"], 1)


if __name__ == "__main__":
    unittest.main()
