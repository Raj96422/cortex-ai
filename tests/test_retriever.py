"""
Unit and Integration Tests for Cortex AI Semantic Retriever.
Tests query validations, standard similarity searches, MMR ranking, metadata filters,
result/embedding caching, statistics collection, and diagnostics.
"""

import sys
import unittest
import time
from unittest.mock import MagicMock, patch
from pathlib import Path
from typing import Any, Dict, List

# Ensure workspace root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.embeddings import EmbeddingService
from core.repository.vector_repository import VectorRepository
from core.retriever.semantic_retriever import SemanticRetriever
from core.retriever.retriever_factory import RetrieverFactory
from core.exceptions import (
    CollectionNotFoundException,
    EmptyQueryException,
    InvalidQueryException,
)

EXPECTED_DIMENSION = 768


class TestSemanticRetriever(unittest.TestCase):
    """Test suite for SemanticRetriever operations."""

    def setUp(self):
        """Prepares mock dependencies and configures SemanticRetriever."""
        self.mock_embeddings = MagicMock(spec=EmbeddingService)
        self.mock_repo = MagicMock(spec=VectorRepository)

        # Standard configuration
        self.retriever = SemanticRetriever(
            embedding_service=self.mock_embeddings,
            vector_repository=self.mock_repo,
            search_strategy="similarity",
            lambda_mult=0.5,
            max_query_length=100
        )

        self.collection_name = "test_cortex_collection"
        self.query_vector = [0.1] * EXPECTED_DIMENSION
        self.mock_embeddings.embed_text.return_value = self.query_vector

        # Setup mock chunks in repository
        self.mock_search_results = [
            {
                "chunk_id": "chunk_1",
                "text": "This is text chunk 1, discussing agentic frameworks.",
                "score": 0.95,
                "metadata": {
                    "document_id": "doc1",
                    "page": 1,
                    "source": "agents.pdf",
                    "embedding_provider": "Google Gemini",
                    "embedding_model": "models/text-embedding-004",
                    "file_hash": "hash1"
                }
            },
            {
                "chunk_id": "chunk_2",
                "text": "This is text chunk 2, discussing vector space indexing.",
                "score": 0.88,
                "metadata": {
                    "document_id": "doc2",
                    "page": 5,
                    "source": "indexing.pdf",
                    "embedding_provider": "Google Gemini",
                    "embedding_model": "models/text-embedding-004",
                    "file_hash": "hash2"
                }
            }
        ]
        self.mock_repo.similarity_search.return_value = self.mock_search_results

    def test_factory_creation(self):
        """Test that the RetrieverFactory constructs retriever instances correctly."""
        ret = RetrieverFactory.get_retriever(
            retriever_type="semantic",
            embedding_service=self.mock_embeddings,
            vector_repository=self.mock_repo
        )
        self.assertIsInstance(ret, SemanticRetriever)

        # Missing DI parameter
        with self.assertRaises(ValueError):
            RetrieverFactory.get_retriever(
                retriever_type="semantic",
                embedding_service=self.mock_embeddings
            )

        # Invalid retriever type
        with self.assertRaises(ValueError):
            RetrieverFactory.get_retriever(retriever_type="hybrid")

    def test_query_validation(self):
        """Test validation exceptions for invalid or empty queries."""
        with self.assertRaises(InvalidQueryException):
            self.retriever.retrieve(None, self.collection_name)  # type: ignore

        with self.assertRaises(InvalidQueryException):
            self.retriever.retrieve(12345, self.collection_name)  # type: ignore

        with self.assertRaises(EmptyQueryException):
            self.retriever.retrieve("", self.collection_name)

        with self.assertRaises(EmptyQueryException):
            self.retriever.retrieve("     \n    ", self.collection_name)

        # Exceeds max length (configured to 100 in setUp)
        too_long = "a" * 101
        with self.assertRaises(InvalidQueryException):
            self.retriever.retrieve(too_long, self.collection_name)

    def test_standard_similarity_retrieval(self):
        """Test successful query retrieval and chunk metadata packing."""
        results = self.retriever.retrieve("agentic frameworks", self.collection_name, k=2)

        self.assertEqual(len(results), 2)
        
        # Verify first item contents
        item_1 = results[0]
        self.assertEqual(item_1["chunk_id"], "chunk_1")
        self.assertEqual(item_1["text"], "This is text chunk 1, discussing agentic frameworks.")
        self.assertEqual(item_1["similarity_score"], 0.95)
        self.assertEqual(item_1["document_id"], "doc1")
        self.assertEqual(item_1["page"], 1)
        self.assertEqual(item_1["source"], "agents.pdf")
        self.assertEqual(item_1["embedding_provider"], "Google Gemini")
        self.assertEqual(item_1["embedding_model"], "models/text-embedding-004")

        # Verify repo search was called correctly
        self.mock_repo.similarity_search.assert_called_once_with(
            collection_name=self.collection_name,
            query_vector=self.query_vector,
            k=2,
            score_threshold=None
        )

    def test_result_caching_efficiency(self):
        """Test that identical queries trigger cache hits and skip redundant vector lookups."""
        # Query 1 (Cache Miss)
        self.retriever.retrieve("agentic frameworks", self.collection_name, k=2)
        stats = self.retriever.get_statistics()
        self.assertEqual(stats["cache_misses"], 1)
        self.assertEqual(stats["cache_hits"], 0)
        self.assertEqual(self.mock_embeddings.embed_text.call_count, 1)

        # Query 2 (Cache Hit)
        self.retriever.retrieve("agentic frameworks", self.collection_name, k=2)
        stats = self.retriever.get_statistics()
        self.assertEqual(stats["cache_misses"], 1)
        self.assertEqual(stats["cache_hits"], 1)
        # Verify embedding service was NOT called again
        self.assertEqual(self.mock_embeddings.embed_text.call_count, 1)

    def test_metadata_and_document_filtering(self):
        """Test metadata and document ID query scoping filters."""
        self.mock_repo.filter_search.return_value = [self.mock_search_results[1]]

        filters = {"source": "indexing.pdf"}
        results = self.retriever.retrieve_with_metadata(
            query="indexing",
            collection_name=self.collection_name,
            metadata_filters=filters,
            k=1
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "chunk_2")

        # Verify filter_search was dispatched to repo
        self.mock_repo.filter_search.assert_called_once_with(
            collection_name=self.collection_name,
            query_vector=self.query_vector,
            metadata_filters=filters,
            k=1
        )

        # Test document scoping wrapper
        self.mock_repo.filter_search.reset_mock()
        self.retriever.retrieve_by_document(
            query="indexing",
            collection_name=self.collection_name,
            document_id="doc2",
            k=1
        )
        self.mock_repo.filter_search.assert_called_once_with(
            collection_name=self.collection_name,
            query_vector=self.query_vector,
            metadata_filters={"document_id": "doc2"},
            k=1
        )

    def test_mmr_diversification(self):
        """Test MMR reranking selects a diversified candidate list."""
        mmr_retriever = SemanticRetriever(
            embedding_service=self.mock_embeddings,
            vector_repository=self.mock_repo,
            search_strategy="mmr",
            lambda_mult=0.5
        )

        # Vector coordinates skewing to check MMR diversity selection
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.95, 0.1, 0.0]  # Very close to vec_a (redundant)
        vec_c = [0.1, 0.9, 0.0]   # Different direction (diverse)

        candidates = [
            {"chunk_id": "c1", "text": "text_a", "score": 0.9, "metadata": {}},
            {"chunk_id": "c2", "text": "text_b", "score": 0.85, "metadata": {}},
            {"chunk_id": "c3", "text": "text_c", "score": 0.7, "metadata": {}},
        ]
        self.mock_repo.similarity_search.return_value = candidates

        # Configure mock embedding return values for candidates
        def embed_text_side_effect(text):
            if text == "text_a":
                return vec_a
            if text == "text_b":
                return vec_b
            return vec_c

        self.mock_embeddings.embed_text.side_effect = embed_text_side_effect

        # Retrieve (k=2) -> should select c1 (highest similarity) and c3 (diverse), skipping c2
        results = mmr_retriever.retrieve("diverse query", self.collection_name, k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "c1")
        self.assertEqual(results[1]["chunk_id"], "c3")  # Selected over c2 due to diversity!

    def test_health_check_diagnostics(self):
        """Test structured health report maps provider states."""
        self.mock_repo.health_check.return_value = {
            "status": "healthy",
            "total_vectors": 10,
            "collection_status": {"collections": [self.collection_name]}
        }
        self.mock_embeddings.health_check.return_value = {
            "status": "healthy"
        }

        report = self.retriever.health_check()
        self.assertEqual(report["status"], "healthy")
        self.assertEqual(report["repository_status"], "healthy")
        self.assertEqual(report["embedding_provider_status"], "healthy")
        self.assertEqual(report["total_indexed_chunks"], 10)
        self.assertIn(self.collection_name, report["collection_availability"])

    def test_statistics_aggregation(self):
        """Test metrics tracking updates and reset routines."""
        # Initial checks
        stats = self.retriever.get_statistics()
        self.assertEqual(stats["total_queries"], 0)

        # Run lookup
        self.retriever.retrieve("indexing", self.collection_name, k=2)
        stats = self.retriever.get_statistics()
        self.assertEqual(stats["total_queries"], 1)
        self.assertEqual(stats["successful_retrievals"], 1)
        self.assertEqual(stats["average_returned_chunks"], 2.0)
        self.assertGreater(stats["average_retrieval_time"], 0.0)

        # Reset checks
        self.retriever.reset_statistics()
        reset_stats = self.retriever.get_statistics()
        self.assertEqual(reset_stats["total_queries"], 0)
        self.assertEqual(reset_stats["average_retrieval_time"], 0.0)


if __name__ == "__main__":
    unittest.main()
