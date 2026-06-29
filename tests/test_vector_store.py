"""
Unit and Integration Tests for Cortex AI Vector Store Layer.
Tests collection creation, insertion, updates, deletion, similarity search,
metadata filtering, duplicate detection exceptions, stats, diagnostics, and persistence.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, List

# Ensure workspace root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.embeddings import EmbeddedChunk
from core.vector_store.chroma_vector_store import ChromaVectorStore
from core.vector_store.vector_store_factory import VectorStoreFactory
from core.exceptions import (
    CollectionNotFoundException,
    DuplicateIdException,
    InvalidVectorException,
)
from core.repository.vector_repository import VectorRepository
EXPECTED_DIMENSION = 768


class TestChromaVectorStore(unittest.TestCase):
    """Test suite for ChromaVectorStore operations."""

    def setUp(self):
        """Creates a temporary folder and instantiates ChromaVectorStore."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name)
        self.store = ChromaVectorStore(persist_dir=self.db_path)
        
        # Test collection name
        self.collection_name = "test_cortex_collection"
        
        # Setup standard mock chunks
        self.vector_a = [0.1] * EXPECTED_DIMENSION
        # Opposite direction vector to guarantee low cosine similarity
        self.vector_b = [-0.1] * EXPECTED_DIMENSION

        self.chunk_a = EmbeddedChunk(
            chunk_id="chunk_a_123",
            text="This is document chunk A content. Discussing neural architectures.",
            vector=self.vector_a,
            metadata={
                "document_id": "doc_1",
                "chunk_id": "chunk_a_123",
                "chunk_index": 0,
                "source": "paper1.pdf",
                "page": 1,
                "total_pages": 5,
                "file_hash": "hash_abc",
                "created_at": "2026-06-29 17:00:00",
                "embedding_model": "models/text-embedding-004",
                "embedding_provider": "Google Gemini",
                "embedding_dimension": 768,
                "embedding_version": "1.0.0",
                "generated_at": "2026-06-29 17:01:00"
            }
        )

        self.chunk_b = EmbeddedChunk(
            chunk_id="chunk_b_123",
            text="This is document chunk B content. Discussing RAG databases.",
            vector=self.vector_b,
            metadata={
                "document_id": "doc_2",
                "chunk_id": "chunk_b_123",
                "chunk_index": 0,
                "source": "paper2.pdf",
                "page": 3,
                "total_pages": 12,
                "file_hash": "hash_xyz",
                "created_at": "2026-06-29 17:00:00",
                "embedding_model": "models/text-embedding-004",
                "embedding_provider": "Google Gemini",
                "embedding_dimension": 768,
                "embedding_version": "1.0.0",
                "generated_at": "2026-06-29 17:01:00"
            }
        )

    def tearDown(self):
        """Cleans up database and closes active clients, releasing locked file handles."""
        self.store.close()
        import gc
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_factory_creation(self):
        """Test that the VectorStoreFactory constructs instances correctly."""
        factory_store = VectorStoreFactory.get_vector_store(
            provider_name="chromadb",
            persist_dir=self.db_path
        )
        self.assertIsInstance(factory_store, ChromaVectorStore)

        # Invalid provider
        with self.assertRaises(ValueError):
            VectorStoreFactory.get_vector_store("pinecone")

    def test_collection_lifecycle(self):
        """Test creating and deleting collections."""
        # Active client check
        client = self.store._get_client()
        
        # Verify collection count starts at 0
        self.assertEqual(len(client.list_collections()), 0)

        # Create
        self.store.create_collection(self.collection_name)
        self.assertEqual(len(client.list_collections()), 1)
        self.assertEqual(client.list_collections()[0].name, self.collection_name)

        # Delete
        self.store.delete_collection(self.collection_name)
        self.assertEqual(len(client.list_collections()), 0)

        # Delete non-existent collection raises CollectionNotFoundException
        with self.assertRaises(CollectionNotFoundException):
            self.store.delete_collection("invalid_col")

    def test_add_embeddings_and_duplicate_prevention(self):
        """Test inserting EmbeddedChunks and ensuring duplicate key protection triggers exceptions."""
        self.store.create_collection(self.collection_name)

        # Insert A and B
        self.store.add_embeddings(self.collection_name, [self.chunk_a, self.chunk_b])
        
        stats = self.store.collection_statistics(self.collection_name)
        self.assertEqual(stats["document_count"], 2)

        # Try to insert chunk A again (duplicate ID check) -> raises DuplicateIdException
        with self.assertRaises(DuplicateIdException):
            self.store.add_embeddings(self.collection_name, [self.chunk_a])

        # Try to insert a list containing duplicate IDs within the list -> raises DuplicateIdException
        chunk_dup = EmbeddedChunk(
            chunk_id="chunk_dup",
            text="text",
            vector=self.vector_a,
            metadata=self.chunk_a.metadata.copy()
        )
        # Modify copy IDs to match list duplication
        chunk_dup.metadata["chunk_id"] = "chunk_dup"
        with self.assertRaises(DuplicateIdException):
            self.store.add_embeddings(self.collection_name, [chunk_dup, chunk_dup])

    def test_invalid_vector_dimensions(self):
        """Test dimension checking rejects invalid vector lengths."""
        self.store.create_collection(self.collection_name)
        
        invalid_vector = [0.1] * 128  # Wrong dimension (expects 768)
        chunk_invalid = EmbeddedChunk(
            chunk_id="chunk_invalid",
            text="invalid vector size",
            vector=invalid_vector,
            metadata=self.chunk_a.metadata.copy()
        )
        chunk_invalid.metadata["chunk_id"] = "chunk_invalid"

        with self.assertRaises(InvalidVectorException):
            self.store.add_embeddings(self.collection_name, [chunk_invalid])

    def test_update_and_delete_embeddings(self):
        """Test updating text content/vectors and deleting records."""
        self.store.create_collection(self.collection_name)
        self.store.add_embeddings(self.collection_name, [self.chunk_a])

        # Update text content and vector
        updated_vector = [0.9] * EXPECTED_DIMENSION
        updated_chunk = EmbeddedChunk(
            chunk_id="chunk_a_123",
            text="Updated content showing learning theory.",
            vector=updated_vector,
            metadata=self.chunk_a.metadata.copy()
        )
        # Change source metadata inside update
        updated_chunk.metadata["source"] = "updated_paper.pdf"

        # Execute Update
        self.store.update_embeddings(self.collection_name, [updated_chunk])

        # Verify update via retrieve
        client = self.store._get_client()
        col = client.get_collection(self.collection_name)
        res = col.get(ids=["chunk_a_123"])
        
        self.assertEqual(res["documents"][0], "Updated content showing learning theory.")
        self.assertEqual(res["metadatas"][0]["source"], "updated_paper.pdf")

        # Delete record
        self.store.delete_embeddings(self.collection_name, ["chunk_a_123"])
        self.assertEqual(col.count(), 0)

    def test_similarity_search_and_score_threshold(self):
        """Test similarity search retrieves expected results and filters by score threshold."""
        self.store.create_collection(self.collection_name)
        self.store.add_embeddings(self.collection_name, [self.chunk_a, self.chunk_b])

        # Query using vector_a (exact match for chunk A)
        results = self.store.similarity_search(
            self.collection_name,
            query_vector=self.vector_a,
            k=2
        )

        self.assertEqual(len(results), 2)
        # Cosine similarity for exact vector A match should be 1.0 (or very close due to floats)
        self.assertEqual(results[0]["chunk_id"], "chunk_a_123")
        self.assertAlmostEqual(results[0]["score"], 1.0, places=4)
        
        # Verify result format
        self.assertEqual(results[0]["text"], self.chunk_a.text)
        self.assertEqual(results[0]["metadata"]["source"], "paper1.pdf")

        # Test score threshold (set threshold to 0.99)
        # Vector B similarity is less than 0.99 compared to vector A, so it should be filtered out
        results_threshold = self.store.similarity_search(
            self.collection_name,
            query_vector=self.vector_a,
            k=2,
            score_threshold=0.99
        )
        self.assertEqual(len(results_threshold), 1)
        self.assertEqual(results_threshold[0]["chunk_id"], "chunk_a_123")

    def test_metadata_filtering(self):
        """Test filter searches using metadata parameters."""
        self.store.create_collection(self.collection_name)
        self.store.add_embeddings(self.collection_name, [self.chunk_a, self.chunk_b])

        # Filter by source: 'paper2.pdf'
        filter_meta = {"source": "paper2.pdf"}
        results = self.store.filter_search(
            self.collection_name,
            query_vector=self.vector_a,
            metadata_filters=filter_meta,
            k=2
        )

        # Should only return chunk B (even though vector A is closer to chunk A)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "chunk_b_123")
        self.assertEqual(results[0]["metadata"]["source"], "paper2.pdf")

    def test_health_check_diagnostics(self):
        """Test structured health check diagnostics report."""
        self.store.create_collection(self.collection_name)
        self.store.add_embeddings(self.collection_name, [self.chunk_a])

        report = self.store.health_check()
        
        self.assertEqual(report["status"], "healthy")
        self.assertEqual(report["provider"], "ChromaDB")
        self.assertEqual(report["database_path"], str(self.db_path.resolve()))
        self.assertEqual(report["collection_count"], 1)
        self.assertEqual(report["total_vectors"], 1)
        self.assertGreater(report["latency_ms"], 0.0)
        self.assertIn("Version", report.get("details", "") or report.get("storage_size", ""))

    def test_persistence_between_connections(self):
        """Test that data persists locally when store client is closed and reopened."""
        self.store.create_collection(self.collection_name)
        self.store.add_embeddings(self.collection_name, [self.chunk_a])

        # Close the connection (self.store._client set to None)
        self.store.close()
        self.assertIsNone(self.store._client)

        # Open a new store instance pointing to same directory
        reopened_store = ChromaVectorStore(persist_dir=self.db_path)
        stats = reopened_store.collection_statistics(self.collection_name)
        
        self.assertEqual(stats["document_count"], 1)
        reopened_store.close()


class TestVectorRepository(unittest.TestCase):
    """Test suite for the VectorRepository layer."""

    def setUp(self):
        """Creates a temporary folder and instantiates VectorRepository."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name)
        self.store = ChromaVectorStore(persist_dir=self.db_path)
        self.repo = VectorRepository(vector_store=self.store)
        
        self.collection_name = "test_repo_collection"
        
        self.vector_a = [0.1] * EXPECTED_DIMENSION
        self.chunk_a = EmbeddedChunk(
            chunk_id="repo_chunk_a",
            text="This is repo document content chunk A.",
            vector=self.vector_a,
            metadata={
                "document_id": "doc_1",
                "chunk_id": "repo_chunk_a",
                "chunk_index": 0,
                "source": "manual.pdf",
                "page": 1,
                "total_pages": 3,
                "file_hash": "hash_aaa",
                "created_at": "2026-06-29 17:00:00"
            }
        )

    def tearDown(self):
        """Cleans up database and repository connections."""
        self.store.close()
        import gc
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_repository_collection_metadata(self):
        """Test collection creation and metadata parameter initialization."""
        self.repo.create_collection(
            collection_name=self.collection_name,
            embedding_provider="Google Gemini",
            embedding_model="models/text-embedding-004",
            embedding_dimension=768,
            version="1.2.0"
        )

        meta = self.store.get_collection_metadata(self.collection_name)
        self.assertEqual(meta["collection_name"], self.collection_name)
        self.assertEqual(meta["collection_version"], "1.2.0")
        self.assertEqual(meta["embedding_provider"], "Google Gemini")
        self.assertEqual(meta["embedding_model"], "models/text-embedding-004")
        self.assertEqual(meta["embedding_dimension"], 768)
        self.assertEqual(meta["total_chunks"], 0)

    def test_version_management(self):
        """Test version compatibility check and upgrading version numbers."""
        self.repo.create_collection(
            collection_name=self.collection_name,
            embedding_provider="Google Gemini",
            embedding_model="models/text-embedding-004",
            embedding_dimension=768,
            version="1.2.0"
        )

        # Compatible checks (matching major version '1')
        self.assertTrue(self.repo.check_version_compatibility(self.collection_name, "1.0.0"))
        self.assertTrue(self.repo.check_version_compatibility(self.collection_name, "1.5.0"))
        
        # Incompatible checks (different major version '2')
        self.assertFalse(self.repo.check_version_compatibility(self.collection_name, "2.0.0"))

        # Upgrade version
        self.repo.upgrade_collection_version(self.collection_name, "2.1.0")
        meta = self.store.get_collection_metadata(self.collection_name)
        self.assertEqual(meta["collection_version"], "2.1.0")
        self.assertTrue(self.repo.check_version_compatibility(self.collection_name, "2.0.0"))

    def test_indexing_operations_and_statistics(self):
        """Test document batch indexing and statistics monitoring."""
        self.repo.create_collection(
            collection_name=self.collection_name,
            embedding_provider="Google Gemini",
            embedding_model="models/text-embedding-004",
            embedding_dimension=768,
            version="1.0.0"
        )

        # Assert initial statistics
        stats = self.repo.get_index_statistics(self.collection_name)
        self.assertEqual(stats["total_indexed_chunks"], 0)
        self.assertEqual(stats["total_indexed_documents"], 0)

        # Perform index operations
        summary = self.repo.add_embeddings(self.collection_name, [self.chunk_a])
        self.assertTrue(summary["success"])
        self.assertEqual(summary["inserted_count"], 1)

        # Verify stats changes
        stats = self.repo.get_index_statistics(self.collection_name)
        self.assertEqual(stats["total_indexed_chunks"], 1)
        self.assertEqual(stats["total_indexed_documents"], 1)
        self.assertEqual(stats["collection_size"], 1)
        self.assertGreater(stats["average_indexing_time"], 0.0)
        self.assertNotEqual(stats["last_indexing_timestamp"], "none")

        # Verify collection metadata fields were incremented
        meta = self.store.get_collection_metadata(self.collection_name)
        self.assertEqual(meta["total_chunks"], 1)
        self.assertEqual(meta["total_documents"], 1)

        # Reset statistics
        self.repo.reset_statistics()
        reset_stats = self.repo.get_index_statistics(self.collection_name)
        self.assertEqual(reset_stats["total_indexed_chunks"], 0)
        # Verify collection metadata still has count (only in-memory metrics reset)
        self.assertEqual(reset_stats["collection_size"], 1)

    def test_failed_insertion_rollback(self):
        """Test transaction support rollback when inserting a batch containing an invalid chunk."""
        self.repo.create_collection(
            collection_name=self.collection_name,
            embedding_provider="Google Gemini",
            embedding_model="models/text-embedding-004",
            embedding_dimension=768,
            version="1.0.0"
        )

        # Second chunk with invalid dimensions (128 instead of 768)
        chunk_bad = EmbeddedChunk(
            chunk_id="repo_chunk_bad",
            text="invalid vector dimension size",
            vector=[0.5] * 128,
            metadata=self.chunk_a.metadata.copy()
        )
        chunk_bad.metadata["chunk_id"] = "repo_chunk_bad"

        # Attempt to insert batch -> validation layer throws InvalidVectorException
        with self.assertRaises(InvalidVectorException):
            self.repo.add_embeddings(self.collection_name, [self.chunk_a, chunk_bad])

        # Verify rollback: self.chunk_a should NOT exist in the collection
        client = self.store._get_client()
        col = client.get_collection(self.collection_name)
        self.assertEqual(col.count(), 0)

    def test_repository_health_check(self):
        """Test repository health diagnostics report updates last successful operation timestamp."""
        self.repo.create_collection(
            collection_name=self.collection_name,
            embedding_provider="Google Gemini",
            embedding_model="models/text-embedding-004",
            embedding_dimension=768,
            version="1.0.0"
        )

        # Confirm health report has active diagnostics
        report = self.repo.health_check()
        self.assertEqual(report["status"], "healthy")
        self.assertEqual(report["collection_versions"][self.collection_name], "1.0.0")
        self.assertEqual(report["initialization_state"], "initialized")
        self.assertNotEqual(report["last_successful_operation"], "none")


if __name__ == "__main__":
    unittest.main()
