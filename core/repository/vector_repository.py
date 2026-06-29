"""
Vector Repository Module for Cortex AI.
Coordinates indexing operations, manages collection versioning/metadata,
aggregates metrics, validates chunk parameters, and simulates atomic rollbacks on failure.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.embeddings import EmbeddedChunk
from core.vector_store.base_vector_store import BaseVectorStore
from core.exceptions import (
    CollectionNotFoundException,
    DuplicateIdException,
    InvalidMetadataException,
    InvalidVectorException,
)
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)


class VectorRepository:
    """
    Repository Layer managing business logic for Vector Databases.
    
    Coordinates writes, queries, versioning metadata updates, transaction boundaries,
    and validation constraints on top of any abstract BaseVectorStore instance.
    """

    def __init__(self, vector_store: BaseVectorStore):
        """
        Initializes the VectorRepository.

        Args:
            vector_store (BaseVectorStore): Concrete vector store client backend.
        """
        self.vector_store = vector_store
        self._stats: Dict[str, Any] = {}
        self.reset_statistics()
        
        logger.info(f"VectorRepository initialized with store type: '{type(vector_store).__name__}'")

    def create_collection(
        self,
        collection_name: str,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimension: int,
        version: str = "1.0.0"
    ) -> None:
        """
        Creates a new collection and initializes its metadata record.

        Args:
            collection_name (str): Collection name.
            embedding_provider (str): Provider name (e.g. 'Google Gemini').
            embedding_model (str): Model name (e.g. 'models/text-embedding-004').
            embedding_dimension (int): Vector dimension length (e.g. 768).
            version (str): Custom semantic versioning.
        """
        logger.info(f"Repository: Creating collection '{collection_name}'...")
        self.vector_store.create_collection(collection_name)
        
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        metadata = {
            "collection_name": collection_name,
            "collection_version": version,
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
            "created_at": now_str,
            "updated_at": now_str,
            "total_documents": 0,
            "total_chunks": 0,
            "total_vectors": 0
        }
        self.vector_store.modify_collection_metadata(collection_name, metadata)
        logger.info(f"Repository: Collection metadata initialized for '{collection_name}' Version: {version}.")

    def delete_collection(self, collection_name: str) -> None:
        """
        Deletes a collection and clears its data.

        Args:
            collection_name (str): Target collection.
        """
        logger.info(f"Repository: Deleting collection '{collection_name}'...")
        self.vector_store.delete_collection(collection_name)

    def check_version_compatibility(self, collection_name: str, expected_version: str) -> bool:
        """
        Validates collection version compatibility.

        Verifies that the major version matches the expected semantic version.

        Args:
            collection_name (str): Collection to test.
            expected_version (str): Target version.

        Returns:
            bool: True if compatible.
        """
        meta = self.vector_store.get_collection_metadata(collection_name)
        col_ver = meta.get("collection_version", "1.0.0")
        
        col_major = col_ver.split(".")[0]
        exp_major = expected_version.split(".")[0]
        
        compatible = col_major == exp_major
        logger.debug(
            f"Version check for '{collection_name}': collection version is '{col_ver}', "
            f"expected major matches: {compatible}"
        )
        return compatible

    def upgrade_collection_version(self, collection_name: str, new_version: str) -> None:
        """
        Upgrades the collection version inside its metadata.

        Args:
            collection_name (str): Target collection.
            new_version (str): New semantic version.
        """
        logger.info(f"Repository: Upgrading collection '{collection_name}' to version '{new_version}'...")
        meta = self.vector_store.get_collection_metadata(collection_name)
        meta["collection_version"] = new_version
        meta["updated_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        self.vector_store.modify_collection_metadata(collection_name, meta)

    def _validate_batch(self, collection_name: str, chunks: List[EmbeddedChunk]) -> None:
        """
        Runs comprehensive validation constraints on an insertion batch.

        Args:
            collection_name (str): Collection name.
            chunks (List[EmbeddedChunk]): Incoming batch of chunks.
        """
        if not chunks:
            raise ValueError("Insertion batch is empty. Cannot add embeddings.")

        meta = self.vector_store.get_collection_metadata(collection_name)
        expected_dim = meta.get("embedding_dimension")

        ids: List[str] = []
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, EmbeddedChunk):
                raise ValueError(f"Batch item at index {idx} must be an EmbeddedChunk.")

            if not chunk.chunk_id or not isinstance(chunk.chunk_id, str):
                raise InvalidMetadataException(f"Chunk at index {idx} has missing or invalid chunk_id.")
                
            if not chunk.text or not chunk.text.strip():
                raise ValueError(f"Chunk '{chunk.chunk_id}' has missing or empty document text.")
                
            if not chunk.vector:
                raise InvalidVectorException(f"Chunk '{chunk.chunk_id}' is missing its vector embedding.")

            if expected_dim is not None and len(chunk.vector) != expected_dim:
                raise InvalidVectorException(
                    f"Chunk '{chunk.chunk_id}' vector dimension ({len(chunk.vector)}) "
                    f"does not match collection metadata expectation ({expected_dim})."
                )

            if not chunk.metadata or not isinstance(chunk.metadata, dict):
                raise InvalidMetadataException(f"Chunk '{chunk.chunk_id}' is missing a valid metadata dict.")

            ids.append(chunk.chunk_id)

        # Duplicate key check within the batch list
        if len(ids) != len(set(ids)):
            raise DuplicateIdException("Duplicate chunk IDs detected inside the incoming batch.")

    def add_embeddings(self, collection_name: str, chunks: List[EmbeddedChunk]) -> Dict[str, Any]:
        """
        Performs validation, statistics accumulation, duplicate detection, 
        and dispatches writes to the store with transactional rollback capability.

        Args:
            collection_name (str): Target collection.
            chunks (List[EmbeddedChunk]): Chunks to insert.

        Returns:
            Dict[str, Any]: Transaction summary dictionary.
        """
        start_time = time.perf_counter()
        self._stats["indexing_calls"] += 1
        
        summary = {
            "success": False,
            "inserted_count": 0,
            "skipped_count": 0,
            "rolled_back": False,
            "error_message": None
        }

        try:
            # 1. Validation
            self._validate_batch(collection_name, chunks)
            
            # 2. Check for duplicate ID keys in store
            ids = [c.chunk_id for c in chunks]
            try:
                # Direct check via store pre-queries
                store_stats = self.vector_store.collection_statistics(collection_name)
                # Verify duplicates if collection has records
                if store_stats["document_count"] > 0:
                    for chunk_id in ids:
                        # Test if ID already exists
                        # In ChromaDB, we can test existence via get query
                        # To keep it provider-agnostic, the store's add_embeddings checks this
                        pass
            except CollectionNotFoundException:
                raise

            # 3. Dispatched insert
            logger.info(f"Repository: Storing batch of {len(chunks)} chunks in '{collection_name}'...")
            self.vector_store.add_embeddings(collection_name, chunks)
            
            # 4. Successful batch - Update stats and collection metadata
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self._stats["total_indexing_time_ms"] += elapsed_ms
            self._stats["total_indexed_chunks"] += len(chunks)
            
            # Deduplicate file_hash strings to count unique documents
            doc_hashes = {c.metadata.get("file_hash") for c in chunks if c.metadata.get("file_hash")}
            self._stats["total_indexed_documents"] += len(doc_hashes)
            self._stats["last_indexing_timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

            # Update collection metadata attributes
            meta = self.vector_store.get_collection_metadata(collection_name)
            meta["total_chunks"] = meta.get("total_chunks", 0) + len(chunks)
            meta["total_vectors"] = meta["total_chunks"]
            meta["total_documents"] = meta.get("total_documents", 0) + len(doc_hashes)
            meta["updated_at"] = self._stats["last_indexing_timestamp"]
            self.vector_store.modify_collection_metadata(collection_name, meta)

            summary["success"] = True
            summary["inserted_count"] = len(chunks)
            
        except DuplicateIdException as dup_err:
            self._stats["duplicate_chunks_skipped"] += len(chunks)
            self._stats["failed_insertions"] += 1
            summary["error_message"] = str(dup_err)
            logger.warning(f"Repository: Duplicate insertion skipped: {dup_err}")
            raise
            
        except Exception as err:
            self._stats["failed_insertions"] += 1
            summary["error_message"] = str(err)
            logger.error(f"Repository: Failed batch insertion: {err}. Executing atomic rollback cleanups...")
            
            # Rollback: delete any chunk IDs present in the failed batch to prevent partial indexes
            try:
                self.vector_store.delete_embeddings(collection_name, [c.chunk_id for c in chunks])
                summary["rolled_back"] = True
                logger.info("Repository: Batch transaction rolled back successfully.")
            except Exception as rollback_err:
                logger.error(f"Repository: Rollback failed during cleanup execution: {rollback_err}")
                
            raise
            
        return summary

    def similarity_search(
        self,
        collection_name: str,
        query_vector: List[float],
        k: int = 4,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Passes similarity query down to the vector store.

        Args:
            collection_name (str): Collection name.
            query_vector (List[float]): Embedding search vector.
            k (int): Retrieve top-K.
            score_threshold (Optional[float]): Minimum match confidence.

        Returns:
            List[Dict[str, Any]]: Semantic query results list.
        """
        return self.vector_store.similarity_search(
            collection_name,
            query_vector,
            k=k,
            score_threshold=score_threshold
        )

    def filter_search(
        self,
        collection_name: str,
        query_vector: List[float],
        metadata_filters: Dict[str, Any],
        k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Passes filtered similarity search down to the vector store.

        Args:
            collection_name (str): Collection name.
            query_vector (List[float]): Search vector.
            metadata_filters (Dict[str, Any]): Filter matching criteria.
            k (int): Top-K match count.

        Returns:
            List[Dict[str, Any]]: Semantic matches.
        """
        return self.vector_store.filter_search(
            collection_name,
            query_vector,
            metadata_filters=metadata_filters,
            k=k
        )

    def reset_statistics(self) -> None:
        """Resets all metrics and indexing performance stats to zero."""
        self._stats = {
            "total_indexed_documents": 0,
            "total_indexed_chunks": 0,
            "duplicate_chunks_skipped": 0,
            "total_indexing_time_ms": 0.0,
            "indexing_calls": 0,
            "failed_insertions": 0,
            "last_indexing_timestamp": "none"
        }
        logger.info("VectorRepository indexing statistics have been reset.")

    def get_index_statistics(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns accumulated index stats merged with current collection metadata sizes.

        Args:
            collection_name (Optional[str]): Target collection to retrieve sizes.

        Returns:
            Dict[str, Any]: Statistics report dictionary.
        """
        stats_copy = self._stats.copy()
        
        # Calculate average indexing time
        calls = stats_copy.get("indexing_calls", 0)
        time_spent = stats_copy.get("total_indexing_time_ms", 0.0)
        stats_copy["average_indexing_time"] = time_spent / calls if calls > 0 else 0.0
        
        # Fetch physical sizes if collection is defined
        if collection_name:
            try:
                meta = self.vector_store.get_collection_metadata(collection_name)
                stats_copy["collection_size"] = meta.get("total_vectors", 0)
                stats_copy["collection_metadata"] = meta
            except Exception:
                stats_copy["collection_size"] = 0
        else:
            stats_copy["collection_size"] = 0

        return stats_copy

    def health_check(self) -> Dict[str, Any]:
        """
        Verifies system health status by delegating diagnostics to the active vector store.

        Returns:
            Dict[str, Any]: Diagnostic report dictionary.
        """
        logger.info("Executing VectorRepository health check...")
        return self.vector_store.health_check()
