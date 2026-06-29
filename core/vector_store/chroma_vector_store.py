"""
ChromaDB Vector Store Implementation for Cortex AI.
Provides persistent storage, management, similarity search, and diagnostics for 
EmbeddedChunk objects using ChromaDB.
"""

import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chromadb
from chromadb.config import Settings
from chromadb.errors import ChromaError

from core.embeddings import EmbeddedChunk
from core.vector_store.base_vector_store import BaseVectorStore
from core.exceptions import (
    CollectionNotFoundException,
    ConnectionFailureException,
    DuplicateIdException,
    EmptyCollectionException,
    InvalidVectorException,
    StorageCorruptionException,
)
from utils.config import CHROMA_DB_DIR
from utils.helpers import format_file_size
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)

# Expected dimensions matching Gemini text-embedding-004
EXPECTED_DIMENSION: int = 768


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB implementation of BaseVectorStore.
    
    Provides persistent local vector database storage with lazy initialization,
    caching, search thresholds, and diagnostic reporting.
    """

    def __init__(self, persist_dir: Optional[Union[str, Path]] = None):
        """
        Initializes the ChromaVectorStore.

        Args:
            persist_dir (Optional[Union[str, Path]]): Path to local persistence folder. 
                                                     Defaults to CHROMA_DB_DIR.
        """
        self.persist_dir = Path(persist_dir) if persist_dir is not None else CHROMA_DB_DIR
        self._client: Optional[chromadb.PersistentClient] = None
        self._last_successful_op_timestamp: Optional[str] = None
        
        logger.info(f"ChromaVectorStore configured at: {self.persist_dir.resolve()}")

    def _get_client(self) -> chromadb.PersistentClient:
        """
        Accesses or lazily instantiates the persistent ChromaDB client.

        Returns:
            chromadb.PersistentClient: Active database client.
        """
        if self._client is None:
            self.initialize()
        assert self._client is not None
        return self._client

    def _update_last_op(self) -> None:
        """Updates the timestamp of the last successful database action."""
        from datetime import datetime
        self._last_successful_op_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    def initialize(self) -> None:
        """
        Initializes the persistent client connection to local storage.

        Raises:
            ConnectionFailureException: If client instantiation fails.
            StorageCorruptionException: If SQLite/index files are corrupted.
        """
        try:
            logger.info("Initializing persistent ChromaDB client...")
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize persistent client (connection pool reuse is handled internally by client)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir.resolve()),
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info("ChromaDB client connection established successfully.")
            self._update_last_op()
        except sqlite3.DatabaseError as sqlite_err: # type: ignore
            # Handle SQLite corruption errors
            raise StorageCorruptionException(
                f"ChromaDB storage file corruption detected at '{self.persist_dir}': {sqlite_err}"
            )
        except Exception as e:
            if "sqlite" in str(e).lower() or "corrupt" in str(e).lower():
                raise StorageCorruptionException(
                    f"Local database files at '{self.persist_dir}' appear corrupted: {e}"
                )
            raise ConnectionFailureException(f"Failed to connect to ChromaDB: {e}")

    def _get_collection_object(self, collection_name: str) -> Any:
        """
        Retrieves the collection object or raises CollectionNotFoundException if missing.

        Args:
            collection_name (str): Collection name to fetch.

        Returns:
            Collection: ChromaDB collection.
        """
        client = self._get_client()
        try:
            # list_collections() returns collection objects or names depending on version
            # To be version compatible, we fetch or raise
            return client.get_collection(name=collection_name)
        except ValueError as val_err:
            raise CollectionNotFoundException(
                f"Collection '{collection_name}' does not exist: {val_err}"
            )
        except Exception as e:
            # Some versions throw general errors for missing collections
            raise CollectionNotFoundException(f"Could not retrieve collection '{collection_name}': {e}")

    def create_collection(self, collection_name: str) -> None:
        """
        Creates a new collection using cosine distance space.

        Args:
            collection_name (str): Unique collection name.
        """
        client = self._get_client()
        try:
            logger.info(f"Creating ChromaDB collection: '{collection_name}' (Distance Space: Cosine)")
            client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Similarity score maps to 1 - distance
            )
        except Exception as e:
            raise ConnectionFailureException(f"Failed to create collection '{collection_name}': {e}")

    def delete_collection(self, collection_name: str) -> None:
        """
        Deletes a collection and all associated records.

        Args:
            collection_name (str): Name of the collection.
        """
        client = self._get_client()
        try:
            logger.info(f"Deleting collection: '{collection_name}'")
            client.delete_collection(name=collection_name)
        except (ValueError, Exception) as e:
            err_msg = str(e).lower()
            if "not exist" in err_msg or "not found" in err_msg or isinstance(e, ValueError):
                raise CollectionNotFoundException(f"Cannot delete collection '{collection_name}': Collection not found.")
            raise ConnectionFailureException(f"Failed to delete collection '{collection_name}': {e}")

    def add_embeddings(self, collection_name: str, chunks: List[EmbeddedChunk]) -> None:
        """
        Inserts a batch of EmbeddedChunk objects. Performs duplicate detection and size validation.

        Args:
            collection_name (str): Target collection name.
            chunks (List[EmbeddedChunk]): Chunks to insert.
        """
        if not chunks:
            logger.warning("No chunks provided for insertion. Skipping add_embeddings.")
            return

        collection = self._get_collection_object(collection_name)

        ids: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []
        documents: List[str] = []

        # 1. Validate inputs and structures
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, EmbeddedChunk):
                raise ValueError(f"Batch item at index {idx} must be an EmbeddedChunk.")
            
            # Dimension Check
            if len(chunk.vector) != EXPECTED_DIMENSION:
                raise InvalidVectorException(
                    f"Vector dimension mismatch at index {idx}. "
                    f"Expected {EXPECTED_DIMENSION}, got {len(chunk.vector)}."
                )

            ids.append(chunk.chunk_id)
            embeddings.append(chunk.vector)
            metadatas.append(chunk.metadata)
            documents.append(chunk.text)

        # 2. Check for duplicate IDs in the incoming batch
        if len(ids) != len(set(ids)):
            raise DuplicateIdException("Duplicate chunk IDs found inside the provided insertion batch.")

        # 3. Check for existing IDs in database to prevent overwriting
        try:
            existing = collection.get(ids=ids)
            if existing and existing.get("ids"):
                raise DuplicateIdException(
                    f"Conflict detected: Chunk IDs {existing['ids']} already exist in collection '{collection_name}'."
                )
        except DuplicateIdException:
            raise
        except Exception as e:
            # If get raises, log and proceed with insertion attempt
            logger.debug(f"Precheck get call returned: {e}")

        # 4. Batch Insertion
        try:
            logger.info(f"Inserting batch of {len(chunks)} vectors into collection '{collection_name}'...")
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            logger.info(f"Successfully inserted {len(chunks)} vector(s) into database.")
        except Exception as e:
            raise ConnectionFailureException(f"ChromaDB insert failed: {e}")

    def update_embeddings(self, collection_name: str, chunks: List[EmbeddedChunk]) -> None:
        """
        Updates text contents, vectors, or metadata for existing chunks.

        Args:
            collection_name (str): Target collection.
            chunks (List[EmbeddedChunk]): Updated chunks.
        """
        if not chunks:
            return

        collection = self._get_collection_object(collection_name)

        ids = [c.chunk_id for c in chunks]
        embeddings = [c.vector for c in chunks]
        metadatas = [c.metadata for c in chunks]
        documents = [c.text for c in chunks]

        # Dimension Check
        for idx, vec in enumerate(embeddings):
            if len(vec) != EXPECTED_DIMENSION:
                raise InvalidVectorException(
                    f"Vector dimension mismatch at index {idx} in update. "
                    f"Expected {EXPECTED_DIMENSION}, got {len(vec)}."
                )

        try:
            logger.info(f"Updating batch of {len(chunks)} records in collection '{collection_name}'...")
            collection.update(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            logger.info("Successfully updated vector database records.")
        except Exception as e:
            raise ConnectionFailureException(f"ChromaDB update failed: {e}")

    def delete_embeddings(self, collection_name: str, chunk_ids: List[str]) -> None:
        """
        Deletes vector records from collection by their IDs.

        Args:
            collection_name (str): Target collection.
            chunk_ids (List[str]): Unique IDs to delete.
        """
        if not chunk_ids:
            return

        collection = self._get_collection_object(collection_name)

        try:
            logger.info(f"Deleting {len(chunk_ids)} records from collection '{collection_name}'...")
            collection.delete(ids=chunk_ids)
            logger.info("Successfully deleted specified vector database records.")
        except Exception as e:
            raise ConnectionFailureException(f"ChromaDB delete failed: {e}")

    def similarity_search(
        self,
        collection_name: str,
        query_vector: List[float],
        k: int = 4,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-K matches using cosine similarity, with score threshold filtering.

        Args:
            collection_name (str): Collection name.
            query_vector (List[float]): User query vector.
            k (int): Retrieve top K.
            score_threshold (Optional[float]): Minimum similarity confidence.

        Returns:
            List[Dict[str, Any]]: Unpacked matches in descending score order.
        """
        if len(query_vector) != EXPECTED_DIMENSION:
            raise InvalidVectorException(
                f"Query vector dimension mismatch. Expected {EXPECTED_DIMENSION}, got {len(query_vector)}."
            )

        collection = self._get_collection_object(collection_name)

        try:
            logger.info(f"Querying nearest neighbors (K={k}) in collection '{collection_name}'...")
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=k
            )
            return self._unpack_query_results(results, score_threshold)
        except Exception as e:
            raise ConnectionFailureException(f"Similarity search failed: {e}")

    def filter_search(
        self,
        collection_name: str,
        query_vector: List[float],
        metadata_filters: Dict[str, Any],
        k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Searches nearest neighbors applying metadata filters (where conditions).

        Args:
            collection_name (str): Target collection.
            query_vector (List[float]): Query vector coords.
            metadata_filters (Dict[str, Any]): Filter expressions (e.g. {"source": "doc.pdf"}).
            k (int): Top-K results.

        Returns:
            List[Dict[str, Any]]: Search results.
        """
        if len(query_vector) != EXPECTED_DIMENSION:
            raise InvalidVectorException(
                f"Query vector dimension mismatch. Expected {EXPECTED_DIMENSION}, got {len(query_vector)}."
            )

        collection = self._get_collection_object(collection_name)

        try:
            logger.info(f"Querying nearest neighbors (K={k}) with filters: {metadata_filters}...")
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=k,
                where=metadata_filters
            )
            return self._unpack_query_results(results, None)
        except Exception as e:
            raise ConnectionFailureException(f"Filtered search failed: {e}")

    def _unpack_query_results(
        self,
        results: Dict[str, Any],
        score_threshold: Optional[float]
    ) -> List[Dict[str, Any]]:
        """
        Helper to unpack ChromaDB query outputs and compute Cosine similarity scores.

        Args:
            results (Dict[str, Any]): Direct output from collection.query.
            score_threshold (Optional[float]): Minimum score bounds.

        Returns:
            List[Dict[str, Any]]: Results with 'chunk_id', 'text', 'metadata', and 'score' keys.
        """
        search_results: List[Dict[str, Any]] = []
        if not results or not results.get("ids") or len(results["ids"]) == 0:
            return search_results

        ids = results["ids"][0]
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
        documents = results["documents"][0] if results.get("documents") else [""] * len(ids)

        for idx in range(len(ids)):
            distance = distances[idx]
            # ChromaDB cosine space outputs Cosine Distance (1 - similarity).
            # Similarity = 1 - distance.
            similarity = 1.0 - distance

            if score_threshold is not None and similarity < score_threshold:
                logger.debug(f"Skipped match '{ids[idx]}' with similarity {similarity:.4f} < threshold {score_threshold}")
                continue

            search_results.append({
                "chunk_id": ids[idx],
                "text": documents[idx],
                "metadata": metadatas[idx],
                "score": similarity
            })

        # Sort results by similarity score descending (highest similarity first)
        search_results.sort(key=lambda r: r["score"], reverse=True)
        return search_results

    def collection_statistics(self, collection_name: str) -> Dict[str, Any]:
        """
        Retrieves statistics for the specified collection.

        Args:
            collection_name (str): Target collection.

        Returns:
            Dict[str, Any]: Stats summary including document count.
        """
        collection = self._get_collection_object(collection_name)
        try:
            count = collection.count()
            return {
                "collection_name": collection_name,
                "document_count": count,
                "is_empty": count == 0
            }
        except Exception as e:
            raise ConnectionFailureException(f"Failed to retrieve collection stats: {e}")

    def health_check(self) -> Dict[str, Any]:
        """
        Runs diagnostics assessing SQLite status, latency, folder metrics, and collection versions.

        Returns:
            Dict[str, Any]: Detailed status report mapping.
        """
        import shutil
        status = "healthy"
        details = "ChromaDB connection is active."
        version = "unknown"
        latency_ms = 0.0
        collection_count = 0
        total_vectors = 0
        storage_size_str = "0 Bytes"
        collection_versions = {}
        available_disk_space = "unknown"
        collections_list = []

        try:
            # 1. Connection check
            client = self._get_client()
            version = client.get_version()
            details = f"ChromaDB connection is active. Version: {version}"
            
            # 2. Latency check
            start = time.perf_counter()
            collections = client.list_collections()
            latency_ms = (time.perf_counter() - start) * 1000.0
            
            # 3. Collection details
            collection_count = len(collections)
            for col in collections:
                total_vectors += col.count()
                collections_list.append(col.name)
                # Read collection_version from metadata if available
                meta = col.metadata or {}
                collection_versions[col.name] = meta.get("collection_version", "unknown")

            # 4. Storage size details
            if self.persist_dir.exists():
                total_size = sum(p.stat().st_size for p in self.persist_dir.glob('**/*') if p.is_file())
                storage_size_str = format_file_size(total_size)
                
                # Available space
                total_b, used_b, free_b = shutil.disk_usage(self.persist_dir)
                available_disk_space = format_file_size(free_b)

            self._update_last_op()

        except Exception as e:
            status = "unhealthy"
            details = f"Diagnostics execution failed: {e}"

        return {
            "status": status,
            "provider": "ChromaDB",
            "database_path": str(self.persist_dir.resolve()),
            "version": version,
            "latency_ms": latency_ms,
            "collection_count": collection_count,
            "total_vectors": total_vectors,
            "storage_size": storage_size_str,
            "available_disk_space": available_disk_space,
            "collection_status": {"collections": collections_list},
            "collection_versions": collection_versions,
            "initialization_state": "initialized" if self._client is not None else "uninitialized",
            "last_successful_operation": self._last_successful_op_timestamp or "none",
            "details": details
        }

    def modify_collection_metadata(self, collection_name: str, metadata: Dict[str, Any]) -> None:
        """
        Updates the metadata dict attached directly to the collection.
        """
        collection = self._get_collection_object(collection_name)
        try:
            logger.info(f"Modifying collection metadata for '{collection_name}': {metadata}")
            current_metadata = collection.metadata or {}
            combined = {**current_metadata, **metadata}
            filtered = {k: v for k, v in combined.items() if not k.startswith("hnsw:")}
            collection.modify(metadata=filtered)
            self._update_last_op()
        except Exception as e:
            raise ConnectionFailureException(f"Failed to modify collection metadata: {e}")

    def get_collection_metadata(self, collection_name: str) -> Dict[str, Any]:
        """
        Retrieves the metadata dictionary of the collection.
        """
        collection = self._get_collection_object(collection_name)
        try:
            self._update_last_op()
            return collection.metadata or {}
        except Exception as e:
            raise ConnectionFailureException(f"Failed to retrieve collection metadata: {e}")

    def close(self) -> None:
        """Resets active client to trigger garbage collection on references."""
        # ChromaDB persistent client disconnect is handled via garbage collection
        self._client = None
        logger.info("ChromaDB vector store connection closed.")
