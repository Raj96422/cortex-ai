"""
Base Vector Store Interface for Cortex AI.
Defines the abstract BaseVectorStore class that concrete database wrappers 
must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from core.embeddings import EmbeddedChunk


class BaseVectorStore(ABC):
    """
    Abstract Base Class (ABC) defining the interface for a Vector Database.
    
    Provides extensibility to support multiple backends (ChromaDB, Pinecone, Weaviate, Qdrant).
    """

    @abstractmethod
    def initialize(self) -> None:
        """
        Initializes the connection to the vector database client.

        Raises:
            ConnectionFailureException: If connection to the database cannot be established.
            StorageCorruptionException: If the underlying storage files are corrupted.
        """
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Performs a diagnostic check on the database instance and checks its health.

        Returns:
            Dict[str, Any]: Diagnostic report containing status, storage, version, latency.
        """
        pass

    @abstractmethod
    def create_collection(self, collection_name: str) -> None:
        """
        Creates a new collection within the vector store.

        Args:
            collection_name (str): Unique identifier for the collection.

        Raises:
            ConnectionFailureException: If the query fails due to connection issues.
        """
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """
        Deletes a collection and all associated records from the vector store.

        Args:
            collection_name (str): Name of the collection to delete.

        Raises:
            CollectionNotFoundException: If the collection does not exist.
        """
        pass

    @abstractmethod
    def add_embeddings(self, collection_name: str, chunks: List[EmbeddedChunk]) -> None:
        """
        Inserts a batch of EmbeddedChunk objects into the specified collection.

        Args:
            collection_name (str): Target collection name.
            chunks (List[EmbeddedChunk]): List of processed text chunks with vectors.

        Raises:
            CollectionNotFoundException: If the collection does not exist.
            DuplicateIdException: If any chunk ID already exists in the collection.
            InvalidVectorException: If vector dimensions or types do not match requirements.
        """
        pass

    @abstractmethod
    def update_embeddings(self, collection_name: str, chunks: List[EmbeddedChunk]) -> None:
        """
        Updates the vectors or metadata for existing chunks in the collection.

        Args:
            collection_name (str): Target collection name.
            chunks (List[EmbeddedChunk]): Chunks containing updated text/embeddings/metadata.

        Raises:
            CollectionNotFoundException: If the collection does not exist.
            InvalidVectorException: If updated vector properties are invalid.
        """
        pass

    @abstractmethod
    def delete_embeddings(self, collection_name: str, chunk_ids: List[str]) -> None:
        """
        Removes specific text chunks from the vector database by their IDs.

        Args:
            collection_name (str): Target collection name.
            chunk_ids (List[str]): List of unique identifiers of the chunks to delete.

        Raises:
            CollectionNotFoundException: If the collection does not exist.
        """
        pass

    @abstractmethod
    def similarity_search(
        self,
        collection_name: str,
        query_vector: List[float],
        k: int = 4,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-K most semantically similar chunks to the query vector.

        Args:
            collection_name (str): Collection to search within.
            query_vector (List[float]): Vector representation of the user query.
            k (int): Number of nearest neighbors to retrieve.
            score_threshold (Optional[float]): Optional threshold to filter low-confidence hits.

        Returns:
            List[Dict[str, Any]]: List of dictionary results representing matches, 
                                  ordered by score (highest similarity first).
                                  Each dict contains 'chunk_id', 'text', 'metadata', and 'score'.

        Raises:
            CollectionNotFoundException: If the collection does not exist.
            InvalidVectorException: If the query vector format or dimension is invalid.
        """
        pass

    @abstractmethod
    def filter_search(
        self,
        collection_name: str,
        query_vector: List[float],
        metadata_filters: Dict[str, Any],
        k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Searches nearest neighbors with metadata filters applied.

        Args:
            collection_name (str): Target collection name.
            query_vector (List[float]): Query vector coords.
            metadata_filters (Dict[str, Any]): Filter expressions.
            k (int): Number of nearest neighbors to retrieve.

        Returns:
            List[Dict[str, Any]]: Filtered similarity search matches.
        """
        pass

    @abstractmethod
    def collection_statistics(self, collection_name: str) -> Dict[str, Any]:
        """
        Retrieves internal stats for the collection, such as count of entries.

        Args:
            collection_name (str): Target collection name.

        Returns:
            Dict[str, Any]: Dictionary containing counts and configuration stats.

        Raises:
            CollectionNotFoundException: If the collection does not exist.
        """
        pass

    @abstractmethod
    def modify_collection_metadata(self, collection_name: str, metadata: Dict[str, Any]) -> None:
        """
        Updates the key-value metadata attached directly to the collection.

        Args:
            collection_name (str): Target collection.
            metadata (Dict[str, Any]): Dictionary of updated properties.

        Raises:
            CollectionNotFoundException: If the collection does not exist.
        """
        pass

    @abstractmethod
    def get_collection_metadata(self, collection_name: str) -> Dict[str, Any]:
        """
        Retrieves the key-value metadata dictionary attached directly to the collection.

        Args:
            collection_name (str): Target collection.

        Returns:
            Dict[str, Any]: Collection metadata dictionary.

        Raises:
            CollectionNotFoundException: If the collection does not exist.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Closes any active database client connections and releases resources."""
        pass
