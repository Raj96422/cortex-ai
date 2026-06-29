"""
Base Retriever Interface for Cortex AI.
Defines the abstract BaseRetriever class that concrete search rankers 
must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseRetriever(ABC):
    """
    Abstract Base Class (ABC) defining the interface for document retrieval.
    
    Provides extensibility to support multiple retrieval strategies (Semantic, Hybrid, MMR).
    """

    @abstractmethod
    def initialize(self) -> None:
        """
        Initializes the retriever dependencies (e.g. models or connections).
        """
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        collection_name: str,
        k: int = 4,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the top-K relevant chunks for the given query text.

        Args:
            query (str): The search query.
            collection_name (str): The database collection to search within.
            k (int): Number of chunks to retrieve.
            score_threshold (Optional[float]): Optional threshold to filter low confidence hits.

        Returns:
            List[Dict[str, Any]]: List of retrieved matches containing content and metadata.

        Raises:
            EmptyQueryException: If query is empty.
            InvalidQueryException: If query violates constraints.
            CollectionNotFoundException: If the target collection is missing.
            RetrievalFailureException: If retrieval execution fails.
        """
        pass

    @abstractmethod
    def retrieve_with_metadata(
        self,
        query: str,
        collection_name: str,
        metadata_filters: Dict[str, Any],
        k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Retrieves documents matching the query vector and metadata filter conditions.

        Args:
            query (str): Search query.
            collection_name (str): Target collection.
            metadata_filters (Dict[str, Any]): Filter matches.
            k (int): Top-K search limit.

        Returns:
            List[Dict[str, Any]]: List of matching results.
        """
        pass

    @abstractmethod
    def retrieve_by_document(
        self,
        query: str,
        collection_name: str,
        document_id: str,
        k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Retrieves document chunks matching query, scoped strictly to a single document.

        Args:
            query (str): Search query.
            collection_name (str): Target collection.
            document_id (str): Unique document identifier.
            k (int): Top-K limits.

        Returns:
            List[Dict[str, Any]]: Scoped document matches.
        """
        pass

    @abstractmethod
    def retrieve_similar_chunks(
        self,
        chunk_id: str,
        collection_name: str,
        k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Retrieves other chunks close to the specified chunk ID (recommendations / context).

        Args:
            chunk_id (str): Target chunk key.
            collection_name (str): Target collection.
            k (int): Retrieve top-K.

        Returns:
            List[Dict[str, Any]]: Semantically similar chunks.
        """
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Runs a diagnostic status check on the retriever and its dependencies.

        Returns:
            Dict[str, Any]: Status diagnostics mapping.
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieves current retrieval operations statistics.

        Returns:
            Dict[str, Any]: Metrics report mapping.
        """
        pass

    @abstractmethod
    def reset_statistics(self) -> None:
        """Resets all metrics counters to zero."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Closes connection clients and releases system resources."""
        pass
