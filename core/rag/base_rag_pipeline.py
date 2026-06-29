"""
Base RAG Pipeline Interface for Cortex AI.
Defines the abstract BaseRAGPipeline class and the structured RAGResponse dataclass.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RAGResponse:
    """
    Structured data container representing the final RAG pipeline response.
    """
    answer: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    similarity_scores: List[float] = field(default_factory=list)
    prompt: str = ""
    llm_response: Any = None
    latency: float = 0.0
    query: str = ""
    timestamp: str = ""


class BaseRAGPipeline(ABC):
    """
    Abstract Base Class (ABC) defining the orchestration workflow for RAG systems.
    
    Integrates loaders, embeddings, databases, search retrievers, prompts, and LLMs.
    """

    @abstractmethod
    def ask(
        self,
        question: str,
        collection_name: str,
        session_id: Optional[str] = None,
        k: int = 4,
        score_threshold: Optional[float] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> RAGResponse:
        """
        Runs the complete end-to-end question answering pipeline:
        Query Validation -> Retrieval -> Prompt Assembly -> LLM Inference -> Normalization.

        Args:
            question (str): User query.
            collection_name (str): Database collection to search within.
            session_id (Optional[str]): Conversation session identifier.
            k (int): Number of context chunks to retrieve.
            score_threshold (Optional[float]): Filtering threshold.
            metadata_filters (Optional[Dict[str, Any]]): Metadata queries filters.
            **kwargs (Any): Additional options (temperature, streaming options, etc.).

        Returns:
            RAGResponse: Complete packaged RAG answer and tracing metrics.
        """
        pass

    @abstractmethod
    def ingest_document(
        self,
        file_path: str,
        collection_name: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> Dict[str, Any]:
        """
        Loads, chunks, embeds, and indexes a PDF document.

        Args:
            file_path (str): Path to the target PDF file.
            collection_name (str): Vector store collection target.
            chunk_size (int): Size of text chunks.
            chunk_overlap (int): Overlap size between adjacent chunks.

        Returns:
            Dict[str, Any]: Report containing document_id and total chunk counts indexed.
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieves aggregated performance and count metrics across the pipeline.

        Returns:
            Dict[str, Any]: Metrics report summary mapping.
        """
        pass

    @abstractmethod
    def reset_statistics(self) -> None:
        """Resets all metrics counters to zero."""
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Aggregates health diagnostics reports from all pipeline dependencies.

        Returns:
            Dict[str, Any]: Consolidated health check mapping.
        """
        pass
