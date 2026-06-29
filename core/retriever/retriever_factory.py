"""
Retriever Factory Module for Cortex AI.
Implements the Factory pattern to initialize concrete retrievers dynamically.
"""

from typing import Any

from core.retriever.base_retriever import BaseRetriever
from core.retriever.semantic_retriever import SemanticRetriever


class RetrieverFactory:
    """
    Factory for producing BaseRetriever instances.
    
    Promotes loose coupling by hiding retriever initialization mechanics from business logic.
    """

    @staticmethod
    def get_retriever(retriever_type: str = "semantic", **kwargs: Any) -> BaseRetriever:
        """
        Creates and returns a concrete Retriever instance.

        Args:
            retriever_type (str): Type of retriever (defaults to 'semantic').
            **kwargs (Any): Arguments passed directly to the concrete constructor.

        Returns:
            BaseRetriever: The configured retriever instance.

        Raises:
            ValueError: If the retriever type is unknown or unsupported.
        """
        retriever_key = retriever_type.lower().strip()
        
        if retriever_key == "semantic":
            # SemanticRetriever requires embedding_service and vector_repository
            # Check presence of mandatory DI arguments
            if "embedding_service" not in kwargs:
                raise ValueError("Missing mandatory constructor parameter: 'embedding_service'")
            if "vector_repository" not in kwargs:
                raise ValueError("Missing mandatory constructor parameter: 'vector_repository'")
                
            return SemanticRetriever(**kwargs)
        else:
            raise ValueError(
                f"Unsupported Retriever type: '{retriever_type}'. "
                f"Currently supported options: ['semantic']."
            )
