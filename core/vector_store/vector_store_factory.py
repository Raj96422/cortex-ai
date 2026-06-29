"""
Vector Store Factory Module for Cortex AI.
Implements the Factory pattern to initialize concrete vector stores dynamically.
"""

from typing import Any

from core.vector_store.base_vector_store import BaseVectorStore
from core.vector_store.chroma_vector_store import ChromaVectorStore


class VectorStoreFactory:
    """
    Factory for producing BaseVectorStore instances.
    
    Promotes loose coupling by hiding database instantiation mechanics from business logic.
    """

    @staticmethod
    def get_vector_store(provider_name: str = "chromadb", **kwargs: Any) -> BaseVectorStore:
        """
        Creates and returns a concrete Vector Store wrapper.

        Args:
            provider_name (str): Database provider selector (defaults to 'chromadb').
            **kwargs (Any): Arguments passed directly to the concrete constructor.

        Returns:
            BaseVectorStore: The configured database manager instance.

        Raises:
            ValueError: If the provider name is unknown or unsupported.
        """
        provider_key = provider_name.lower().strip()
        
        if provider_key == "chromadb":
            return ChromaVectorStore(**kwargs)
        else:
            raise ValueError(
                f"Unsupported Vector Store provider: '{provider_name}'. "
                f"Currently supported options: ['chromadb']."
            )
