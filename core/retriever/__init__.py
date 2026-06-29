"""
Retriever Package for Cortex AI.
Exposes the abstract base retriever interface, SemanticRetriever, and factory.
"""

from core.retriever.base_retriever import BaseRetriever
from core.retriever.semantic_retriever import SemanticRetriever
from core.retriever.retriever_factory import RetrieverFactory

__all__ = ["BaseRetriever", "SemanticRetriever", "RetrieverFactory"]
