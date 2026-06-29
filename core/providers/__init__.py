"""
Providers Package for Cortex AI Embedding Service.
Exposes the abstract base class and concrete vendor implementations.
"""

from core.providers.base_embedding_provider import EmbeddingProvider
from core.providers.gemini_embedding_provider import GeminiEmbeddingProvider

__all__ = ["EmbeddingProvider", "GeminiEmbeddingProvider"]
