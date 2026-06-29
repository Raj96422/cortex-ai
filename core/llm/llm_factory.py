"""
LLM Factory Module for Cortex AI.
Implements the Factory pattern to initialize concrete LLM wrappers dynamically.
"""

from typing import Any

from core.llm.base_llm import BaseLLM
from core.llm.gemini_llm import GeminiLLM


class LLMFactory:
    """
    Factory for producing BaseLLM instances.
    
    Promotes loose coupling by hiding API initialization details from RAG orchestrators.
    """

    @staticmethod
    def get_llm_service(provider: str = "gemini", **kwargs: Any) -> BaseLLM:
        """
        Creates and returns a concrete LLM Service instance.

        Args:
            provider (str): Name of LLM API vendor (defaults to 'gemini').
            **kwargs (Any): Arguments passed directly to the concrete constructor.

        Returns:
            BaseLLM: The configured LLM client instance.

        Raises:
            ValueError: If the provider is unknown or unsupported.
        """
        provider_key = provider.lower().strip()
        
        if provider_key == "gemini":
            return GeminiLLM(**kwargs)
        else:
            raise ValueError(
                f"Unsupported LLM provider: '{provider}'. "
                f"Currently supported options: ['gemini']."
            )
