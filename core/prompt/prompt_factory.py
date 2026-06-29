"""
Prompt Factory Module for Cortex AI.
Implements the Factory pattern to initialize concrete prompt builders dynamically.
"""

from typing import Any

from core.prompt.base_prompt_builder import BasePromptBuilder
from core.prompt.rag_prompt_builder import RAGPromptBuilder


class PromptFactory:
    """
    Factory for producing BasePromptBuilder instances.
    
    Promotes loose coupling by hiding builder initialization mechanics from business logic.
    """

    @staticmethod
    def get_prompt_builder(builder_type: str = "rag", **kwargs: Any) -> BasePromptBuilder:
        """
        Creates and returns a concrete Prompt Builder.

        Args:
            builder_type (str): Type of builder (defaults to 'rag').
            **kwargs (Any): Arguments passed directly to the concrete constructor.

        Returns:
            BasePromptBuilder: The configured prompt builder instance.

        Raises:
            ValueError: If the builder type is unknown or unsupported.
        """
        builder_key = builder_type.lower().strip()
        
        if builder_key == "rag":
            return RAGPromptBuilder(**kwargs)
        else:
            raise ValueError(
                f"Unsupported Prompt Builder type: '{builder_type}'. "
                f"Currently supported options: ['rag']."
            )
