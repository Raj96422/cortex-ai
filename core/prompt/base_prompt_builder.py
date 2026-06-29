"""
Base Prompt Builder Interface for Cortex AI.
Defines the abstract BasePromptBuilder class that concrete prompt generation engines 
must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BasePromptBuilder(ABC):
    """
    Abstract Base Class (ABC) defining the interface for LLM prompt assembly.
    
    Provides extensibility to support multiple prompt types (RAG, Summarization, QA, Chat).
    """

    @abstractmethod
    def build_prompt(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        template_name: str = "qa",
        **kwargs: Any
    ) -> str:
        """
        Constructs a complete, formatted prompt for the LLM.

        Args:
            question (str): User question.
            context_chunks (List[Dict[str, Any]]): Retrieved relevant context chunks.
            conversation_history (Optional[List[Dict[str, str]]]): Optional list of past messages,
                                                                   e.g., [{'role': 'user', 'content': '...'}].
            template_name (str): Identifier of the template to use (e.g. 'qa', 'summary').
            **kwargs (Any): Additional properties for formatting templates (e.g. prompt version overrides).

        Returns:
            str: The fully assembled prompt text.

        Raises:
            EmptyQueryException: If query is empty.
            PromptTooLargeException: If final prompt exceeds budgets.
            TemplateNotFoundException: If the requested template doesn't exist.
            InvalidTemplateException: If formatting parameters fail constraints.
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieves current prompt generation operations statistics.

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
        Performs diagnostic checks on templates, configurations, and builder versions.

        Returns:
            Dict[str, Any]: Structured status diagnostics report.
        """
        pass
