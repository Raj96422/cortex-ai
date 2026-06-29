"""
Base LLM Interface for Cortex AI.
Defines the abstract BaseLLM class that concrete language model clients 
must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Optional

from core.llm.response_parser import LLMResponse


class BaseLLM(ABC):
    """
    Abstract Base Class (ABC) defining the interface for language model inference.
    
    Provides extensibility to support multiple provider integrations (Gemini, Claude, OpenAI).
    """

    @abstractmethod
    def initialize_client(self) -> None:
        """
        Initializes the model API client connection.
        """
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Executes a prompt query and returns a structured LLMResponse.

        Args:
            prompt (str): The compiled input prompt text.
            **kwargs (Any): Inference overrides (temperature, top_p, top_k, safety).

        Returns:
            LLMResponse: Structured token estimates, citations, and output text response.

        Raises:
            APIKeyMissingException: If API credential is empty.
            LLMTimeoutException: If inference request exceeds timeout limit.
            RateLimitException: If throttled by API limits.
            SafetyBlockException: If response content was blocked.
            EmptyResponseException: If response is null.
            NetworkFailureException: For generic request/network failures.
        """
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, None]:
        """
        Streams generated text response token by token.

        Args:
            prompt (str): The compiled input prompt text.
            **kwargs (Any): Inference overrides.

        Yields:
            str: Next chunk of output text tokens.
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieves performance metrics and statistics counters.

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
        Runs diagnostic checks on connection and client states.

        Returns:
            Dict[str, Any]: Diagnostic mapping details.
        """
        pass
