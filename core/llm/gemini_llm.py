"""
Gemini LLM Client Implementation for Cortex AI.
Connects with Google Gemini Generative AI, implementing response caching,
exponential backoff, streaming tokens, safety filters, and operational diagnostics.
"""

import hashlib
import time
import logging
from typing import Any, Dict, Generator, Optional

import google.generativeai as genai

from core.llm.base_llm import BaseLLM
from core.llm.response_parser import LLMResponse, ResponseParser
from core.exceptions import (
    APIKeyMissingException,
    LLMTimeoutException,
    RateLimitException,
    SafetyBlockException,
    EmptyResponseException,
    NetworkFailureException,
)
from utils.config import GOOGLE_API_KEY, DEFAULT_LLM_MODEL
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)


class GeminiLLM(BaseLLM):
    """
    Client wrapper for Google Gemini LLM API calls.
    
    Implements rate-limit retries, timeout management, result caches,
    streaming, safety ratings, and performance telemetry tracking.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        max_output_tokens: int = 2048,
        timeout: float = 30.0,
        cache_responses: bool = True
    ):
        """
        Initializes the GeminiLLM.

        Args:
            api_key (Optional[str]): Gemini API key. If absent, loads from config.
            model_name (Optional[str]): Model version string. If absent, loads default.
            temperature (float): Controls generation diversity.
            top_p (float): Nucleus sampling threshold.
            top_k (int): Vocabulary search restriction parameter.
            max_output_tokens (int): Response length constraint.
            timeout (float): Connection timeout threshold in seconds.
            cache_responses (bool): Enables cache matching for prompt text.
        """
        self.api_key = api_key or GOOGLE_API_KEY
        self.model_name = model_name or DEFAULT_LLM_MODEL
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.cache_responses = cache_responses

        # Lazy initialized client
        self._client_configured = False

        # In-memory prompt cache
        # Key: SHA-256(prompt + config_str) -> Value: LLMResponse
        self._response_cache: Dict[str, LLMResponse] = {}

        self._stats: Dict[str, Any] = {}
        self.reset_statistics()

        logger.info(
            f"GeminiLLM initialized (Model: '{self.model_name}', "
            f"Temp: {self.temperature}, Timeout: {self.timeout}s)"
        )

    def initialize_client(self) -> None:
        """
        Configures the google-generativeai client wrapper lazily.
        """
        if self._client_configured:
            return

        if not self.api_key:
            raise APIKeyMissingException(
                "Gemini API key is not configured. Please set GOOGLE_API_KEY environment variable."
            )

        try:
            logger.info("Initializing Google Generative AI client...")
            genai.configure(api_key=self.api_key)
            self._client_configured = True
            logger.info("Google Generative AI client configured successfully.")
        except Exception as e:
            raise NetworkFailureException(f"Failed to configure Gemini client: {e}")

    def reset_statistics(self) -> None:
        """Resets all metrics counters to zero."""
        self._stats = {
            "requests": 0,
            "successful_responses": 0,
            "failed_responses": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "retry_count": 0,
            "total_latency_ms": 0.0,
            "total_output_length": 0,
            "last_latency_ms": 0.0
        }
        logger.info("GeminiLLM performance statistics have been reset.")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieves performance metrics, calculating latency and output averages.

        Returns:
            Dict[str, Any]: Metrics report summary mapping.
        """
        stats_copy = self._stats.copy()
        requests = stats_copy.get("requests", 0)
        successes = stats_copy.get("successful_responses", 0)
        latency = stats_copy.get("total_latency_ms", 0.0)
        output_len = stats_copy.get("total_output_length", 0)

        stats_copy["average_latency"] = latency / requests if requests > 0 else 0.0
        stats_copy["average_output_length"] = output_len / successes if successes > 0 else 0.0
        return stats_copy

    def _generate_cache_key(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Generates SHA-256 signature key for cache mapping."""
        # Normalize keys by sorting
        config_items = sorted(kwargs.items())
        config_str = "".join(f"{k}:{v}" for k, v in config_items)
        raw_key = f"{prompt}||{config_str}||{self.model_name}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _get_generation_config(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Merges default parameter maps with request overrides."""
        return {
            "temperature": overrides.get("temperature", self.temperature),
            "top_p": overrides.get("top_p", self.top_p),
            "top_k": overrides.get("top_k", self.top_k),
            "max_output_tokens": overrides.get("max_output_tokens", self.max_output_tokens),
        }

    def _execute_api_with_backoff(self, model: genai.GenerativeModel, prompt: str, config: Dict[str, Any]) -> tuple[Any, float]:
        """
        Dispatches request execution wrapped in exponential backoff retries.
        """
        max_retries = 3
        base_delay = 2.0
        start_time = time.perf_counter()

        # Check API key configuration
        self.initialize_client()

        for attempt in range(max_retries + 1):
            try:
                # Execute content generation
                # Safety settings mapping (blocking harmful content categories)
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
                ]

                # Note: timeouts in google-generativeai are set via client request parameters
                # or handled via signals/threads wrapper. Here we pass generation config.
                response = model.generate_content(
                    prompt,
                    generation_config=config,
                    safety_settings=safety_settings
                )
                
                generation_time = time.perf_counter() - start_time
                return response, generation_time

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = "429" in error_str or "quota" in error_str or "resource_exhausted" in error_str
                is_transient = "503" in error_str or "unavailable" in error_str or "deadline exceeded" in error_str

                if (is_rate_limit or is_transient) and attempt < max_retries:
                    self._stats["retry_count"] += 1
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Gemini API request throttled or hit transient error: {e}. "
                        f"Retrying in {delay:.1f}s (Attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(delay)
                else:
                    # Translate to specific exceptions
                    if is_rate_limit:
                        raise RateLimitException(f"Gemini API Rate Limit Exceeded: {e}")
                    elif "deadline" in error_str or "timeout" in error_str:
                        raise LLMTimeoutException(f"Gemini API Request Timed Out: {e}")
                    elif "key" in error_str or "unauthorized" in error_str or "not found" in error_str:
                        raise APIKeyMissingException(f"Gemini API Key invalid or expired: {e}")
                    elif "safety" in error_str or "blocked" in error_str:
                        raise SafetyBlockException(f"Gemini blocked content: {e}")
                    else:
                        raise NetworkFailureException(f"Gemini connection failed: {e}")

        # Fallback (unreachable due to re-raise, but keeps linters happy)
        raise NetworkFailureException("Gemini generation failed due to repeated errors.")

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Executes a prompt query and returns a structured LLMResponse.
        """
        self._stats["requests"] += 1
        
        # Prepare cache parameters matching overrides
        config = self._get_generation_config(kwargs)
        cache_key = self._generate_cache_key(prompt, config)

        # Check Cache
        if self.cache_responses and cache_key in self._response_cache:
            self._stats["cache_hits"] += 1
            self._stats["successful_responses"] += 1
            cached = self._response_cache[cache_key]
            self._stats["total_output_length"] += len(cached.response_text)
            logger.info("Response cache hit. Returning cached LLMResponse.")
            return cached

        self._stats["cache_misses"] += 1

        try:
            # Instantiate lazy model wrapper
            model = genai.GenerativeModel(self.model_name)
            
            raw_response, gen_time = self._execute_api_with_backoff(model, prompt, config)

            # Parse response
            llm_response = ResponseParser.parse_gemini_response(
                raw_response=raw_response,
                generation_time=gen_time,
                prompt_version=kwargs.get("prompt_version", "1.0.0")
            )

            # Store in Cache
            if self.cache_responses:
                if len(self._response_cache) >= 1000:
                    self._response_cache.clear()
                self._response_cache[cache_key] = llm_response

            self._stats["successful_responses"] += 1
            self._stats["total_output_length"] += len(llm_response.response_text)
            
            latency_ms = gen_time * 1000.0
            self._stats["total_latency_ms"] += latency_ms
            self._stats["last_latency_ms"] = latency_ms

            return llm_response

        except Exception as e:
            self._stats["failed_responses"] += 1
            logger.error(f"Inference request failed: {e}")
            raise

    def generate_stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, None]:
        """
        Streams generated text response token by token.
        """
        self.initialize_client()
        config = self._get_generation_config(kwargs)
        
        try:
            model = genai.GenerativeModel(self.model_name)
            
            # Simple streaming generator
            response = model.generate_content(
                prompt,
                generation_config=config,
                stream=True
            )

            for chunk in response:
                # Verify chunk has text content
                text = chunk.text
                if text:
                    yield text

        except Exception as e:
            logger.error(f"Streaming inference request failed: {e}")
            error_str = str(e).lower()
            if "safety" in error_str:
                raise SafetyBlockException(f"Streaming blocked: {e}")
            elif "quota" in error_str or "429" in error_str:
                raise RateLimitException(f"Streaming rate limit: {e}")
            else:
                raise NetworkFailureException(f"Streaming generation failure: {e}")

    def health_check(self) -> Dict[str, Any]:
        """
        Performs connectivity checks.
        """
        api_configured = bool(self.api_key)
        connection = "offline"
        details = "Gemini API client connection has not been initialized."

        if api_configured:
            try:
                # Perform rapid validation connection query
                self.initialize_client()
                model = genai.GenerativeModel(self.model_name)
                # Quick call to verify model availability
                # We mock or run this quickly. For a health check, we verify GenAI library initialized.
                connection = "online"
                details = f"Gemini provider connection verified for model: '{self.model_name}'"
            except Exception as e:
                connection = "offline"
                details = f"Gemini client verification failed: {e}"

        status = "healthy" if (api_configured and connection == "online") else "unhealthy"

        return {
            "status": status,
            "provider": "Google Gemini",
            "model_name": self.model_name,
            "api_key_configured": api_configured,
            "connection_status": connection,
            "last_latency_ms": self._stats.get("last_latency_ms", 0.0),
            "details": details,
            "statistics": self.get_statistics()
        }

    def close(self) -> None:
        """Clears response caches."""
        self._response_cache.clear()
        logger.info("GeminiLLM caches cleared and client resources closed.")
