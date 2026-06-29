"""
Unit and Integration Tests for Cortex AI LLM Service.
Tests initialization, normal content generation, streaming tokens, response parser,
exponential backoff, response caching, and diagnostics health reports.
"""

import sys
import unittest
import time
from unittest.mock import MagicMock, patch
from pathlib import Path
from typing import Any, Dict, Generator, List

# Ensure workspace root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm.base_llm import BaseLLM
from core.llm.gemini_llm import GeminiLLM
from core.llm.response_parser import LLMResponse, ResponseParser
from core.llm.llm_factory import LLMFactory
from core.exceptions import (
    APIKeyMissingException,
    LLMTimeoutException,
    RateLimitException,
    SafetyBlockException,
    EmptyResponseException,
)


class TestResponseParser(unittest.TestCase):
    """Test suite for ResponseParser citation extraction and response normalizer."""

    def test_citation_parsing(self):
        """Test regex extraction of document citations from generated text."""
        # Standard citation
        text = "Deep learning uses backpropagation [Source: deep_learning.pdf, Page: 10] to adjust weights."
        citations = ResponseParser.parse_citations(text)
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["source"], "deep_learning.pdf")
        self.assertEqual(citations[0]["page"], "10")

        # Formatting variance (Page without colon, spaces)
        text_var = "RAG expands model context [Source: rag_paper.pdf, Page 2] through databases."
        citations_var = ResponseParser.parse_citations(text_var)
        self.assertEqual(len(citations_var), 1)
        self.assertEqual(citations_var[0]["source"], "rag_paper.pdf")
        self.assertEqual(citations_var[0]["page"], "2")

        # Multiple citations and duplicate filter checks
        text_multi = (
            "We check citations. "
            "Point A [Source: doc_a.pdf, Page 1] and Point B [Source: doc_b.pdf, Page 5]. "
            "Repeat Point A [Source: doc_a.pdf, Page 1]."
        )
        citations_multi = ResponseParser.parse_citations(text_multi)
        self.assertEqual(len(citations_multi), 2)
        self.assertEqual(citations_multi[0]["source"], "doc_a.pdf")
        self.assertEqual(citations_multi[1]["source"], "doc_b.pdf")

    def test_gemini_parser_empty_response(self):
        """Test parser validates and raises exceptions for empty responses."""
        with self.assertRaises(EmptyResponseException):
            ResponseParser.parse_gemini_response(None, 0.0)

        # Mock mock response object with empty text
        mock_response = MagicMock()
        mock_response.text = ""
        mock_response.prompt_feedback = None
        mock_response.candidates = []

        with self.assertRaises(EmptyResponseException):
            ResponseParser.parse_gemini_response(mock_response, 0.0)

    def test_gemini_parser_safety_blocks(self):
        """Test parser catches prompt-level or candidate-level safety blocks."""
        # 1. Prompt level feedback blocks
        mock_block = MagicMock()
        mock_block.prompt_feedback.block_reason = 1  # Blocked
        with self.assertRaises(SafetyBlockException):
            ResponseParser.parse_gemini_response(mock_block, 0.0)

        # 2. Candidate level finish reason safety blocks
        mock_cand_block = MagicMock()
        mock_cand_block.prompt_feedback = None
        
        mock_cand = MagicMock()
        mock_cand.finish_reason = "SAFETY"
        mock_cand_block.candidates = [mock_cand]

        with self.assertRaises(SafetyBlockException):
            ResponseParser.parse_gemini_response(mock_cand_block, 0.0)


class TestGeminiLLM(unittest.TestCase):
    """Test suite for GeminiLLM inference engine client."""

    def setUp(self):
        """Sets up GeminiLLM with mocked client configs."""
        self.llm = GeminiLLM(
            api_key="mock_key_12345",
            model_name="models/gemini-1.5-flash",
            temperature=0.2,
            timeout=10.0,
            cache_responses=True
        )

        # Mock standard API response payload
        self.mock_api_response = MagicMock()
        self.mock_api_response.text = "In backpropagation, models adjust weights [Source: ml.pdf, Page 5]."
        self.mock_api_response.model_version = "models/gemini-1.5-flash"
        self.mock_api_response.prompt_feedback = None
        self.mock_api_response.candidates = []

    def test_factory_creation(self):
        """Test that the LLMFactory constructs model clients correctly."""
        client = LLMFactory.get_llm_service("gemini", api_key="key")
        self.assertIsInstance(client, GeminiLLM)

        with self.assertRaises(ValueError):
            LLMFactory.get_llm_service("openai")

    @patch("google.generativeai.configure")
    def test_lazy_initialization(self, mock_configure):
        """Test that the client initialization runs lazily."""
        # Not configured yet
        self.assertFalse(self.llm._client_configured)

        # Trigger client configuration
        self.llm.initialize_client()
        self.assertTrue(self.llm._client_configured)
        mock_configure.assert_called_once_with(api_key="mock_key_12345")

        # Missing key raises APIKeyMissingException
        broken_llm = GeminiLLM(api_key="")
        with self.assertRaises(APIKeyMissingException):
            broken_llm.initialize_client()

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_successful_generation(self, mock_configure, mock_model_class):
        """Test prompt query constructs structured response container."""
        mock_model = MagicMock()
        mock_model.generate_content.return_value = self.mock_api_response
        mock_model_class.return_value = mock_model

        response = self.llm.generate("How do neural networks learn?")

        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.provider, "Google Gemini")
        self.assertEqual(response.model_name, "models/gemini-1.5-flash")
        self.assertIn("backpropagation", response.response_text)
        self.assertEqual(len(response.citations), 1)
        self.assertEqual(response.citations[0]["source"], "ml.pdf")

        # Check metrics updated
        stats = self.llm.get_statistics()
        self.assertEqual(stats["requests"], 1)
        self.assertEqual(stats["successful_responses"], 1)

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_response_caching(self, mock_configure, mock_model_class):
        """Test response caching matches identical prompt text inputs."""
        mock_model = MagicMock()
        mock_model.generate_content.return_value = self.mock_api_response
        mock_model_class.return_value = mock_model

        # Query 1 (Cache Miss)
        self.llm.generate("Prompt Text")
        stats = self.llm.get_statistics()
        self.assertEqual(stats["cache_misses"], 1)
        self.assertEqual(stats["cache_hits"], 0)
        self.assertEqual(mock_model.generate_content.call_count, 1)

        # Query 2 (Cache Hit)
        self.llm.generate("Prompt Text")
        stats = self.llm.get_statistics()
        self.assertEqual(stats["cache_misses"], 1)
        self.assertEqual(stats["cache_hits"], 1)
        # Verify generate_content was NOT called again
        self.assertEqual(mock_model.generate_content.call_count, 1)

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_exponential_backoff_retry(self, mock_configure, mock_model_class):
        """Test backoff retry triggers on rate limits or API throttles."""
        mock_model = MagicMock()
        # Side effect: throw rate limit exception first, then succeed
        mock_model.generate_content.side_effect = [
            Exception("ResourceExhausted: 429 Rate limit exceeded"),
            self.mock_api_response
        ]
        mock_model_class.return_value = mock_model

        # Mock time.sleep to accelerate test runtime
        with patch("time.sleep") as mock_sleep:
            response = self.llm.generate("Prompt retry test")
            self.assertIn("backpropagation", response.response_text)
            self.assertEqual(mock_model.generate_content.call_count, 2)
            mock_sleep.assert_called_once()  # Slept once

            stats = self.llm.get_statistics()
            self.assertEqual(stats["retry_count"], 1)

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_rate_limit_failure(self, mock_configure, mock_model_class):
        """Test that persistent rate limits raise RateLimitException after exhausting retries."""
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("ResourceExhausted: 429 Rate limit exceeded")
        mock_model_class.return_value = mock_model

        with patch("time.sleep"):
            with self.assertRaises(RateLimitException):
                self.llm.generate("Prompt fail test")

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_streaming_tokens(self, mock_configure, mock_model_class):
        """Test streaming tokens generator yield sequence."""
        mock_model = MagicMock()
        # Mock generator chunks
        chunk1 = MagicMock()
        chunk1.text = "Token A "
        chunk2 = MagicMock()
        chunk2.text = "Token B"
        
        mock_model.generate_content.return_value = [chunk1, chunk2]
        mock_model_class.return_value = mock_model

        tokens = list(self.llm.generate_stream("Stream test"))
        self.assertEqual(len(tokens), 2)
        self.assertEqual(tokens[0], "Token A ")
        self.assertEqual(tokens[1], "Token B")


if __name__ == "__main__":
    unittest.main()
