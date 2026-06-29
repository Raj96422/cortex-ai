"""
Unit and Integration Tests for Cortex AI Prompt Engineering Layer.
Tests prompt assembly, citation formatters, templates QA/Summary selections,
statistics collection, compression budgets, and exceptions.
"""

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

# Ensure workspace root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.prompt.rag_prompt_builder import RAGPromptBuilder
from core.prompt.prompt_factory import PromptFactory
from core.exceptions import (
    EmptyQueryException,
    InvalidMetadataException,
    PromptTooLargeException,
    TemplateNotFoundException,
)


class TestRAGPromptBuilder(unittest.TestCase):
    """Test suite for RAGPromptBuilder operations."""

    def setUp(self):
        """Sets up default configuration builder."""
        self.builder = RAGPromptBuilder(
            max_context_chars=300,
            max_prompt_chars=1000,
            strict_mode=False
        )

        self.context_chunks = [
            {
                "chunk_id": "chunk1",
                "text": "Neural networks learn parameters via backpropagation.",
                "similarity_score": 0.95,
                "source": "deep_learning.pdf",
                "page": 10
            },
            {
                "chunk_id": "chunk2",
                "text": "RAG models retrieve external documents to extend context.",
                "similarity_score": 0.88,
                "source": "rag_paper.pdf",
                "page": 2
            }
        ]

    def test_factory_creation(self):
        """Test that the PromptFactory constructs prompt builders correctly."""
        b = PromptFactory.get_prompt_builder("rag", max_context_chars=1000)
        self.assertIsInstance(b, RAGPromptBuilder)
        self.assertEqual(b.max_context_chars, 1000)

        with self.assertRaises(ValueError):
            PromptFactory.get_prompt_builder("invalid")

    def test_empty_query_exceptions(self):
        """Test that empty queries raise EmptyQueryException."""
        with self.assertRaises(EmptyQueryException):
            self.builder.build_prompt("", self.context_chunks)

        with self.assertRaises(EmptyQueryException):
            self.builder.build_prompt("   \n  ", self.context_chunks)

        with self.assertRaises(EmptyQueryException):
            self.builder.build_prompt(None, self.context_chunks)  # type: ignore

    def test_strict_mode_context_check(self):
        """Test that strict mode raises exceptions for missing metadata and contexts."""
        strict_builder = RAGPromptBuilder(strict_mode=True)
        
        # 1. Missing context chunks raises ValueError
        with self.assertRaises(ValueError):
            strict_builder.build_prompt("What is backprop?", [])

        # 2. Chunk missing chunk_id raises InvalidMetadataException
        bad_chunk = {"text": "text", "source": "doc.pdf"}  # lacks chunk_id
        with self.assertRaises(InvalidMetadataException):
            strict_builder.build_prompt("What is backprop?", [bad_chunk])

    def test_inline_citation_formatting(self):
        """Test that context chunks format inline citations correctly."""
        prompt = self.builder.build_prompt(
            question="What is backpropagation?",
            context_chunks=self.context_chunks,
            template_name="qa"
        )

        self.assertIn("Neural networks learn parameters via backpropagation.", prompt)
        self.assertIn("[Source: deep_learning.pdf, Page: 10]", prompt)
        self.assertIn("RAG models retrieve external documents to extend context.", prompt)
        self.assertIn("[Source: rag_paper.pdf, Page: 2]", prompt)

    def test_conversation_history_formatting(self):
        """Test formatting dialog history strings."""
        history = [
            {"role": "user", "content": "Hello AI assistant."},
            {"role": "assistant", "content": "Hello user, how can I help today?"}
        ]

        prompt = self.builder.build_prompt(
            question="Tell me about neural nets.",
            context_chunks=self.context_chunks,
            conversation_history=history,
            template_name="qa"
        )

        self.assertIn("User: Hello AI assistant.", prompt)
        self.assertIn("Assistant: Hello user, how can I help today?", prompt)

    def test_context_budget_truncation(self):
        """Test that context block truncates chunks exceeding character constraints."""
        # max_context_chars configured to 300 in setUp
        # Total original context length of chunk1 block + chunk2 block is ~230 chars.
        # Let's set max_context_chars to 100 to force truncation.
        small_builder = RAGPromptBuilder(
            max_context_chars=110,
            max_prompt_chars=1000
        )

        prompt = small_builder.build_prompt(
            question="What is backpropagation?",
            context_chunks=self.context_chunks,
            template_name="qa"
        )

        # Chunk 1 (~105 chars formatted) fits. Chunk 2 (~125 chars) will exceed the 110 char limit and gets truncated.
        self.assertIn("Neural networks learn parameters via backpropagation.", prompt)
        self.assertIn("[Truncated to fit context budget]", prompt)
        # Check statistics registered truncation
        stats = small_builder.get_statistics()
        self.assertEqual(stats["truncated_prompts_count"], 1)
        self.assertGreater(stats["compression_ratio"], 0.0)

    def test_prompt_exceeds_budget_exception(self):
        """Test that final prompt exceeding max_prompt_chars raises PromptTooLargeException."""
        oversized_builder = RAGPromptBuilder(
            max_context_chars=3000,
            max_prompt_chars=400  # Small prompt ceiling
        )

        with self.assertRaises(PromptTooLargeException):
            oversized_builder.build_prompt(
                question="Oversized Query" * 10,
                context_chunks=self.context_chunks,
                template_name="qa"
            )

    def test_template_manager_selections(self):
        """Test fetching and utilizing standard summarization templates."""
        # Summarization template
        summary_prompt = self.builder.build_prompt(
            question="Give me a summary",
            context_chunks=self.context_chunks,
            template_name="summary"
        )
        self.assertIn("Summarize the context documents. Give me a summary", summary_prompt)

        # Bullet list template
        bullets_prompt = self.builder.build_prompt(
            question="Extract points",
            context_chunks=self.context_chunks,
            template_name="bullets"
        )
        self.assertIn("Use a clean markdown bullet list.", bullets_prompt)

        # Non-existent template raises TemplateNotFoundException
        with self.assertRaises(TemplateNotFoundException):
            self.builder.build_prompt(
                question="Compare",
                context_chunks=self.context_chunks,
                template_name="invalid_template"
            )

    def test_statistics_aggregation(self):
        """Test statistics trackers incrementing and reset operations."""
        stats = self.builder.get_statistics()
        self.assertEqual(stats["prompts_generated"], 0)

        # Generate QA prompt
        self.builder.build_prompt(
            question="QA test question",
            context_chunks=self.context_chunks,
            template_name="qa"
        )

        stats = self.builder.get_statistics()
        self.assertEqual(stats["prompts_generated"], 1)
        self.assertGreater(stats["average_prompt_length"], 0.0)
        self.assertGreater(stats["average_context_length"], 0.0)

        # Reset
        self.builder.reset_statistics()
        reset_stats = self.builder.get_statistics()
        self.assertEqual(reset_stats["prompts_generated"], 0)
        self.assertEqual(reset_stats["average_prompt_length"], 0.0)

    def test_health_check_status(self):
        """Test structured health check reports."""
        report = self.builder.health_check()
        self.assertEqual(report["status"], "healthy")
        self.assertEqual(report["template_status"], "active")
        self.assertIn("qa", report["available_templates"])
        self.assertIn("summary", report["available_templates"])


if __name__ == "__main__":
    unittest.main()
