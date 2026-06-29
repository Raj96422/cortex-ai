"""
RAG Prompt Builder Implementation for Cortex AI.
Constructs optimized, versioned prompt inputs for the LLM using context formatting,
inline citations, hallucination prevention directives, conversation history,
and token/character budget compression.
"""

import logging
from typing import Any, Dict, List, Optional

from core.exceptions import (
    EmptyQueryException,
    InvalidMetadataException,
    PromptTooLargeException,
)
from core.prompt.base_prompt_builder import BasePromptBuilder  # Note: we will correct this import
from core.prompt.prompt_template_manager import PromptTemplateManager
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)

BUILDER_VERSION: str = "1.0.0"


class RAGPromptBuilder(BasePromptBuilder):
    """
    Prompt Builder implementing Retrieval-Augmented Generation schemas.
    
    Validates queries and chunk metadata, enforces context character budgets,
    compiles inline citations, and reports formatting metrics.
    """

    def __init__(
        self,
        template_manager: Optional[PromptTemplateManager] = None,
        max_context_chars: int = 5000,
        max_prompt_chars: int = 10000,
        strict_mode: bool = False
    ):
        """
        Initializes the RAGPromptBuilder.

        Args:
            template_manager (Optional[PromptTemplateManager]): Manager for templates.
            max_context_chars (int): Character budget for the context block.
            max_prompt_chars (int): Maximum allowed characters in the final prompt.
            strict_mode (bool): If True, raises exceptions for missing contexts or metadata.
        """
        self.template_manager = template_manager if template_manager is not None else PromptTemplateManager()
        self.max_context_chars = max_context_chars
        self.max_prompt_chars = max_prompt_chars
        self.strict_mode = strict_mode
        self.version = BUILDER_VERSION

        self._stats: Dict[str, Any] = {}
        self.reset_statistics()

        logger.info(
            f"RAGPromptBuilder active (Version: {self.version}, "
            f"Max Context Chars: {self.max_context_chars}, Max Prompt Chars: {self.max_prompt_chars})"
        )

    def reset_statistics(self) -> None:
        """Resets all metrics counters to zero."""
        self._stats = {
            "prompts_generated": 0,
            "total_prompt_length": 0,
            "total_context_length": 0,
            "truncated_prompts_count": 0,
            "total_original_context_length": 0,
        }
        logger.info("RAGPromptBuilder statistics have been reset.")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieves active metrics, calculating averages and compression ratios.

        Returns:
            Dict[str, Any]: Metrics report summary mapping.
        """
        stats_copy = self._stats.copy()
        generated = stats_copy.get("prompts_generated", 0)
        prompt_len = stats_copy.get("total_prompt_length", 0)
        context_len = stats_copy.get("total_context_length", 0)
        original_context_len = stats_copy.get("total_original_context_length", 0)

        stats_copy["average_prompt_length"] = prompt_len / generated if generated > 0 else 0.0
        stats_copy["average_context_length"] = context_len / generated if generated > 0 else 0.0
        
        # Compression ratio represents budgeted context characters vs. original context characters
        stats_copy["compression_ratio"] = (
            context_len / original_context_len if original_context_len > 0 else 1.0
        )
        
        return stats_copy

    def _validate_inputs(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Asserts validity of question and chunk parameters.

        Args:
            question (str): User question.
            context_chunks (List[Dict[str, Any]]): Retrieved chunks.

        Returns:
            str: Stripped query string.
        """
        if question is None:
            raise EmptyQueryException("Question input is None. Expected string.")
            
        if not isinstance(question, str):
            raise ValueError(f"Question input must be a string, got {type(question).__name__}.")
            
        stripped = question.strip()
        if not stripped:
            raise EmptyQueryException("Question is empty or whitespace-only.")

        if self.strict_mode and (context_chunks is None or len(context_chunks) == 0):
            raise ValueError("Strict Mode: Context chunks list is empty or None.")

        return stripped

    def _format_context(self, context_chunks: List[Dict[str, Any]]) -> tuple[str, bool, int]:
        """
        Deduplicates, sorts, formats citations, and compresses chunks to fit context budget.

        Args:
            context_chunks (List[Dict[str, Any]]): Chunks to process.

        Returns:
            tuple[str, bool, int]: Formatted context block string, truncated flag, original length.
        """
        if not context_chunks:
            return "No retrieved context available.", False, 0

        # Sort candidates by similarity score descending (highest priority first)
        # Handle cases where score is missing gracefully
        sorted_chunks = sorted(
            context_chunks,
            key=lambda c: c.get("similarity_score", c.get("score", 0.0)),
            reverse=True
        )

        seen_ids = set()
        formatted_parts = []
        truncated = False
        original_length = 0
        current_length = 0

        for chunk in sorted_chunks:
            # Metadata Checks
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                if self.strict_mode:
                    raise InvalidMetadataException("Chunk missing mandatory 'chunk_id' key.")
                chunk_id = "unknown"

            # Deduplication
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)

            text = chunk.get("text", "").strip()
            if not text:
                continue

            source = chunk.get("source", "unknown")
            page = chunk.get("page", "unknown")
            
            citation = f"[Source: {source}, Page: {page}]"
            chunk_block = f"Document Chunk:\n{text}\nCitation: {citation}\n---\n"
            
            original_length += len(chunk_block)

            # Check Context Budget
            if current_length + len(chunk_block) > self.max_context_chars:
                truncated = True
                # Add partial chunk text up to budget ceiling
                allowed_chars = self.max_context_chars - current_length
                if allowed_chars > 50:  # Only add if we can fit meaningful text
                    partial_chunk = chunk_block[:allowed_chars] + "\n[Truncated to fit context budget]\n---\n"
                    formatted_parts.append(partial_chunk)
                    current_length += len(partial_chunk)
                break
            else:
                formatted_parts.append(chunk_block)
                current_length += len(chunk_block)

        context_string = "".join(formatted_parts).strip()
        return context_string, truncated, original_length

    def _format_history(self, history: Optional[List[Dict[str, str]]]) -> str:
        """
        Converts dialog logs array to clean dialogue string.

        Args:
            history (Optional[List[Dict[str, str]]]): Dialog logs.

        Returns:
            str: Dialog string representation.
        """
        if not history:
            return "No past conversation history."

        parts = []
        for idx, msg in enumerate(history):
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "").strip()
            if not content:
                continue
            parts.append(f"{role}: {content}")

        return "\n".join(parts).strip()

    def build_prompt(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        template_name: str = "qa",
        **kwargs: Any
    ) -> str:
        """
        Assembles templates, system instructions, and formats contexts into LLM inputs.
        """
        self._stats["prompts_generated"] += 1
        
        try:
            # 1. Validate
            normalized_query = self._validate_inputs(question, context_chunks)

            # 2. Format Context (enforces budgets and builds citations)
            context_str, truncated, original_len = self._format_context(context_chunks)
            self._stats["total_context_length"] += len(context_str)
            self._stats["total_original_context_length"] += original_len
            if truncated:
                self._stats["truncated_prompts_count"] += 1

            # 3. Format History
            history_str = self._format_history(conversation_history)

            # 4. Fetch Template
            tpl_data = self.template_manager.get_template(template_name)
            system_instruction = tpl_data["system_instruction"]
            template_body = tpl_data["template"]

            # 5. Populate Template
            # Safe populate to avoid KeyError for extra/missing template parameters
            prompt = template_body.format(
                system_instruction=system_instruction,
                context=context_str,
                conversation_history=history_str,
                question=normalized_query
            )

            # 6. Size Budget check
            if len(prompt) > self.max_prompt_chars:
                raise PromptTooLargeException(
                    f"Generated prompt size ({len(prompt)} chars) exceeds maximum allowed "
                    f"size of {self.max_prompt_chars} chars."
                )

            self._stats["total_prompt_length"] += len(prompt)
            return prompt

        except Exception:
            # Decrement generated count if execution failed
            self._stats["prompts_generated"] -= 1
            raise

    def health_check(self) -> Dict[str, Any]:
        """
        Validates template status registry configurations.
        """
        templates = self.template_manager.list_templates()
        stats = self.get_statistics()
        
        return {
            "status": "healthy",
            "provider": "RAGPromptBuilder",
            "current_version": self.version,
            "template_status": "active",
            "available_templates": list(templates.keys()),
            "average_prompt_size_chars": stats.get("average_prompt_length", 0.0),
            "max_context_chars": self.max_context_chars,
            "max_prompt_chars": self.max_prompt_chars
        }
