"""
Response Parser Module for Cortex AI.
Defines the LLMResponse dataclass and ResponseParser utility to normalize model outputs,
parse inline citations, estimate token metrics, and validate content.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.exceptions import (
    EmptyResponseException,
    SafetyBlockException,
)


@dataclass
class LLMResponse:
    """
    Standard container representing a structured response from the LLM.
    """
    response_text: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    model_name: str = "unknown"
    provider: str = "unknown"
    finish_reason: str = "stop"
    token_estimate: int = 0
    generation_time: float = 0.0
    timestamp: str = ""
    prompt_version: str = "1.0.0"
    raw_response: Any = None


class ResponseParser:
    """
    Normalizer and citation parser for language model responses.
    
    Parses string outputs to extract structured metadata and isolate citations.
    """

    # Matches pattern: [Source: filename.pdf, Page: 12] or [Source: filename.pdf, Page 12]
    CITATION_REGEX = re.compile(
        r"\[Source:\s*([^,\]]+),\s*(?:Page:?\s*)?([^\]]+)\]",
        re.IGNORECASE
    )

    @staticmethod
    def parse_citations(text: str) -> List[Dict[str, Any]]:
        """
        Parses text content to extract unique inline document source citations.

        Args:
            text (str): Output response string from model.

        Returns:
            List[Dict[str, Any]]: Unique source citations list (dicts of source and page).
        """
        if not text:
            return []

        matches = ResponseParser.CITATION_REGEX.findall(text)
        citations = []
        seen = set()

        for source, page in matches:
            src = source.strip()
            pg = page.strip()
            key = (src.lower(), pg.lower())
            
            if key not in seen:
                seen.add(key)
                citations.append({
                    "source": src,
                    "page": pg
                })

        return citations

    @staticmethod
    def parse_gemini_response(
        raw_response: Any,
        generation_time: float,
        prompt_version: str = "1.0.0"
    ) -> LLMResponse:
        """
        Extracts and normalizes raw Gemini generation output into a structured LLMResponse.

        Args:
            raw_response (Any): Raw response object from Google Generative AI client.
            generation_time (float): Request processing duration in seconds.
            prompt_version (str): Prompt template schema version.

        Returns:
            LLMResponse: The structured response container.

        Raises:
            EmptyResponseException: If text response is empty or absent.
            SafetyBlockException: If response was blocked due to safety flags.
        """
        if raw_response is None:
            raise EmptyResponseException("Received null response from Gemini API.")

        # Check if response was blocked by safety metadata
        # Gemini structure: raw_response.prompt_feedback.block_reason
        prompt_feedback = getattr(raw_response, "prompt_feedback", None)
        if prompt_feedback:
            block_reason = getattr(prompt_feedback, "block_reason", None)
            if block_reason and block_reason != 0:
                raise SafetyBlockException(
                    f"Prompt was blocked by Gemini safety filters. Reason Code: {block_reason}"
                )

        # Check safety ratings inside candidates if they exist
        candidates = getattr(raw_response, "candidates", None)
        if candidates and len(candidates) > 0:
            candidate = candidates[0]
            finish_reason_code = getattr(candidate, "finish_reason", None)
            # 1 = STOP, 2 = MAX_TOKENS, 3 = SAFETY, 4 = RECITATION, 5 = OTHER
            if finish_reason_code in (3, "SAFETY"):
                raise SafetyBlockException(
                    "Generative output was blocked by safety filters."
                )

        # Try to extract text content
        try:
            text = raw_response.text
        except Exception as e:
            # Check if block occurred on parts/candidates
            raise EmptyResponseException(
                f"Failed to extract text from Gemini response (possibly blocked or empty): {e}"
            )

        if not text or not text.strip():
            raise EmptyResponseException("Gemini response text is empty or blank.")

        # Resolve model name
        model_name = getattr(raw_response, "model_version", "models/gemini-pro")
        if not model_name:
            model_name = "models/gemini-1.5-flash"  # standard fallback

        # Estimate token counts
        # Rough estimation: 1 token = 4 characters
        token_estimate = len(text) // 4 + 1

        citations = ResponseParser.parse_citations(text)
        
        timestamp_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        return LLMResponse(
            response_text=text,
            citations=citations,
            model_name=model_name,
            provider="Google Gemini",
            finish_reason="stop",
            token_estimate=token_estimate,
            generation_time=generation_time,
            timestamp=timestamp_str,
            prompt_version=prompt_version,
            raw_response=raw_response
        )
