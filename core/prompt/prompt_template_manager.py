"""
Prompt Template Manager Module for Cortex AI.
Registers, manages, and validates default system instructions and prompt templates 
utilized during prompt generation.
"""

from typing import Any, Dict

from core.exceptions import (
    InvalidTemplateException,
    TemplateNotFoundException,
)


class PromptTemplateManager:
    """
    Registry for managing prompt templates and system instructions.
    
    Ensures that prompt structures are uniform, validated, and easily hot-swapped.
    """

    def __init__(self) -> None:
        """Initializes the manager and loads default templates."""
        self._templates: Dict[str, Dict[str, str]] = {}
        self._load_defaults()

    def _load_defaults(self) -> None:
        """Loads default templates for standard tasks."""
        # QA template
        self.register_template(
            name="qa",
            system_instruction=(
                "You are Antigravity, a professional AI research assistant. Your task is to answer "
                "the user's question using only the provided context. Follow these strict rules:\n"
                "1. Cite sources inline like '[Source: filename.pdf, Page 12]' where you got the facts.\n"
                "2. If the answer cannot be found in the provided context, state: 'I cannot answer this "
                "based on the provided context.' Do not make up facts or hallucinate."
            ),
            template=(
                "=== System Instructions ===\n"
                "{system_instruction}\n\n"
                "=== Retrieved Context ===\n"
                "{context}\n\n"
                "=== Conversation History ===\n"
                "{conversation_history}\n\n"
                "=== User Question ===\n"
                "{question}\n\n"
                "=== Output Format ===\n"
                "Answer the question clearly. For every claim, add the exact source citation at the end of the sentence."
            )
        )

        # Summarization template
        self.register_template(
            name="summary",
            system_instruction=(
                "Summarize the provided context documents clearly. Highlight key takeaways, "
                "objectives, and conclusions. Cite the sources of your key points."
            ),
            template=(
                "=== System Instructions ===\n"
                "{system_instruction}\n\n"
                "=== Retrieved Context ===\n"
                "{context}\n\n"
                "=== User Question ===\n"
                "Summarize the context documents. {question}\n\n"
                "=== Output Format ===\n"
                "Provide a structured executive summary followed by key takeaways with inline citations."
            )
        )

        # Comparison template
        self.register_template(
            name="compare",
            system_instruction=(
                "Compare the distinct products, theories, or metrics mentioned in the context. "
                "Use the provided context to build a comparative analysis. Cite sources."
            ),
            template=(
                "=== System Instructions ===\n"
                "{system_instruction}\n\n"
                "=== Retrieved Context ===\n"
                "{context}\n\n"
                "=== User Question ===\n"
                "Provide a comparison based on the context. {question}\n\n"
                "=== Output Format ===\n"
                "Present a comparative analysis detailing similarities, differences, and contrasting metrics."
            )
        )

        # Explain Concept template
        self.register_template(
            name="explain",
            system_instruction=(
                "Explain the specified concept from the context in simple, accessible terms. "
                "Define terminology and provide illustrative explanations using only the context facts. "
                "Cite sources."
            ),
            template=(
                "=== System Instructions ===\n"
                "{system_instruction}\n\n"
                "=== Retrieved Context ===\n"
                "{context}\n\n"
                "=== User Question ===\n"
                "Explain this concept: {question}\n\n"
                "=== Output Format ===\n"
                "Break down the concept using simple headings, definitions, and explanations."
            )
        )

        # Bullet Summary template
        self.register_template(
            name="bullets",
            system_instruction=(
                "Extract the critical facts from the context and present them as a clean, "
                "non-redundant bulleted list. Cite the source of every bullet point."
            ),
            template=(
                "=== System Instructions ===\n"
                "{system_instruction}\n\n"
                "=== Retrieved Context ===\n"
                "{context}\n\n"
                "=== User Question ===\n"
                "Provide a bulleted summary. {question}\n\n"
                "=== Output Format ===\n"
                "Use a clean markdown bullet list. Every bullet must end with its source citation."
            )
        )

    def register_template(self, name: str, system_instruction: str, template: str) -> None:
        """
        Registers a new template in the manager.

        Args:
            name (str): Selector name.
            system_instruction (str): Core guidelines for the model.
            template (str): String template containing formatting placeholders.

        Raises:
            InvalidTemplateException: If the template string format is invalid.
        """
        if not name or not isinstance(name, str):
            raise ValueError("Template name must be a non-empty string.")
        
        # Simple validations: template should be string and contain required placeholders
        if not isinstance(template, str) or not isinstance(system_instruction, str):
            raise InvalidTemplateException("Template and system instructions must be strings.")

        required_placeholders = ["{context}", "{question}", "{system_instruction}"]
        for p in required_placeholders:
            if p not in template:
                raise InvalidTemplateException(
                    f"Template '{name}' is missing mandatory placeholder: '{p}'."
                )

        self._templates[name.lower().strip()] = {
            "system_instruction": system_instruction,
            "template": template
        }

    def get_template(self, name: str) -> Dict[str, str]:
        """
        Retrieves a template by name.

        Args:
            name (str): Template identifier.

        Returns:
            Dict[str, str]: Map containing 'system_instruction' and 'template'.

        Raises:
            TemplateNotFoundException: If the template name is not registered.
        """
        key = name.lower().strip()
        if key not in self._templates:
            raise TemplateNotFoundException(
                f"Requested prompt template '{name}' does not exist in registry. "
                f"Available templates: {list(self._templates.keys())}"
            )
        return self._templates[key]

    def list_templates(self) -> Dict[str, Dict[str, str]]:
        """Returns a copy of all registered templates."""
        return self._templates.copy()
