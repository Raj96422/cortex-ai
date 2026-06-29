"""
RAG Session Module for Cortex AI.
Manages conversational session history, dialog logs memory, and turn metrics.
"""

import time
from typing import Any, Dict, List


class RAGSession:
    """
    Manages conversation memory and metrics for a single RAG conversation session.
    """

    def __init__(self, session_id: str):
        """
        Initializes a new RAGSession.

        Args:
            session_id (str): Unique session identifier.
        """
        self.session_id = session_id
        self.history: List[Dict[str, str]] = []
        self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.updated_at = self.created_at
        
        # Turn-level statistics
        self.total_turns = 0

    def add_message(self, role: str, content: str) -> None:
        """
        Appends a message to the dialogue history.

        Args:
            role (str): Sender role ('user' or 'assistant').
            content (str): Text message content.
        """
        self.history.append({
            "role": role.lower().strip(),
            "content": content.strip()
        })
        self.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if role.lower().strip() == "user":
            self.total_turns += 1

    def clear_history(self) -> None:
        """Clears dialog logs history and resets session counters."""
        self.history.clear()
        self.total_turns = 0
        self.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def get_history(self) -> List[Dict[str, str]]:
        """
        Retrieves the conversation dialogue log array.

        Returns:
            List[Dict[str, str]]: Messages list.
        """
        return self.history

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieves session metrics.

        Returns:
            Dict[str, Any]: Statistics summary.
        """
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "total_turns": self.total_turns,
            "history_length": len(self.history)
        }
