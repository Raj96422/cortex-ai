"""
Integration and Orchestration Tests for Cortex AI Complete RAG Pipeline.
Tests end-to-end QA flow, document ingestion, session memory preservation,
statistics aggregation, health reporting, and request tracing.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from typing import Any, Dict, List

# Ensure workspace root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.rag.rag_pipeline import CortexRAGPipeline
from core.rag.rag_session import RAGSession
from core.rag.base_rag_pipeline import RAGResponse
from core.rag.rag_factory import RAGFactory
from core.llm.response_parser import LLMResponse
from core.exceptions import EmptyQueryException


class TestCortexRAGPipeline(unittest.TestCase):
    """Test suite for RAG Pipeline orchestration workflows."""

    def setUp(self):
        """Prepares mock services for dependency injection."""
        self.mock_doc_processor = MagicMock()
        self.mock_embeddings = MagicMock()
        self.mock_repo = MagicMock()
        self.mock_retriever = MagicMock()
        self.mock_prompt_builder = MagicMock()
        self.mock_llm = MagicMock()

        self.pipeline = CortexRAGPipeline(
            document_processor=self.mock_doc_processor,
            embedding_service=self.mock_embeddings,
            vector_repository=self.mock_repo,
            semantic_retriever=self.mock_retriever,
            rag_prompt_builder=self.mock_prompt_builder,
            gemini_llm=self.mock_llm
        )

        self.collection_name = "test_rag_collection"

    def test_factory_creation(self):
        """Test that the RAGFactory builds the RAG orchestrator correctly."""
        p = RAGFactory.get_pipeline()
        self.assertIsInstance(p, CortexRAGPipeline)

    def test_ask_e2e_successful_answering(self):
        """Test complete QA pipeline from question input to structured output response."""
        # 1. Setup mocks outputs
        mock_retrieved_chunks = [
            {"chunk_id": "c1", "text": "RAG connects retrievers to generators.", "score": 0.9}
        ]
        self.mock_retriever.retrieve.return_value = mock_retrieved_chunks
        
        mock_prompt_string = "Prompt Context: RAG connects retrievers to generators. Question: Explain RAG."
        self.mock_prompt_builder.build_prompt.return_value = mock_prompt_string
        self.mock_prompt_builder.version = "1.0.0"

        mock_llm_response = LLMResponse(
            response_text="RAG is an AI framework that connects retrievers to generators.",
            citations=[{"source": "rag.pdf", "page": "2"}],
            model_name="gemini-1.5-flash",
            provider="Google Gemini"
        )
        self.mock_llm.generate.return_value = mock_llm_response

        # 2. Run Pipeline Ask
        response = self.pipeline.ask("Explain RAG", self.collection_name)

        # 3. Assertions
        self.assertIsInstance(response, RAGResponse)
        self.assertEqual(response.answer, "RAG is an AI framework that connects retrievers to generators.")
        self.assertEqual(response.citations, [{"source": "rag.pdf", "page": "2"}])
        self.assertEqual(response.retrieved_chunks, mock_retrieved_chunks)
        self.assertEqual(response.prompt, mock_prompt_string)
        self.assertEqual(response.query, "Explain RAG")
        self.assertGreater(response.latency, 0.0)

        # Verify calls occurred in correct order
        self.mock_retriever.retrieve.assert_called_once_with(
            query="Explain RAG",
            collection_name=self.collection_name,
            k=4,
            score_threshold=None
        )
        self.mock_prompt_builder.build_prompt.assert_called_once_with(
            question="Explain RAG",
            context_chunks=mock_retrieved_chunks,
            conversation_history=None,
            template_name="qa"
        )
        self.mock_llm.generate.assert_called_once_with(
            prompt=mock_prompt_string,
            prompt_version="1.0.0"
        )

    def test_ask_empty_query_raises_exception(self):
        """Test pipeline rejects empty query inputs immediately."""
        with self.assertRaises(EmptyQueryException):
            self.pipeline.ask("", self.collection_name)

        with self.assertRaises(EmptyQueryException):
            self.pipeline.ask("     ", self.collection_name)

    def test_document_ingestion_orchestration(self):
        """Test document loading, chunking, embedding, and indexing sequence."""
        # 1. Setup mocks outputs
        mock_chunks = [MagicMock(), MagicMock()]
        # Mock metadata
        mock_chunks[0].metadata = {"document_id": "doc123"}
        self.mock_doc_processor.process_uploaded_files.return_value = mock_chunks

        mock_embedded_chunks = [MagicMock(), MagicMock()]
        self.mock_embeddings.embed_documents.return_value = mock_embedded_chunks

        # Create temporary dummy file to read
        temp_file = Path("tests/temp_test_ingest.pdf")
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file.write_bytes(b"dummy pdf content bytes")

        try:
            # 2. Run Pipeline Ingest
            report = self.pipeline.ingest_document(str(temp_file), self.collection_name)

            # 3. Assertions
            self.assertEqual(report["status"], "success")
            self.assertEqual(report["document_id"], "doc123")
            self.assertEqual(report["chunks_indexed"], 2)

            self.mock_doc_processor.process_uploaded_files.assert_called_once()
            self.mock_embeddings.embed_documents.assert_called_once_with(mock_chunks)
            self.mock_repo.add_embeddings.assert_called_once_with(self.collection_name, mock_embedded_chunks)

        finally:
            if temp_file.exists():
                temp_file.unlink()

    def test_session_conversational_history(self):
        """Test dialogue memory preservation across multiple session turns."""
        session_id = "session_abc"
        
        # turn 1 Mocks
        self.mock_retriever.retrieve.return_value = []
        self.mock_prompt_builder.build_prompt.return_value = "prompt_1"
        self.mock_prompt_builder.version = "1.0.0"
        
        self.mock_llm.generate.return_value = LLMResponse(response_text="Answer 1")

        # Call turn 1
        self.pipeline.ask("Question 1", self.collection_name, session_id=session_id)

        # Check session history contains turn 1
        session = self.pipeline.get_session(session_id)
        history = session.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], "Question 1")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(history[1]["content"], "Answer 1")

        # turn 2
        self.pipeline.ask("Question 2", self.collection_name, session_id=session_id)
        self.mock_prompt_builder.build_prompt.assert_called_with(
            question="Question 2",
            context_chunks=[],
            conversation_history=history, # Passed history of previous turn
            template_name="qa"
        )
        self.assertEqual(len(session.get_history()), 4)

        # Reset Session checks
        self.pipeline.close_session(session_id)
        self.assertNotIn(session_id, self.pipeline._sessions)

    def test_health_checks_aggregation(self):
        """Test health diagnostics aggregate dependency statuses."""
        self.mock_embeddings.health_check.return_value = {"status": "healthy"}
        self.mock_repo.health_check.return_value = {"status": "healthy"}
        self.mock_retriever.health_check.return_value = {"status": "healthy"}
        self.mock_prompt_builder.health_check.return_value = {"status": "healthy"}
        self.mock_llm.health_check.return_value = {"status": "healthy"}

        report = self.pipeline.health_check()
        self.assertEqual(report["status"], "healthy")
        self.assertEqual(report["dependencies"]["embedding_service"]["status"], "healthy")
        self.assertEqual(report["dependencies"]["llm_service"]["status"], "healthy")

        # Test unhealthy propagation
        self.mock_llm.health_check.return_value = {"status": "unhealthy"}
        report_unhealthy = self.pipeline.health_check()
        self.assertEqual(report_unhealthy["status"], "unhealthy")

    def test_statistics_aggregation(self):
        """Test pipeline metrics tracking increment correctly."""
        stats = self.pipeline.get_statistics()
        self.assertEqual(stats["questions_asked"], 0)

        # Mock calls
        self.mock_retriever.retrieve.return_value = [{"text": "data", "score": 0.9}]
        self.mock_prompt_builder.build_prompt.return_value = "prompt"
        self.mock_prompt_builder.version = "1.0.0"
        self.mock_llm.generate.return_value = LLMResponse(response_text="answer")

        self.pipeline.ask("Question", self.collection_name)

        stats = self.pipeline.get_statistics()
        self.assertEqual(stats["questions_asked"], 1)
        self.assertEqual(stats["average_chunks_retrieved"], 1.0)
        self.assertGreater(stats["average_response_time"], 0.0)


if __name__ == "__main__":
    unittest.main()
