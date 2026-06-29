"""
RAG Pipeline Factory Module for Cortex AI.
Implements the Factory pattern to assemble the complete RAG Pipeline.
"""

from typing import Any

from core.rag.base_rag_pipeline import BaseRAGPipeline
from core.rag.rag_pipeline import CortexRAGPipeline
from core.embeddings import EmbeddingService
from core.vector_store.vector_store_factory import VectorStoreFactory
from core.repository.vector_repository import VectorRepository
from core.retriever.semantic_retriever import SemanticRetriever
from core.prompt.rag_prompt_builder import RAGPromptBuilder
from core.llm.gemini_llm import GeminiLLM
import core.document_processor as document_processor


class RAGFactory:
    """
    Factory class responsible for assembling RAG pipelines with resolved dependencies.
    """

    @staticmethod
    def get_pipeline(
        store_type: str = "chromadb",
        search_strategy: str = "similarity",
        **kwargs: Any
    ) -> BaseRAGPipeline:
        """
        Assembles and returns a complete RAG Pipeline instance.

        Args:
            store_type (str): Database type (defaults to 'chromadb').
            search_strategy (str): Search algorithm ('similarity' or 'mmr').
            **kwargs (Any): Additional properties.

        Returns:
            BaseRAGPipeline: The assembled pipeline instance.
        """
        # 1. Resolve DocumentProcessor (injecting module functions)
        doc_proc = document_processor

        # 2. Resolve EmbeddingService
        emb_service = EmbeddingService()

        # 3. Resolve Vector Store & VectorRepository
        store = VectorStoreFactory.get_vector_store(store_type)
        repo = VectorRepository(vector_store=store)

        # 4. Resolve Retriever
        retriever = SemanticRetriever(
            embedding_service=emb_service,
            vector_repository=repo,
            search_strategy=search_strategy
        )

        # 5. Resolve Prompt Builder
        prompt_builder = RAGPromptBuilder()

        # 6. Resolve LLM Service
        llm_service = GeminiLLM()

        # Assemble pipeline orchestrator
        return CortexRAGPipeline(
            document_processor=doc_proc,
            embedding_service=emb_service,
            vector_repository=repo,
            semantic_retriever=retriever,
            rag_prompt_builder=prompt_builder,
            gemini_llm=llm_service
        )
