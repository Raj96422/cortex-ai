"""
CortexRAGPipeline Implementation for Cortex AI.
Wires together Modules 3-8 into a complete Retrieval-Augmented Generation orchestrator.
Manages sessions, query validations, statistics, and health aggregations.
"""

import time
import uuid
import logging
from typing import Any, Dict, List, Optional

from core.rag.base_rag_pipeline import BaseRAGPipeline, RAGResponse
from core.rag.rag_session import RAGSession
from core.exceptions import EmptyQueryException
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)


class CortexRAGPipeline(BaseRAGPipeline):
    """
    Orchestration layer connecting DocumentProcessor, EmbeddingService,
    VectorRepository, SemanticRetriever, RAGPromptBuilder, and GeminiLLM.
    """

    def __init__(
        self,
        document_processor: Any,
        embedding_service: Any,
        vector_repository: Any,
        semantic_retriever: Any,
        rag_prompt_builder: Any,
        gemini_llm: Any
    ):
        """
        Initializes the CortexRAGPipeline using Dependency Injection.

        Args:
            document_processor (Any): Module or service handling PDF loading/chunking.
            embedding_service (Any): Service generating vector embeddings.
            vector_repository (Any): Central storage and rollback layer.
            semantic_retriever (Any): Retriever looking up matching chunks.
            rag_prompt_builder (Any): Prompt builder compiling formatting styles.
            gemini_llm (Any): Client executing model inference.
        """
        self.doc_processor = document_processor
        self.embedding_service = embedding_service
        self.repo = vector_repository
        self.retriever = semantic_retriever
        self.prompt_builder = rag_prompt_builder
        self.llm = gemini_llm

        # Sessions storage mapping
        self._sessions: Dict[str, RAGSession] = {}

        self._stats: Dict[str, Any] = {}
        self.reset_statistics()

        logger.info("CortexRAGPipeline orchestrator initialized successfully.")

    def reset_statistics(self) -> None:
        """Resets all metrics counters to zero."""
        self._stats = {
            "questions_asked": 0,
            "total_response_time_ms": 0.0,
            "total_retrieval_time_ms": 0.0,
            "total_generation_time_ms": 0.0,
            "total_chunks_retrieved": 0
        }
        logger.info("CortexRAGPipeline statistics have been reset.")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieves aggregated performance and count metrics across the pipeline.

        Returns:
            Dict[str, Any]: Metrics report summary mapping.
        """
        stats_copy = self._stats.copy()
        questions = stats_copy.get("questions_asked", 0)
        resp_time = stats_copy.get("total_response_time_ms", 0.0)
        ret_time = stats_copy.get("total_retrieval_time_ms", 0.0)
        gen_time = stats_copy.get("total_generation_time_ms", 0.0)
        chunks = stats_copy.get("total_chunks_retrieved", 0)

        stats_copy["average_response_time"] = resp_time / questions if questions > 0 else 0.0
        stats_copy["average_retrieval_time"] = ret_time / questions if questions > 0 else 0.0
        stats_copy["average_generation_time"] = gen_time / questions if questions > 0 else 0.0
        stats_copy["average_chunks_retrieved"] = chunks / questions if questions > 0 else 0.0

        # Inject cache efficiency from retriever
        try:
            retriever_stats = self.retriever.get_statistics()
            stats_copy["cache_efficiency"] = retriever_stats.get("cache_efficiency_percentage", 0.0)
        except Exception:
            stats_copy["cache_efficiency"] = 0.0

        # Inject total indexed documents from repository
        try:
            repo_stats = self.repo.get_statistics()
            stats_copy["total_indexed_documents"] = repo_stats.get("total_indexed_documents", 0)
        except Exception:
            stats_copy["total_indexed_documents"] = 0

        return stats_copy

    def get_session(self, session_id: str) -> RAGSession:
        """
        Resolves or instantiates a conversation session container.

        Args:
            session_id (str): Unique session identifier.

        Returns:
            RAGSession: Resolved session object.
        """
        sid = session_id.strip()
        if not sid:
            raise ValueError("session_id must be a non-empty string.")
            
        if sid not in self._sessions:
            logger.info(f"Creating new conversation session: '{sid}'")
            self._sessions[sid] = RAGSession(sid)
        return self._sessions[sid]

    def close_session(self, session_id: str) -> None:
        """Clears memory buffers and evicts active session registry."""
        sid = session_id.strip()
        if sid in self._sessions:
            self._sessions[sid].clear_history()
            self._sessions.pop(sid)
            logger.info(f"Session '{sid}' closed and deleted from registry.")

    def ingest_document(
        self,
        file_path: str,
        collection_name: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> Dict[str, Any]:
        """
        Loads, chunks, embeds, and indexes a PDF document.
        """
        logger.info(f"RAG Pipeline: Ingestion request received for file: '{file_path}'")
        
        # Open file in binary mode to simulate uploaded file wrapper
        try:
            with open(file_path, "rb") as f:
                # process_uploaded_files expects a list of file streams
                chunks = self.doc_processor.process_uploaded_files([f])
        except FileNotFoundError:
            logger.error(f"Ingestion failed: File '{file_path}' not found.")
            raise
        except Exception as e:
            logger.error(f"Ingestion processor failure: {e}")
            raise

        if not chunks:
            logger.warning(f"RAG Pipeline: Ingestion skipped. No valid chunks parsed for '{file_path}'.")
            return {
                "document_id": "none",
                "chunks_indexed": 0,
                "status": "skipped"
            }

        # Embed document chunks
        try:
            logger.info(f"RAG Pipeline: Generating embeddings for {len(chunks)} chunk(s)...")
            embedded_chunks = self.embedding_service.embed_documents(chunks)
        except Exception as e:
            logger.error(f"Ingestion embedding generation failed: {e}")
            raise

        # Save to database via repository
        try:
            logger.info(f"RAG Pipeline: Indexing {len(embedded_chunks)} chunk(s) in repository...")
            self.repo.add_embeddings(collection_name, embedded_chunks)
        except Exception as e:
            logger.error(f"Ingestion indexing store failure: {e}")
            raise

        doc_id = chunks[0].metadata.get("document_id", "unknown")
        logger.info(f"RAG Pipeline: Successfully completed ingestion for document ID: '{doc_id}'")

        return {
            "document_id": doc_id,
            "chunks_indexed": len(embedded_chunks),
            "status": "success"
        }

    def ask(
        self,
        question: str,
        collection_name: str,
        session_id: Optional[str] = None,
        k: int = 4,
        score_threshold: Optional[float] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> RAGResponse:
        """
        Runs the complete end-to-end question answering pipeline:
        Query Validation -> Retrieval -> Prompt Assembly -> LLM Inference -> Normalization.
        """
        request_id = str(uuid.uuid4())
        start_total = time.perf_counter()
        
        logger.info(f"[{request_id}] CortexRAGPipeline: Received question query.")

        # 1. Validate Question
        if not question or not question.strip():
            logger.error(f"[{request_id}] Question query is empty or whitespace.")
            raise EmptyQueryException("Question input is empty or null.")

        # Resolve session context
        session = None
        history = None
        if session_id:
            session = self.get_session(session_id)
            history = session.get_history()

        # 2. Retrieve context chunks
        logger.info(f"[{request_id}] Pipeline Stage: Retrieving context matches...")
        start_ret = time.perf_counter()
        try:
            if metadata_filters:
                retrieved = self.retriever.retrieve_with_metadata(
                    query=question,
                    collection_name=collection_name,
                    metadata_filters=metadata_filters,
                    k=k
                )
            else:
                retrieved = self.retriever.retrieve(
                    query=question,
                    collection_name=collection_name,
                    k=k,
                    score_threshold=score_threshold
                )
        except Exception as e:
            logger.error(f"[{request_id}] Pipeline Stage: Retrieval failed: {e}")
            raise
        ret_time = (time.perf_counter() - start_ret) * 1000.0
        logger.info(f"[{request_id}] Pipeline Stage: Retrieval completed in {ret_time:.1f}ms (Chunks: {len(retrieved)})")

        # 3. Build Prompt
        logger.info(f"[{request_id}] Pipeline Stage: Formatting prompt templates...")
        start_prompt = time.perf_counter()
        try:
            prompt = self.prompt_builder.build_prompt(
                question=question,
                context_chunks=retrieved,
                conversation_history=history,
                template_name=kwargs.get("template_name", "qa")
            )
        except Exception as e:
            logger.error(f"[{request_id}] Pipeline Stage: Prompt formatting failed: {e}")
            raise
        prompt_time = (time.perf_counter() - start_prompt) * 1000.0
        logger.info(f"[{request_id}] Pipeline Stage: Prompt built in {prompt_time:.1f}ms")

        # 4. Generate Response
        logger.info(f"[{request_id}] Pipeline Stage: Requesting LLM generation...")
        start_gen = time.perf_counter()
        try:
            llm_res = self.llm.generate(
                prompt=prompt,
                prompt_version=self.prompt_builder.version,
                **kwargs
            )
        except Exception as e:
            logger.error(f"[{request_id}] Pipeline Stage: LLM generation failed: {e}")
            raise
        gen_time = (time.perf_counter() - start_gen) * 1000.0
        logger.info(f"[{request_id}] Pipeline Stage: Generation completed in {gen_time:.1f}ms")

        # 5. Update session memory if session is active
        if session:
            session.add_message("user", question)
            session.add_message("assistant", llm_res.response_text)
            logger.info(f"[{request_id}] Dialog logs memory updated for session: '{session_id}'")

        # Package response
        timestamp_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        latency_total = (time.perf_counter() - start_total) * 1000.0

        # Update metrics counters
        self._stats["questions_asked"] += 1
        self._stats["total_retrieval_time_ms"] += ret_time
        self._stats["total_generation_time_ms"] += gen_time
        self._stats["total_response_time_ms"] += latency_total
        self._stats["total_chunks_retrieved"] += len(retrieved)

        logger.info(f"[{request_id}] CortexRAGPipeline: Successfully compiled structured RAGResponse.")

        return RAGResponse(
            answer=llm_res.response_text,
            citations=llm_res.citations,
            retrieved_chunks=retrieved,
            similarity_scores=[c.get("similarity_score", c.get("score", 0.0)) for c in retrieved],
            prompt=prompt,
            llm_response=llm_res,
            latency=latency_total,
            query=question,
            timestamp=timestamp_str
        )

    def health_check(self) -> Dict[str, Any]:
        """
        Aggregates health diagnostics reports from all pipeline dependencies.
        """
        logger.info("Orchestrating consolidated health check diagnostics...")
        
        # Default report
        report = {
            "status": "healthy",
            "orchestrator_status": "healthy",
            "dependencies": {}
        }
        
        # 1. Embedding health
        try:
            emb_check = self.embedding_service.health_check()
            report["dependencies"]["embedding_service"] = emb_check
            if emb_check.get("status") != "healthy":
                report["status"] = "unhealthy"
        except Exception as e:
            report["status"] = "unhealthy"
            report["dependencies"]["embedding_service"] = {"status": "unhealthy", "error": str(e)}

        # 2. Repo health
        try:
            repo_check = self.repo.health_check()
            report["dependencies"]["vector_repository"] = repo_check
            if repo_check.get("status") != "healthy":
                report["status"] = "unhealthy"
        except Exception as e:
            report["status"] = "unhealthy"
            report["dependencies"]["vector_repository"] = {"status": "unhealthy", "error": str(e)}

        # 3. Retriever health
        try:
            ret_check = self.retriever.health_check()
            report["dependencies"]["retriever"] = ret_check
            if ret_check.get("status") != "healthy":
                report["status"] = "unhealthy"
        except Exception as e:
            report["status"] = "unhealthy"
            report["dependencies"]["retriever"] = {"status": "unhealthy", "error": str(e)}

        # 4. Prompt Builder health
        try:
            prompt_check = self.prompt_builder.health_check()
            report["dependencies"]["prompt_builder"] = prompt_check
            if prompt_check.get("status") != "healthy":
                report["status"] = "unhealthy"
        except Exception as e:
            report["status"] = "unhealthy"
            report["dependencies"]["prompt_builder"] = {"status": "unhealthy", "error": str(e)}

        # 5. Gemini LLM health
        try:
            llm_check = self.llm.health_check()
            report["dependencies"]["llm_service"] = llm_check
            if llm_check.get("status") != "healthy":
                report["status"] = "unhealthy"
        except Exception as e:
            report["status"] = "unhealthy"
            report["dependencies"]["llm_service"] = {"status": "unhealthy", "error": str(e)}

        return report
