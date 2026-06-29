"""
Semantic Retriever Implementation for Cortex AI.
Generates query embeddings via EmbeddingService, queries VectorRepository,
supports top-K, score thresholds, metadata filtering, MMR diversification,
and caching.
"""

import time
import logging
from typing import Any, Dict, List, Optional

from core.embeddings import EmbeddingService
from core.repository.vector_repository import VectorRepository
from core.retriever.base_retriever import BaseRetriever
from core.exceptions import (
    CollectionNotFoundException,
    EmptyQueryException,
    EmbeddingFailureException,
    InvalidQueryException,
    RetrievalFailureException,
)
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)


class SemanticRetriever(BaseRetriever):
    """
    Concrete implementation of BaseRetriever executing semantic query lookups.
    
    Acts as the main retrieval engine, handling query caching, validations,
    MMR rankings, and database metrics logging.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_repository: VectorRepository,
        search_strategy: str = "similarity",
        lambda_mult: float = 0.5,
        max_query_length: int = 1000
    ):
        """
        Initializes the SemanticRetriever.

        Args:
            embedding_service (EmbeddingService): Core embedding generator.
            vector_repository (VectorRepository): Central database access layer.
            search_strategy (str): Default ranking algorithm ('similarity' or 'mmr').
            lambda_mult (float): Diversity penalty multiplier for MMR search.
            max_query_length (int): Characters threshold for input queries.
        """
        self.embedding_service = embedding_service
        self.repo = vector_repository
        self.search_strategy = search_strategy.lower().strip()
        self.lambda_mult = lambda_mult
        self.max_query_length = max_query_length

        # In-memory query embedding and results caches
        self._embedding_cache: Dict[str, List[float]] = {}
        # Key: (query, collection_name, strategy, k, score_threshold, frozenset(filters))
        self._result_cache: Dict[Any, List[Dict[str, Any]]] = {}

        self._stats: Dict[str, Any] = {}
        self.reset_statistics()

        logger.info(
            f"SemanticRetriever initialized (Strategy: '{self.search_strategy}', "
            f"MMR Lambda: {self.lambda_mult})"
        )

    def initialize(self) -> None:
        """
        Ensures underlying embedding client and repository connections are warm.
        """
        self.embedding_service.initialize_model()

    def reset_statistics(self) -> None:
        """Resets all metrics counters to zero."""
        self._stats = {
            "total_queries": 0,
            "successful_retrievals": 0,
            "failed_retrievals": 0,
            "total_retrieval_time_ms": 0.0,
            "total_returned_chunks": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        logger.info("SemanticRetriever statistics have been reset.")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieves active metrics, calculating averages and hit ratios.

        Returns:
            Dict[str, Any]: Performance metrics summary.
        """
        stats_copy = self._stats.copy()
        queries = stats_copy.get("total_queries", 0)
        time_spent = stats_copy.get("total_retrieval_time_ms", 0.0)
        chunks = stats_copy.get("total_returned_chunks", 0)
        successes = stats_copy.get("successful_retrievals", 0)

        stats_copy["average_retrieval_time"] = time_spent / queries if queries > 0 else 0.0
        stats_copy["average_returned_chunks"] = chunks / successes if successes > 0 else 0.0
        
        # Calculate cache efficiency
        hits = stats_copy.get("cache_hits", 0)
        misses = stats_copy.get("cache_misses", 0)
        total_cache_lookups = hits + misses
        stats_copy["cache_efficiency_percentage"] = (
            (hits / total_cache_lookups) * 100.0 if total_cache_lookups > 0 else 0.0
        )
        
        return stats_copy

    def _validate_query(self, query: str) -> str:
        """
        Enforces character lengths and empties check on input search query.

        Args:
            query (str): The search text.

        Returns:
            str: Normalized stripped search text.
        """
        if query is None:
            raise InvalidQueryException("Query input is None. Expected string.")
            
        if not isinstance(query, str):
            raise InvalidQueryException(
                f"Query input must be a string, got {type(query).__name__}."
            )
            
        stripped = query.strip()
        if not stripped:
            raise EmptyQueryException("Query is empty or whitespace-only.")

        if len(stripped) > self.max_query_length:
            raise InvalidQueryException(
                f"Query length ({len(stripped)} chars) exceeds maximum allowed size "
                f"of {self.max_query_length} chars."
            )
            
        return stripped

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Resolves query embeddings utilizing cache mapping.

        Args:
            query (str): Validated text.

        Returns:
            List[float]: Core vector embedding.
        """
        if query in self._embedding_cache:
            return self._embedding_cache[query]

        try:
            vector = self.embedding_service.embed_text(query)
            self._embedding_cache[query] = vector
            return vector
        except Exception as e:
            raise EmbeddingFailureException(f"Failed to generate query embedding: {e}")

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Pure Python cosine similarity calculation to prevent numpy overhead.

        Args:
            a (List[float]): Vector A.
            b (List[float]): Vector B.

        Returns:
            float: Cosine similarity score.
        """
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def _compute_mmr(
        self,
        query_vector: List[float],
        candidates: List[Dict[str, Any]],
        k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Executes MMR selection to diversify returned candidates.

        Args:
            query_vector (List[float]): Normalized query coordinates.
            candidates (List[Dict[str, Any]]): Chunk matches returned by database query.
            k (int): Target output results count.

        Returns:
            List[Dict[str, Any]]: Diversified list of candidates.
        """
        if not candidates or k <= 0:
            return []

        if len(candidates) <= k:
            return candidates

        # 1. Fetch vector embeddings for candidates
        # Since EmbeddingService has a fast hash-based cache, calling embed_text 
        # for candidate text strings is very fast (100% cache hits) and database-agnostic.
        candidate_embeddings: List[List[float]] = []
        for cand in candidates:
            try:
                # Retrieve from cache or re-embed (instantly resolved)
                vector = self.embedding_service.embed_text(cand["text"])
                candidate_embeddings.append(vector)
            except Exception:
                # Fallback to zero vector if embedding fails (unlikely due to cache)
                candidate_embeddings.append([0.0] * len(query_vector))

        selected_indices: List[int] = []

        # Start by selecting the candidate closest to the query (index 0)
        selected_indices.append(0)

        while len(selected_indices) < k and len(selected_indices) < len(candidates):
            best_mmr_score = -float("inf")
            best_idx = -1

            for i in range(len(candidates)):
                if i in selected_indices:
                    continue

                # Query similarity: similarity score returned by store query
                sim_query = candidates[i]["score"]

                # Max similarity to already selected candidates
                max_sim_selected = -float("inf")
                for j in selected_indices:
                    sim_selected = self._cosine_similarity(candidate_embeddings[i], candidate_embeddings[j])
                    if sim_selected > max_sim_selected:
                        max_sim_selected = sim_selected

                # MMR Selection Score
                mmr_score = self.lambda_mult * sim_query - (1.0 - self.lambda_mult) * max_sim_selected
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = i

            if best_idx != -1:
                selected_indices.append(best_idx)
            else:
                break

        return [candidates[i] for i in selected_indices]

    def _pack_and_deduplicate(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Formats output results and eliminates duplicate entries by chunk_id.

        Args:
            matches (List[Dict[str, Any]]): Database records.

        Returns:
            List[Dict[str, Any]]: Ordered deduplicated output dictionaries.
        """
        seen_ids = set()
        deduplicated = []

        for m in matches:
            chunk_id = m.get("chunk_id")
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)

            meta = m.get("metadata", {})
            
            # Populate standard metadata fields
            result_item = {
                "document_id": meta.get("document_id"),
                "chunk_id": chunk_id,
                "page": meta.get("page"),
                "source": meta.get("source"),
                "text": m.get("text"),
                "similarity_score": m.get("score"),
                "embedding_provider": meta.get("embedding_provider"),
                "embedding_model": meta.get("embedding_model")
            }
            deduplicated.append(result_item)

        return deduplicated

    def _resolve_cached_results(self, cache_key: Any) -> Optional[List[Dict[str, Any]]]:
        """Resolves results from the result cache, updating statistics."""
        if cache_key in self._result_cache:
            self._stats["cache_hits"] += 1
            self._stats["successful_retrievals"] += 1
            cached = self._result_cache[cache_key]
            self._stats["total_returned_chunks"] += len(cached)
            logger.info("Result cache hit. Returning cached chunks.")
            return cached
            
        self._stats["cache_misses"] += 1
        return None

    def retrieve(
        self,
        query: str,
        collection_name: str,
        k: int = 4,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-K relevant chunks for the given query text.
        """
        start_time = time.perf_counter()
        self._stats["total_queries"] += 1

        try:
            # 1. Validate query
            normalized_query = self._validate_query(query)
            
            # Check Result Cache
            cache_key = (normalized_query, collection_name, self.search_strategy, k, score_threshold, None)
            cached = self._resolve_cached_results(cache_key)
            if cached is not None:
                return cached

            # 2. Get Embedding (uses query cache internally)
            query_vector = self._get_query_embedding(normalized_query)

            # 3. Fetch Candidates from Database
            # For MMR, we pull a larger pool of candidates to pick the best diversified set of K items
            fetch_k = k * 2 if self.search_strategy == "mmr" else k
            
            logger.info(f"Retrieving from vector repository. Strategy: '{self.search_strategy}'")
            candidates = self.repo.similarity_search(
                collection_name=collection_name,
                query_vector=query_vector,
                k=fetch_k,
                score_threshold=score_threshold
            )

            # 4. Diversify via MMR if requested
            if self.search_strategy == "mmr":
                ranked = self._compute_mmr(query_vector, candidates, k=k)
            else:
                ranked = candidates

            # 5. Pack and deduplicate output
            results = self._pack_and_deduplicate(ranked)
            
            # Store in cache
            if len(self._result_cache) >= 1000:
                # Prevent memory leaks by clearing old cache entries
                self._result_cache.clear()
            self._result_cache[cache_key] = results

            # Successful Metrics
            self._stats["successful_retrievals"] += 1
            self._stats["total_returned_chunks"] += len(results)
            
            return results

        except (EmptyQueryException, InvalidQueryException, CollectionNotFoundException):
            self._stats["failed_retrievals"] += 1
            raise
        except Exception as e:
            self._stats["failed_retrievals"] += 1
            raise RetrievalFailureException(f"Retrieval operation failed: {e}")
        finally:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            self._stats["total_retrieval_time_ms"] += elapsed

    def retrieve_with_metadata(
        self,
        query: str,
        collection_name: str,
        metadata_filters: Dict[str, Any],
        k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Retrieves documents matching the query vector and metadata filter conditions.
        """
        start_time = time.perf_counter()
        self._stats["total_queries"] += 1

        try:
            normalized_query = self._validate_query(query)
            
            # Convert filter dictionary to frozenset for caching
            filter_key = frozenset(metadata_filters.items())
            cache_key = (normalized_query, collection_name, "filter_search", k, None, filter_key)
            
            cached = self._resolve_cached_results(cache_key)
            if cached is not None:
                return cached

            # Query Embedding
            query_vector = self._get_query_embedding(normalized_query)

            # Query database applying filters
            logger.info(f"Retrieving using metadata filters: {metadata_filters}")
            candidates = self.repo.filter_search(
                collection_name=collection_name,
                query_vector=query_vector,
                metadata_filters=metadata_filters,
                k=k
            )

            # Pack and deduplicate
            results = self._pack_and_deduplicate(candidates)
            self._result_cache[cache_key] = results

            # Metrics
            self._stats["successful_retrievals"] += 1
            self._stats["total_returned_chunks"] += len(results)
            
            return results

        except (EmptyQueryException, InvalidQueryException, CollectionNotFoundException):
            self._stats["failed_retrievals"] += 1
            raise
        except Exception as e:
            self._stats["failed_retrievals"] += 1
            raise RetrievalFailureException(f"Retrieval with filters failed: {e}")
        finally:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            self._stats["total_retrieval_time_ms"] += elapsed

    def retrieve_by_document(
        self,
        query: str,
        collection_name: str,
        document_id: str,
        k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Retrieves document chunks matching query, scoped strictly to a single document.
        """
        logger.info(f"Retrieving context scoped to document_id: '{document_id}'")
        # Apply standard metadata filter for document_id
        filters = {"document_id": document_id}
        return self.retrieve_with_metadata(
            query=query,
            collection_name=collection_name,
            metadata_filters=filters,
            k=k
        )

    def retrieve_similar_chunks(
        self,
        chunk_id: str,
        collection_name: str,
        k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Retrieves other chunks close to the specified chunk ID.
        """
        start_time = time.perf_counter()
        self._stats["total_queries"] += 1

        if not chunk_id or not isinstance(chunk_id, str):
            self._stats["failed_retrievals"] += 1
            raise InvalidQueryException("Target chunk_id is empty or invalid.")

        try:
            # Check cache
            cache_key = (chunk_id, collection_name, "similar_chunks", k, None, None)
            cached = self._resolve_cached_results(cache_key)
            if cached is not None:
                return cached

            # To retrieve similar chunks, we first fetch the target chunk's vector from database
            client = self.repo.vector_store._get_client()
            col = client.get_collection(collection_name)
            res = col.get(ids=[chunk_id], include=["embeddings"])
            
            if not res or not res.get("embeddings") or len(res["embeddings"]) == 0:
                raise RetrievalFailureException(f"Target chunk_id '{chunk_id}' was not found in index.")

            target_vector = res["embeddings"][0]

            # Similarity search utilizing the chunk's vector
            logger.info(f"Retrieving semantically similar chunks for chunk_id: '{chunk_id}'")
            candidates = self.repo.similarity_search(
                collection_name=collection_name,
                query_vector=target_vector,
                k=k + 1  # Get K+1 because candidate vector includes target vector itself
            )

            # Filter out the target chunk itself
            filtered_candidates = [c for c in candidates if c["chunk_id"] != chunk_id][:k]

            results = self._pack_and_deduplicate(filtered_candidates)
            self._result_cache[cache_key] = results

            # Metrics
            self._stats["successful_retrievals"] += 1
            self._stats["total_returned_chunks"] += len(results)
            
            return results

        except CollectionNotFoundException:
            self._stats["failed_retrievals"] += 1
            raise
        except Exception as e:
            self._stats["failed_retrievals"] += 1
            raise RetrievalFailureException(f"Failed to retrieve similar chunks: {e}")
        finally:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            self._stats["total_retrieval_time_ms"] += elapsed

    def health_check(self) -> Dict[str, Any]:
        """
        Runs diagnostic status check on the retriever, repository, and embedding provider.

        Returns:
            Dict[str, Any]: Structured health diagnostics report mapping.
        """
        logger.info("Executing retriever diagnostics...")
        status = "healthy"
        details = "Retriever and dependencies are fully operational."
        
        # Dependency 1: Repository health check
        try:
            repo_report = self.repo.health_check()
            repo_status = repo_report.get("status", "unknown")
            collection_availability = repo_report.get("collection_status", {}).get("collections", [])
            total_indexed_chunks = repo_report.get("total_vectors", 0)
        except Exception as e:
            status = "unhealthy"
            repo_status = f"error: {e}"
            collection_availability = []
            total_indexed_chunks = 0
            details = f"Repository connection diagnostics failed: {e}"

        # Dependency 2: Embedding service health check
        try:
            embedding_report = self.embedding_service.health_check()
            embedding_status = embedding_report.get("status", "unknown")
        except Exception as e:
            status = "unhealthy"
            embedding_status = f"error: {e}"
            details = f"Embedding service diagnostics failed: {e}"

        # Average latency from statistics
        stats = self.get_statistics()
        average_latency = stats.get("average_retrieval_time", 0.0)

        return {
            "status": status,
            "retriever_status": "healthy",
            "repository_status": repo_status,
            "embedding_provider_status": embedding_status,
            "average_latency_ms": average_latency,
            "total_indexed_chunks": total_indexed_chunks,
            "collection_availability": collection_availability,
            "details": details
        }

    def close(self) -> None:
        """Clears memory caches and delegates close to repository."""
        self._embedding_cache.clear()
        self._result_cache.clear()
        logger.info("Retriever caches cleared and resources closed.")
