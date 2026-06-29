"""
Embedding Service Module for Cortex AI.
Provides vector embedding generation for document chunks using Google Gemini Embeddings
via LangChain, complete with hash-based caching, validation, detailed metrics,
batch processing, and backoff retries.
Supports multi-provider architectures via abstract provider dependency injection.
"""

import hashlib
import logging
import time
import random
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document

from core.exceptions import (
    DocumentProcessingException,
    EmbeddingGenerationException,
    EmbeddingTimeoutException,
    EmptyTextException,
    GeminiAPIException,
    InvalidMetadataException,
    InvalidTextException,
    MissingAPIKeyException,
    RateLimitException,
)
from core.providers.base_embedding_provider import EmbeddingProvider
from utils.constants import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_RETRY_DELAY,
    EMBEDDING_TIMEOUT,
    MAX_EMBEDDING_TEXT_LENGTH,
    EMBEDDING_CACHE_LIMIT,
)
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)


@dataclass
class EmbeddedChunk:
    """Dataclass holding a document chunk's text, generated vector embedding, and metadata."""
    chunk_id: str
    text: str
    vector: List[float]
    metadata: Dict[str, Any]


class EmbeddingService:
    """
    Service for generating vector embeddings for text chunks.
    
    Acts as an orchestrator delegating the core embedding calls to an injected EmbeddingProvider,
    implementing validation, caching, and robust metrics tracing around it.
    """

    def __init__(
        self,
        provider: Optional[EmbeddingProvider] = None,
        model_name: Optional[str] = None
    ):
        """
        Initializes the EmbeddingService.

        Args:
            provider (Optional[EmbeddingProvider]): Injected concrete embedding provider. 
                                                    Defaults to GeminiEmbeddingProvider if None.
            model_name (Optional[str]): Backward-compatibility fallback. Overrides provider's model if provided.
        """
        if provider is None:
            from core.providers.gemini_embedding_provider import GeminiEmbeddingProvider
            provider = GeminiEmbeddingProvider(model_name=model_name)
        
        self.provider = provider
        self._cache: Dict[str, List[float]] = {}
        self._stats: Dict[str, Any] = {}
        self.reset_statistics()
        
        logger.info(
            f"EmbeddingService initialized with provider: '{self.provider.provider_name}', "
            f"model: '{self.provider.model_name}'"
        )

    def initialize_model(self) -> None:
        """
        Delegates model client initialization to the provider.
        """
        self.provider.initialize_model()

    def validate_embedding_dimension(self, vector: List[float]) -> bool:
        """
        Delegates vector dimension verification to the active provider.

        Args:
            vector (List[float]): A list of floats representing the embedding vector.

        Returns:
            bool: True if the vector dimension matches expectations.
        """
        return self.provider.validate_embedding_dimension(vector)

    def reset_statistics(self) -> None:
        """Resets all metrics and performance statistics back to zero."""
        self._stats = {
            "total_requests": 0,
            "successful_embeddings": 0,
            "failed_embeddings": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "api_requests": 0,
            "total_api_time_ms": 0.0,
            "total_api_chunks": 0,
            "total_tokens_processed": 0,
            "total_processing_time_ms": 0.0,
        }
        logger.info("Embedding service statistics have been reset.")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieves a copy of the current embedding generation performance statistics.

        Returns:
            Dict[str, Any]: Map of metrics like hits, misses, latency, and estimated token counts.
        """
        stats_copy = self._stats.copy()
        
        # Injected provider details
        stats_copy["provider_name"] = self.provider.provider_name
        stats_copy["embedding_model"] = self.provider.model_name
        
        total_reqs = stats_copy.get("total_requests", 0)
        api_reqs = stats_copy.get("api_requests", 0)
        total_api_time = stats_copy.get("total_api_time_ms", 0.0)
        total_api_chunks = stats_copy.get("total_api_chunks", 0)
        total_proc_time = stats_copy.get("total_processing_time_ms", 0.0)
        cache_hits = stats_copy.get("cache_hits", 0)
        
        # Computed averages
        stats_copy["avg_embedding_time_ms"] = (
            total_api_time / api_reqs if api_reqs > 0 else 0.0
        )
        stats_copy["average_batch_size"] = (
            total_api_chunks / api_reqs if api_reqs > 0 else 0.0
        )
        stats_copy["cache_efficiency_percentage"] = (
            (cache_hits / total_reqs) * 100.0 if total_reqs > 0 else 0.0
        )
        stats_copy["average_request_latency"] = (
            total_proc_time / total_reqs if total_reqs > 0 else 0.0
        )
        
        return stats_copy

    def health_check(self) -> Dict[str, Any]:
        """
        Verifies system health status by delegating diagnostics to the active provider.

        Returns:
            Dict[str, Any]: Diagnostic report dictionary.
        """
        logger.info("Executing embedding service health check...")
        return self.provider.health_check()

    def _validate_text(self, text: Optional[str]) -> str:
        """
        Validates text input for single or batch embedding requests.

        Args:
            text (Optional[str]): Text string to validate.

        Returns:
            str: Stripped version of the valid input text.

        Raises:
            InvalidTextException: If text is None, not a string, or exceeds maximum length.
            EmptyTextException: If text is empty or whitespace-only.
        """
        if text is None:
            raise InvalidTextException("Input text is None. Expected string.")
            
        if not isinstance(text, str):
            raise InvalidTextException(
                f"Input text must be a string, got {type(text).__name__}."
            )
            
        stripped = text.strip()
        if not stripped:
            raise EmptyTextException("Input text is empty or whitespace-only.")

        if len(stripped) > MAX_EMBEDDING_TEXT_LENGTH:
            raise InvalidTextException(
                f"Input text length ({len(stripped)} chars) exceeds maximum allowed "
                f"size of {MAX_EMBEDDING_TEXT_LENGTH} chars."
            )
            
        return stripped

    def _validate_chunk_metadata(self, chunk: Document) -> None:
        """
        Validates that a document chunk has a valid metadata dictionary with required parameters.

        Args:
            chunk (Document): Document chunk to check.

        Raises:
            InvalidMetadataException: If metadata is missing or does not contain required fields.
        """
        if not hasattr(chunk, "metadata") or chunk.metadata is None:
            raise InvalidMetadataException("Chunk is missing metadata dictionary.")

        if not isinstance(chunk.metadata, dict):
            raise InvalidMetadataException("Chunk metadata must be a dictionary.")

        # Ensure critical keys exist
        required_keys = [
            "chunk_id", "document_id", "chunk_index", "source", "page",
            "total_pages", "file_hash", "created_at"
        ]
        
        for key in required_keys:
            if key not in chunk.metadata:
                raise InvalidMetadataException(
                    f"Chunk metadata is missing required key: '{key}'."
                )

        chunk_id = chunk.metadata.get("chunk_id")
        if not chunk_id or not isinstance(chunk_id, str):
            raise InvalidMetadataException("Chunk metadata 'chunk_id' is missing or not a string.")

    def _generate_cache_key(self, text: str) -> str:
        """
        Generates a SHA-256 hash to use as a cache key for the given text.

        Args:
            text (str): The validated text string.

        Returns:
            str: SHA-256 hexadecimal hash key.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _add_to_cache(self, cache_key: str, vector: List[float]) -> None:
        """
        Adds a vector embedding to the in-memory cache, enforcing size limits.

        Args:
            cache_key (str): SHA-256 hash key.
            vector (List[float]): Vector embedding list.
        """
        if cache_key in self._cache:
            self._cache.pop(cache_key)  # Evict to insert at end (LRU order)
        elif len(self._cache) >= EMBEDDING_CACHE_LIMIT:
            # Python dict preserves insertion order; pop the first element (oldest insertion)
            oldest_key = next(iter(self._cache))
            self._cache.pop(oldest_key)
            logger.debug(f"Cache limit ({EMBEDDING_CACHE_LIMIT}) reached. Evicted oldest entry.")
            
        self._cache[cache_key] = vector

    def _retry_with_backoff(
        self,
        func: Any,
        *args: Any,
        max_retries: int = EMBEDDING_MAX_RETRIES,
        initial_delay: float = EMBEDDING_RETRY_DELAY,
        **kwargs: Any
    ) -> Any:
        """
        Executes a callable with exponential backoff retry.

        Args:
            func (Any): The function to execute.
            max_retries (int): Number of retries on failure.
            initial_delay (float): Initial delay in seconds.

        Returns:
            Any: The return value of the function.
        """
        delay = initial_delay
        
        # Calculate chunks count for this request
        if args and isinstance(args[0], list):
            chunks_count = len(args[0])
        else:
            chunks_count = 1

        for attempt in range(max_retries + 1):
            try:
                self._stats["api_requests"] += 1
                start_api_time = time.perf_counter()
                
                result = func(*args, **kwargs)
                
                # Record successful API metrics
                elapsed_api_ms = (time.perf_counter() - start_api_time) * 1000.0
                self._stats["total_api_time_ms"] += elapsed_api_ms
                self._stats["total_api_chunks"] += chunks_count
                return result
            except Exception as e:
                # Decrement api_requests since this attempt failed and did not yield a vector
                self._stats["api_requests"] -= 1
                
                err_msg = str(e).lower()
                is_rate_limit = "429" in err_msg or "rate limit" in err_msg or "quota" in err_msg
                is_timeout = "timeout" in err_msg or "deadline" in err_msg

                if attempt == max_retries:
                    if is_rate_limit:
                        raise RateLimitException(
                            f"Gemini API rate limit exceeded. Attempted {max_retries} retries. Error: {e}"
                        )
                    elif is_timeout:
                        raise EmbeddingTimeoutException(
                            f"Gemini API request timed out. Attempted {max_retries} retries. Error: {e}"
                        )
                    else:
                        raise GeminiAPIException(
                            f"Gemini API error after {max_retries} retries: {e}"
                        )

                sleep_time = delay + random.uniform(0, 0.5)
                logger.warning(
                    f"Embedding generation attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {sleep_time:.2f} seconds..."
                )
                time.sleep(sleep_time)
                delay *= 2

    def embed_text(self, text: str) -> List[float]:
        """
        Generates a vector embedding for a single string.

        Checks the in-memory cache first to avoid redundant API requests.

        Args:
            text (str): String content to be embedded.

        Returns:
            List[float]: The generated vector embedding.
        """
        start_processing = time.perf_counter()
        self._stats["total_requests"] += 1
        
        try:
            # 1. Validate
            validated_text = self._validate_text(text)
            self._stats["total_tokens_processed"] += len(validated_text) // 4
            
            # Generate cache key
            cache_key = self._generate_cache_key(validated_text)
            logger.info("Embedding started for single text chunk.")

            # 2. Check Cache
            if cache_key in self._cache:
                self._stats["cache_hits"] += 1
                self._stats["successful_embeddings"] += 1
                logger.info(f"Cache hit for key hash '{cache_key[:8]}...'.")
                return self._cache[cache_key]

            # Cache Miss
            self._stats["cache_misses"] += 1
            logger.info(f"Cache miss for key hash '{cache_key[:8]}...'. Querying provider API.")

            # 3. Request from Provider
            vector = self._retry_with_backoff(self.provider.embed_text, validated_text)
            
            # 4. Dimension Verification
            if not self.validate_embedding_dimension(vector):
                raise EmbeddingGenerationException(
                    f"Generated embedding dimension ({len(vector)}) does not match expected {self.provider.dimension}."
                )

            # Store in cache
            self._add_to_cache(cache_key, vector)
            self._stats["successful_embeddings"] += 1
            logger.info("Embedding completed successfully.")
            return vector
            
        except Exception:
            self._stats["failed_embeddings"] += 1
            raise
        finally:
            elapsed = (time.perf_counter() - start_processing) * 1000.0
            self._stats["total_processing_time_ms"] += elapsed

    def batch_embed(self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> List[List[float]]:
        """
        Generates embeddings for a batch of text strings.

        Optimizes calls by retrieving cached strings and only invoking the Gemini API 
        for new/uncached strings in grouped batches.

        Args:
            texts (List[str]): List of string contents to embed.
            batch_size (int): Max number of strings processed in a single API call.

        Returns:
            List[List[float]]: List of vector embeddings matching the order of input texts.
        """
        start_processing = time.perf_counter()
        
        if not isinstance(texts, list):
            raise InvalidTextException("Input texts must be a list of strings.")
            
        if not texts:
            raise EmptyTextException("Batch text list is empty.")

        logger.info(f"Batch processing started for {len(texts)} texts. Batch size: {batch_size}")

        results: Dict[int, List[float]] = {}
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        try:
            # 1. Pre-process, Validate, check cache
            for idx, text in enumerate(texts):
                self._stats["total_requests"] += 1
                validated = self._validate_text(text)
                self._stats["total_tokens_processed"] += len(validated) // 4
                cache_key = self._generate_cache_key(validated)

                if cache_key in self._cache:
                    self._stats["cache_hits"] += 1
                    self._stats["successful_embeddings"] += 1
                    results[idx] = self._cache[cache_key]
                else:
                    self._stats["cache_misses"] += 1
                    uncached_indices.append(idx)
                    uncached_texts.append(validated)

            # 2. Query API in batches for uncached texts
            if uncached_texts:
                logger.info(
                    f"Requesting embeddings for {len(uncached_texts)} uncached items "
                    f"in slices of {batch_size}."
                )

                for i in range(0, len(uncached_texts), batch_size):
                    batch_slice = uncached_texts[i : i + batch_size]
                    batch_indices_slice = uncached_indices[i : i + batch_size]
                    
                    # Call embed_documents on the slice through retry handler
                    logger.debug(f"Calling embed_documents for batch slice of size {len(batch_slice)}")
                    batch_vectors = self._retry_with_backoff(self.provider.embed_documents, batch_slice)
                    
                    # Verify vector count
                    if len(batch_vectors) != len(batch_slice):
                        raise EmbeddingGenerationException(
                            f"API returned {len(batch_vectors)} vectors, expected {len(batch_slice)}."
                        )

                    # Store in cache and populate results map
                    for slice_idx, vector in enumerate(batch_vectors):
                        if not self.validate_embedding_dimension(vector):
                            raise EmbeddingGenerationException(
                                f"Generated vector at slice index {slice_idx} has invalid dimensions."
                            )
                        
                        original_idx = batch_indices_slice[slice_idx]
                        text_value = batch_slice[slice_idx]
                        cache_key = self._generate_cache_key(text_value)
                        
                        self._add_to_cache(cache_key, vector)
                        self._stats["successful_embeddings"] += 1
                        results[original_idx] = vector

            # 3. Re-assemble results in original index order
            ordered_vectors: List[List[float]] = [results[i] for i in range(len(texts))]
            logger.info("Batch processing completed successfully.")
            
            # Print stats summary in debug
            stats = self.get_statistics()
            logger.debug(
                f"Stats Summary -> Hits: {stats['cache_hits']}, Misses: {stats['cache_misses']}, "
                f"API Calls: {stats['api_requests']}, Avg API Time: {stats['avg_embedding_time_ms']:.2f}ms"
            )
            
            return ordered_vectors
            
        except Exception:
            self._stats["failed_embeddings"] += len(uncached_texts) - len(results)
            raise
        finally:
            elapsed = (time.perf_counter() - start_processing) * 1000.0
            self._stats["total_processing_time_ms"] += elapsed

    def embed_documents(self, documents: List[Document]) -> List[EmbeddedChunk]:
        """
        Accepts a list of LangChain Document objects and converts them to EmbeddedChunks.

        Validates all metadata parameters and appends embedding-specific attributes.

        Args:
            documents (List[Document]): List of chunked Document objects.

        Returns:
            List[EmbeddedChunk]: A list of EmbeddedChunk objects containing vectors and metadata.
        """
        start_processing = time.perf_counter()
        
        if not documents:
            logger.warning("No documents passed to embed_documents. Returning empty list.")
            return []

        logger.info(f"Starting chunk embedding batch of size: {len(documents)}")

        try:
            # 1. Validate all document chunks and metadata first
            for idx, doc in enumerate(documents):
                if not isinstance(doc, Document):
                    raise InvalidTextException(
                        f"Item at index {idx} is not a LangChain Document object."
                    )
                self._validate_chunk_metadata(doc)

            # 2. Extract text list
            texts = [doc.page_content for doc in documents]
            
            # 3. Batch embed texts (handles caching, batching, stats update internally)
            vectors = self.batch_embed(texts)

            # 4. Map to EmbeddedChunk objects
            embedded_chunks: List[EmbeddedChunk] = []
            created_at_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            
            for idx, doc in enumerate(documents):
                metadata = doc.metadata
                chunk_id = metadata["chunk_id"]
                
                # Format clean preserved metadata dictionary
                preserved_metadata = {
                    "document_id": metadata["document_id"],
                    "chunk_id": chunk_id,
                    "chunk_index": metadata["chunk_index"],
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "total_pages": metadata["total_pages"],
                    "file_hash": metadata["file_hash"],
                    "created_at": metadata["created_at"],
                }

                # Inject additional embedding-specific metadata parameters
                enhanced_metadata = {
                    **preserved_metadata,
                    "embedding_model": self.provider.model_name,
                    "embedding_provider": self.provider.provider_name,
                    "embedding_dimension": self.provider.dimension,
                    "embedding_version": "1.0.0",
                    "generated_at": created_at_str
                }

                embedded_chunks.append(
                    EmbeddedChunk(
                        chunk_id=chunk_id,
                        text=doc.page_content,
                        vector=vectors[idx],
                        metadata=enhanced_metadata,
                    )
                )

            logger.info(f"Successfully processed and embedded {len(embedded_chunks)} document chunk(s).")
            return embedded_chunks
            
        finally:
            elapsed = (time.perf_counter() - start_processing) * 1000.0
            self._stats["total_processing_time_ms"] += elapsed
