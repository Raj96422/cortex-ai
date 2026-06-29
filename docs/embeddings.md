# Embedding Service Documentation

This document explains vector embeddings, their role in Retrieval-Augmented Generation (RAG) architectures, and the design of the Cortex AI Embedding Service.

---

## 1. What are Embeddings?

A **vector embedding** is a numerical representation of a piece of data (in our case, a text chunk) in a high-dimensional space. Unlike raw text or keywords, embeddings capture the **semantic meaning** and context of words and sentences.

In the Google Gemini `models/text-embedding-004` model used by this service:
- Each text segment is mapped to a vector containing **768 floating-point numbers**.
- Words and phrases with similar meanings (e.g., "annual earnings" and "financial profits") are placed close to each other in this high-dimensional vector space.

---

## 2. Why are Embeddings Required in RAG?

In RAG, when a user asks a question, the system must search the knowledge base for relevant passages to provide context to the LLM. 

Traditional keyword search fails when synonyms are used or when semantic intent differs from syntactic words. Vector embeddings solve this by enabling **semantic search**. We generate vector embeddings for both:
1. All document chunks in the database.
2. The user's query.

We then calculate the distance (e.g., Cosine Similarity) between the query vector and the document vectors to retrieve the top $K$ most semantically relevant passages.

---

## 3. Why are Embeddings Generated before ChromaDB?

ChromaDB is a **vector database**. Separating the **Embedding Service** from the **Vector Database Layer** offers several architectural benefits:
- **Separation of Concerns**: The embedding generation phase is a computational operation requiring network calls, whereas ChromaDB is a storage and indexing layer.
- **Efficiency**: By generating embeddings independently, we apply custom in-memory caching and batch throttling. This prevents redundant API requests.
- **Flexibility**: We can swap vector databases or embedding models without rewriting the other.

---

## 4. Pipeline Data Flow

```mermaid
graph TD
    subgraph Module 3: Ingestion
        A[Uploaded PDFs] --> B[PDF Loader]
        B -->|Parsed Pages| C[Chunker]
        C -->|Document Chunks| D[Preserved Metadata]
    end

    subgraph Module 4: Embedding Service Refactored
        D -->|Document Chunks| E[Validation Layer]
        E -->|Validate Text & Metadata| F[Cache Check: SHA-256 Hash]
        F -->|Cache Hit| G[Resolve Cached Vector]
        F -->|Cache Miss| H[Gemini API via Exponential Backoff]
        H -->|Verify Dimensions 768| I[Save to Cache with FIFO Eviction]
        G --> J[EmbeddedChunk Dataclass]
        I --> J
    end

    subgraph Module 5: Storage (Future)
        J -->|Prepared IDs, Vectors, Metadata| K[(ChromaDB Vector Store)]
    end
```

---

## 5. Embedding Lifecycle & Workflow

The Embedding Service processes each text segment through a strict lifecycle:

### A. Validation Layer
Before any API or cache operations occur, the service performs validation:
* **Text Validation**: Rejects `None` inputs, non-string types (`InvalidTextException`), empty/whitespace strings (`EmptyTextException`), and strings exceeding `MAX_EMBEDDING_TEXT_LENGTH` (10,000 characters).
* **Metadata Validation**: Ensures the chunk contains a valid metadata dictionary with critical tracking keys: `chunk_id`, `document_id`, `chunk_index`, `source`, `page`, `total_pages`, `file_hash`, and `created_at`. If missing, raises `InvalidMetadataException`.

### B. Hash-Based Cache Workflow
Instead of raw text keys, cache keys are calculated using the **SHA-256 hash** of the text. This improves memory performance and lookup efficiency:
1. The text is stripped and validated.
2. A SHA-256 hash string (64 characters) is generated as the cache key.
3. If the hash exists in the cache dictionary, it retrieves the vector immediately (Cache Hit).
4. If it is a cache miss:
   * The text is embedded via the API.
   * If the cache size exceeds `EMBEDDING_CACHE_LIMIT` (10,000 entries), the oldest inserted entry is evicted (First-In, First-Out) to prevent memory leaks.
   * The new vector is added to the cache.

### C. Retry Flow with Exponential Backoff
Requests to Google's API are wrapped in a retry handler:
* If the API fails with **Rate Limits (HTTP 429)** or **Timeout/Deadlines**, the handler logs a warning and sleeps.
* The sleep duration is determined using exponential backoff: $Delay = InitialDelay \times 2^{Attempt} + RandomJitter$.
* It retries up to `EMBEDDING_MAX_RETRIES` (3 times). If it still fails, it raises `RateLimitException` or `EmbeddingTimeoutException`. Other API errors raise `GeminiAPIException`.

---

## 6. Statistics & Performance Tracking

The service aggregates operation metrics in an internal collector. Statistics can be retrieved using `get_statistics()` and reset using `reset_statistics()`.

Tracked metrics include:
* **`total_requests`**: Total number of text chunk embedding requests processed.
* **`successful_embeddings`**: Number of vectors successfully resolved (from cache or API).
* **`failed_embeddings`**: Number of requests that threw an exception.
* **`cache_hits`**: Count of items resolved from the local hash-cache.
* **`cache_misses`**: Count of items requiring API generation.
* **`api_requests`**: Total number of actual calls dispatched to the Gemini API.
* **`avg_embedding_time_ms`**: Average call duration for successful API calls.
* **`total_processing_time_ms`**: Total latency in milliseconds spent inside the service.
