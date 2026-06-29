# Performance Tuning and Optimization Guide

This document describes latency profiles, memory footprints, and parameters configurations inside the Cortex AI pipeline.

---

## 1. Chunk Sizing Trade-Offs

Document splitting is configured via `core/chunker.py` and `ui/pages/2_📂_Documents.py`.
* **Default Settings**: `chunk_size = 500` characters, `chunk_overlap = 50` characters.
* **Trade-Off**:
  * **Smaller Chunks (e.g. 200-500)**: Yield higher semantic precision, retrieve highly relevant sentences, but can lose macro-document context.
  * **Larger Chunks (e.g. 1000-2000)**: Preserve cohesive document context but increase token consumption, increase prompt latency, and risk diluting specific details.

---

## 2. Embedding Caching

Generating embeddings requires making network requests to Google AI Studio.
* **SHA-256 Caching**: The `EmbeddingService` checks an in-memory dictionary mapped to the SHA-256 hash of the input text chunk.
* **Impact**: During chunk re-indexing or repeated queries, cache hit ratios approach **100%**, reducing latency from ~300ms down to `< 0.1ms` and completely eliminating API costs.

---

## 3. Retriever Optimization (MMR vs Similarity)

* **Similarity Search**: Retrieves the closest Top-K documents in vector cosine space. Fast, but can return redundant chunks if a document repeats the same concepts.
* **Max Marginal Relevance (MMR)**: Balances similarity to the query with diversity among the retrieved chunks.
  * **Lambda factor ($\lambda$)**: Set between `0` and `1`.
    * **$\lambda = 1$**: Identical to similarity search.
    * **$\lambda = 0.5$ (Default)**: Balanced search.
    * **$\lambda = 0$**: Prioritizes diversity above query matching.

---

## 4. Prompt Engineering Context Budgets

To prevent prompt overflow issues when dealing with large contexts:
* **Context Characters Budget**: `RAGPromptBuilder` enforces a max budget limit of `5000` context characters.
* **Truncation**: If retrieved chunks exceed this budget, they are compressed sequentially.
* **Citations**: Compiles inline source citations using index maps to keep LLM outputs highly readable.

---

## 5. ChromaDB Database Persistence

ChromaDB is configured in persistent mode (`core/vector_store/chroma_vector_store.py`).
* **Disk IO**: Operations write vector indices directly to `chroma_db/` folder on disk.
* **Optimization**: Batch writes are compiled and executed in a single transaction rather than making individual file-system write queries, which speeds up indexing.
