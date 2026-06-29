# Comprehensive System Architecture

This document describes the architectural flow, component relationships, and software engineering principles of the Cortex AI RAG system.

---

## 1. End-to-End System Diagrams

Cortex AI partitions tasks into two main workflows: **Document Ingestion** and **Conversational Query Retrieval (RAG)**.

### Ingestion Flow
```mermaid
graph TD
    PDF[PDF File Upload] -->|1. Stream uploaded bytes| DP[DocumentProcessor]
    DP -->|2. Validate extension, size, and emptiness| VAL{IsValid?}
    VAL -->|No| ERR[Raise Custom Ingestion Exception]
    VAL -->|Yes| HASH[Calculate SHA-256 Hash]
    HASH --> DUP{Hash in Registry?}
    DUP -->|Yes| SKIP[Skip Duplicate File]
    DUP -->|No| SAVE[Save to disk in pdfs/ & Suffix collision names]
    SAVE --> LOAD[Load PDF Text page-by-page]
    LOAD --> SPLIT[Split into Chunks using Recursive Splitter]
    SPLIT --> EMB[EmbeddingService]
    EMB -->|3. Query Gemini Embeddings| API[models/text-embedding-004]
    API -->|4. Return 768-dim Vector| EMB
    EMB -->|5. Convert to EmbeddedChunk| REPO[VectorRepository]
    REPO -->|6. Store vectors & metadata| DB[(ChromaDB)]
```

### Query and Retrieval Flow
```mermaid
graph TD
    User([User Question]) -->|1. Submit text| UI[Streamlit Page]
    UI -->|2. call ask| Pipe[CortexRAGPipeline]
    
    %% Session Resolution
    Pipe -->|3. Resolve active session| SES[RAGSession]
    SES -->|4. Fetch past turns| Pipe
    
    %% Semantic retrieval
    Pipe -->|5. retrieve| Ret[SemanticRetriever]
    Ret -->|6. embed query| ES[EmbeddingService]
    ES -->|7. Generate query vector| API[models/text-embedding-004]
    API -->|8. Return query vector| ES
    ES -->|9. Query matching candidates| Repo[VectorRepository]
    Repo -->|10. Cosine/MMR search| DB[(ChromaDB)]
    DB -->|11. Return retrieved chunks| Repo
    Repo -->|12. Return list of EmbeddedChunks| Ret
    Ret -->|13. Perform MMR diversity re-ranking| Pipe
    
    %% Prompt Compilation
    Pipe -->|14. build_prompt| PB[RAGPromptBuilder]
    PB -->|15. Compress context & inject citations| Pipe
    
    %% Inference
    Pipe -->|16. generate response| LLM[GeminiLLM]
    LLM -->|17. Execute inference request| LLM_API[gemini-1.5-flash]
    LLM_API -->|18. Return generated text| LLM
    LLM -->|19. Parse citations| Pipe
    
    %% Memory updates and Return
    Pipe -->|20. Save turns| SES
    Pipe -->|21. Return RAGResponse| UI
    UI -->|22. Render markdown & citation cards| User
```

---

## 2. Decoupled Service Layers

* **Presentation Layer (`ui/`)**: Reusable UI blocks and multi-page configurations using Streamlit.
* **Orchestration Layer (`core/rag/`)**: Coordinates conversational state context and coordinates pipelines (`CortexRAGPipeline`, `RAGSession`).
* **Domain Layer (`core/`)**:
  * **`EmbeddingService`**: Embeds text using pluggable providers and caches vector calculations.
  * **`SemanticRetriever`**: Implements MMR and Similarity algorithms.
  * **`RAGPromptBuilder`**: Templates instructions and enforces token-character budgets.
  * **`GeminiLLM`**: Wrapper around Google Gemini API featuring exponential backoff.
  * **`DocumentProcessor`**: Orchestrates PDF loading, page sanitization, validation, hashing, and character chunking.
* **Data Storage Layer (`core/vector_store/` & `core/repository/`)**:
  * **`VectorRepository`**: Manages writes, updates, rollback states, collection versioning, and statistics calculations.
  * **`ChromaVectorStore`**: Adapter for persistent ChromaDB client transactions.
