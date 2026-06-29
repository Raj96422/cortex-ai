# Changelog

All notable changes to the Cortex AI project are documented in this file.

---

## [v1.0.0] - 2026-06-29

Cortex AI v1.0.0 represents the first production-ready release of the Retrieval-Augmented Generation (RAG) knowledge assistant.

### Completed Modules Summary

#### Module 3: Document Ingest
- Integrated `PyPDF` loader to parse PDF files.
- Implemented `RecursiveCharacterTextSplitter` chunking, generating sequential document-relative indices.
- Configured registry storage tracking processed file hashes to prevent duplicates.

#### Module 4: Embedding Service
- Integrated Google Gemini text embeddings model (`models/text-embedding-004`).
- Implemented SHA-256 hash-based in-memory cache to prevent redundant API queries.
- Added abstract base providers allowing future embedding client extensions.

#### Module 5: Vector Repository
- Configured persistent local ChromaDB client storage.
- Created `VectorRepository` to handle transactional database interactions, collection creation, metadata updates, and diagnostic statistics.

#### Module 6: Semantic Retriever
- Implemented abstract retrieval interface and concrete `SemanticRetriever`.
- Implemented Maximum Marginal Relevance (MMR) and Similarity search strategies.
- Added query caching, threshold filters, and retrieval statistics.

#### Module 7: Prompt Engineering
- Created `RAGPromptBuilder` using budget managers to compress contexts under token boundaries.
- Formatted QA prompt templates with inline source citation indices (`[Source: file.pdf, Page X]`).

#### Module 8: LLM Service
- Created `GeminiLLM` wrapping inference with Google Gemini (`gemini-1.5-flash`).
- Added response parsing support returning structured answer text and citation arrays.
- Implemented exponential backoff retry algorithms to handle network and quota limits.

#### Module 9: RAG Pipeline Orchestrator
- Created `CortexRAGPipeline` coordinating modules 3 through 8.
- Added turn-based chat session memory (`RAGSession`).
- Integrated request-traced logging using UUIDv4 values.

#### Module 10: Streamlit Frontend
- Built a multi-page web application (Home, Chat, Documents, Analytics, Settings, About).
- Styled panels with custom glassmorphism cards and dark slate variables.
- Added drag-and-drop document upload pipelines and performance dashboards.

#### Module 11: Production Polish
- Added Docker support (`Dockerfile`, `docker-compose.yml`, `.dockerignore`).
- Configured pre-commit hooks and style rules (`pyproject.toml`, `.pre-commit-config.yaml`).
- Configured automated GitHub Actions workflows (`ci.yml`).
- Implemented an automated project validation verification script (`scripts/validate_project.py`).
