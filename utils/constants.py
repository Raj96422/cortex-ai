from typing import Final

# Centralized App Details
APP_NAME: Final[str] = "🧠 Cortex AI"
APP_SUBTITLE: Final[str] = "Intelligent Knowledge Assistant"
APP_VERSION: Final[str] = "1.0.0"

# PDF Validation Limits
ALLOWED_EXTENSIONS: Final[set[str]] = {".pdf"}
MAX_FILE_SIZE_MB: Final[int] = 50  # Limit single file size to 50MB for memory efficiency
MAX_TOTAL_FILES: Final[int] = 10   # Limit parallel processing to 10 files

# RAG / Text Splitter Parameters
DEFAULT_CHUNK_SIZE: Final[int] = 1000
DEFAULT_CHUNK_OVERLAP: Final[int] = 200

# Vector Database Settings
CHROMA_COLLECTION_NAME: Final[str] = "cortex_knowledge_base"

# Retrieval Settings
DEFAULT_RETRIEVAL_K: Final[int] = 4  # Retrieve top 4 most relevant chunks

# LLM Parameter Settings
DEFAULT_LLM_TEMPERATURE: Final[float] = 0.2  # Keep temperature low to prevent hallucination

# System Prompt Templates
RAG_SYSTEM_PROMPT: Final[str] = (
    "You are Cortex AI, a highly precise Intelligent Knowledge Assistant. "
    "Your objective is to answer user questions using only the retrieved source document text. "
    "Follow these strict directives:\n"
    "1. Rely ONLY on the clear facts provided in the Context section below.\n"
    "2. If the context does not contain enough information to answer the question, state politely "
    "that you do not know the answer based on the provided documents. Do not attempt to make up "
    "or extrapolate any information.\n"
    "3. Keep your answers concise, well-structured, and factual.\n"
    "4. Cite the exact file names and page numbers in your answer when referencing facts.\n\n"
    "Context:\n{context}"
)

# Export Config
DEFAULT_EXPORT_FILENAME: Final[str] = "cortex_chat_history.md"

# Embedding Service Configuration Parameters
EMBEDDING_BATCH_SIZE: Final[int] = 16
EMBEDDING_MAX_RETRIES: Final[int] = 3
EMBEDDING_RETRY_DELAY: Final[float] = 1.0
EMBEDDING_TIMEOUT: Final[float] = 30.0
MAX_EMBEDDING_TEXT_LENGTH: Final[int] = 10000       # Maximum character length per text chunk
EMBEDDING_CACHE_LIMIT: Final[int] = 10000           # Cap to prevent memory leaks from excessive entries
