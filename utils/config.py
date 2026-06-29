import os
from pathlib import Path
from dotenv import load_dotenv

# Define project root directory path
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# Core directories configuration
CHROMA_DB_DIR: Path = PROJECT_ROOT / "chroma_db"
PDFS_DIR: Path = PROJECT_ROOT / "pdfs"
EXPORTS_DIR: Path = PROJECT_ROOT / "exports"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

# Ensure all critical folders are created at runtime
for directory in [CHROMA_DB_DIR, PDFS_DIR, EXPORTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Google Gemini settings
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# Default Model configurations (centralized)
# Using Gemini 1.5 Flash for fast/cost-effective RAG pipelines
DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gemini-1.5-flash")
# Google GenAI Embeddings model identifier
DEFAULT_EMBEDDINGS_MODEL: str = os.getenv("DEFAULT_EMBEDDINGS_MODEL", "models/text-embedding-004")

# UI and Styling Constants
APP_TITLE: str = "🧠 Cortex AI"
APP_SUBTITLE: str = "Intelligent Knowledge Assistant"

def validate_config() -> bool:
    """
    Validates that essential configuration variables are present.
    Returns:
        bool: True if configuration is valid.
    Raises:
        ValueError: If a required configuration (like GOOGLE_API_KEY) is missing.
    """
    if not GOOGLE_API_KEY:
        raise ValueError(
            "Missing GOOGLE_API_KEY environment variable. "
            "Please create a local .env file and set GOOGLE_API_KEY."
        )
    return True
