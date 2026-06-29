"""
Cortex AI - Automated Project Validation Verification Script.
Checks workspace layout, file structures, environment keys, packages imports,
and runs live orchestrator health checks.
"""

import os
import sys
from pathlib import Path

# Fix Windows cp1252 encoding crashes on emojis
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# Ensure workspace root is in path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from core.rag.rag_factory import RAGFactory


def validate_project() -> bool:
    print("=" * 60)
    print("🧠 CORTEX AI - AUTOMATED PROJECT VALIDATION REPORT")
    print("=" * 60)

    success = True

    # 1. Check Required Folders
    required_folders = [
        "core", "core/vector_store", "core/repository", "core/retriever",
        "core/prompt", "core/llm", "core/rag", "ui", "ui/pages",
        "ui/components", "ui/styles", "tests", "docs", "scripts"
    ]
    print("\n[1] Checking Directory Layout:")
    for folder in required_folders:
        path = root / folder
        if path.exists() and path.is_dir():
            print(f"  🟢 {folder}/ found.")
        else:
            print(f"  🔴 {folder}/ missing!")
            success = False

    # 2. Check Required Config/Main files
    required_files = [
        "VERSION", "README.md", "Dockerfile", "docker-compose.yml",
        "pyproject.toml", ".pre-commit-config.yaml", "ui/app.py",
        "ui/styles/custom.css", "ui/components/ui_components.py",
        "ui/pages/1_💬_Chat.py", "ui/pages/2_📂_Documents.py"
    ]
    print("\n[2] Checking Required Config & UI Files:")
    for file in required_files:
        path = root / file
        if path.exists() and path.is_file():
            print(f"  🟢 {file} found.")
        else:
            print(f"  🔴 {file} missing!")
            success = False

    # 3. Check Environment Variables
    print("\n[3] Checking Environment Keys:")
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        masked = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "***"
        print(f"  🟢 GEMINI_API_KEY is configured in env: {masked}")
    else:
        print("  ⚠️  GEMINI_API_KEY is not set in current shell environment.")
        dotenv_path = root / ".env"
        if dotenv_path.exists():
            print("  🟢 .env file found at root. Checking keys...")
            with open(dotenv_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            has_key = any(line.strip().startswith("GEMINI_API_KEY") for line in lines)
            if has_key:
                print("  🟢 GEMINI_API_KEY found inside .env file.")
            else:
                print("  🔴 GEMINI_API_KEY missing from .env file!")
        else:
            print("  🔴 No .env file found at root workspace!")

    # 4. Check Package Imports
    print("\n[4] Checking Package Imports:")
    try:
        from core.document_processor import process_uploaded_files
        from core.embeddings import EmbeddingService
        from core.vector_store.vector_store_factory import VectorStoreFactory
        from core.repository.vector_repository import VectorRepository
        from core.retriever.semantic_retriever import SemanticRetriever
        from core.prompt.rag_prompt_builder import RAGPromptBuilder
        from core.llm.gemini_llm import GeminiLLM
        from core.rag.rag_pipeline import CortexRAGPipeline
        
        print("  🟢 Core architectural modules imported successfully.")
    except Exception as e:
        print(f"  🔴 Import failure: {e}")
        success = False

    # 5. Live Orchestrator Diagnostics
    print("\n[5] Executing Live Orchestrator Diagnostics:")
    try:
        pipeline = RAGFactory.get_pipeline()
        health = pipeline.health_check()
        
        print(f"  System Health status: {health.get('status').upper()}")
        deps = health.get("dependencies", {})
        for dep, details in deps.items():
            status = details.get("status", "unknown").upper()
            if status == "HEALTHY":
                print(f"    🟢 {dep}: {status}")
            else:
                print(f"    🔴 {dep}: {status} (Error: {details.get('error', 'none')})")
    except Exception as e:
        print(f"  🔴 Live health check diagnostics failed to run: {e}")
        success = False

    print("\n" + "=" * 60)
    if success:
        print("🎉 CORTEX AI v1.0.0 VALIDATION: SUCCESSFUL")
    else:
        print("❌ CORTEX AI v1.0.0 VALIDATION: FAILED (Check errors above)")
    print("=" * 60)

    return success


if __name__ == "__main__":
    sys.exit(0 if validate_project() else 1)
