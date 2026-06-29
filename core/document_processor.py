"""
Document Processor Module for Cortex AI.
Coordinates the document ingestion flow: validates files, computes hashes,
detects duplicates, saves to disk, and runs the parsing and chunking pipeline.
Handles all custom pipeline exceptions gracefully.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Union
from langchain_core.documents import Document

from core.exceptions import (
    CorruptedPDFException,
    DocumentProcessingException,
    DuplicateDocumentException,
    EmptyPDFException,
    InvalidPDFException,
)
from core.pdf_loader import load_pdf
from core.chunker import chunk_documents
from utils.config import PDFS_DIR
from utils.helpers import calculate_file_hash, validate_file_extension
from utils.constants import MAX_FILE_SIZE_MB
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)

# Registry path to persist metadata and hashes of processed files
REGISTRY_PATH: Path = PDFS_DIR / "processed_files.json"


def _load_registry() -> Dict[str, Any]:
    """
    Loads the processed files registry from the local JSON file.

    Returns:
        Dict[str, Any]: Dictionary mapping SHA-256 hashes to file information.
    """
    if not REGISTRY_PATH.exists():
        logger.debug("Registry file does not exist. Initializing empty registry.")
        return {}
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            registry = json.load(f)
            logger.debug(f"Loaded {len(registry)} entry/entries from registry.")
            return registry
    except Exception as e:
        logger.error(f"Failed to load processed files registry: {e}")
        return {}


def _save_registry(registry: Dict[str, Any]) -> None:
    """
    Saves the updated processed files registry to the local JSON file.

    Args:
        registry (Dict[str, Any]): Dictionary mapping SHA-256 hashes to file information.
    """
    try:
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=4, ensure_ascii=False)
        logger.debug("Registry updated and saved successfully.")
    except Exception as e:
        logger.error(f"Failed to write processed files registry: {e}")


def process_uploaded_files(uploaded_files: List[Any]) -> List[Document]:
    """
    Validates, deduplicates, saves, parses, and chunks uploaded PDF files.

    Maintains the processed files registry, raising and handling custom exceptions
    on a per-file basis to ensure the ingestion of other valid files is not interrupted.

    Args:
        uploaded_files (List[Any]): A list of file-like objects (e.g., Streamlit 
                                    UploadedFile) containing PDF documents.

    Returns:
        List[Document]: Combined list of chunked Document objects from all newly 
                        processed PDF files.
    """
    all_new_chunks: List[Document] = []
    
    if not uploaded_files:
        logger.info("No files provided for processing.")
        return all_new_chunks

    # Ensure PDFs folder exists
    PDFS_DIR.mkdir(parents=True, exist_ok=True)

    # Load registry of already processed files
    registry = _load_registry()
    registry_updated = False

    logger.info(f"Received {len(uploaded_files)} file(s) for ingestion.")

    for uploaded_file in uploaded_files:
        filename = getattr(uploaded_file, "name", "unknown_document.pdf")
        
        try:
            logger.info(f"Starting ingestion process for file: '{filename}'")

            # 1. Validate File Extension
            if not validate_file_extension(filename):
                raise InvalidPDFException(
                    f"Invalid file extension for '{filename}'. Only PDF files (.pdf) are allowed."
                )

            # Read file bytes for hash calculation and saving
            file_bytes = uploaded_file.read()
            
            # Reset file pointer if the object supports seek (critical for Streamlit file uploads)
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
                
            # 2. Validate Empty File
            if not file_bytes or len(file_bytes) == 0:
                raise EmptyPDFException(f"Uploaded file '{filename}' is empty (0 bytes).")

            # 3. Validate File Size
            max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
            file_size_bytes = len(file_bytes)
            if file_size_bytes > max_size_bytes:
                raise InvalidPDFException(
                    f"File '{filename}' size ({file_size_bytes / (1024*1024):.2f} MB) "
                    f"exceeds the maximum allowed size of {MAX_FILE_SIZE_MB} MB."
                )

            # 4. Generate SHA-256 Hash
            file_hash = calculate_file_hash(file_bytes)
            logger.info(f"File '{filename}' SHA-256 hash generated: {file_hash}")

            # 5. Check for Duplicate Files (by Hash)
            if file_hash in registry:
                existing_info = registry[file_hash]
                raise DuplicateDocumentException(
                    f"Duplicate document detected: '{filename}' has the identical content hash as "
                    f"previously processed file '{existing_info.get('filename')}' (processed at {existing_info.get('processed_at')})."
                )

            # 6. Resolve filename collisions and determine target path
            target_path = PDFS_DIR / filename
            if target_path.exists():
                # Avoid filename conflicts by suffixing with first 8 characters of the SHA-256 hash
                stem = target_path.stem
                suffix = target_path.suffix
                target_path = PDFS_DIR / f"{stem}_{file_hash[:8]}{suffix}"
                logger.info(f"Filename collision detected. Saving to alternative path: {target_path.name}")

            # 7. Save uploaded PDF to the local pdfs folder
            logger.info(f"Saving '{filename}' to disk at: {target_path}")
            with open(target_path, "wb") as f:
                f.write(file_bytes)

            # 8. Call PDF Loader
            logger.info(f"Parsing pages from '{target_path.name}'...")
            try:
                loaded_docs = load_pdf(target_path)
            except (InvalidPDFException, EmptyPDFException, CorruptedPDFException, DocumentProcessingException):
                # Clean up saved file if parsing failed to keep system clean
                if target_path.exists():
                    target_path.unlink()
                    logger.info(f"Deleted saved file '{target_path.name}' due to parsing failure.")
                raise

            # 9. Call Chunker
            logger.info(f"Chunking document pages for '{target_path.name}'...")
            chunks = chunk_documents(loaded_docs)
            
            if chunks:
                all_new_chunks.extend(chunks)
                
                # 10. Record in Hash Registry
                registry[file_hash] = {
                    "filename": filename,
                    "saved_filename": target_path.name,
                    "saved_path": str(target_path.resolve()),
                    "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "num_pages": len(loaded_docs),
                    "num_chunks": len(chunks)
                }
                registry_updated = True
                logger.info(f"Successfully processed '{filename}' into {len(chunks)} chunk(s).")
            else:
                raise EmptyPDFException(f"No text chunks were generated for '{filename}'.")

        # Handle all custom exceptions gracefully
        except InvalidPDFException as e:
            logger.error(f"Invalid PDF Error: {e.message}")
        except CorruptedPDFException as e:
            logger.error(f"Corrupted PDF Error: {e.message}")
        except EmptyPDFException as e:
            logger.error(f"Empty PDF Error: {e.message}")
        except DuplicateDocumentException as e:
            logger.warning(f"Duplicate Skip: {e.message}")
        except DocumentProcessingException as e:
            logger.error(f"Document Ingestion Error: {e.message}")
        except Exception as e:
            logger.error(f"Unexpected error processing '{filename}': {e}", exc_info=True)

    # Save registry changes if any new files were successfully processed
    if registry_updated:
        _save_registry(registry)

    logger.info(f"Completed ingestion processing. Generated {len(all_new_chunks)} total new chunk(s).")
    return all_new_chunks
