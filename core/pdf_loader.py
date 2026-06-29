"""
PDF Loader Module for Cortex AI.
Responsible for loading single or multiple PDF files, extracting text page-by-page,
handling encrypted or corrupted PDFs, skipping blank pages, and returning LangChain 
Document objects with comprehensive metadata.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union
from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from core.exceptions import (
    CorruptedPDFException,
    DocumentProcessingException,
    EmptyPDFException,
    InvalidPDFException,
)
from utils.helpers import calculate_file_hash
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)


def _load_single_pdf(file_path: Path, file_hash: Optional[str] = None) -> List[Document]:
    """
    Loads, validates, and extracts text page-by-page from a single PDF file.

    Args:
        file_path (Path): Path to the PDF file on disk.
        file_hash (Optional[str]): Precalculated SHA-256 hash of the PDF. If not provided,
                                   it will be calculated.

    Returns:
        List[Document]: List of LangChain Document objects representing the non-blank pages.

    Raises:
        InvalidPDFException: If the file does not exist, is encrypted and cannot be decrypted,
                             or is not a file.
        EmptyPDFException: If the file is 0 bytes, has 0 pages, or contains no readable text.
        CorruptedPDFException: If the PDF structure is corrupted.
        DocumentProcessingException: For general system or unexpected errors.
    """
    if not file_path.exists():
        raise InvalidPDFException(f"PDF file does not exist at '{file_path}'")
    if not file_path.is_file():
        raise InvalidPDFException(f"Path is not a valid file: '{file_path}'")

    # 1. Check size and calculate hash if not provided
    try:
        file_size = file_path.stat().st_size
        if file_size == 0:
            raise EmptyPDFException(f"PDF file '{file_path.name}' is empty (0 bytes).")

        if not file_hash:
            logger.debug(f"Calculating SHA-256 hash for '{file_path.name}'")
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            file_hash = calculate_file_hash(file_bytes)
    except EmptyPDFException:
        raise
    except Exception as e:
        raise DocumentProcessingException(f"Failed to read properties of '{file_path.name}': {e}")

    # 2. Extract pages using pypdf
    try:
        logger.info(f"Loading PDF: '{file_path.name}'")
        reader = PdfReader(file_path)

        # Handle Encrypted PDFs
        if reader.is_encrypted:
            logger.warning(f"PDF '{file_path.name}' is encrypted. Attempting decryption with empty password.")
            try:
                reader.decrypt("")
            except Exception as decrypt_err:
                raise InvalidPDFException(
                    f"Failed to decrypt PDF '{file_path.name}' using empty password: {decrypt_err}"
                )
            if reader.is_encrypted:
                raise InvalidPDFException(
                    f"PDF '{file_path.name}' remains encrypted. Password is required to decrypt and load."
                )
            logger.info(f"Successfully decrypted PDF '{file_path.name}' with empty password.")

        total_pages = len(reader.pages)
        if total_pages == 0:
            raise EmptyPDFException(f"PDF file '{file_path.name}' has 0 pages.")

        documents: List[Document] = []
        created_at_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # Extract text page-by-page
        for page_idx in range(total_pages):
            try:
                page = reader.pages[page_idx]
                text = page.extract_text()

                # Skip blank pages
                if not text or not text.strip():
                    logger.debug(f"Skipping blank page {page_idx + 1} of '{file_path.name}'")
                    continue

                # Construct precise metadata matching requirements
                metadata = {
                    "document_id": file_hash,                # ID based on content hash
                    "source": str(file_path.resolve()),       # Absolute path
                    "filename": file_path.name,               # Base filename
                    "page": page_idx,                         # 0-indexed page number
                    "total_pages": total_pages,               # Total page count of source
                    "file_hash": file_hash,                   # File content hash
                    "file_size": file_size,                   # File size in bytes
                    "content_type": "application/pdf",        # Format type
                    "created_at": created_at_str              # Processed timestamp
                }

                documents.append(Document(page_content=text, metadata=metadata))
            except Exception as page_err:
                logger.warning(
                    f"Error extracting text from page {page_idx + 1} of '{file_path.name}': {page_err}"
                )

        if not documents:
            raise EmptyPDFException(f"PDF file '{file_path.name}' contains no readable text.")

        logger.info(f"Successfully loaded {len(documents)} page(s) from '{file_path.name}'")
        return documents

    except PdfReadError as pdf_err:
        raise CorruptedPDFException(f"Corrupted or invalid PDF format in '{file_path.name}': {pdf_err}")
    except (InvalidPDFException, EmptyPDFException):
        raise
    except Exception as e:
        raise DocumentProcessingException(f"Unexpected error loading PDF '{file_path.name}': {e}")


def load_pdf(file_paths: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
    """
    Loads one or multiple PDF files and extracts non-blank pages into LangChain Document objects.

    Args:
        file_paths (Union[str, Path, List[Union[str, Path]]]): A single path or list of paths.

    Returns:
        List[Document]: Combined list of LangChain Document objects from all loaded files.
        
    Raises:
        InvalidPDFException: If input validation fails or a file cannot be decrypted.
        CorruptedPDFException: If any PDF file is corrupted.
        EmptyPDFException: If any PDF file contains no readable text.
        DocumentProcessingException: If processing fails due to other system issues.
    """
    paths_to_process: List[Path] = []
    if isinstance(file_paths, list):
        for p in file_paths:
            paths_to_process.append(Path(p))
    else:
        paths_to_process.append(Path(file_paths))

    all_documents: List[Document] = []

    for path in paths_to_process:
        docs = _load_single_pdf(path)
        all_documents.extend(docs)

    return all_documents
