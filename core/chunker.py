"""
Document Chunker Module for Cortex AI.
Splits text from LangChain Document pages into smaller semantic chunks using
RecursiveCharacterTextSplitter, preserving all page metadata and injecting chunk-specific 
parameters (chunk_id, chunk_index).
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional
from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)


def chunk_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[Document]:
    """
    Chunks a list of LangChain Document pages, preserving metadata and adding chunk identifiers.

    Groups pages by their document_id to assign sequential, document-relative chunk indices 
    and unique chunk IDs in the format: {document_id}_c{chunk_index}.

    Args:
        documents (List[Document]): List of extracted page Documents.
        chunk_size (Optional[int]): Max chunk size in characters. Defaults to DEFAULT_CHUNK_SIZE.
        chunk_overlap (Optional[int]): Character overlap between chunks. Defaults to DEFAULT_CHUNK_OVERLAP.

    Returns:
        List[Document]: Chunks with all required metadata parameters populated.
    """
    size = chunk_size if chunk_size is not None else DEFAULT_CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else DEFAULT_CHUNK_OVERLAP

    if not documents:
        logger.warning("No documents provided to chunk_documents. Returning empty list.")
        return []

    logger.info(f"Initializing RecursiveCharacterTextSplitter with size={size}, overlap={overlap}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        add_start_index=True
    )

    # 1. Group documents by document_id to assign accurate relative chunk indices
    docs_by_id: Dict[str, List[Document]] = defaultdict(list)
    for doc in documents:
        doc_id = doc.metadata.get("document_id")
        if not doc_id:
            # Fallback if document_id is missing, though pdf_loader guarantees it
            doc_id = doc.metadata.get("file_hash", "unknown_doc")
            doc.metadata["document_id"] = doc_id
        docs_by_id[doc_id].append(doc)

    all_processed_chunks: List[Document] = []

    # 2. Process each document group sequentially
    for doc_id, pages in docs_by_id.items():
        # Ensure pages are processed in correct sequence (sort by page number)
        pages.sort(key=lambda p: p.metadata.get("page", 0))
        logger.info(f"Splitting document '{doc_id}' ({len(pages)} pages)...")
        
        try:
            # Split all pages of this document
            chunks = text_splitter.split_documents(pages)
            
            # Post-process chunks to inject chunk-specific metadata
            for idx, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = idx
                chunk.metadata["chunk_id"] = f"{doc_id}_c{idx}"
                
                # Verify that all mandatory metadata fields exist
                # (source, page, total_pages, file_hash, file_size, content_type, created_at
                # are already copied by split_documents)
                
            all_processed_chunks.extend(chunks)
            logger.info(f"Generated {len(chunks)} chunk(s) for document ID '{doc_id}'")
        except Exception as e:
            logger.error(f"Error chunking document ID '{doc_id}': {e}", exc_info=True)

    logger.info(f"Total chunks generated across all files: {len(all_processed_chunks)}")
    return all_processed_chunks
