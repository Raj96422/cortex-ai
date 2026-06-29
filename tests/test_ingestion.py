"""
Unit and Integration Tests for Cortex AI Document Ingestion Pipeline.
Tests loading, validation, hashing, deduplication, custom exceptions, 
and chunking parameters with metadata preservation.
"""

import json
import os
import unittest
from pathlib import Path
from typing import Any
import sys

# Ensure workspace root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.exceptions import (
    CorruptedPDFException,
    DuplicateDocumentException,
    EmptyPDFException,
    InvalidPDFException,
)
from core.pdf_loader import _load_single_pdf, load_pdf
from core.chunker import chunk_documents
from core.document_processor import (
    process_uploaded_files,
    REGISTRY_PATH,
    _load_registry,
    _save_registry,
)
from utils.config import PDFS_DIR


class MockUploadedFile:
    """Mock file-like object simulating Streamlit's UploadedFile."""
    def __init__(self, name: str, content: bytes):
        self.name = name
        self.content = content
        self.offset = 0

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            data = self.content[self.offset:]
            self.offset = len(self.content)
            return data
        else:
            data = self.content[self.offset:self.offset + size]
            self.offset += size
            return data

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.offset = offset
        elif whence == 1:
            self.offset += offset
        elif whence == 2:
            self.offset = len(self.content) + offset
        return self.offset


def generate_minimal_pdf_bytes(text_content: str) -> bytes:
    """Helper to generate standard binary content of a minimal single-page PDF."""
    stream_content = f"BT\n/F1 12 Tf\n100 700 Td\n({text_content}) Tj\nET\n".encode("latin1")
    stream_len = len(stream_content)
    
    pdf_parts = [
        b"%PDF-1.4\n",
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>\nendobj\n",
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        f"5 0 obj\n<< /Length {stream_len} >>\nstream\n".encode("latin1"),
        stream_content,
        b"\nendstream\nendobj\n",
        b"xref\n0 6\n0000000000 65535 f\n",
        b"0000000009 00000 n\n",
        b"0000000058 00000 n\n",
        b"0000000115 00000 n\n",
        b"0000000227 00000 n\n",
        b"0000000302 00000 n\n",
        b"trailer\n<< /Size 6 /Root 1 0 R >>\n",
        b"startxref\n350\n%%EOF\n"
    ]
    return b"".join(pdf_parts)


class TestDocumentIngestion(unittest.TestCase):
    """Test suite for the Cortex AI Document Ingestion Pipeline."""

    def setUp(self):
        """Prepare clean test state before each test case."""
        # Ensure PDFs directory exists
        PDFS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Back up existing registry if any
        self.registry_backup = None
        if REGISTRY_PATH.exists():
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                self.registry_backup = f.read()
            REGISTRY_PATH.unlink()

        # Clean up any existing test PDFs in the folder
        self._cleanup_test_pdfs()

    def tearDown(self):
        """Restore workspace state after each test case."""
        self._cleanup_test_pdfs()
        
        # Restore backup registry
        if self.registry_backup is not None:
            with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
                f.write(self.registry_backup)
        elif REGISTRY_PATH.exists():
            REGISTRY_PATH.unlink()

    def _cleanup_test_pdfs(self):
        """Removes test generated PDFs from the pdfs/ directory."""
        for p in PDFS_DIR.glob("test_doc*.pdf"):
            try:
                p.unlink()
            except OSError:
                pass

    def test_valid_pdf_ingestion(self):
        """Test processing a valid PDF file succeeds and outputs chunks with correct metadata."""
        text_content = "This is a valid test PDF document. It contains custom text to verify that parsing and chunking work."
        pdf_bytes = generate_minimal_pdf_bytes(text_content)
        uploaded_file = MockUploadedFile("test_doc_valid.pdf", pdf_bytes)

        # Process uploaded file
        chunks = process_uploaded_files([uploaded_file])

        # Assertions
        self.assertGreater(len(chunks), 0, "Should generate at least one chunk.")
        
        # Verify metadata keys on first chunk
        chunk = chunks[0]
        required_keys = {
            "document_id", "chunk_id", "chunk_index", "source", "page",
            "total_pages", "file_hash", "file_size", "content_type", "created_at"
        }
        for key in required_keys:
            self.assertIn(key, chunk.metadata, f"Metadata key '{key}' should be present in chunk.")

        self.assertEqual(chunk.metadata["total_pages"], 1)
        self.assertEqual(chunk.metadata["page"], 0)
        self.assertEqual(chunk.metadata["chunk_index"], 0)
        self.assertEqual(chunk.metadata["content_type"], "application/pdf")
        self.assertEqual(chunk.metadata["filename"], "test_doc_valid.pdf")
        
        # Verify content was extracted
        self.assertIn("valid test PDF", chunk.page_content)

    def test_invalid_extension(self):
        """Test that files with invalid extensions (e.g., .txt) are skipped and return 0 chunks."""
        text_content = "Hello, this is a plain text file."
        file_bytes = text_content.encode("utf-8")
        uploaded_file = MockUploadedFile("test_doc_invalid.txt", file_bytes)

        chunks = process_uploaded_files([uploaded_file])
        self.assertEqual(len(chunks), 0, "Should skip non-PDF file and return 0 chunks.")

    def test_corrupted_pdf(self):
        """Test that corrupted PDF structures are handled gracefully and generate 0 chunks."""
        corrupted_bytes = b"%PDF-1.4\n%%EOF\nThis is not a real PDF structure"
        uploaded_file = MockUploadedFile("test_doc_corrupted.pdf", corrupted_bytes)

        chunks = process_uploaded_files([uploaded_file])
        self.assertEqual(len(chunks), 0, "Should handle corrupted PDF gracefully and return 0 chunks.")
        
        # Ensure file was not saved
        saved_file = PDFS_DIR / "test_doc_corrupted.pdf"
        self.assertFalse(saved_file.exists(), "Corrupted file should not remain in storage.")

    def test_empty_pdf(self):
        """Test that empty PDF uploads (0 bytes) are skipped and return 0 chunks."""
        uploaded_file = MockUploadedFile("test_doc_empty.pdf", b"")

        chunks = process_uploaded_files([uploaded_file])
        self.assertEqual(len(chunks), 0, "Should skip empty PDF and return 0 chunks.")

    def test_duplicate_detection(self):
        """Test that upload of identical document is detected and skipped."""
        text_content = "This text must be identical in both documents."
        pdf_bytes = generate_minimal_pdf_bytes(text_content)
        
        uploaded_file1 = MockUploadedFile("test_doc_1.pdf", pdf_bytes)
        uploaded_file2 = MockUploadedFile("test_doc_2.pdf", pdf_bytes)  # Duplicate hash

        # First upload
        chunks1 = process_uploaded_files([uploaded_file1])
        self.assertGreater(len(chunks1), 0, "First upload should succeed.")

        # Second upload (duplicate)
        chunks2 = process_uploaded_files([uploaded_file2])
        self.assertEqual(len(chunks2), 0, "Second duplicate upload should be skipped and return 0 chunks.")

    def test_metadata_preservation_and_chunk_generation(self):
        """Test that chunking parameters and metadata preservation operate correctly."""
        # Create a document containing text longer than chunk size to force split
        long_text = " ".join([f"Word{i}" for i in range(400)])  # Long string
        pdf_bytes = generate_minimal_pdf_bytes(long_text)
        uploaded_file = MockUploadedFile("test_doc_long.pdf", pdf_bytes)

        # Process with custom chunk parameters (override size to 100, overlap 20)
        registry = _load_registry()
        # Mock processing by saving, loading, chunking
        target_path = PDFS_DIR / "test_doc_long.pdf"
        with open(target_path, "wb") as f:
            f.write(pdf_bytes)
            
        docs = load_pdf(target_path)
        chunks = chunk_documents(docs, chunk_size=150, chunk_overlap=30)

        # Assertions
        self.assertGreater(len(chunks), 1, "Should split into multiple chunks.")
        
        # Verify details
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.metadata["chunk_index"], i)
            self.assertEqual(chunk.metadata["chunk_id"], f"{chunk.metadata['document_id']}_c{i}")
            self.assertEqual(chunk.metadata["page"], 0)
            self.assertLessEqual(len(chunk.page_content), 150, "Chunk size should be less than or equal to limits.")
            self.assertEqual(chunk.metadata["filename"], "test_doc_long.pdf")


if __name__ == "__main__":
    unittest.main()
