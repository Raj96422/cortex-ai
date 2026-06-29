"""
Custom Exceptions for the Cortex AI Document Ingestion Pipeline.
Provides descriptive exceptions for PDF validation, parsing, chunking, and deduplication.
"""

class DocumentProcessingException(Exception):
    """Base exception for all document processing and ingestion errors."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidPDFException(DocumentProcessingException):
    """Raised when the uploaded file is not a valid PDF or fails basic extension validation."""
    def __init__(self, message: str):
        super().__init__(message)


class CorruptedPDFException(DocumentProcessingException):
    """Raised when the PDF file structure is corrupted or unreadable."""
    def __init__(self, message: str):
        super().__init__(message)


class EmptyPDFException(DocumentProcessingException):
    """Raised when the PDF file is empty (0 bytes) or contains no readable text pages."""
    def __init__(self, message: str):
        super().__init__(message)


class DuplicateDocumentException(DocumentProcessingException):
    """Raised when a document with an identical SHA-256 hash has already been processed."""
    def __init__(self, message: str):
        super().__init__(message)
