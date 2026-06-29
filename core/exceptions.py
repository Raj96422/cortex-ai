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


class MissingAPIKeyException(DocumentProcessingException):
    """Raised when the Google GenAI API key is missing or not set in configuration."""
    def __init__(self, message: str):
        super().__init__(message)


class InvalidTextException(DocumentProcessingException):
    """Raised when the text content provided for embedding is invalid (e.g., not a string)."""
    def __init__(self, message: str):
        super().__init__(message)


class EmptyTextException(DocumentProcessingException):
    """Raised when the text content provided for embedding is empty or whitespace-only."""
    def __init__(self, message: str):
        super().__init__(message)


class GeminiAPIException(DocumentProcessingException):
    """Raised when the underlying Gemini API returns an error during embedding generation."""
    def __init__(self, message: str):
        super().__init__(message)


class EmbeddingTimeoutException(DocumentProcessingException):
    """Raised when the connection or request to the Google GenAI API times out."""
    def __init__(self, message: str):
        super().__init__(message)


class RateLimitException(DocumentProcessingException):
    """Raised when the Google GenAI API quota or rate limits are exceeded."""
    def __init__(self, message: str):
        super().__init__(message)


class EmbeddingGenerationException(DocumentProcessingException):
    """Raised when vector embedding generation fails due to internal errors."""
    def __init__(self, message: str):
        super().__init__(message)


class InvalidMetadataException(DocumentProcessingException):
    """Raised when a chunk lacks required metadata keys (e.g. chunk_id)."""
    def __init__(self, message: str):
        super().__init__(message)
