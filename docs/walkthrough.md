# Walkthrough of the Ingestion Pipeline Codebase

This document provides a line-by-line file guide explaining the roles of all generated source files and the implementation details of each critical function.

---

## File Structure

```
core/
├── __init__.py               # Package exports
├── exceptions.py             # Custom domain exceptions
├── pdf_loader.py             # PDF text extractor
├── chunker.py                # Semantic text splitter
└── document_processor.py     # Pipeline orchestrator
```

---

## 1. Package Exports: `core/__init__.py`

Exposes the primary functions of each module for cleaner imports throughout the application.

Exposes:
- `load_pdf` from `core.pdf_loader`
- `chunk_documents` from `core.chunker`
- `process_uploaded_files` from `core.document_processor`

---

## 2. Domain Exceptions: `core/exceptions.py`

Defines the hierarchy of custom exceptions used to report specific ingestion pipeline failures:

* **`DocumentProcessingException`**: Base custom class for all pipeline errors (inherits from `Exception`).
* **`InvalidPDFException`**: Raised if a file has an incorrect file extension or if it is encrypted and decryption fails.
* **`CorruptedPDFException`**: Raised when file format errors occur, such as a missing EOF or an invalid PDF stream structure.
* **`EmptyPDFException`**: Raised if a PDF is 0 bytes on disk or if no text can be extracted from any of its pages.
* **`DuplicateDocumentException`**: Raised when a file content hash matches a document that was already successfully processed.

---

## 3. PDF Ingestion: `core/pdf_loader.py`

### Important Functions

#### `_load_single_pdf(file_path: Path, file_hash: Optional[str] = None) -> List[Document]`
* **Purpose**: Performs validation, decryption, and text extraction for a single PDF.
* **Logic**:
  1. Validates existence and checks if size is 0 (raises `EmptyPDFException`).
  2. Generates the content hash if not provided.
  3. Uses `pypdf.PdfReader` to open the file.
  4. If encrypted, attempts decryption with `""`. If still encrypted, raises `InvalidPDFException`.
  5. Extracts text page-by-page.
  6. Skips a page if the text is empty or only contains whitespace.
  7. Populates metadata dictionary:
     - `document_id`: the content hash.
     - `source`: absolute path on disk.
     - `filename`: filename.
     - `page`: 0-indexed page number.
     - `total_pages`: total pages in file.
     - `file_hash`: SHA-256 hash.
     - `file_size`: size in bytes.
     - `content_type`: "application/pdf".
     - `created_at`: processing timestamp.
  8. If no pages return text, raises `EmptyPDFException`.
  9. Catch `PdfReadError` to raise `CorruptedPDFException`.

#### `load_pdf(file_paths: Union[str, Path, List[Union[str, Path]]]) -> List[Document]`
* **Purpose**: Converts a single path or list of paths into a unified list of LangChain `Document` pages.
* **Logic**: Iterates over resolved path arguments, calls `_load_single_pdf` on each, aggregates and returns the page documents.

---

## 4. Document Chunker: `core/chunker.py`

### Important Functions

#### `chunk_documents(documents: List[Document], chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> List[Document]`
* **Purpose**: Splits page documents into cohesive, overlapping text chunks.
* **Logic**:
  1. Defaults sizes to configuration parameters defined in [constants.py](file:///e:/Cortex%20Ai/utils/constants.py) (`DEFAULT_CHUNK_SIZE` and `DEFAULT_CHUNK_OVERLAP`).
  2. Initializes a `RecursiveCharacterTextSplitter`.
  3. Groups the pages by `document_id` so that relative chunk indices are unique to each document.
  4. Sorts page documents by their `page` numbers to enforce sequential reading order.
  5. Splitting operations automatically copy the page-level metadata.
  6. Post-processes each chunk to inject:
     - `chunk_index`: the sequential index of the chunk relative to this document.
     - `chunk_id`: `{document_id}_c{chunk_index}` which forms a globally unique ID.

---

## 5. Ingestion Orchestrator: `core/document_processor.py`

### Important Functions

#### `_load_registry() -> Dict[str, Any]` & `_save_registry(registry: Dict[str, Any]) -> None`
* **Purpose**: Manage the state of `pdfs/processed_files.json`, reading and writing metadata registry logs of processed file hashes.

#### `process_uploaded_files(uploaded_files: List[Any]) -> List[Document]`
* **Purpose**: Orchestrates validation, persistence, parsing, chunking, and registration of uploaded files.
* **Logic**:
  1. Iterates over the batch of uploads.
  2. Validates filename extensions using `validate_file_extension` (raises `InvalidPDFException` on mismatch).
  3. Reads file bytes and asserts they are not empty (raises `EmptyPDFException` on 0 bytes).
  4. Validates size against maximum limit (raises `InvalidPDFException` if size > limit).
  5. Computes SHA-256 hash.
  6. Check registry: if hash is present, raises `DuplicateDocumentException`.
  7. Resolves file collisions: if a file with the same name exists, it appends the first 8 characters of the hash to the name to prevent overwriting other files.
  8. Saves the file to disk at `pdfs/`.
  9. Calls `load_pdf` on the path. If exceptions are thrown, deletes the saved file to clean up storage and re-raises.
  10. Calls `chunk_documents` on page documents.
  11. Adds the successful file entry and hash metadata to the registry.
  12. Wraps everything in a per-file `try-except` block to catch and handle all pipeline custom exceptions individually. Logs detailed production-level logs for each issue.
  13. Persists the registry to disk and returns the generated list of chunks.
