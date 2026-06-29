# Ingestion Pipeline Testing Guide

This document describes how to execute tests and verify the behavior of the Intelligent Document Ingestion Pipeline.

## 1. Running Unit Tests

The test suite in [test_ingestion.py](file:///e:/Cortex%20Ai/tests/test_ingestion.py) covers all specifications. You can execute these tests directly from your command line.

### Run All Tests
Execute the following command in the workspace root directory:

```bash
python -m unittest tests/test_ingestion.py
```

### Run a Specific Test Case
To run a specific test, append the test class and method name:

```bash
# Run duplicate detection test only
python -m unittest tests.test_ingestion.TestDocumentIngestion.test_duplicate_detection

# Run valid pdf ingestion test only
python -m unittest tests.test_ingestion.TestDocumentIngestion.test_valid_pdf_ingestion
```

---

## 2. Test Coverage

The suite contains tests covering:

* **Valid PDF Ingestion**: Verifies parsing of mock PDF bytes, metadata extraction, page counting, and document chunking output structure.
* **Invalid Extension**: Asserts that uploading non-PDF extensions (like `.txt`) fails and returns 0 chunks.
* **Corrupted PDF**: Asserts that feeding non-PDF bytes under a `.pdf` name raises `CorruptedPDFException`, returns 0 chunks, and cleans up storage.
* **Empty PDF**: Asserts that empty file streams (0 bytes) raise `EmptyPDFException` and are skipped.
* **Duplicate Detection**: Asserts that uploading identical files twice processes only the first file and ignores the duplicate.
* **Metadata Preservation**: Verifies that custom chunking limits are enforced, chunk indexing is incremental, and parent page metadata is copied.

---

## 3. Manual Testing / Verification

You can also perform manual testing by running our scratch test pipeline:

```bash
python "C:\Users\RAJA SEKHAR\.gemini\antigravity-ide\brain\02c6239d-0f58-489e-950b-85991801402f\scratch\test_pipeline.py"
```

This script generates actual mock files in the `pdfs/` folder, processes them, prints detailed metadata mapping for the resulting chunks, and demonstrates duplicate skipping and corrupted file validation.
