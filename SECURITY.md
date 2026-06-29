# Security Policy

This document outlines the guidelines and best practices for secure configuration, credential handling, and reporting security vulnerabilities in Cortex AI.

---

## 1. Credentials and API Keys

Cortex AI integrates with the Google Gemini API, which requires a private API Key.
* **Strictly No Hardcoding**: Do not hardcode the API Key or any other credentials anywhere in the repository.
* **Environment Variables**: Store the API key inside a local `.env` file or export it to the environment as `GEMINI_API_KEY`.
* **Exclusions**: The `.gitignore` and `.dockerignore` files are configured to ignore `.env` files to prevent committing secrets to version control.

---

## 2. Sensitive Data Logging

Cortex AI uses structured logging (`utils/logger.py`) across all modules.
* **Logging Boundary**: The application strictly avoids logging actual parsed document text chunks, user query values, or generated answer texts.
* **Metadata only**: Log statements are restricted to tracing operational metadata (e.g. file hashes, chunk numbers, query request IDs, processing latencies) to prevent leaking confidential user content.

---

## 3. Vulnerability Reporting

If you identify a security vulnerability in this project, please report it responsibly by contacting the repository maintainers directly rather than opening a public GitHub issue.

Please provide:
* A detailed description of the vulnerability.
* Steps to reproduce the issue.
* Any proof-of-concept scripts or steps.

We will acknowledge your report within 48 hours and work with you to patch the issue before making it public.
