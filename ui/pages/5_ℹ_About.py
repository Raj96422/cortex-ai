"""
Cortex AI Streamlit Application - About Page.
Renders project architectural details, tech stack listings, licensing, and repository links.
"""

import streamlit as st
from pathlib import Path
import sys

# Ensure workspace root is in path
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from ui.components import (
    inject_custom_css,
    render_header,
    render_footer,
)

# Page Configuration
st.set_page_config(
    page_title="Cortex AI - About",
    page_icon="ℹ",
    layout="wide"
)

# Inject custom CSS
inject_custom_css()

# Render Header
render_header(title="ℹ️ About Cortex AI", subtitle="System architecture and developer info")

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("System Architecture")
st.markdown(
    """
    Cortex AI is structured in layers following **clean architecture** guidelines to prevent circular dependencies:
    - **Presentation (Frontend)**: Renders multi-page Streamlit dashboards.
    - **Orchestration Layer**: Manages sessions and transactions (`CortexRAGPipeline`).
    - **Domain Layer**: Core services representing independent business domains:
      - `EmbeddingService` - Handles client and cache wrapping.
      - `SemanticRetriever` - Implements MMR and similarity matches.
      - `RAGPromptBuilder` - Enforces character-token budgets.
      - `GeminiLLM` - Communicates with Gemini API with backoff.
      - `DocumentProcessor` - Loads and splits PDF texts.
    - **Data Layer (Storage)**:
      - `VectorRepository` - Handles business rules and transactions.
      - `ChromaVectorStore` - Communicates with SQLite database.
    """
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Technical Stack")
st.markdown(
    """
    - **Language**: Python 3.11+
    - **Frontend**: Streamlit
    - **Vector Storage**: ChromaDB (Embedded SQLite)
    - **Embedding Provider**: Google Gemini Embeddings (`models/text-embedding-004`)
    - **LLM Provider**: Google Gemini LLM Service (`gemini-1.5-flash`)
    - **Document Parsing**: PyPDF
    - **Text Chunking**: LangChain Text Splitters (Recursive Character Chunker)
    """
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Project Metadata")
st.markdown(
    """
    - **Current Version**: `1.0.0`
    - **License**: `MIT License`
    - **GitHub Repository**: [Raj96422/cortex-ai](https://github.com/Raj96422/cortex-ai.git)
    """
)
st.markdown('</div>', unsafe_allow_html=True)

# Render Footer
render_footer()
