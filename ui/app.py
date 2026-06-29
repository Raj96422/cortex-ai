"""
Cortex AI Streamlit Application - Home Page.
Renders project overview, RAG system workflow diagrams, quickstart steps, and unified system health diagnostics.
"""

import streamlit as st
from pathlib import Path
import sys

# Ensure workspace root is in path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from ui.components import (
    get_pipeline,
    inject_custom_css,
    render_header,
    render_footer,
    render_health_dot,
)

# Page Configuration
st.set_page_config(
    page_title="Cortex AI - Home",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
inject_custom_css()

# Render Header
render_header(title="🧠 Cortex AI", subtitle="Production-Ready Document Ingestion and Semantic Retrieval System")

# Initialize Pipeline
try:
    pipeline = get_pipeline()
    health = pipeline.health_check()
    health_status = health.get("status") == "healthy"
except Exception as e:
    pipeline = None
    health = {}
    health_status = False
    st.error(f"Failed to initialize RAG Pipeline: {e}")

# Layout Columns
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Welcome to Cortex AI")
    st.markdown(
        """
        Cortex AI is an advanced, production-grade **Retrieval-Augmented Generation (RAG)** pipeline. 
        It enables intelligent search and context retrieval over uploaded PDF documents by combining 
        vector indexing and semantic search with large language models.
        
        This system is designed using a modular, decoupled architecture following **SOLID principles**:
        - **Module 3**: Intelligent Document Processor (loads, validates, hashes, and chunks PDFs).
        - **Module 4**: Embedding Service (converts text chunks to 768-dim vector embeddings).
        - **Module 5**: Vector Repository (manages ChromaDB storage, collection versioning, and transaction rollbacks).
        - **Module 6**: Semantic Retriever (performs similarity and MMR searches with metadata scope filtering).
        - **Module 7**: Prompt Builder (manages templates and compresses contexts under token budgets).
        - **Module 8**: LLM Service (handles Google Gemini API integrations with backoff retries and safety checks).
        - **Module 9**: Orchestration Pipeline (wires all modules together under request-traced transaction IDs).
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("RAG Orchestration Workflow")
    st.markdown(
        """
        ```text
        [User Query]
             │
             ▼
        [CortexRAGPipeline] ── (Generate Request ID)
             │
             ├─► [Semantic Retriever] ──► (Generates query vector embedding)
             │                                   │
             │                                   ▼
             │                             [Vector Repository] (Similarity / MMR lookup)
             │                                   │
             │◄──────────────────────────────────┘
             │
             ├─► [RAG Prompt Builder] ──► (Formats Context, Citations & Memory)
             │
             ├─► [Gemini LLM Client] ──► (Inference with Safety filters & Retries)
             │
             ▼
        [Packaged RAG Response] ──► (Includes Answer, Citations & Source Chunks)
        ```
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("System Status")
    
    if health_status:
        st.success("🟢 All Services Operational")
    else:
        st.error("🔴 Service Outage Detected")
        
    st.markdown("---")
    
    if health:
        deps = health.get("dependencies", {})
        
        # Ingestion / Document Processor Health (always operational since it's local)
        render_health_dot(True, "Document Ingestion Service")
        
        # Embedding Service
        emb_health = deps.get("embedding_service", {}).get("status") == "healthy"
        render_health_dot(emb_health, "Embedding Generation Service")
        
        # Vector Store
        repo_health = deps.get("vector_repository", {}).get("status") == "healthy"
        render_health_dot(repo_health, "Vector Store (ChromaDB)")
        
        # Retriever
        ret_health = deps.get("retriever", {}).get("status") == "healthy"
        render_health_dot(ret_health, "Semantic Search Retriever")
        
        # Prompt Builder
        prompt_health = deps.get("prompt_builder", {}).get("status") == "healthy"
        render_health_dot(prompt_health, "Prompt Engineering Builder")
        
        # LLM Client
        llm_health = deps.get("llm_service", {}).get("status") == "healthy"
        render_health_dot(llm_health, "Google Gemini LLM Service")
    else:
        st.warning("Could not retrieve detailed status reports.")
        
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Quick Start Steps")
    st.markdown(
        """
        1. Go to **📂 Documents** in the sidebar to upload and index PDF reports.
        2. Adjust search and model settings inside the **⚙ Settings** tab.
        3. Start chatting with your documents in the **💬 Chat** workspace!
        4. Track system latencies and cache efficiency metrics in **📊 Analytics**.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Render Footer
render_footer()
