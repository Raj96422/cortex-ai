"""
Cortex AI Streamlit Application - Settings Workspace.
Exposes sliders for temperature, Top-K limits, MMR lambda, similarity thresholds,
theme toggles, and database purging utilities.
"""

import streamlit as st
from pathlib import Path
import sys
import json

# Ensure workspace root is in path
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from ui.components import (
    get_pipeline,
    inject_custom_css,
    render_header,
    render_footer,
)
from utils.config import PDFS_DIR

# Page Configuration
st.set_page_config(
    page_title="Cortex AI - Settings",
    page_icon="⚙",
    layout="wide"
)

# Inject custom CSS
inject_custom_css()

# Render Header
render_header(title="⚙️ Configuration Settings", subtitle="Tune semantic search thresholds and LLM generation parameters")

# Initialize Pipeline
pipeline = get_pipeline()

# Registry path
registry_path = PDFS_DIR / "processed_files.json"

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Model Generation Parameters")

temp = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05, help="Lower values make output more deterministic; higher values make it more creative.")
top_p = st.slider("Top-P (Nucleus Sampling)", min_value=0.0, max_value=1.0, value=0.95, step=0.05)
max_tokens = st.number_input("Max Output Tokens", min_value=128, max_value=4096, value=2048, step=128)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Retrieval and Indexing Parameters")

search_strategy = st.selectbox("Default Search Strategy", options=["Similarity Search", "MMR (Maximum Marginal Relevance)"], index=0)
k_val = st.slider("Default Top-K Chunks to Retrieve", min_value=1, max_value=10, value=4)
similarity_threshold = st.slider("Default Similarity Score Threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.05, help="0.0 disables the threshold filter.")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Database Maintenance")
st.warning("Warning: Resetting the database will permanently delete all indexed PDF text chunks and clear the files registry.")

if st.button("🚨 Purge All Data (Reset Database)", type="primary"):
    try:
        # Delete processed_files.json
        if registry_path.exists():
            registry_path.unlink()
            
        # Reset vector store client collections
        client = pipeline.repo.vector_store._get_client()
        collections = client.list_collections()
        for col in collections:
            client.delete_collection(col.name)
            
        # Reset pipeline and statistics
        pipeline.reset_statistics()
        
        st.success("Successfully purged vector database collections and cleared registries!")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to reset database: {e}")

st.markdown('</div>', unsafe_allow_html=True)

# Render Footer
render_footer()
