"""
Cortex AI Streamlit Application - Document Manager.
Handles PDF document uploads, indexing operations, metadata listings, and registries deletions.
"""

import streamlit as st
import tempfile
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
    render_metric_card,
)
from utils.config import PDFS_DIR
from core.exceptions import DuplicateDocumentException

# Page Configuration
st.set_page_config(
    page_title="Cortex AI - Documents",
    page_icon="📂",
    layout="wide"
)

# Inject custom CSS
inject_custom_css()

# Render Header
render_header(title="📂 Document Repository Manager", subtitle="Upload, index, and organize files in the knowledge base")

# Initialize Pipeline
pipeline = get_pipeline()

# Registry path
registry_path = PDFS_DIR / "processed_files.json"

def load_local_registry():
    if not registry_path.exists():
        return {}
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

# Registry details
registry = load_local_registry()

# Layout Columns
col_upload, col_list = st.columns([1, 2], gap="large")

with col_upload:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Upload Documents")
    
    collection_name = st.text_input("Ingest Collection Target", value="default_collection")
    uploaded_files = st.file_uploader("Drag & Drop PDFs here", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        st.markdown("---")
        st.markdown("##### Processing Uploads")
        
        # Ingest files
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            st.info(f"Ingesting: `{filename}`")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # 1. Save uploaded file to temp file inside workspace
                temp_dir = Path("temp_uploads")
                temp_dir.mkdir(exist_ok=True)
                temp_file_path = temp_dir / filename
                
                status_text.text("Saving file to temporary path...")
                progress_bar.progress(25)
                
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                status_text.text("Chunking and generating embeddings...")
                progress_bar.progress(60)
                
                # 2. Ingest via RAG Pipeline
                report = pipeline.ingest_document(
                    file_path=str(temp_file_path.resolve()),
                    collection_name=collection_name
                )
                
                # 3. Clean up temp file
                if temp_file_path.exists():
                    temp_file_path.unlink()
                
                progress_bar.progress(100)
                status_text.text("Ingestion completed successfully!")
                st.success(f"Successfully indexed document: `{filename}` ({report.get('chunks_indexed')} chunks)")
                
                # Refresh registry after ingest
                st.rerun()
                
            except DuplicateDocumentException as e:
                progress_bar.progress(100)
                status_text.text("Ignored duplicate.")
                st.warning(f"Skipped duplicate file: {filename}")
            except Exception as e:
                progress_bar.progress(100)
                status_text.text("Failed.")
                st.error(f"Failed to ingest '{filename}': {e}")
                
    st.markdown('</div>', unsafe_allow_html=True)

with col_list:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Indexed Documents Registry")
    
    if not registry:
        st.info("No documents are currently indexed in the repository.")
    else:
        st.markdown("The following files are stored in the local vector database:")
        
        # Display each file as a clean sub-card
        for file_hash, info in list(registry.items()):
            col_doc_info, col_doc_delete = st.columns([4, 1])
            
            with col_doc_info:
                st.markdown(
                    f"📄 **{info.get('filename')}**  \n"
                    f"⏱️ *Processed at:* `{info.get('processed_at')}` | "
                    f"Pages: `{info.get('num_pages')}` | Chunks: `{info.get('num_chunks')}`"
                )
            
            with col_doc_delete:
                # Add unique key to avoid Streamlit widget ID collisions
                if st.button("🗑️ Delete", key=f"del_{file_hash}"):
                    # In real RAG, delete from vector database
                    # Here we simulate deletion by removing from registry JSON and vector store
                    try:
                        # Deleting chunks of this file hash by querying collection keys
                        # For simplicity, we remove from registry and vector store collection
                        client = pipeline.repo.vector_store._get_client()
                        col = client.get_collection(collection_name)
                        
                        # Find matching IDs to delete
                        # The chunk IDs are in format {document_id}_c{chunk_index}
                        # document_id is either file_hash or derived
                        # We delete by document_id metadata filter or ids matching prefix
                        col.delete(where={"file_hash": file_hash})
                        
                        # Remove from registry JSON
                        if file_hash in registry:
                            registry.pop(file_hash)
                            with open(registry_path, "w", encoding="utf-8") as f:
                                json.dump(registry, f, indent=4, ensure_ascii=False)
                                
                        st.success(f"Deleted document registry: {info.get('filename')}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete registry item: {e}")
                        
            st.markdown("---")
            
    st.markdown('</div>', unsafe_allow_html=True)

# Render Footer
render_footer()
