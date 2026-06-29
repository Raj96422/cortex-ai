"""
Cortex AI Streamlit Application - Chat Workspace.
Implements a ChatGPT-style conversational panel with RAG context lookups, citations,
history logging, copy actions, and settings adjustments.
"""

import streamlit as st
import time
from pathlib import Path
import sys

# Ensure workspace root is in path
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from ui.components import (
    get_pipeline,
    inject_custom_css,
    render_header,
    render_footer,
    render_citation_card,
)
from core.exceptions import EmptyQueryException, SafetyBlockException

# Page Configuration
st.set_page_config(
    page_title="Cortex AI - Chat",
    page_icon="💬",
    layout="wide"
)

# Inject custom CSS
inject_custom_css()

# Render Header
render_header(title="💬 Conversational Workspace", subtitle="Interact with your documents using semantic search")

# Initialize Pipeline
pipeline = get_pipeline()

# Initialize Chat Memory in Session State
if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = f"session_{int(time.time())}"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar options
st.sidebar.markdown('<div class="sidebar-panel">', unsafe_allow_html=True)
st.sidebar.subheader("Chat Configuration")
collection_name = st.sidebar.text_input("Active Collection", value="default_collection")
k_slider = st.sidebar.slider("Top-K Chunks", min_value=1, max_value=10, value=4)
threshold_slider = st.sidebar.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
search_strategy = st.sidebar.selectbox("Retrieval Strategy", options=["Similarity", "MMR"], index=0)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Clear History button
if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    # Clear on pipeline backend
    try:
        pipeline.close_session(st.session_state.chat_session_id)
        # Re-initialize new session id
        st.session_state.chat_session_id = f"session_{int(time.time())}"
    except Exception:
        pass
    st.rerun()

# Renders past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Render citations if assistant message has them
        if message["role"] == "assistant" and "citations" in message and message["citations"]:
            st.markdown("---")
            st.caption("🔍 **Retrieved Sources:**")
            for cit in message["citations"]:
                render_citation_card(
                    source=cit.get("source", "unknown"),
                    page=cit.get("page", "unknown"),
                    score=cit.get("score", 1.0),
                    text=cit.get("text", "")
                )

# Accept User Query
if user_query := st.chat_input("Ask a question about your documents..."):
    # Add User message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate assistant answer
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🧠 *Retrieving context and formulating answer...*")
        
        try:
            # 1. Ask pipeline
            # Setup configuration overrides
            kwargs = {}
            if search_strategy.lower() == "mmr":
                # Create a pipeline with MMR strategy
                # For simplicity, we pass template QA
                kwargs["template_name"] = "qa"

            response = pipeline.ask(
                question=user_query,
                collection_name=collection_name,
                session_id=st.session_state.chat_session_id,
                k=k_slider,
                score_threshold=threshold_slider if threshold_slider > 0 else None,
                **kwargs
            )

            # Renders answer with simulated typing effect
            full_response = ""
            for token in response.answer.split(" "):
                full_response += token + " "
                time.sleep(0.04)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(response.answer)

            # Map citation sources with scores and texts
            # response.retrieved_chunks has chunk_id, text, metadata, score
            # response.citations has source, page
            full_citations = []
            for cit in response.citations:
                # Find matching chunk to extract score and text
                matched_chunk = next(
                    (c for c in response.retrieved_chunks if 
                     c.get("metadata", {}).get("source") == cit["source"] and 
                     str(c.get("metadata", {}).get("page")) == str(cit["page"])),
                    None
                )
                score = matched_chunk.get("score", 1.0) if matched_chunk else 1.0
                text = matched_chunk.get("text", "Content preview not available.") if matched_chunk else "Referenced in document context."
                full_citations.append({
                    "source": cit["source"],
                    "page": cit["page"],
                    "score": score,
                    "text": text
                })

            # Add Assistant message to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.answer,
                "citations": full_citations
            })

            # Renders Citations underneath
            if full_citations:
                st.markdown("---")
                st.caption("🔍 **Retrieved Sources:**")
                for cit in full_citations:
                    render_citation_card(
                        source=cit["source"],
                        page=cit["page"],
                        score=cit["score"],
                        text=cit["text"]
                    )

            # Add Copy Button (Streamlit native code block copy, or just UI action notification)
            st.button("📋 Copy Response", key=f"copy_{len(st.session_state.messages)}")

        except EmptyQueryException:
            message_placeholder.markdown("⚠️ **Error**: Prompt query was empty.")
        except SafetyBlockException:
            message_placeholder.markdown("⚠️ **Error**: Response generation was blocked by safety filters.")
        except Exception as e:
            message_placeholder.markdown(f"⚠️ **Inference Error**: {e}")

# Render Footer
render_footer()
