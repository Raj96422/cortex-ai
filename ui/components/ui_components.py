"""
Reusable UI Components for the Cortex AI Streamlit Application.
Provides styling injectors, headers, footers, glassmorphism cards, and citation components.
"""

import streamlit as st
from pathlib import Path
from typing import Any


def get_pipeline():
    """
    Returns the RAG pipeline instance, initializing it in session state if needed.
    """
    import sys
    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
        
    from core.rag.rag_factory import RAGFactory
    
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGFactory.get_pipeline()
    return st.session_state.pipeline


def inject_custom_css() -> None:
    """
    Reads ui/styles/custom.css and injects it into the Streamlit app.
    """
    css_path = Path(__file__).resolve().parent.parent / "styles" / "custom.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        st.warning("Custom CSS file not found.")


def render_header(title: str = "🧠 Cortex AI", subtitle: str = "Intelligent Knowledge Assistant") -> None:
    """
    Renders a unified header with gradient styling.
    """
    st.markdown(f'<h1 class="gradient-title" style="margin-bottom:0px;">{title}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle-text">{subtitle}</p>', unsafe_allow_html=True)
    st.markdown("---")


def render_footer() -> None:
    """
    Renders a sticky footer at the bottom of the page.
    """
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #64748B; font-size: 0.8rem;">'
        'Cortex AI © 2026 | Built with Streamlit, ChromaDB, and Google Gemini | License: MIT'
        '</p>',
        unsafe_allow_html=True
    )


def render_metric_card(title: str, value: Any, subtitle: str = "") -> None:
    """
    Renders a glassmorphism metric card.
    """
    st.markdown(
        f'<div class="glass-card">'
        f'<p style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 5px; text-transform: uppercase; font-weight: 600;">{title}</p>'
        f'<h2 style="color: #F8FAFC; margin: 0px; font-weight: 700; font-size: 2rem;">{value}</h2>'
        f'<p style="color: #38BDF8; font-size: 0.8rem; margin-top: 5px; margin-bottom: 0px;">{subtitle}</p>'
        f'</div>',
        unsafe_allow_html=True
    )


def render_citation_card(source: str, page: str, score: float, text: str) -> None:
    """
    Renders an expandable citation card with text highlighting.
    """
    with st.expander(f"📄 {source} (Page {page}) - Similarity: {score:.1%}"):
        st.markdown(f'<div class="citation-box">"{text}"</div>', unsafe_allow_html=True)
        st.caption(f"Source file: `{source}` | Chunk Page: `{page}`")


def render_health_dot(is_healthy: bool, name: str) -> None:
    """
    Renders a health status dot with label text.
    """
    status_class = "status-healthy" if is_healthy else "status-unhealthy"
    label = "Healthy" if is_healthy else "Unhealthy"
    st.markdown(
        f'<p style="margin: 0px; display: flex; align-items: center;">'
        f'<span class="status-dot {status_class}"></span>'
        f'<strong style="color: #E2E8F0;">{name}:</strong>'
        f'<span style="color: #94A3B8; margin-left: 8px;">{label}</span>'
        f'</p>',
        unsafe_allow_html=True
    )
