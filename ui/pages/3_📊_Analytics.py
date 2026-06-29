"""
Cortex AI Streamlit Application - Analytics Dashboard.
Visualizes performance statistics, latencies breakdown, cache hit efficiencies, and index sizes.
"""

import streamlit as st
import pandas as pd
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
    render_metric_card,
)

# Page Configuration
st.set_page_config(
    page_title="Cortex AI - Analytics",
    page_icon="📊",
    layout="wide"
)

# Inject custom CSS
inject_custom_css()

# Render Header
render_header(title="📊 Performance Analytics", subtitle="Monitor RAG pipeline latency breakdown and cache statistics")

# Initialize Pipeline
pipeline = get_pipeline()

# Fetch Stats
try:
    stats = pipeline.get_statistics()
except Exception:
    stats = {}

# Stats Grid
col1, col2, col3, col4 = st.columns(4)

with col1:
    render_metric_card(
        title="Questions Asked",
        value=stats.get("questions_asked", 0),
        subtitle="Total queries processed"
    )

with col2:
    avg_latency = stats.get("average_response_time", 0.0)
    render_metric_card(
        title="Avg Response Latency",
        value=f"{avg_latency:.1f} ms" if avg_latency > 0 else "0 ms",
        subtitle="End-to-end turnaround"
    )

with col3:
    cache_eff = stats.get("cache_efficiency", 0.0)
    render_metric_card(
        title="Cache Efficiency",
        value=f"{cache_eff:.1f} %" if cache_eff > 0 else "0 %",
        subtitle="Retriever cache hit ratio"
    )

with col4:
    render_metric_card(
        title="Indexed Documents",
        value=stats.get("total_indexed_documents", 0),
        subtitle="Registry count"
    )

st.markdown("---")

# Visualizations
col_chart, col_details = st.columns([2, 1], gap="large")

with col_chart:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Latency Breakdown Comparison")
    
    # Compile latency data
    ret_time = stats.get("average_retrieval_time", 0.0)
    gen_time = stats.get("average_generation_time", 0.0)
    
    if ret_time > 0 or gen_time > 0:
        data = {
            "Stage": ["Semantic Retrieval", "LLM Generation"],
            "Time (ms)": [ret_time, gen_time]
        }
        df = pd.DataFrame(data)
        st.bar_chart(df.set_index("Stage"), color="#38BDF8")
    else:
        st.info("Start asking questions in the Chat workspace to view latency charts.")
    st.markdown('</div>', unsafe_allow_html=True)

with col_details:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Database & Index Stats")
    
    # Retrieve registry count from folder
    try:
        client = pipeline.repo.vector_store._get_client()
        collections = client.list_collections()
        num_cols = len(collections)
    except Exception:
        num_cols = 0

    st.markdown(
        f"- **Active Vector Database:** `ChromaDB`  \n"
        f"- **Index Space Model:** `Cosine Space`  \n"
        f"- **Active Collection Count:** `{num_cols}`  \n"
        f"- **Avg Retrieved Chunks:** `{stats.get('average_chunks_retrieved', 0.0):.1f} chunks`  \n"
        f"- **Total Chunks Retrieved:** `{stats.get('total_chunks_retrieved', 0)} chunks`  \n"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Render Footer
render_footer()
