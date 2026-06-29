import streamlit as st

# Custom CSS for modern glassmorphic dark theme styling
DARK_THEME_CSS: str = """
<style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

    /* Global CSS Variable definitions */
    :root {
        --primary-bg: #0D0E15;
        --secondary-bg: #151824;
        --accent-cyan: #00F2FE;
        --accent-blue: #4FACFE;
        --text-color: #E2E8F0;
        --text-muted: #94A3B8;
        --glass-bg: rgba(21, 24, 36, 0.7);
        --glass-border: rgba(255, 255, 255, 0.05);
    }

    /* Apply global font defaults */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-color);
    }

    /* Style titles with a gradient and modern header font */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
    }

    /* Custom Header Gradient Effect */
    .gradient-text {
        background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-cyan) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* Sidebar aesthetics */
    [data-testid="stSidebar"] {
        background-color: var(--secondary-bg) !important;
        border-right: 1px solid var(--glass-border) !important;
    }

    /* Custom main content card panel */
    .cortex-card {
        background-color: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.25);
    }

    /* Citation badge styling */
    .citation-badge {
        display: inline-flex;
        align-items: center;
        background-color: rgba(79, 172, 254, 0.15);
        color: #7FCDFF;
        border: 1px solid rgba(79, 172, 254, 0.3);
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        margin-right: 6px;
        margin-bottom: 6px;
        text-decoration: none;
    }
    
    .citation-badge:hover {
        background-color: rgba(79, 172, 254, 0.25);
        border-color: rgba(0, 242, 254, 0.5);
        color: #FFFFFF;
    }

    /* Custom styling for status messages */
    .stAlert {
        border-radius: 8px !important;
        background-color: var(--secondary-bg) !important;
        border: 1px solid var(--glass-border) !important;
    }

    /* Footer / branding styling */
    .footer-text {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--glass-border);
    }
</style>
"""

def apply_custom_css() -> None:
    """
    Injects custom CSS styling into the Streamlit page to achieve 
    a dark, modern, glassmorphic layout.
    """
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
