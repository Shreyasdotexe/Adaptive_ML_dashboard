import os
import streamlit as st

LIGHT_THEME = {
    "base": "light",
    "primaryColor": "#4f4ff0",
    "backgroundColor": "#ffffff",
    "secondaryBackgroundColor": "#f8f9fa",
    "textColor": "#1a1a1a",
}

DARK_THEME = {
    "base": "dark",
    "primaryColor": "#4f4ff0",
    "backgroundColor": "#0e1117",
    "secondaryBackgroundColor": "#262730",
    "textColor": "#fafafa",
}

def _write_config(theme_dict):
    config_dir = os.path.join(os.path.dirname(__file__), ".streamlit")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "config.toml")
    
    with open(config_path, "w") as f:
        f.write("[theme]\n")
        for k, v in theme_dict.items():
            f.write(f'{k}="{v}"\n')

def apply_theme():
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
        _write_config(LIGHT_THEME)
        
    st.sidebar.markdown("### Appearance")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Light", use_container_width=True, disabled=st.session_state.theme=="light"):
            st.session_state.theme = "light"
            _write_config(LIGHT_THEME)
            st.rerun()
    with col2:
        if st.button("Dark", use_container_width=True, disabled=st.session_state.theme=="dark"):
            st.session_state.theme = "dark"
            _write_config(DARK_THEME)
            st.rerun()

    if st.session_state.theme == "light":
        theme_inline = {
            "color_text": "#212529",
            "color_muted": "#6c757d",
            "color_faded": "#adb5bd",
            "color_highlight": "#4f4ff0",
            "bg_card": "#ffffff",
            "border_card": "#e9ecef"
        }
        theme_css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        *, html, body { font-family: 'Inter', sans-serif; box-sizing: border-box; }
        
        /* Force explicit text and background colors so they never invert */
        [data-testid="stAppViewContainer"] {
            background-color: #ffffff !important;
            color: #1a1a1a !important;
        }
        [data-testid="stSidebar"] {
            background-color: #f8f9fa !important;
        }
        
        .card {
            background-color: #ffffff !important;
            color: #212529 !important;
            border: 1px solid #e9ecef !important;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }
        .card h1, .card h2, .card h3, .card h4, .card p, .card div, .card span {
            color: inherit;
        }

        .terminal {
            background-color: #1e1e1e !important;
            color: #f8f8f2 !important;
            font-family: 'Fira Code', 'Cascadia Code', monospace;
            font-size: 0.85rem;
            padding: 1.5rem;
            border-radius: 8px;
            height: 400px;
            overflow-y: auto;
            border: 1px solid #333 !important;
            line-height: 1.5;
        }
        .t-ok { color: #a6e22e; }
        .t-drift { color: #f92672; font-weight: bold; }
        .t-warn { color: #fd971f; }
        .t-info { color: #66d9ef; }
        .dot-idle { display:inline-block; width:8px; height:8px; border-radius:50%; background:#6c757d; margin-right:8px; }
        .dot-running { display:inline-block; width:8px; height:8px; border-radius:50%; background:#f5a623; margin-right:8px; box-shadow: 0 0 8px rgba(245,166,35,0.6); }
        .dot-done { display:inline-block; width:8px; height:8px; border-radius:50%; background:#a6e22e; margin-right:8px; }
        .dot-stopped { display:inline-block; width:8px; height:8px; border-radius:50%; background:#f92672; margin-right:8px; }
        </style>
        """
    else:
        theme_inline = {
            "color_text": "#fafafa",
            "color_muted": "#a0aab4",
            "color_faded": "#6c757d",
            "color_highlight": "#4f4ff0",
            "bg_card": "#262730",
            "border_card": "#3b3d45"
        }
        theme_css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        *, html, body { font-family: 'Inter', sans-serif; box-sizing: border-box; }
        
        /* Force explicit text and background colors so they never invert */
        [data-testid="stAppViewContainer"] {
            background-color: #0e1117 !important;
            color: #fafafa !important;
        }
        [data-testid="stSidebar"] {
            background-color: #262730 !important;
        }
        
        .card {
            background-color: #262730 !important;
            color: #fafafa !important;
            border: 1px solid #3b3d45 !important;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
            margin-bottom: 1rem;
        }
        .card h1, .card h2, .card h3, .card h4, .card p, .card div, .card span {
            color: inherit;
        }

        .terminal {
            background-color: #1e1e1e !important;
            color: #f8f8f2 !important;
            font-family: 'Fira Code', 'Cascadia Code', monospace;
            font-size: 0.85rem;
            padding: 1.5rem;
            border-radius: 8px;
            height: 400px;
            overflow-y: auto;
            border: 1px solid #333 !important;
            line-height: 1.5;
        }
        .t-ok { color: #a6e22e; }
        .t-drift { color: #f92672; font-weight: bold; }
        .t-warn { color: #fd971f; }
        .t-info { color: #66d9ef; }
        .dot-idle { display:inline-block; width:8px; height:8px; border-radius:50%; background:#6c757d; margin-right:8px; }
        .dot-running { display:inline-block; width:8px; height:8px; border-radius:50%; background:#f5a623; margin-right:8px; box-shadow: 0 0 8px rgba(245,166,35,0.6); }
        .dot-done { display:inline-block; width:8px; height:8px; border-radius:50%; background:#a6e22e; margin-right:8px; }
        .dot-stopped { display:inline-block; width:8px; height:8px; border-radius:50%; background:#f92672; margin-right:8px; }
        </style>
        """
    return theme_css, theme_inline
