"""
dashboard.py  —  home page for the Adaptive ML multi-page app.
Run with:  python -m streamlit run dashboard.py
"""

import streamlit as st

st.set_page_config(
    page_title="Adaptive ML — Concept Drift",
    layout="wide",
    initial_sidebar_state="expanded",
)

from theme import apply_theme
theme_css, theme_inline = apply_theme()
st.markdown(theme_css, unsafe_allow_html=True)

# sidebar label
st.sidebar.markdown("### Adaptive ML")
st.sidebar.markdown("Concept drift detection for streaming data.")
st.sidebar.markdown("---")
st.sidebar.caption("Navigate using the links above.")

# main content
st.markdown("## Adaptive ML System")
st.markdown(
    "This project detects when a machine learning model's accuracy drops due to "
    "**concept drift** — shifts in the underlying data distribution over time. "
    "When drift is detected, the model is automatically retrained."
)
st.markdown("---")

col1, col2 = st.columns(2, gap="large")

with col1:
    html_content = f"""
    <div class="card">
        <h3 style="margin-bottom:0.6rem;">Run Detection</h3>
        <p>
            Start the detection pipeline from the browser. Choose which drift detector
            should trigger retraining, tune a few parameters, then hit start and watch the
            output come in live — same as running it in a terminal.
        </p>
        <p style="margin-top:1rem;color:{theme_inline['color_highlight']};font-size:0.85rem;">
            → Open "Run Detection" in the sidebar
        </p>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)

with col2:
    html_content = f"""
    <div class="card">
        <h3 style="margin-bottom:0.6rem;">Results</h3>
        <p>
            After a run completes, explore what happened — accuracy over time, when drift
            was detected, how each detector performed, and a breakdown of every retraining
            event. All plots from the run are here too.
        </p>
        <p style="margin-top:1rem;color:{theme_inline['color_highlight']};font-size:0.85rem;">
            → Open "Results" in the sidebar
        </p>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### How it works")

s1, s2, s3, s4 = st.columns(4, gap="medium")
steps = [
    ("1", "Stream data", "45k electricity price samples are fed one at a time, simulating a live data stream."),
    ("2", "Predict & learn", "The model predicts each sample, then immediately learns from the true label."),
    ("3", "Watch for drift", "ADWIN, DDM, and EDDM all monitor the error rate — the chosen one triggers retraining."),
    ("4", "Retrain", "When drift is confirmed, the model resets and relearns from the current distribution."),
]
for col, (num, title, desc) in zip([s1, s2, s3, s4], steps):
    col.markdown(f"""
    <div class="card" style="text-align:left;">
        <div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;
                    color:{theme_inline['color_muted']};text-transform:uppercase;margin-bottom:0.4rem;">
            Step {num}
        </div>
        <div style="font-weight:600;color:{theme_inline['color_text']};margin-bottom:0.4rem;">{title}</div>
        <p style="font-size:0.85rem;color:{theme_inline['color_muted']};">{desc}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("### Case Study: Streaming Data Drift Detection")
st.markdown(f"""
<div class="card" style="margin-bottom: 2rem; border-left: 4px solid #4f4ff0;">
    <h4 style="margin-bottom:0.8rem; color:{theme_inline['color_text']}; font-weight: 600;">The Challenge of Real-Time Environments</h4>
    <p style="margin-bottom:1rem;">
        In traditional machine learning, models are trained on static datasets and deployed with the assumption that future data will follow the same distribution. However, in real-world streaming scenarios—like electricity pricing, financial markets, or sensor data—the underlying concepts frequently change. This phenomenon is known as <strong>Concept Drift</strong>.
    </p>
    <h4 style="margin-bottom:0.8rem; color:{theme_inline['color_text']}; font-weight: 600;">Our Approach</h4>
    <p style="margin-bottom:1rem;">
        This framework tackles concept drift by utilizing continuous learning algorithms combined with robust drift detectors:
    </p>
    <ul style="margin-left: 1.5rem; margin-bottom: 1rem; color:{theme_inline['color_muted']}; line-height: 1.6;">
        <li style="margin-bottom: 0.5rem;"><strong>ADWIN (Adaptive Windowing):</strong> Dynamically adjusts its window size to track recent data, quickly identifying distribution shifts.</li>
        <li style="margin-bottom: 0.5rem;"><strong>DDM (Drift Detection Method):</strong> Monitors the model's error rate and triggers warnings when the error statistically increases.</li>
        <li style="margin-bottom: 0.5rem;"><strong>EDDM (Early Drift Detection Method):</strong> Analyzes the distance between errors to detect gradual drifts before accuracy degrades severely.</li>
    </ul>
    <p>
        By observing these metrics in a simulated live stream, our system can autonomously trigger retraining events, allowing the model to adapt to new patterns without manual intervention.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    f"<p style='font-size:0.82rem;color:{theme_inline['color_faded']};'>"
    "Final Year Capstone Project &nbsp;·&nbsp; "
    "Adaptive Machine Learning for Real-Time Systems"
    "</p>",
    unsafe_allow_html=True
)
