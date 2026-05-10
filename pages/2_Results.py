"""
pages/2_Results.py
View results from the most recent (or any previous) run.
"""

import os
import json
import glob

import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Results",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from theme import apply_theme
theme_css, theme_inline = apply_theme()
st.markdown(theme_css, unsafe_allow_html=True)

# ── helpers ────────────────────────────────────────────────────────────────────

REPORTS_DIR = os.path.join("outputs", "reports")


def load_all_runs():
    files = sorted(glob.glob(os.path.join(REPORTS_DIR, "meta_*.json")), reverse=True)
    runs  = []
    for path in files:
        with open(path) as f:
            meta = json.load(f)
        runs.append((meta.get("run_id", os.path.basename(path)), meta))
    return runs


def safe_img(path):
    if path and os.path.exists(path):
        return Image.open(path)
    return None


# ── sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.markdown("### Results")
st.sidebar.markdown("---")

all_runs = load_all_runs()

if not all_runs:
    st.markdown("## Results")
    st.warning(
        "No runs yet. Go to **Run Detection** and start a pipeline run first."
    )
    st.stop()

run_labels    = [r[0] for r in all_runs]
selected_id   = st.sidebar.selectbox("Run", run_labels, index=0,
                                      help="Most recent run is at the top.")
meta          = dict(next(m for (r, m) in all_runs if r == selected_id))

st.sidebar.markdown("---")
st.sidebar.caption(f"Run ID: `{meta.get('run_id', '—')}`")
st.sidebar.caption(f"Dataset: {os.path.basename(meta.get('dataset', '—'))}")

ev   = meta.get("evaluation", {})
cm_s = ev.get("confusion_matrix", {})
if ev:
    st.sidebar.markdown("---")
    st.sidebar.caption(f"F1 {cm_s.get('f1','—')}  ·  κ {ev.get('kappa','—')}")

# ── page ───────────────────────────────────────────────────────────────────────

comp   = meta.get("detector_comparison", {})
rt     = meta.get("retraining_events", {})
plots  = meta.get("plots", {})

st.markdown("## Results")
st.markdown(
    f"Showing results for run `{meta.get('run_id','—')}`. "
    f"Use the sidebar to switch between runs."
)
st.markdown("---")

# ─ Summary numbers ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-label">Overview</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Samples",       f"{meta.get('n_samples', 0):,}")
c2.metric("Final accuracy",f"{meta.get('final_acc', 0):.1%}")
c3.metric("F1",            f"{cm_s.get('f1', 0):.3f}")
c4.metric("Cohen's κ",     f"{ev.get('kappa', 0):.3f}")
c5.metric("Retrains",      rt.get("total_retraining_events", 0))

c6, c7, c8, c9, c10 = st.columns(5)
c6.metric("Precision",      f"{cm_s.get('precision', 0):.3f}")
c7.metric("Recall",         f"{cm_s.get('recall', 0):.3f}")
c8.metric("First-half acc", f"{meta.get('first_acc', 0):.1%}")
c9.metric("Second-half acc",f"{meta.get('second_acc', 0):.1%}")
_delta = meta.get('second_acc', 0) - meta.get('first_acc', 0)
c10.metric("Acc change",    f"{_delta:+.1%}")

st.markdown("---")

# ─ Accuracy over time ──────────────────────────────────────────────────────────

st.markdown("### Accuracy over time")
st.markdown(
    '<p class="note">'
    "Running accuracy smoothed over recent predictions. Vertical lines mark ADWIN-triggered "
    "retraining events — you'll see a V-shaped dip before each one, followed by recovery. "
    "If second-half accuracy stays close to first-half, drift adaptation is working."
    "</p>",
    unsafe_allow_html=True,
)

_fig2 = os.path.join("outputs", "paper_figures", "fig2_running_accuracy.png")
if os.path.exists(_fig2):
    st.image(Image.open(_fig2), use_container_width=True)
else:
    acc_img = safe_img(plots.get("accuracy", ""))
    if acc_img:
        st.image(acc_img, use_container_width=True)
    else:
        st.caption("Accuracy plot not found.")

st.markdown("---")

# ─ Confusion matrix ────────────────────────────────────────────────────────────

st.markdown("### Confusion matrix & classification breakdown")
st.markdown(
    '<p class="note">'
    "Overall confusion matrix across all predictions in test-then-train order. "
    "Precision = of the times we predicted UP, how often were we right. "
    "Recall = of the actual UP events, how many did we catch."
    "</p>",
    unsafe_allow_html=True,
)

if ev:
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        cm_img = safe_img(plots.get("confusion_matrix", ""))
        if cm_img:
            st.image(cm_img, use_container_width=True)
        else:
            cm_raw = ev.get("confusion_matrix_raw", [[0,0],[0,0]])
            st.dataframe(
                pd.DataFrame(cm_raw,
                    index=["Actual DOWN", "Actual UP"],
                    columns=["Pred DOWN", "Pred UP"]),
                use_container_width=True,
            )

    with col_r:
        class_report = ev.get("class_report", {})
        if class_report:
            st.markdown("**Per-class breakdown**")
            rows = []
            for cls, d in class_report.items():
                rows.append({
                    "Class":     cls.replace("class_", ""),
                    "Precision": f"{d.get('precision', 0):.3f}",
                    "Recall":    f"{d.get('recall', 0):.3f}",
                    "F1":        f"{d.get('f1', 0):.3f}",
                    "Support":   d.get("support", 0),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        first_cm  = ev.get("first_half_cm", {})
        second_cm = ev.get("second_half_cm", {})
        if first_cm and second_cm:
            st.markdown("**First half vs second half**")
            st.dataframe(pd.DataFrame([
                {"Period": "First half",
                 "Accuracy": f"{first_cm.get('accuracy',0):.3f}",
                 "F1": f"{first_cm.get('f1',0):.3f}",
                 "Kappa": f"{first_cm.get('kappa',0):.3f}"},
                {"Period": "Second half",
                 "Accuracy": f"{second_cm.get('accuracy',0):.3f}",
                 "F1": f"{second_cm.get('f1',0):.3f}",
                 "Kappa": f"{second_cm.get('kappa',0):.3f}"},
            ]), hide_index=True, use_container_width=True)

st.markdown("---")

# ─ Recovery analysis ───────────────────────────────────────────────────────────

st.markdown("### Recovery after retraining")
st.markdown(
    '<p class="note">'
    "After each retrain, how long did it take for the model to get back above 85% "
    "windowed accuracy? Short recovery = the model adapted quickly. "
    "No recovery = the distribution may still be shifting."
    "</p>",
    unsafe_allow_html=True,
)

recovery = ev.get("recovery_analysis", {})
if recovery and recovery.get("events", 0) > 0:
    r1, r2 = st.columns(2)
    r1.metric("Retraining events", recovery.get("events", 0))
    avg = recovery.get("avg_recovery_time")
    r2.metric("Avg recovery time", f"{avg} samples" if avg else "—")

    details = recovery.get("details", [])
    if details:
        rec_df = pd.DataFrame(details).rename(columns={
            "retrain_at":        "Retrain at",
            "pre_drift_acc":     "Acc before",
            "recovery_at":       "Recovered at",
            "recovery_samples":  "Samples to recover",
            "post_recovery_acc": "Acc after",
            "recovered":         "Recovered?",
        })
        st.dataframe(rec_df, hide_index=True, use_container_width=True)
else:
    st.caption("No recovery data for this run.")

st.markdown("---")

# ─ Detector comparison ─────────────────────────────────────────────────────────

st.markdown("### Detector comparison — ADWIN vs DDM vs EDDM")
st.markdown(
    '<p class="note">'
    "All three detectors ran on the same error stream. "
    "Only ADWIN triggered retraining — DDM and EDDM ran in parallel for comparison. "
    "The chart below shows total detections, average inter-drift gap, and the full "
    "detection timeline across all 45,312 samples."
    "</p>",
    unsafe_allow_html=True,
)

_fig1 = os.path.join("outputs", "paper_figures", "fig1_detector_comparison.png")
if os.path.exists(_fig1):
    st.image(Image.open(_fig1), use_container_width=True)
else:
    if comp:
        rows = []
        for name, stats in comp.items():
            gap = stats.get("avg_inter_drift_gap")
            rows.append({
                "Detector":          name,
                "Drifts detected":   stats.get("total_drifts", 0),
                "Avg gap (samples)": f"{gap:,.0f}" if gap else "—",
                "First 5 detections": str(stats.get("drift_indices", [])[:5]),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

if comp:
    st.markdown("**Detector summary table**")
    rows = []
    for name, stats in comp.items():
        gap = stats.get("avg_inter_drift_gap")
        rows.append({
            "Detector":          name,
            "Drifts detected":   stats.get("total_drifts", 0),
            "Avg gap (samples)": f"{gap:,.0f}" if gap else "—",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.markdown("---")

# ─ Retraining events ───────────────────────────────────────────────────────────

st.markdown("### Retraining events")
st.markdown(
    '<p class="note">'
    "Every row is one retrain. The accuracy shown is the model's running accuracy "
    "right at the moment drift was called — usually at or near a local low."
    "</p>",
    unsafe_allow_html=True,
)

rt_idx  = rt.get("retraining_indices", [])
rt_accs = rt.get("accuracy_at_each_drift", [])

if rt_idx:
    rt_df = pd.DataFrame({
        "#":              range(1, len(rt_idx) + 1),
        "Sample":         rt_idx,
        "Accuracy then":  [f"{a:.1%}" for a in rt_accs],
    })
    st.dataframe(rt_df, hide_index=True, use_container_width=True)

    with st.expander("Chart"):
        st.line_chart(
            pd.DataFrame({"Accuracy at retrain": rt_accs}, index=rt_idx),
            use_container_width=True,
        )
else:
    st.caption("No retraining events recorded.")

st.markdown("---")

# ─ Model comparison ─────────────────────────────────────────────────────────────

st.markdown("### Model comparison — Static vs Continuous vs Adaptive")
st.markdown(
    '<p class="note">'
    "Comparing three setups: a frozen static model, a continuously-updated model "
    "(no resets), and the adaptive ADWIN model. The static model's accuracy drop "
    "confirms concept drift is real. ADWIN outperforms the static baseline by over "
    "15 percentage points across accuracy, F1, and kappa."
    "</p>",
    unsafe_allow_html=True,
)

_fig3 = os.path.join("outputs", "paper_figures", "fig3_model_comparison.png")
if os.path.exists(_fig3):
    st.image(Image.open(_fig3), use_container_width=True)
else:
    st.caption("Model comparison figure not found. Run comparison.py to generate it.")

st.markdown("---")

# ─ Delta ablation ────────────────────────────────────────────────────────────────

st.markdown("### ADWIN delta sensitivity")
st.markdown(
    '<p class="note">'
    "How does changing ADWIN's delta parameter affect performance? Lower delta = "
    "fewer resets, less post-reset accuracy loss, better overall accuracy. "
    "At delta=0.001, only 5 resets were needed and accuracy reached 91.09%."
    "</p>",
    unsafe_allow_html=True,
)

_fig4 = os.path.join("outputs", "paper_figures", "fig4_delta_ablation.png")
if os.path.exists(_fig4):
    st.image(Image.open(_fig4), use_container_width=True)
else:
    st.caption("Delta ablation figure not found. Run comparison.py to generate it.")

st.markdown("---")

# ─ Confusion matrix + recovery ───────────────────────────────────────────────────

st.markdown("### Confusion matrix & recovery analysis")
st.markdown(
    '<p class="note">'
    "The confusion matrix confirms no majority-class bias (balanced TP/TN). "
    "The recovery histogram shows how many samples the model needed after each reset "
    "to climb back above 85% accuracy — all 27 drift events were fully recovered."
    "</p>",
    unsafe_allow_html=True,
)

_fig5 = os.path.join("outputs", "paper_figures", "fig5_cm_and_recovery.png")
if os.path.exists(_fig5):
    st.image(Image.open(_fig5), use_container_width=True)
else:
    cm_img = safe_img(plots.get("confusion_matrix", ""))
    if cm_img:
        st.image(cm_img, use_container_width=True)
    else:
        st.caption("Confusion matrix figure not found.")

st.markdown("---")

# ─ Raw log ─────────────────────────────────────────────────────────────────────

st.markdown("### Raw log")
log_path = meta.get("log_path", "")
if log_path and os.path.exists(log_path):
    with st.expander("Browse log data"):
        log_df  = pd.read_csv(log_path)
        min_idx = int(log_df["sample_index"].min())
        max_idx = int(log_df["sample_index"].max())

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            rng = st.slider("Sample range", min_idx, max_idx, (min_idx, max_idx))
        with fc2:
            only_drift = st.checkbox("Drift events only")
        with fc3:
            only_error = st.checkbox("Errors only")

        filt = log_df[(log_df["sample_index"] >= rng[0]) & (log_df["sample_index"] <= rng[1])]
        if only_drift:
            filt = filt[filt["drift_detected"] == 1]
        if only_error:
            filt = filt[filt["error"] == 1]

        st.caption(f"{len(filt):,} of {len(log_df):,} rows")
        st.dataframe(filt, use_container_width=True, height=380)

        st.download_button(
            "Download CSV", filt.to_csv(index=False),
            file_name="filtered_log.csv", mime="text/csv",
        )
else:
    st.caption("Log file not found. Run detection first.")

st.markdown("---")
st.markdown(
    f"<p style='font-size:0.82rem;color:{theme_inline['color_faded']};'>"
    "Final Year Capstone Project &nbsp;·&nbsp; "
    "Adaptive Machine Learning for Real-Time Systems"
    "</p>",
    unsafe_allow_html=True,
)
