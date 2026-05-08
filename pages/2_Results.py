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
    "The top panel shows running accuracy smoothed over recent predictions. "
    "The vertical lines mark retraining events — you'll usually see a dip right "
    "before each one, followed by recovery as the model adapts. "
    "If second-half accuracy stays close to first-half, drift adaptation is working."
    "</p>",
    unsafe_allow_html=True,
)

acc_img = safe_img(plots.get("accuracy", ""))
if acc_img:
    st.image(acc_img, use_container_width=True)
else:
    st.caption("Accuracy plot not found.")

st.markdown("---")

# ─ Drift signal ────────────────────────────────────────────────────────────────

st.markdown("### When did ADWIN see drift?")
st.markdown(
    '<p class="note">'
    "This shows the rolling error rate fed into ADWIN. "
    "Each red marker is a drift call — the point where the error rate "
    "shifted enough for ADWIN to act. A cluster of markers suggests "
    "a prolonged unstable period."
    "</p>",
    unsafe_allow_html=True,
)

drift_img = safe_img(plots.get("drift", ""))
if drift_img:
    st.image(drift_img, use_container_width=True)
else:
    st.caption("Drift signal plot not found.")

st.markdown("---")

# ─ Detection timeline ──────────────────────────────────────────────────────────

st.markdown("### Detection timeline")
st.markdown(
    '<p class="note">'
    "A stem plot of every individual drift event from ADWIN. "
    "Even spacing suggests the detector is picking up periodic shifts. "
    "Bursts of activity in one region point to concentrated instability there."
    "</p>",
    unsafe_allow_html=True,
)

tl_img = safe_img(plots.get("timeline", ""))
if tl_img:
    st.image(tl_img, use_container_width=True)
else:
    st.caption("Timeline plot not found.")

st.markdown("---")

# ─ Windowed accuracy ───────────────────────────────────────────────────────────

st.markdown("### Windowed accuracy")
st.markdown(
    '<p class="note">'
    "Unlike the cumulative accuracy above, this only looks at the last 500 "
    "predictions at any point in time. It's a better signal of how the model "
    "is doing *right now* vs. historically. Drops that recover quickly = "
    "the retrain mechanism working. Drops that don't recover = worth investigating."
    "</p>",
    unsafe_allow_html=True,
)

w_img = safe_img(plots.get("windowed_accuracy", ""))
if w_img:
    st.image(w_img, use_container_width=True)

w_acc = ev.get("windowed_accuracy", [])
w_idx = ev.get("windowed_acc_indices", [])
if w_acc and w_idx and len(w_acc) == len(w_idx):
    with st.expander("Interactive chart"):
        chart_df = pd.DataFrame({
            "Sample": w_idx,
            "Windowed accuracy": w_acc,
        }).set_index("Sample")
        st.line_chart(chart_df, use_container_width=True)

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

st.markdown("### Detector comparison")
st.markdown(
    '<p class="note">'
    "All three detectors ran on the same error stream. "
    "Only the primary one triggered retraining — the others just recorded "
    "what they would have done. This table lets you compare their sensitivity."
    "</p>",
    unsafe_allow_html=True,
)

if comp:
    rows = []
    for name, stats in comp.items():
        gap = stats.get("avg_inter_drift_gap")
        rows.append({
            "Detector":         name,
            "Drifts detected":  stats.get("total_drifts", 0),
            "Avg gap (samples)": f"{gap:,.0f}" if gap else "—",
            "First 5 detections": str(stats.get("drift_indices", [])[:5]),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("**Detection count**")
    bar_df = pd.DataFrame({
        "Detector": list(comp.keys()),
        "Detections": [v.get("total_drifts", 0) for v in comp.values()],
    }).set_index("Detector")
    st.bar_chart(bar_df, use_container_width=True)

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

# ─ Feature shift ───────────────────────────────────────────────────────────────

st.markdown("### Feature distribution shift")
st.markdown(
    '<p class="note">'
    "Comparing how each feature's distribution looked in the first half of the "
    "stream versus the second. Big differences between the two halves = "
    "the data really did change, confirming the drift signals above are real."
    "</p>",
    unsafe_allow_html=True,
)

f_img  = safe_img(plots.get("feature", ""))
mf_img = safe_img(plots.get("multi_feature", ""))
if f_img:
    st.image(f_img, use_container_width=True)
if mf_img:
    st.image(mf_img, use_container_width=True)
if not f_img and not mf_img:
    st.caption("Feature distribution plots not found.")

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
