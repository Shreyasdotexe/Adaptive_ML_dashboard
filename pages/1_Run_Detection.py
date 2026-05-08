"""
pages/1_Run_Detection.py
Run the detection pipeline directly from the browser.
"""

import os
import sys
import subprocess
import threading
import queue
import time
import json
import glob

import streamlit as st

st.set_page_config(
    page_title="Run Detection",
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

# ── session state ──────────────────────────────────────────────────────────────

for key, default in [
    ("run_status", "idle"),
    ("terminal_lines", []),
    ("process", None),
    ("output_queue", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── helpers ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "outputs", "reports")

DETECTOR_NOTES = {
    "ADWIN": (
        "Keeps a sliding window of error values. When the error mean in one part of "
        "the window differs enough from another, it calls drift. Works well for both "
        "sudden and gradual shifts. The **delta** parameter controls sensitivity — "
        "lower means it reacts sooner."
    ),
    "DDM": (
        "Tracks the running error rate and standard deviation. Drift is flagged when "
        "the error rate climbs past a statistical threshold. Good for sudden changes. "
        "No extra parameters to tune."
    ),
    "EDDM": (
        "Looks at the gap between consecutive mistakes. If errors start happening "
        "closer together, it considers that a sign of drift. Tends to catch things "
        "early, before accuracy visibly drops."
    ),
}

import re
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def _colorise(line: str) -> str:
    line = ANSI_ESCAPE.sub('', line)
    lo = line.lower()
    if any(k in lo for k in ["drift", "retraining", "↺"]):
        return f'<span class="t-drift">{line}</span>'
    if any(k in lo for k in ["✓", "ok", "saved", "ready", "complete"]):
        return f'<span class="t-ok">{line}</span>'
    if any(k in lo for k in ["warn", "⚠"]):
        return f'<span class="t-warn">{line}</span>'
    if any(k in lo for k in ["run id", "dataset", "samples", "features", "delta", "cooldown", "primary", "comparison"]):
        return f'<span class="t-info">{line}</span>'
    return line


def _render_terminal(lines):
    return "<br>".join(_colorise(ln) for ln in lines)


def _enqueue(stream, q):
    try:
        for line in iter(stream.readline, ""):
            q.put(line)
    finally:
        stream.close()


def _load_latest_meta():
    files = sorted(glob.glob(os.path.join(REPORTS_DIR, "meta_*.json")))
    if not files:
        return {}
    with open(files[-1]) as f:
        return json.load(f)


# ── sidebar config ─────────────────────────────────────────────────────────────

st.sidebar.markdown("### Config")
st.sidebar.markdown("---")

detector = st.sidebar.selectbox(
    "Primary detector",
    ["ADWIN", "DDM", "EDDM"],
    index=0,
    help="This detector drives retraining. The other two just observe.",
)

st.sidebar.markdown(
    f"<p style='font-size:0.82rem;color:{theme_inline['color_muted']};line-height:1.5;'>"
    f"{DETECTOR_NOTES[detector]}</p>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

adwin_delta = st.sidebar.slider(
    "ADWIN delta", 0.001, 0.5, 0.1, 0.001, format="%.3f",
    help="Lower = more sensitive. Only relevant when ADWIN is primary.",
)

max_samples_label = st.sidebar.selectbox(
    "Max samples",
    ["Full dataset (~45k)", "10,000", "5,000", "2,000", "1,000"],
    help="Limit how many samples to process. Useful for a quick test run.",
)
_max_map = {
    "Full dataset (~45k)": "None", "10,000": "10000",
    "5,000": "5000", "2,000": "2000", "1,000": "1000",
}
max_samples_val = _max_map[max_samples_label]

warmup = st.sidebar.slider(
    "Warm-up samples", 50, 500, 200, 50,
    help="Samples used to fit the scaler before predictions start.",
)

cooldown = st.sidebar.slider(
    "Cooldown between retrains", 50, 1000, 300, 50,
    help="Minimum samples that must pass before another retraining can trigger.",
)

# ── page header ────────────────────────────────────────────────────────────────

st.markdown("## Run Detection")
st.markdown(
    "Configure and start the pipeline below. Output streams live — "
    "same as running `python main.py` in a terminal."
)
st.markdown("---")

# ── control row ────────────────────────────────────────────────────────────────

btn_col, status_col = st.columns([2, 5], gap="large")

with btn_col:
    start = st.button(
        "Start",
        type="primary",
        use_container_width=True,
        disabled=(st.session_state.run_status == "running"),
    )
    stop = st.button(
        "Stop",
        use_container_width=True,
        disabled=(st.session_state.run_status != "running"),
    )

status_placeholder = status_col.empty()
def update_status_label():
    _dot = {
        "idle":    "dot-idle",
        "running": "dot-running",
        "done":    "dot-done",
        "stopped": "dot-stopped",
    }.get(st.session_state.run_status, "dot-idle")
    _label = {
        "idle":    "Idle",
        "running": "Running…",
        "done":    "Finished",
        "stopped": "Stopped",
    }.get(st.session_state.run_status, "Idle")
    status_placeholder.markdown(
        f"<p style='margin-top:0.6rem;font-size:0.9rem;color:{theme_inline['color_muted']};'>"
        f"<span class='{_dot}'></span>{_label}</p>",
        unsafe_allow_html=True,
    )

update_status_label()

st.markdown("---")

# ── stop handler ───────────────────────────────────────────────────────────────

if stop and st.session_state.process is not None:
    try:
        st.session_state.process.terminate()
    except Exception:
        pass
    st.session_state.run_status = "stopped"
    st.session_state.process = None
    st.rerun()

# ── start handler ──────────────────────────────────────────────────────────────

if start:
    st.session_state.terminal_lines = []
    st.session_state.run_status = "running"
    update_status_label()

    launcher = f"""
import os, sys
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'): sys.stderr.reconfigure(encoding='utf-8')
sys.path.insert(0, r'{PROJECT_ROOT}')
os.chdir(r'{PROJECT_ROOT}')
from main import run_pipeline
run_pipeline({{
    "csv_path":           os.path.join("data", "electricity.csv"),
    "max_samples":        {max_samples_val},
    "warmup_size":        {warmup},
    "adwin_delta":        {adwin_delta},
    "cooldown_samples":   {cooldown},
    "post_drift_settle":  300,
    "model_random_state": 42,
    "n_features":         8,
    "log_dir":            "data/logs",
    "plots_dir":          "outputs/plots",
    "reports_dir":        "outputs/reports",
    "primary_detector":   "{detector}",
}})
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", launcher],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", bufsize=1, cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
    )
    st.session_state.process = proc
    q: queue.Queue = queue.Queue()
    st.session_state.output_queue = q
    threading.Thread(target=_enqueue, args=(proc.stdout, q), daemon=True).start()

# ── terminal output ─────────────────────────────────────────────────────────────

st.markdown("**Output**")
st.caption("Green = success &nbsp; · &nbsp; Red = drift / retrain &nbsp; · &nbsp; Blue = config info")

term = st.empty()

if st.session_state.run_status == "running" and st.session_state.output_queue is not None:
    q    = st.session_state.output_queue
    proc = st.session_state.process
    MAX  = 600

    try:
        while True:
            st.session_state.terminal_lines.append(q.get_nowait().rstrip())
    except queue.Empty:
        pass

    if len(st.session_state.terminal_lines) > MAX:
        st.session_state.terminal_lines = st.session_state.terminal_lines[-MAX:]

    term.markdown(
        f'<div class="terminal">{_render_terminal(st.session_state.terminal_lines)}</div>',
        unsafe_allow_html=True,
    )

    if proc.poll() is not None:
        # Process finished
        try:
            while True:
                st.session_state.terminal_lines.append(q.get_nowait().rstrip())
        except queue.Empty:
            pass
        term.markdown(
            f'<div class="terminal">{_render_terminal(st.session_state.terminal_lines)}</div>',
            unsafe_allow_html=True,
        )
        st.session_state.run_status = "done"
        st.session_state.process    = None
        st.rerun()
    else:
        # Still running, yield to UI and repeat
        time.sleep(0.3)
        st.rerun()

else:
    if st.session_state.terminal_lines:
        term.markdown(
            f'<div class="terminal">{_render_terminal(st.session_state.terminal_lines)}</div>',
            unsafe_allow_html=True,
        )
    else:
        term.markdown(
            f'<div class="terminal" style="color:{theme_inline["color_muted"]};">No output yet.</div>',
            unsafe_allow_html=True,
        )

# ── quick summary after run ────────────────────────────────────────────────────

if st.session_state.run_status == "done":
    st.markdown("---")
    st.markdown("**Run complete — quick look**")

    meta = _load_latest_meta()
    if meta:
        comp   = meta.get("detector_comparison", {})
        rt     = meta.get("retraining_events", {})
        ev     = meta.get("evaluation", {})
        cm     = ev.get("confusion_matrix", {})

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Samples",     f"{meta.get('n_samples', 0):,}")
        c2.metric("Accuracy",    f"{meta.get('final_acc', 0):.1%}")
        c3.metric("F1",          f"{cm.get('f1', 0):.3f}")
        c4.metric("Kappa",       f"{ev.get('kappa', 0):.3f}")
        c5.metric("Retrains",    rt.get("total_retraining_events", 0))

        d1, d2, d3 = st.columns(3)
        d1.metric("ADWIN detections", comp.get("ADWIN", {}).get("total_drifts", "—"))
        d2.metric("DDM detections",   comp.get("DDM",   {}).get("total_drifts", "—"))
        d3.metric("EDDM detections",  comp.get("EDDM",  {}).get("total_drifts", "—"))

        st.markdown("---")
        drifts = comp.get(detector, {}).get("drift_indices", [])
        if drifts:
            drift_str = ", ".join(map(str, drifts))
        else:
            drift_str = "No drift detected during this run."
            
        st.markdown(f"""
        <div style="background:{theme_inline['bg_card']}; border:1px solid {theme_inline['border_card']}; border-radius:8px; padding:1.2rem; margin-bottom:1rem;">
            <div style="font-size:0.75rem; font-weight:700; color:{theme_inline['color_muted']}; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.5rem;">
                Drift Points Detected by {detector} (Primary)
            </div>
            <div style="font-family:'Fira Mono', monospace; font-size:0.95rem; color:{theme_inline['color_text']}; line-height:1.6;">
                {drift_str}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.info("Open the **Results** page to see all plots and breakdowns.")
    else:
        st.warning("Couldn't find the results file — try refreshing.")
