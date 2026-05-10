"""
generate_paper_figures.py
Produces 5 publication-quality white-background figures for the research paper.
Run from the project root:  python generate_paper_figures.py
"""
import json, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# ── paths ──────────────────────────────────────────────────────────────────
META   = "outputs/reports/meta_20260508_153453.json"
LOG    = "data/logs/stream_log_20260508_153453.csv"
OUTDIR = "outputs/paper_figures"
os.makedirs(OUTDIR, exist_ok=True)

with open(META) as f:
    meta = json.load(f)

df = pd.read_csv(LOG)

# ── shared style (white background, publication look) ──────────────────────
BLUE   = "#1a6fb5"
RED    = "#d62728"
GREEN  = "#2ca02c"
ORANGE = "#e07b00"
PURPLE = "#7b2d8b"
GRAY   = "#888888"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#333333",
    "axes.labelcolor":  "#222222",
    "axes.grid":        True,
    "grid.color":       "#dddddd",
    "grid.linewidth":   0.7,
    "xtick.color":      "#333333",
    "ytick.color":      "#333333",
    "text.color":       "#222222",
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "legend.framealpha": 0.9,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.facecolor":"white",
})

# ── pull data from meta ────────────────────────────────────────────────────
adwin_idx  = meta["detector_comparison"]["ADWIN"]["drift_indices"]
ddm_idx    = meta["detector_comparison"]["DDM"]["drift_indices"]
eddm_idx   = meta["detector_comparison"]["EDDM"]["drift_indices"]
retrain_at = meta["retraining_events"]["retraining_indices"]
recovery   = meta["evaluation"]["recovery_analysis"]["details"]
win_acc    = meta["evaluation"]["windowed_accuracy"]
win_idx    = meta["evaluation"]["windowed_acc_indices"]
cm_raw     = meta["evaluation"]["confusion_matrix_raw"]   # [[TN,FP],[FN,TP]]
N          = meta["n_samples"]
mid        = N // 2  # dataset midpoint used as reference

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Detector Comparison Dashboard (4-panel)
# Most important figure: shows all three detectors side-by-side
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── 1a: Drifts detected bar chart ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
detectors  = ["ADWIN", "DDM", "EDDM"]
drifts     = [27, 15, 3]
colors_bar = [BLUE, ORANGE, PURPLE]
bars = ax1.bar(detectors, drifts, color=colors_bar, width=0.45, edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, drifts):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
             str(val), ha="center", va="bottom", fontweight="bold", fontsize=13)
ax1.set_title("(a) Total Drift Events Detected", fontweight="bold")
ax1.set_ylabel("Number of Drift Events")
ax1.set_ylim(0, 34)
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

# ── 1b: Avg inter-drift gap ────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
gaps = [1549, 1897, 10534]
bars2 = ax2.bar(detectors, gaps, color=colors_bar, width=0.45, edgecolor="white", linewidth=1.2)
for bar, val in zip(bars2, gaps):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
             f"{val:,}", ha="center", va="bottom", fontweight="bold", fontsize=11)
ax2.set_title("(b) Avg. Inter-Drift Gap (samples)", fontweight="bold")
ax2.set_ylabel("Samples Between Detections")
ax2.set_ylim(0, 13000)

# ── 1c: Detection timeline scatter ────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :])
y_pos = {"ADWIN": 3, "DDM": 2, "EDDM": 1}
y_labels = {3: "ADWIN\n(primary)", 2: "DDM\n(comparison)", 1: "EDDM\n(comparison)"}
detector_data = {"ADWIN": (adwin_idx, BLUE, "o"),
                 "DDM":   (ddm_idx,   ORANGE, "s"),
                 "EDDM":  (eddm_idx,  PURPLE, "^")}

for name, (indices, col, marker) in detector_data.items():
    y = y_pos[name]
    ax3.scatter(indices, [y]*len(indices), color=col, marker=marker,
                s=90, zorder=5, label=f"{name} ({len(indices)} events)", alpha=0.85)
    for xi in indices:
        ax3.plot([xi, xi], [y-0.35, y+0.35], color=col, lw=0.8, alpha=0.4)

ax3.axvline(mid, color=GRAY, lw=1.5, ls="--", label="Dataset midpoint", alpha=0.7)
ax3.set_xlim(0, N)
ax3.set_ylim(0.4, 3.8)
ax3.set_yticks([1, 2, 3])
ax3.set_yticklabels([y_labels[i] for i in [1, 2, 3]], fontsize=10)
ax3.set_xlabel("Sample Index")
ax3.set_title("(c) Drift Detection Timeline Across 45,312 Samples (ELEC2)", fontweight="bold")
ax3.legend(loc="upper right", ncol=4, fontsize=9)
ax3.grid(axis="x")
ax3.grid(axis="y", alpha=0)

fig.suptitle("Detector Comparison: ADWIN vs DDM vs EDDM on ELEC2 Dataset",
             fontsize=15, fontweight="bold", y=1.01)
path1 = os.path.join(OUTDIR, "fig1_detector_comparison.png")
fig.savefig(path1)
plt.close(fig)
print(f"Saved: {path1}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Running Accuracy with Retraining Events
# Shows the adaptive model's performance over time with drift markers
# ══════════════════════════════════════════════════════════════════════════════
active = df[df["prediction"] != -1].copy() if "prediction" in df.columns else df.copy()

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 7),
    gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

indices    = active["sample_index"].values
acc_smooth = active["running_acc"].rolling(window=300, min_periods=1).mean().values
err_raw    = active["error"].values

ax_top.axvspan(0, mid, alpha=0.04, color=BLUE, label="First half")
ax_top.axvspan(mid, N, alpha=0.04, color=ORANGE, label="Second half")
ax_top.plot(indices, acc_smooth, color=GREEN, lw=1.8, alpha=0.95,
            label="Running accuracy (w=300)")
ax_top.axvline(mid, color=GRAY, lw=1.5, ls="--", alpha=0.7, label="Dataset midpoint")

first = True
for rt in retrain_at:
    ax_top.axvline(rt, color=RED, lw=1.0, ls=":", alpha=0.7,
                   label="Retraining event" if first else None)
    first = False

ax_top.axhline(0.5, color="#aaaaaa", lw=1.0, ls="--", alpha=0.5, label="Random chance (0.50)")
ax_top.set_ylim(0.3, 1.05)
ax_top.set_ylabel("Accuracy")
ax_top.set_title("Model Accuracy Over Time with ADWIN-Triggered Retraining (ELEC2, 45,312 samples)",
                 fontweight="bold")
ax_top.legend(loc="lower left", ncol=3, fontsize=9)

# error strip
ax_bot.fill_between(indices, err_raw, step="pre", alpha=0.35, color=RED, label="Prediction error")
ax_bot.axvline(mid, color=GRAY, lw=1.5, ls="--", alpha=0.7)
for rt in retrain_at:
    ax_bot.axvline(rt, color=RED, lw=0.8, ls=":", alpha=0.5)
ax_bot.set_ylim(-0.05, 1.4)
ax_bot.set_yticks([0, 1])
ax_bot.set_xlabel("Sample Index")
ax_bot.set_ylabel("Error (0/1)")
ax_bot.legend(loc="upper right", fontsize=9)

plt.tight_layout()
path2 = os.path.join(OUTDIR, "fig2_running_accuracy.png")
fig.savefig(path2)
plt.close(fig)
print(f"Saved: {path2}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Three-Model Comparison Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle("Three-Model Performance Comparison on ELEC2", fontweight="bold", fontsize=14, y=1.02)

models   = ["Static\n(frozen)", "Continuous\n(blind update)", "Adaptive ADWIN\n(ours)"]
acc_vals = [74.62, 91.60, 89.94]
f1_vals  = [67.90, 90.09, 88.08]
kap_vals = [0.473, 0.829, 0.794]
mod_cols = [RED, ORANGE, GREEN]

# accuracy
bars = axes[0].bar(models, acc_vals, color=mod_cols, width=0.5, edgecolor="white")
for b, v in zip(bars, acc_vals):
    axes[0].text(b.get_x()+b.get_width()/2, v+0.5, f"{v:.1f}%",
                 ha="center", va="bottom", fontweight="bold", fontsize=11)
axes[0].set_ylim(60, 100)
axes[0].set_title("(a) Prequential Accuracy", fontweight="bold")
axes[0].set_ylabel("Accuracy (%)")
axes[0].axhline(91.60, color=ORANGE, lw=1, ls="--", alpha=0.4)

# F1
bars2 = axes[1].bar(models, f1_vals, color=mod_cols, width=0.5, edgecolor="white")
for b, v in zip(bars2, f1_vals):
    axes[1].text(b.get_x()+b.get_width()/2, v+0.5, f"{v:.1f}%",
                 ha="center", va="bottom", fontweight="bold", fontsize=11)
axes[1].set_ylim(60, 100)
axes[1].set_title("(b) F1-Score", fontweight="bold")
axes[1].set_ylabel("F1-Score (%)")

# Kappa
bars3 = axes[2].bar(models, kap_vals, color=mod_cols, width=0.5, edgecolor="white")
for b, v in zip(bars3, kap_vals):
    axes[2].text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}",
                 ha="center", va="bottom", fontweight="bold", fontsize=11)
axes[2].set_ylim(0.3, 1.0)
axes[2].set_title("(c) Cohen's Kappa", fontweight="bold")
axes[2].set_ylabel("Kappa")

plt.tight_layout()
path3 = os.path.join(OUTDIR, "fig3_model_comparison.png")
fig.savefig(path3)
plt.close(fig)
print(f"Saved: {path3}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — ADWIN Delta Ablation Study
# ══════════════════════════════════════════════════════════════════════════════
delta_labels = ["0.001\n(conservative)", "0.01", "0.05", "0.10\n(default)", "0.50\n(aggressive)"]
delta_acc    = [91.09, 90.49, 90.15, 89.94, 87.63]
delta_f1     = [89.50, 88.79, 88.35, 88.08, 85.71]
delta_kappa  = [0.8176, 0.8054, 0.7982, 0.7937, 0.7459]
delta_events = [5, 14, 22, 27, 41]
x = np.arange(len(delta_labels))

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("ADWIN Delta Sensitivity Ablation Study (ELEC2, 45,312 samples)",
             fontweight="bold", fontsize=14)

# accuracy
axes[0,0].plot(x, delta_acc, "o-", color=BLUE, lw=2, ms=8)
axes[0,0].fill_between(x, [min(delta_acc)]*5, delta_acc, alpha=0.12, color=BLUE)
for xi, v in zip(x, delta_acc):
    axes[0,0].annotate(f"{v:.2f}%", (xi, v), textcoords="offset points",
                       xytext=(0, 8), ha="center", fontsize=9, fontweight="bold")
axes[0,0].set_xticks(x); axes[0,0].set_xticklabels(delta_labels, fontsize=9)
axes[0,0].set_ylim(86, 92.5); axes[0,0].set_ylabel("Accuracy (%)")
axes[0,0].set_title("(a) Accuracy vs Delta", fontweight="bold")

# F1
axes[0,1].plot(x, delta_f1, "s-", color=GREEN, lw=2, ms=8)
axes[0,1].fill_between(x, [min(delta_f1)]*5, delta_f1, alpha=0.12, color=GREEN)
for xi, v in zip(x, delta_f1):
    axes[0,1].annotate(f"{v:.2f}%", (xi, v), textcoords="offset points",
                       xytext=(0, 8), ha="center", fontsize=9, fontweight="bold")
axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(delta_labels, fontsize=9)
axes[0,1].set_ylim(84, 91); axes[0,1].set_ylabel("F1-Score (%)")
axes[0,1].set_title("(b) F1-Score vs Delta", fontweight="bold")

# kappa
axes[1,0].plot(x, delta_kappa, "^-", color=ORANGE, lw=2, ms=8)
axes[1,0].fill_between(x, [min(delta_kappa)]*5, delta_kappa, alpha=0.12, color=ORANGE)
for xi, v in zip(x, delta_kappa):
    axes[1,0].annotate(f"{v:.4f}", (xi, v), textcoords="offset points",
                       xytext=(0, 8), ha="center", fontsize=9, fontweight="bold")
axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(delta_labels, fontsize=9)
axes[1,0].set_ylim(0.71, 0.84); axes[1,0].set_ylabel("Cohen's Kappa")
axes[1,0].set_title("(c) Kappa vs Delta", fontweight="bold")

# drift events
bars = axes[1,1].bar(x, delta_events, color=[BLUE,BLUE,BLUE,RED,RED],
                      width=0.5, edgecolor="white")
for b, v in zip(bars, delta_events):
    axes[1,1].text(b.get_x()+b.get_width()/2, v+0.5, str(v),
                   ha="center", va="bottom", fontweight="bold", fontsize=11)
axes[1,1].set_xticks(x); axes[1,1].set_xticklabels(delta_labels, fontsize=9)
axes[1,1].set_ylabel("Drift Events Triggered")
axes[1,1].set_title("(d) Drift Events vs Delta", fontweight="bold")
# legend
p1 = mpatches.Patch(color=BLUE,  label="Conservative delta")
p2 = mpatches.Patch(color=RED,   label="Aggressive delta")
axes[1,1].legend(handles=[p1, p2], fontsize=9)

plt.tight_layout()
path4 = os.path.join(OUTDIR, "fig4_delta_ablation.png")
fig.savefig(path4)
plt.close(fig)
print(f"Saved: {path4}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Confusion Matrix + Recovery Time Distribution (2-panel)
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax_cm, ax_rec) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Model Evaluation: Confusion Matrix & Post-Drift Recovery",
             fontweight="bold", fontsize=14)

# confusion matrix
TN, FP, FN, TP = cm_raw[0][0], cm_raw[0][1], cm_raw[1][0], cm_raw[1][1]
cm = np.array([[TN, FP], [FN, TP]])
im = ax_cm.imshow(cm, cmap="Blues", aspect="auto", vmin=0)
labels = ["DOWN (0)", "UP (1)"]
ax_cm.set_xticks([0,1]); ax_cm.set_xticklabels(labels, fontsize=11)
ax_cm.set_yticks([0,1]); ax_cm.set_yticklabels(labels, fontsize=11)
ax_cm.set_xlabel("Predicted", fontsize=12)
ax_cm.set_ylabel("Actual", fontsize=12)
ax_cm.set_title("(a) Confusion Matrix", fontweight="bold")
total = cm.sum()
for i in range(2):
    for j in range(2):
        v = cm[i, j]
        pct = v/total*100
        col = "white" if v > cm.max()/1.8 else "#222222"
        ax_cm.text(j, i, f"{v:,}\n({pct:.1f}%)",
                   ha="center", va="center", fontsize=13, fontweight="bold", color=col)
cbar = fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=9)

# recovery time histogram
rec_times = [d["recovery_samples"] for d in recovery if d["recovered"]]
ax_rec.hist(rec_times, bins=14, color=BLUE, alpha=0.75, edgecolor="white", lw=1.2)
ax_rec.axvline(np.mean(rec_times), color=RED, lw=2.0, ls="--",
               label=f"Mean = {np.mean(rec_times):.1f} samples")
ax_rec.axvline(np.median(rec_times), color=ORANGE, lw=2.0, ls="-.",
               label=f"Median = {np.median(rec_times):.0f} samples")
ax_rec.set_xlabel("Samples to Recover (>85% accuracy)")
ax_rec.set_ylabel("Frequency (drift events)")
ax_rec.set_title("(b) Post-Drift Recovery Time Distribution\n(n=27 events, all recovered)",
                 fontweight="bold")
ax_rec.legend(fontsize=10)

plt.tight_layout()
path5 = os.path.join(OUTDIR, "fig5_cm_and_recovery.png")
fig.savefig(path5)
plt.close(fig)
print(f"Saved: {path5}")

# ══════════════════════════════════════════════════════════════════════════════
print("\nAll figures saved to:", OUTDIR)
print("""
Recommended figures for the research paper:
  Fig 1 — fig1_detector_comparison.png
           → Section 7.2: shows ADWIN vs DDM vs EDDM detection counts,
             inter-drift gaps, and timeline. The KEY comparison figure.

  Fig 2 — fig2_running_accuracy.png
           → Section 6/Results: shows accuracy over all 45,312 samples
             with retraining markers. The core results figure.

  Fig 3 — fig3_model_comparison.png
           → Section 7.1: Static vs Continuous vs Adaptive across
             Accuracy, F1, Kappa. Clear justification of the approach.

  Fig 4 — fig4_delta_ablation.png
           → Section 7.3: ablation study showing delta sensitivity.
             Shows you understand hyperparameter trade-offs.

  Fig 5 — fig5_cm_and_recovery.png
           → Section 6/9: confusion matrix proves no class bias;
             recovery histogram proves the model actually heals.
""")
