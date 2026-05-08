import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List

STYLE = {
    "bg":"#0f1117","panel":"#1a1d27","grid":"#2a2d3a","text":"#e0e0e0",
    "accent_blue":"#4a9eff","accent_red":"#ff5c5c","accent_green":"#4ade80",
    "accent_orange":"#fbbf24","font":"monospace",
}

def _apply_dark_style(ax, title, xlabel, ylabel):
    ax.set_facecolor(STYLE["panel"])
    ax.tick_params(colors=STYLE["text"], labelsize=9)
    ax.set_title(title, color=STYLE["text"], fontsize=12, fontfamily=STYLE["font"], pad=10)
    ax.set_xlabel(xlabel, color=STYLE["text"], fontsize=10, fontfamily=STYLE["font"])
    ax.set_ylabel(ylabel, color=STYLE["text"], fontsize=10, fontfamily=STYLE["font"])
    ax.grid(True, color=STYLE["grid"], linewidth=0.6, linestyle="--", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE["grid"])

class ResultVisualizer:
    def __init__(self, plots_dir, run_id):
        self.plots_dir = plots_dir
        self.run_id = run_id
        os.makedirs(plots_dir, exist_ok=True)

    def plot_drift_signal(self, df, true_drift_point, detected_drift_indices,
                          window=150, signal_col="detector_input"):
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor(STYLE["bg"])
        indices = df["sample_index"].values
        signal  = df[signal_col].rolling(window=window, min_periods=1).mean()
        ax.axvspan(indices[0], true_drift_point, alpha=0.06, color=STYLE["accent_blue"], label="First Half")
        ax.axvspan(true_drift_point, indices[-1], alpha=0.06, color=STYLE["accent_orange"], label="Second Half")
        ax.plot(indices, signal, color=STYLE["accent_blue"], linewidth=1.2, alpha=0.9,
                label=f"Error Signal (rolling mean, w={window})")
        ax.axvline(x=true_drift_point, color=STYLE["accent_orange"], linewidth=1.5,
                   linestyle="--", label="Dataset Midpoint")
        first = True
        for idx in detected_drift_indices:
            lbl = "ADWIN Detection" if first else None
            first = False
            ax.axvline(x=idx, color=STYLE["accent_red"], linewidth=1.2, linestyle=":", alpha=0.7, label=lbl)
        _apply_dark_style(ax, "ADWIN Error Signal (ELEC2 Dataset)",
                          "Sample Index", f"Model Error (rolling mean, w={window})")
        ax.legend(facecolor=STYLE["panel"], edgecolor=STYLE["grid"],
                  labelcolor=STYLE["text"], fontsize=9, loc="upper left")
        plt.tight_layout()
        path = os.path.join(self.plots_dir, f"drift_signal_{self.run_id}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
        plt.close(fig)
        return path

    def plot_feature_distribution(self, df, true_drift_point, feature_col="feature_0"):
        fig, ax = plt.subplots(figsize=(11, 5))
        fig.patch.set_facecolor(STYLE["bg"])
        pre  = df[df["sample_index"] < true_drift_point][feature_col]
        post = df[df["sample_index"] >= true_drift_point][feature_col]
        pre_mean, post_mean = pre.mean(), post.mean()
        ax.hist(pre,  bins=80, alpha=0.60, color=STYLE["accent_blue"],
                label=f"First Half  (n={len(pre)}, μ={pre_mean:.3f})", density=True)
        ax.hist(post, bins=80, alpha=0.60, color=STYLE["accent_orange"],
                label=f"Second Half (n={len(post)}, μ={post_mean:.3f})", density=True)
        ax.axvline(pre_mean,  color=STYLE["accent_blue"],   linewidth=1.8, linestyle="--")
        ax.axvline(post_mean, color=STYLE["accent_orange"], linewidth=1.8, linestyle="--")
        _apply_dark_style(ax, f"Feature Distribution ({feature_col})",
                          f"Scaled {feature_col}", "Density")
        ax.legend(facecolor=STYLE["panel"], edgecolor=STYLE["grid"],
                  labelcolor=STYLE["text"], fontsize=9)
        plt.tight_layout()
        path = os.path.join(self.plots_dir, f"feature_distribution_{self.run_id}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
        plt.close(fig)
        return path

    def plot_drift_detections_timeline(self, df, true_drift_point, detected_drift_indices):
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 6),
            gridspec_kw={"height_ratios":[3,1]}, sharex=True)
        fig.patch.set_facecolor(STYLE["bg"])
        indices = df["sample_index"].values
        signal  = df["detector_input"].rolling(window=150, min_periods=1).mean()
        ax_top.axvspan(indices[0], true_drift_point, alpha=0.06, color=STYLE["accent_blue"])
        ax_top.axvspan(true_drift_point, indices[-1], alpha=0.06, color=STYLE["accent_orange"])
        ax_top.plot(indices, signal, color=STYLE["accent_blue"], linewidth=1.2, alpha=0.9)
        ax_top.axvline(x=true_drift_point, color=STYLE["accent_orange"],
                       linewidth=1.5, linestyle="--", label="Dataset Midpoint")
        for i, idx in enumerate(detected_drift_indices):
            ax_top.axvline(x=idx, color=STYLE["accent_red"], linewidth=1.0,
                           linestyle=":", alpha=0.7, label="ADWIN Detection" if i==0 else None)
        _apply_dark_style(ax_top, "Detection Timeline (ELEC2)", "", "Error Signal")
        ax_top.legend(facecolor=STYLE["panel"], edgecolor=STYLE["grid"],
                      labelcolor=STYLE["text"], fontsize=9, loc="upper left")
        ax_bot.set_facecolor(STYLE["panel"])
        if detected_drift_indices:
            ml, sl, bl = ax_bot.stem(detected_drift_indices, [1]*len(detected_drift_indices),
                                     linefmt=STYLE["accent_red"], markerfmt="v", basefmt=" ")
            plt.setp(sl, linewidth=1.2, alpha=0.7)
            plt.setp(ml, color=STYLE["accent_red"], markersize=7)
        ax_bot.axvline(x=true_drift_point, color=STYLE["accent_orange"], linewidth=1.5, linestyle="--")
        _apply_dark_style(ax_bot, "", "Sample Index", "Drift Events")
        ax_bot.set_ylim(0, 1.8)
        ax_bot.set_yticks([])
        plt.tight_layout()
        path = os.path.join(self.plots_dir, f"detection_timeline_{self.run_id}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
        plt.close(fig)
        return path

    def plot_accuracy(self, df, true_drift_point, retraining_indices, window=200):
        if "running_acc" not in df.columns or "error" not in df.columns:
            return ""
        active = df[df["prediction"] != -1].copy()
        if active.empty:
            return ""
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 6),
            gridspec_kw={"height_ratios":[3,1]}, sharex=True)
        fig.patch.set_facecolor(STYLE["bg"])
        indices    = active["sample_index"].values
        err_raw    = active["error"].values
        acc_smooth = active["running_acc"].rolling(window=window, min_periods=1).mean().values
        ax_top.axvspan(indices[0], true_drift_point, alpha=0.06, color=STYLE["accent_blue"])
        ax_top.axvspan(true_drift_point, indices[-1], alpha=0.06, color=STYLE["accent_orange"])
        ax_top.plot(indices, acc_smooth, color=STYLE["accent_green"], linewidth=1.6, alpha=0.95,
                    label=f"Running Accuracy (rolling mean, w={window})")
        ax_top.axvline(x=true_drift_point, color=STYLE["accent_orange"], linewidth=1.5,
                       linestyle="--", label="Dataset Midpoint")
        first = True
        for rt_idx in retraining_indices:
            lbl = "Retraining Event" if first else None
            first = False
            ax_top.axvline(x=rt_idx, color=STYLE["accent_red"], linewidth=1.4,
                           linestyle=":", alpha=0.85, label=lbl)
            row = active[active["sample_index"] == rt_idx]
            if not row.empty:
                ax_top.scatter([rt_idx], [float(row["running_acc"].iloc[0])],
                               color=STYLE["accent_red"], s=60, zorder=6, marker="v")
        ax_top.axhline(y=0.5, color=STYLE["grid"], linewidth=1.0, linestyle="--",
                       alpha=0.6, label="Random Chance (0.50)")
        ax_top.set_ylim(0.0, 1.05)
        _apply_dark_style(ax_top, "Running Accuracy with Retraining (ELEC2)",
                          "", "Accuracy")
        ax_top.legend(facecolor=STYLE["panel"], edgecolor=STYLE["grid"],
                      labelcolor=STYLE["text"], fontsize=9, loc="lower left")
        ax_bot.set_facecolor(STYLE["panel"])
        ax_bot.fill_between(indices, err_raw, step="pre", alpha=0.4,
                            color=STYLE["accent_red"], label="Prediction Error")
        ax_bot.axvline(x=true_drift_point, color=STYLE["accent_orange"], linewidth=1.5, linestyle="--")
        for rt_idx in retraining_indices:
            ax_bot.axvline(x=rt_idx, color=STYLE["accent_red"], linewidth=1.2, linestyle=":", alpha=0.7)
        _apply_dark_style(ax_bot, "", "Sample Index", "Error (0/1)")
        ax_bot.set_ylim(-0.05, 1.4)
        ax_bot.set_yticks([0, 1])
        ax_bot.tick_params(colors=STYLE["text"])
        plt.tight_layout()
        path = os.path.join(self.plots_dir, f"accuracy_{self.run_id}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
        plt.close(fig)
        return path

    def plot_multi_feature_distribution(self, df, true_drift_point,
                                        feature_cols=None):
        """
        Plots distribution comparison (first half vs second half) for
        multiple features side by side.

        Parameters
        ----------
        df               : log DataFrame
        true_drift_point : sample index used as the split point
        feature_cols     : list of feature column names (default: first 4)
        """
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c.startswith("feature_")][:4]

        n = len(feature_cols)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
        fig.patch.set_facecolor(STYLE["bg"])
        if n == 1:
            axes = [axes]

        for ax, col in zip(axes, feature_cols):
            pre  = df[df["sample_index"] < true_drift_point][col]
            post = df[df["sample_index"] >= true_drift_point][col]
            ax.hist(pre,  bins=60, alpha=0.55, color=STYLE["accent_blue"],
                    label=f"1st half", density=True)
            ax.hist(post, bins=60, alpha=0.55, color=STYLE["accent_orange"],
                    label=f"2nd half", density=True)
            ax.axvline(pre.mean(),  color=STYLE["accent_blue"],
                       linewidth=1.5, linestyle="--")
            ax.axvline(post.mean(), color=STYLE["accent_orange"],
                       linewidth=1.5, linestyle="--")
            _apply_dark_style(ax, col, "Value", "Density")
            ax.legend(facecolor=STYLE["panel"], edgecolor=STYLE["grid"],
                      labelcolor=STYLE["text"], fontsize=8)

        fig.suptitle("Multi-Feature Distribution Shift",
                     color=STYLE["text"], fontsize=13,
                     fontfamily=STYLE["font"], y=1.02)
        plt.tight_layout()
        path = os.path.join(self.plots_dir,
                            f"multi_feature_dist_{self.run_id}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
        plt.close(fig)
        return path

    def plot_confusion_matrix(self, cm_matrix, labels=None, title="Confusion Matrix"):
        """
        Plots a 2x2 confusion matrix heatmap.

        Parameters
        ----------
        cm_matrix : [[TN, FP], [FN, TP]] — standard layout
        labels    : list of class labels (default: ["DOWN", "UP"])
        title     : plot title
        """
        if labels is None:
            labels = ["DOWN (0)", "UP (1)"]

        cm = np.array(cm_matrix)
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor(STYLE["bg"])
        ax.set_facecolor(STYLE["panel"])

        im = ax.imshow(cm, cmap="Blues", aspect="auto")

        # Annotate cells
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                total = cm.sum()
                pct = val / total * 100 if total > 0 else 0
                ax.text(j, i, f"{val:,}\n({pct:.1f}%)",
                        ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else STYLE["text"])

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=10, color=STYLE["text"])
        ax.set_yticklabels(labels, fontsize=10, color=STYLE["text"])
        ax.set_xlabel("Predicted", color=STYLE["text"], fontsize=11)
        ax.set_ylabel("Actual", color=STYLE["text"], fontsize=11)
        ax.set_title(title, color=STYLE["text"], fontsize=12,
                     fontfamily=STYLE["font"], pad=10)
        ax.tick_params(colors=STYLE["text"])

        plt.tight_layout()
        safe_title = title.lower().replace(" ", "_").replace("—", "").replace("(", "").replace(")", "")
        path = os.path.join(self.plots_dir,
                            f"cm_{safe_title}_{self.run_id}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
        plt.close(fig)
        return path

    def plot_windowed_accuracy(self, windowed_acc, windowed_indices,
                               retraining_indices, window_size=500):
        """
        Plots windowed (sliding window) accuracy over the stream.

        Parameters
        ----------
        windowed_acc       : list of windowed accuracy values
        windowed_indices   : list of sample indices (aligned with windowed_acc)
        retraining_indices : list of sample indices where retraining occurred
        window_size        : window size label for the plot
        """
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor(STYLE["bg"])

        ax.plot(windowed_indices, windowed_acc,
                color=STYLE["accent_green"], linewidth=1.2, alpha=0.9,
                label=f"Windowed Accuracy (w={window_size})")

        for i, rt_idx in enumerate(retraining_indices):
            ax.axvline(x=rt_idx, color=STYLE["accent_red"], linewidth=1.2,
                       linestyle=":", alpha=0.7,
                       label="Retraining" if i == 0 else None)

        ax.axhline(y=0.5, color=STYLE["grid"], linewidth=0.8,
                   linestyle="--", alpha=0.5, label="Random Chance (0.50)")
        ax.set_ylim(0.3, 1.05)
        _apply_dark_style(ax, f"Windowed Accuracy (w={window_size})",
                          "Sample Index", "Accuracy")
        ax.legend(facecolor=STYLE["panel"], edgecolor=STYLE["grid"],
                  labelcolor=STYLE["text"], fontsize=9, loc="lower left")

        plt.tight_layout()
        path = os.path.join(self.plots_dir,
                            f"windowed_accuracy_{self.run_id}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
        plt.close(fig)
        return path
