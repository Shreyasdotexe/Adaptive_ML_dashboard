"""
experiment_runner.py
--------------------
Structured experiments for the Adaptive ML System.

Provides two key experiments that capstone evaluators expect:

  1. WITH vs WITHOUT retraining comparison
     - Runs the pipeline twice: once with ADWIN-triggered retraining and
       once with retraining disabled. Compares accuracy to prove adaptive
       retraining objectively improves performance under concept drift.

  2. ADWIN delta sensitivity analysis (ablation study)
     - Runs the pipeline with multiple delta values and compares drift
       detection frequency, accuracy, and recovery behaviour.

Results are saved to:
    outputs/experiments/
      comparison_<run_id>.json     — with vs without results
      ablation_<run_id>.json       — delta sensitivity results

Run:
    python -m src.experiment_runner
"""

import os, sys, json, copy
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.real_stream_loader              import RealStreamLoader
from src.preprocessing                   import OnlinePreprocessor
from src.online_model                    import OnlineModel
from src.retraining_manager              import RetrainingManager
from src.drift_detectors.detector_manager import DetectorManager
from src.logger                          import StreamLogger
from src.evaluation                      import (
    StreamingConfusionMatrix,
    WindowedAccuracyTracker,
    RecoveryTimeAnalyser,
)
from src.utils import generate_run_id, Timer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
#  STYLE (matches existing visualizer.py)
# ─────────────────────────────────────────────────────────────────────────────

STYLE = {
    "bg": "#0f1117", "panel": "#1a1d27", "grid": "#2a2d3a", "text": "#e0e0e0",
    "accent_blue": "#4a9eff", "accent_red": "#ff5c5c", "accent_green": "#4ade80",
    "accent_orange": "#fbbf24", "accent_purple": "#a78bfa", "font": "monospace",
}


def _apply_dark_style(ax, title, xlabel, ylabel):
    ax.set_facecolor(STYLE["panel"])
    ax.tick_params(colors=STYLE["text"], labelsize=9)
    ax.set_title(title, color=STYLE["text"], fontsize=12,
                 fontfamily=STYLE["font"], pad=10)
    ax.set_xlabel(xlabel, color=STYLE["text"], fontsize=10,
                  fontfamily=STYLE["font"])
    ax.set_ylabel(ylabel, color=STYLE["text"], fontsize=10,
                  fontfamily=STYLE["font"])
    ax.grid(True, color=STYLE["grid"], linewidth=0.6, linestyle="--", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE["grid"])


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: run pipeline with a config, returning metrics
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline_silent(config: dict, training_mode: str = "adaptive") -> dict:
    """
    Runs the full streaming pipeline silently (no console output).
    Returns detailed metrics for comparison.

    Parameters
    ----------
    config        : pipeline configuration dict
    training_mode : 'static' (trains on first 10% only),
                    'continuous' (trains on all, never resets),
                    'adaptive' (trains on all, resets on ADWIN drift)
    """
    loader = RealStreamLoader(
        csv_path=config["csv_path"],
        max_samples=config.get("max_samples"),
    )
    loader_config = loader.get_config()
    n_samples = loader_config["n_samples"]
    n_features = loader_config["n_features"]

    preprocessor = OnlinePreprocessor(
        n_features=n_features,
        warmup_size=config["warmup_size"],
    )
    model = OnlineModel(
        n_features=n_features,
        random_state=config.get("model_random_state", 42),
    )
    detector_manager = DetectorManager(adwin_delta=config["adwin_delta"])

    last_detected_at = -config["cooldown_samples"]
    correct_total = 0
    active_total = 0

    # Tracking
    cm = StreamingConfusionMatrix()
    windowed = WindowedAccuracyTracker(window_size=500)
    retraining_indices = []
    accuracy_at_drifts = []
    all_errors = []
    all_indices = []
    all_running_acc = []

    for index, raw_features, label, phase in loader.stream():
        scaled_features = preprocessor.process(raw_features)

        prediction = -1
        error = -1
        running_acc = 0.0

        if preprocessor.is_ready:
            prediction = model.predict(scaled_features)

            if prediction is not None:
                error = 0 if prediction == label else 1
                active_total += 1
                correct_total += (1 - error)
                running_acc = correct_total / active_total

                cm.update(label, prediction)
                windowed.update(error)
                all_errors.append(error)
                all_indices.append(index)
                all_running_acc.append(running_acc)

            # Update model based on training mode
            if training_mode == "static":
                # Only train on the first 10% of the stream (e.g. 4500 samples)
                if index < 4500:
                    model.update(scaled_features, label)
            else:
                # continuous and adaptive modes train on everything
                model.update(scaled_features, label)

            if model.is_ready and error != -1:
                detector_input = float(error)
                raw_flag = detector_manager.update(detector_input, index)

                if raw_flag and (index - last_detected_at) >= config["cooldown_samples"]:
                    last_detected_at = index

                    if training_mode == "adaptive":
                        retraining_indices.append(index)
                        accuracy_at_drifts.append(running_acc)
                        model.reset()
                        detector_manager.reset_adwin()

    # Recovery analysis
    recovery = RecoveryTimeAnalyser(recovery_threshold=0.85, window_size=200)
    recovery_results = recovery.analyse(
        errors=all_errors,
        sample_indices=all_indices,
        retraining_indices=retraining_indices,
    )

    comparison = detector_manager.get_comparison_metrics()

    return {
        "n_samples":            n_samples,
        "training_mode":        training_mode,
        "adwin_delta":          config["adwin_delta"],
        "confusion_matrix":     cm.to_dict(),
        "windowed_accuracy":    windowed.history,
        "windowed_indices":     all_indices,
        "running_accuracy":     all_running_acc,
        "retraining_indices":   retraining_indices,
        "accuracy_at_drifts":   accuracy_at_drifts,
        "detector_comparison":  comparison,
        "recovery_analysis":    recovery.get_summary(),
        "final_accuracy":       round(cm.accuracy, 4),
        "kappa":                round(cm.kappa, 4),
        "f1":                   round(cm.f1, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  EXPERIMENT 1: WITH vs WITHOUT RETRAINING
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison_experiment(config: dict, output_dir: str = "outputs/experiments") -> dict:
    """
    Runs the pipeline with three modes:
      1. Static Model    (trains on first 10% only, then stops)
      2. Continuous      (blind adaptation, no resets)
      3. Adaptive        (ADWIN detector triggers resets)

    Returns
    -------
    dict with keys for each mode and improvement metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    run_id = generate_run_id()

    print("\n" + "=" * 64)
    print("  EXPERIMENT: 3-Way Model Benchmarking")
    print("=" * 64)

    # 1. Static
    print("\n  [1/3] Running STATIC model (trains only on first 10%)...")
    static_run = _run_pipeline_silent(config, training_mode="static")
    print(f"        > Accuracy: {static_run['final_accuracy']:.4f}  "
          f"F1: {static_run['f1']:.4f}")

    # 2. Continuous
    print("\n  [2/3] Running CONTINUOUS model (blind adaptation, no resets)...")
    continuous_run = _run_pipeline_silent(config, training_mode="continuous")
    print(f"        > Accuracy: {continuous_run['final_accuracy']:.4f}  "
          f"F1: {continuous_run['f1']:.4f}")

    # 3. Adaptive (ADWIN)
    print("\n  [3/3] Running ADAPTIVE model (ADWIN triggers re-training)...")
    adaptive_run = _run_pipeline_silent(config, training_mode="adaptive")
    print(f"        > Accuracy: {adaptive_run['final_accuracy']:.4f}  "
          f"F1: {adaptive_run['f1']:.4f}  "
          f"Retrains: {len(adaptive_run['retraining_indices'])}")

    # Compute improvement (Adaptive vs Static) -> This is the core thesis!
    acc_improvement = adaptive_run["final_accuracy"] - static_run["final_accuracy"]
    f1_improvement  = adaptive_run["f1"] - static_run["f1"]

    result = {
        "run_id":              run_id,
        "experiment":          "3_way_model_benchmarking",
        "static_model":        _serialisable(static_run),
        "continuous_model":    _serialisable(continuous_run),
        "adaptive_model":      _serialisable(adaptive_run),
        "improvement_over_static": {
            "accuracy_delta":  round(acc_improvement, 4),
            "f1_delta":        round(f1_improvement, 4),
            "adaptation_helps": acc_improvement > 0,
        },
    }

    # Save JSON
    path = os.path.join(output_dir, f"comparison_{run_id}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  * Results saved: {path}")

    # Generate comparison plot
    plot_path = _plot_comparison(
        static_run, continuous_run, adaptive_run, output_dir, run_id
    )
    result["plot_path"] = plot_path
    print(f"  * Plot saved:    {plot_path}")

    print("\n  ---- Summary (Thesis Proof) ----")
    print(f"  Static Model:       acc={static_run['final_accuracy']:.4f}  F1={static_run['f1']:.4f}")
    print(f"  Continuous Model:   acc={continuous_run['final_accuracy']:.4f}  F1={continuous_run['f1']:.4f}")
    print(f"  Adaptive (ADWIN):   acc={adaptive_run['final_accuracy']:.4f}  F1={adaptive_run['f1']:.4f}")
    print(f"  Improvement (Adaptive vs Static): d_acc={acc_improvement:+.4f}  d_F1={f1_improvement:+.4f}")
    print(f"  Concept Drift Detected & Adaptation Required -> {'PROVEN *' if acc_improvement > 0 else 'NO x'}")
    print("=" * 64 + "\n")

    return result


def _plot_comparison(static_data: dict, cont_data: dict, adapt_data: dict,
                     output_dir: str, run_id: str) -> str:
    """Generates a side-by-side comparison plot of the 3 modes."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor(STYLE["bg"])

    # Panel 1 — Running accuracy comparison
    ax1 = axes[0]
    idx_s = static_data["windowed_indices"]
    idx_c = cont_data["windowed_indices"]
    idx_a = adapt_data["windowed_indices"]

    ax1.plot(idx_s, static_data["windowed_accuracy"], color=STYLE["accent_red"], linewidth=1.2,
             alpha=0.7, label="Static Model (no learning)")
    ax1.plot(idx_c, cont_data["windowed_accuracy"], color=STYLE["accent_blue"], linewidth=1.2,
             alpha=0.8, label="Continuous Model")
    ax1.plot(idx_a, adapt_data["windowed_accuracy"], color=STYLE["accent_green"], linewidth=1.2,
             alpha=0.9, label="Adaptive ADWIN Model")

    for rt_idx in adapt_data["retraining_indices"]:
        ax1.axvline(x=rt_idx, color=STYLE["accent_orange"], linewidth=1.0,
                    linestyle=":", alpha=0.6)

    ax1.axhline(y=0.5, color=STYLE["grid"], linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.set_ylim(0.3, 1.05)
    _apply_dark_style(ax1, "3-Way Benchmark — Windowed Accuracy (w=500)",
                      "", "Accuracy")
    ax1.legend(facecolor=STYLE["panel"], edgecolor=STYLE["grid"],
              labelcolor=STYLE["text"], fontsize=9, loc="lower left")

    # Panel 2 — Running accuracy (cumulative) comparison
    ax2 = axes[1]
    ax2.plot(idx_s, static_data["running_accuracy"], color=STYLE["accent_red"], linewidth=1.4, alpha=0.7,
             label="Static Model")
    ax2.plot(idx_c, cont_data["running_accuracy"], color=STYLE["accent_blue"], linewidth=1.4, alpha=0.8,
             label="Continuous Model")
    ax2.plot(idx_a, adapt_data["running_accuracy"], color=STYLE["accent_green"], linewidth=1.4, alpha=0.9,
             label="Adaptive ADWIN Model")

    for rt_idx in adapt_data["retraining_indices"]:
        ax2.axvline(x=rt_idx, color=STYLE["accent_orange"], linewidth=1.0,
                    linestyle=":", alpha=0.6)

    ax2.set_ylim(0.4, 1.0)
    _apply_dark_style(ax2, "3-Way Benchmark — Cumulative Accuracy",
                      "Sample Index", "Accuracy")
    ax2.legend(facecolor=STYLE["panel"], edgecolor=STYLE["grid"],
              labelcolor=STYLE["text"], fontsize=9, loc="lower left")

    plt.tight_layout()
    path = os.path.join(output_dir, f"comparison_{run_id}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  EXPERIMENT 2: ADWIN DELTA SENSITIVITY (ABLATION STUDY)
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_experiment(
    config: dict,
    delta_values: list = None,
    output_dir: str = "outputs/experiments",
) -> dict:
    """
    Runs the pipeline with multiple ADWIN delta values.

    Tests how sensitive drift detection is to the delta hyperparameter.

    Parameters
    ----------
    config       : base pipeline configuration
    delta_values : list of delta values to test (default: 5 values)
    output_dir   : where to save results

    Returns
    -------
    dict with per-delta results and summary table
    """
    if delta_values is None:
        delta_values = [0.001, 0.01, 0.05, 0.1, 0.5]

    os.makedirs(output_dir, exist_ok=True)
    run_id = generate_run_id()

    print("\n" + "=" * 64)
    print("  EXPERIMENT: ADWIN Delta Sensitivity (Ablation Study)")
    print("=" * 64)

    results = {}
    for i, delta in enumerate(delta_values):
        cfg = copy.deepcopy(config)
        cfg["adwin_delta"] = delta
        print(f"\n  [{i+1}/{len(delta_values)}] Running with delta = {delta} ...")
        r = _run_pipeline_silent(cfg, training_mode="adaptive")
        results[str(delta)] = _serialisable(r)
        adwin_drifts = r["detector_comparison"].get("ADWIN", {}).get("total_drifts", 0)
        print(f"        > acc={r['final_accuracy']:.4f}  "
              f"F1={r['f1']:.4f}  "
              f"drifts={adwin_drifts}  "
              f"retrains={len(r['retraining_indices'])}")

    # Summary table
    summary_rows = []
    for delta in delta_values:
        r = results[str(delta)]
        adwin_drifts = r["detector_comparison"].get("ADWIN", {}).get("total_drifts", 0)
        summary_rows.append({
            "delta":      delta,
            "accuracy":   r["final_accuracy"],
            "f1":         r["f1"],
            "kappa":      r["kappa"],
            "drifts":     adwin_drifts,
            "retrains":   len(r["retraining_indices"]),
        })

    output = {
        "run_id":     run_id,
        "experiment": "adwin_delta_sensitivity",
        "results":    results,
        "summary":    summary_rows,
    }

    # Save JSON
    path = os.path.join(output_dir, f"ablation_{run_id}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  * Results saved: {path}")

    # Generate ablation plot
    plot_path = _plot_ablation(summary_rows, results, output_dir, run_id)
    output["plot_path"] = plot_path
    print(f"  * Plot saved:    {plot_path}")

    # Print summary table
    print("\n  ---- Summary Table ----")
    print(f"  {'Delta':<8} {'Accuracy':<10} {'F1':<8} {'Kappa':<8} {'Drifts':<8} {'Retrains':<8}")
    print("  " + "-" * 50)
    for row in summary_rows:
        print(f"  {row['delta']:<8} {row['accuracy']:<10.4f} {row['f1']:<8.4f} "
              f"{row['kappa']:<8.4f} {row['drifts']:<8} {row['retrains']:<8}")
    print("=" * 64 + "\n")

    return output


def _plot_ablation(summary: list, results: dict,
                   output_dir: str, run_id: str) -> str:
    """Generates a multi-panel ablation study plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor(STYLE["bg"])

    deltas = [r["delta"] for r in summary]
    accs   = [r["accuracy"] for r in summary]
    f1s    = [r["f1"] for r in summary]
    kappas = [r["kappa"] for r in summary]
    drifts = [r["drifts"] for r in summary]

    delta_labels = [str(d) for d in deltas]

    # Top-left: Accuracy vs Delta
    ax = axes[0, 0]
    ax.bar(delta_labels, accs, color=STYLE["accent_green"], alpha=0.85)
    _apply_dark_style(ax, "Accuracy vs ADWIN Delta", "Delta (δ)", "Accuracy")

    # Top-right: F1 vs Delta
    ax = axes[0, 1]
    ax.bar(delta_labels, f1s, color=STYLE["accent_blue"], alpha=0.85)
    _apply_dark_style(ax, "F1-Score vs ADWIN Delta", "Delta (δ)", "F1")

    # Bottom-left: Kappa vs Delta
    ax = axes[1, 0]
    ax.bar(delta_labels, kappas, color=STYLE["accent_purple"], alpha=0.85)
    _apply_dark_style(ax, "Cohen's Kappa vs ADWIN Delta", "Delta (δ)", "Kappa")

    # Bottom-right: Drift count vs Delta
    ax = axes[1, 1]
    ax.bar(delta_labels, drifts, color=STYLE["accent_orange"], alpha=0.85)
    _apply_dark_style(ax, "ADWIN Drift Count vs Delta", "Delta (δ)", "Drifts Detected")

    fig.suptitle("ADWIN Delta Sensitivity — Ablation Study",
                 color=STYLE["text"], fontsize=14, fontfamily=STYLE["font"],
                 y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, f"ablation_{run_id}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _serialisable(d: dict) -> dict:
    """Convert a result dict to JSON-serialisable form."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = _serialisable(v)
        elif isinstance(v, list) and len(v) > 200:
            # Don't save huge arrays in JSON (windowed_accuracy etc.)
            out[k] = v[:10] + ["...truncated..."] + v[-10:]
        else:
            out[k] = v
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CONFIG = {
        "csv_path":           os.path.join("data", "electricity.csv"),
        "max_samples":        None,
        "warmup_size":        200,
        "adwin_delta":        0.1,
        "cooldown_samples":   300,
        "post_drift_settle":  300,
        "model_random_state": 42,
        "n_features":         8,
    }

    print("\n  Running both experiments. This may take 15-25 minutes.\n")

    # Experiment 1: With vs Without retraining
    comparison = run_comparison_experiment(CONFIG)

    # Experiment 2: Delta sensitivity
    ablation = run_ablation_experiment(CONFIG)

    print("\n  All experiments complete.\n")
