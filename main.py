"""
main.py
-------
Entry point for the Adaptive ML System — Phase 4.

Phase 4 additions over Phase 3:
  - DetectorManager runs ADWIN, DDM, and EDDM in parallel
  - ADWIN remains the primary detector (triggers retraining)
  - DDM and EDDM are comparison-only
  - Run report extended with detector comparison table
  - All Phase 3 outputs (CSV, 4 plots, report) remain unchanged

Run:
    python main.py

Dashboard:
    streamlit run dashboard.py
"""

import os, sys, json, signal
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.real_stream_loader              import RealStreamLoader
from src.preprocessing                   import OnlinePreprocessor
from src.online_model                    import OnlineModel
from src.retraining_manager              import RetrainingManager
from src.drift_detectors.detector_manager import DetectorManager
from src.logger                          import StreamLogger, RunReporter
from src.visualizer                      import ResultVisualizer
from src.evaluation                      import compute_evaluation_from_log
from src.utils                           import (
    Timer, generate_run_id, validate_config,
    print_banner, print_section, print_ok, print_info,
    print_drift_alert, print_progress, print_summary, print_footer,
)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "csv_path":           os.path.join("data", "electricity.csv"),
    "max_samples":        None,       # None = full 45312-row dataset
    "warmup_size":        200,
    "adwin_delta":        0.1,
    "cooldown_samples":   300,
    "post_drift_settle":  300,
    "model_random_state": 42,
    "n_features":         8,
    "log_dir":            "data/logs",
    "plots_dir":          "outputs/plots",
    "reports_dir":        "outputs/reports",
    # Primary detector: "ADWIN" | "DDM" | "EDDM"
    # The chosen detector triggers model retraining; the other two run in comparison mode.
    "primary_detector":   "ADWIN",
}


# ─────────────────────────────────────────────────────────────────────────────
#  CONSOLE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _print_retrain(index: int, acc: float):
    from src.utils import BOLD, RESET, GREEN, CYAN
    print(
        f"  {GREEN}{BOLD}↺ RETRAINING{RESET}  "
        f"at sample {BOLD}{index:>6}{RESET}  "
        f"acc_at_drift={CYAN}{acc:.3f}{RESET}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(config: dict) -> dict:
    timer = Timer()
    timer.start()
    run_id = generate_run_id()

    # Cleanup old runs to save space (keep only the current run)
    import glob
    for d in [config["plots_dir"], config["reports_dir"], config["log_dir"]]:
        if os.path.exists(d):
            for f in glob.glob(os.path.join(d, "*")):
                try:
                    os.remove(f)
                except Exception:
                    pass

    # Validate configuration before starting
    validate_config(config)

    primary_detector = config.get("primary_detector", "ADWIN").upper()
    if primary_detector not in ("ADWIN", "DDM", "EDDM"):
        primary_detector = "ADWIN"

    print_banner()
    print_section("Initialising Pipeline Modules")
    print_info(f"Run ID        : {run_id}")
    print_info(f"Dataset       : {config['csv_path']}")
    print_info(f"Max samples   : {config['max_samples'] or 'full dataset'}")
    print_info(f"ADWIN delta   : {config['adwin_delta']}")
    print_info(f"Cooldown      : {config['cooldown_samples']} samples")
    others = " | ".join(d for d in ["ADWIN", "DDM", "EDDM"] if d != primary_detector)
    print_info(f"Primary Det.  : {primary_detector} (triggers retraining)")
    print_info(f"Comparison    : {others}")

    # ── Initialise modules ────────────────────────────────────────────────────
    loader = RealStreamLoader(
        csv_path=config["csv_path"],
        max_samples=config["max_samples"],
    )
    loader_config = loader.get_config()
    n_samples  = loader_config["n_samples"]
    n_features = loader_config["n_features"]

    print_info(f"Samples       : {n_samples}")
    print_info(f"Features      : {n_features}")

    preprocessor     = OnlinePreprocessor(n_features=n_features,
                                          warmup_size=config["warmup_size"])
    model            = OnlineModel(n_features=n_features,
                                   random_state=config["model_random_state"])
    detector_manager = DetectorManager(adwin_delta=config["adwin_delta"])
    retrainer        = RetrainingManager(reset_detector=False)  # manager handles reset
    logger           = StreamLogger(log_dir=config["log_dir"],
                                    n_features=n_features, run_id=run_id)

    print_ok("RealStreamLoader       ready")
    print_ok("OnlinePreprocessor     ready")
    print_ok("OnlineModel (SGD)      ready")
    print_ok("DetectorManager        ready  [ADWIN + DDM + EDDM]")
    print_ok("RetrainingManager      ready")
    print_ok(f"StreamLogger           ready  → {logger.get_path()}")

    # ── Stream loop ───────────────────────────────────────────────────────────
    print_section(f"Running Stream Pipeline  [{primary_detector} + ELEC2]")

    # Graceful interrupt handler — ensures log is flushed if Ctrl+C
    _interrupted = False
    def _handle_interrupt(signum, frame):
        nonlocal _interrupted
        _interrupted = True
        print("\n\n  ⚠ Interrupt received — flushing logs and saving partial results...")
    signal.signal(signal.SIGINT, _handle_interrupt)

    logger.open()

    last_detected_at = -config["cooldown_samples"]
    correct_total    = 0
    active_total     = 0

    for index, raw_features, label, phase in loader.stream():
        if _interrupted:
            print_info(f"Stopped early at sample {index}")
            break

        # Step 1 — Preprocess
        scaled_features = preprocessor.process(raw_features)

        prediction     = -1
        error          = -1
        running_acc    = 0.0
        retrained      = 0
        detector_input = None

        if preprocessor.is_ready:

            # Step 2 — Predict
            prediction = model.predict(scaled_features)

            # Step 3+4 — Error and running accuracy
            if prediction is not None:
                error          = 0 if prediction == label else 1
                active_total  += 1
                correct_total += (1 - error)
                running_acc    = correct_total / active_total

            # Step 5 — Online update
            model.update(scaled_features, label)

            # Step 6 — Feed all detectors; primary detector triggers retraining
            raw_flag = False
            if model.is_ready and error != -1:
                detector_input = float(error)
                # DetectorManager feeds ADWIN + DDM + EDDM simultaneously;
                # but we gate retraining on whichever detector was chosen.
                raw_flag = detector_manager.update(
                    detector_input, index,
                    primary=primary_detector,
                )

                # Step 7 — Cooldown-gated retraining
                if raw_flag and (index - last_detected_at) >= config["cooldown_samples"]:
                    last_detected_at = index
                    retrained        = 1
                    print_drift_alert(index, running_acc)

                    # Reset model; reset primary detector via manager
                    retrainer.handle_drift(
                        sample_index=index,
                        model=model,
                        detector=None,
                        accuracy_at_drift=running_acc,
                    )
                    detector_manager.reset_primary(primary_detector)
                    _print_retrain(index, running_acc)

        drift_detected = bool(retrained)

        # Step 8 — Log (same schema as Phase 3)
        logger.log(
            index=index,
            features=scaled_features,
            label=label,
            detector_input=detector_input,
            drift_detected=drift_detected,
            phase=phase,
            prediction=prediction if prediction is not None else -1,
            error=error,
            running_acc=running_acc,
            retrained=retrained,
        )

        print_progress(index, n_samples)

    log_path = logger.close()
    print_ok(f"Stream complete — {logger.get_rows_written()} rows logged")

    # ── Evaluation Metrics ─────────────────────────────────────────────────────
    print_section("Computing Evaluation Metrics")

    retrain_summary    = retrainer.get_summary()
    retraining_indices = retrain_summary["retraining_indices"]

    evaluation = compute_evaluation_from_log(
        log_path=log_path,
        retraining_indices=retraining_indices,
        window_size=500,
        recovery_threshold=0.85,
    )
    print_ok(f"Prequential Accuracy : {evaluation['prequential_accuracy']:.4f}")
    print_ok(f"Cohen's Kappa        : {evaluation['kappa']:.4f}")
    print_ok(f"F1 Score             : {evaluation['confusion_matrix']['f1']:.4f}")
    print_ok(f"Precision            : {evaluation['confusion_matrix']['precision']:.4f}")
    print_ok(f"Recall               : {evaluation['confusion_matrix']['recall']:.4f}")

    recovery = evaluation.get("recovery_analysis", {})
    if recovery.get("avg_recovery_time"):
        print_ok(f"Avg Recovery Time    : {recovery['avg_recovery_time']} samples")

    # ── Visualisations ────────────────────────────────────────────────────────
    print_section("Generating Visualisations")

    df                 = pd.read_csv(log_path)
    adwin_indices      = detector_manager.get_adwin_drift_indices()
    comparison         = detector_manager.get_comparison_metrics()
    mid                = n_samples // 2

    visualizer = ResultVisualizer(plots_dir=config["plots_dir"], run_id=run_id)

    p1 = visualizer.plot_drift_signal(
        df=df, true_drift_point=mid,
        detected_drift_indices=adwin_indices, signal_col="detector_input",
    )
    print_ok(f"Saved: {p1}")

    p2 = visualizer.plot_feature_distribution(
        df=df, true_drift_point=mid, feature_col="feature_0",
    )
    print_ok(f"Saved: {p2}")

    p3 = visualizer.plot_drift_detections_timeline(
        df=df, true_drift_point=mid,
        detected_drift_indices=adwin_indices,
    )
    print_ok(f"Saved: {p3}")

    p4 = visualizer.plot_accuracy(
        df=df, true_drift_point=mid,
        retraining_indices=retraining_indices,
    )
    if p4:
        print_ok(f"Saved: {p4}")

    # New Phase 4+ plots
    p5 = visualizer.plot_multi_feature_distribution(df=df, true_drift_point=mid)
    print_ok(f"Saved: {p5}")

    p6 = visualizer.plot_confusion_matrix(
        cm_matrix=evaluation["confusion_matrix_raw"],
        title="Overall Confusion Matrix",
    )
    print_ok(f"Saved: {p6}")

    p7 = visualizer.plot_windowed_accuracy(
        windowed_acc=evaluation["windowed_accuracy"],
        windowed_indices=evaluation["windowed_acc_indices"],
        retraining_indices=retraining_indices,
    )
    print_ok(f"Saved: {p7}")

    # ── Run report ────────────────────────────────────────────────────────────
    print_section("Saving Run Report")

    model_summary = model.get_summary()
    preproc_stats = preprocessor.get_stats()

    active_df   = df[df["prediction"] != -1]
    final_acc   = round(float(active_df["running_acc"].iloc[-1]), 4) if not active_df.empty else 0.0
    first_half  = active_df[active_df["sample_index"] < mid]
    second_half = active_df[active_df["sample_index"] >= mid + config["post_drift_settle"]]
    first_acc   = round(float(first_half["running_acc"].mean()),  4) if not first_half.empty  else 0.0
    second_acc  = round(float(second_half["running_acc"].mean()), 4) if not second_half.empty else 0.0

    # Detector comparison rows
    comp_rows = []
    for det_name, metrics in comparison.items():
        gap_str = str(metrics["avg_inter_drift_gap"]) if metrics["avg_inter_drift_gap"] else "N/A"
        comp_rows.append(
            f"  {det_name:<8} | {metrics['total_drifts']:<16} | {gap_str}"
        )

    report_lines = [
        "=" * 64,
        "  ADAPTIVE ML SYSTEM — RUN REPORT",
        "  Real Dataset: ELEC2 | Detectors: ADWIN + DDM + EDDM",
        "=" * 64,
        f"  Run ID              : {run_id}",
        "",
        "  DATASET",
        f"  Source              : {loader_config['source']}",
        f"  Total samples       : {n_samples}",
        f"  Features            : {loader_config['features']}",
        "",
        "  CONFIGURATION",
        f"  warmup_size         : {config['warmup_size']}",
        f"  adwin_delta         : {config['adwin_delta']}",
        f"  cooldown_samples    : {config['cooldown_samples']}",
        f"  post_drift_settle   : {config['post_drift_settle']}",
        "",
        "  ONLINE MODEL",
        f"  Model type          : {model_summary['model_type']}",
        f"  Samples trained on  : {model_summary['samples_seen']}",
        f"  Retrain count       : {model_summary['retrain_count']}",
        "",
        "  DETECTOR COMPARISON",
        f"  {'Detector':<8} | {'Drifts Detected':<16} | Avg Inter-Drift Gap",
        "  " + "-" * 50,
    ] + comp_rows + [
        "",
        "  RETRAINING (ADWIN-triggered)",
        f"  Total events        : {retrain_summary['total_retraining_events']}",
        f"  Retrain at          : {retrain_summary['retraining_indices']}",
        f"  Accuracy at drift   : {retrain_summary['accuracy_at_each_drift']}",
        "",
        "  ACCURACY METRICS",
        f"  Final accuracy      : {final_acc}",
        f"  First-half avg acc  : {first_acc}",
        f"  Second-half avg acc : {second_acc}",
        "",
        "  OUTPUT FILES",
        f"  CSV Log             : {log_path}",
        f"  Drift Signal Plot   : {p1}",
        f"  Feature Dist Plot   : {p2}",
        f"  Timeline Plot       : {p3}",
        f"  Accuracy Plot       : {p4}",
        f"  Multi-Feature Plot  : {p5}",
        f"  Confusion Matrix    : {p6}",
        f"  Windowed Accuracy   : {p7}",
        "",
        "  EVALUATION METRICS",
        f"  Prequential Accuracy: {evaluation['prequential_accuracy']}",
        f"  Cohen's Kappa       : {evaluation['kappa']}",
        f"  F1 Score            : {evaluation['confusion_matrix']['f1']}",
        f"  Precision           : {evaluation['confusion_matrix']['precision']}",
        f"  Recall              : {evaluation['confusion_matrix']['recall']}",
        f"  Confusion Matrix    : TP={evaluation['confusion_matrix']['tp']}  "
        f"TN={evaluation['confusion_matrix']['tn']}  "
        f"FP={evaluation['confusion_matrix']['fp']}  "
        f"FN={evaluation['confusion_matrix']['fn']}",
        "",
        "=" * 64,
    ]

    reporter = RunReporter(report_dir=config["reports_dir"])
    rpt_path = reporter.save("\n".join(report_lines), run_id=run_id)
    print_ok(f"Saved: {rpt_path}")

    # ── Save machine-readable metadata for dashboard ──────────────────────────
    meta = {
        "run_id":            run_id,
        "dataset":           config["csv_path"],
        "n_samples":         n_samples,
        "final_acc":         final_acc,
        "first_acc":         first_acc,
        "second_acc":        second_acc,
        "retraining_events": retrain_summary,
        "detector_comparison": comparison,
        "evaluation":        evaluation,
        "log_path":          log_path,
        "plots": {
            "accuracy":          p4,
            "drift":             p1,
            "timeline":          p3,
            "feature":           p2,
            "multi_feature":     p5,
            "confusion_matrix":  p6,
            "windowed_accuracy": p7,
        },
        "report_path":       rpt_path,
    }
    meta_path = os.path.join(config["reports_dir"], f"meta_{run_id}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print_ok(f"Saved: {meta_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print_section("Run Summary")
    elapsed = timer.elapsed()
    summary = {
        "Run ID":                    run_id,
        "Samples Processed":         n_samples,
        "ADWIN Drifts":              comparison["ADWIN"]["total_drifts"],
        "DDM Drifts":                comparison["DDM"]["total_drifts"],
        "EDDM Drifts":               comparison["EDDM"]["total_drifts"],
        "Retraining Events":         retrain_summary["total_retraining_events"],
        "Final Accuracy":            f"{final_acc:.4f}",
        "First-Half Accuracy":       f"{first_acc:.4f}",
        "Second-Half Accuracy":      f"{second_acc:.4f}",
        "F1 Score":                  f"{evaluation['confusion_matrix']['f1']:.4f}",
        "Cohen's Kappa":             f"{evaluation['kappa']:.4f}",
        "CSV Log":                   log_path,
        "Metadata":                  meta_path,
    }
    print_summary(summary)
    print_footer(elapsed)
    return summary


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline(CONFIG)
