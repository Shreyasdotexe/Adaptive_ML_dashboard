"""
test_integration.py
-------------------
Integration test: runs the full pipeline on a small subset and
verifies that all outputs are produced correctly.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import json
import pandas as pd

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "electricity.csv")
DATASET_EXISTS = os.path.exists(DATASET_PATH)


@pytest.mark.skipif(not DATASET_EXISTS, reason="electricity.csv not available")
class TestIntegration:
    """End-to-end integration tests on a small subset of the ELEC2 dataset."""

    def test_full_pipeline_small_run(self, tmp_path):
        """
        Run the pipeline on 500 samples and verify:
        - CSV log is created with correct columns
        - Plots are generated
        - Report and metadata JSON are created
        - Metadata has expected keys
        """
        from main import run_pipeline

        log_dir    = str(tmp_path / "logs")
        plots_dir  = str(tmp_path / "plots")
        reports_dir = str(tmp_path / "reports")

        config = {
            "csv_path":           DATASET_PATH,
            "max_samples":        500,
            "warmup_size":        50,
            "adwin_delta":        0.1,
            "cooldown_samples":   100,
            "post_drift_settle":  100,
            "model_random_state": 42,
            "n_features":         8,
            "log_dir":            log_dir,
            "plots_dir":          plots_dir,
            "reports_dir":        reports_dir,
        }

        summary = run_pipeline(config)

        # Check summary keys
        assert "Run ID" in summary
        assert "Samples Processed" in summary

        # Check CSV log
        log_files = [f for f in os.listdir(log_dir) if f.endswith(".csv")]
        assert len(log_files) == 1
        df = pd.read_csv(os.path.join(log_dir, log_files[0]))
        assert len(df) == 500
        expected_cols = ["sample_index", "true_label", "prediction",
                         "error", "running_acc", "retrained"]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Check plots
        plot_files = os.listdir(plots_dir)
        assert len(plot_files) >= 4, f"Expected at least 4 plots, got {len(plot_files)}"

        # Check for new Phase 4+ plots
        plot_names = " ".join(plot_files)
        assert "accuracy" in plot_names
        assert "drift_signal" in plot_names
        assert "detection_timeline" in plot_names

        # Check metadata JSON
        meta_files = [f for f in os.listdir(reports_dir) if f.startswith("meta_")]
        assert len(meta_files) == 1
        with open(os.path.join(reports_dir, meta_files[0])) as f:
            meta = json.load(f)
        assert "run_id" in meta
        assert "final_acc" in meta
        assert "detector_comparison" in meta
        assert "evaluation" in meta
        assert "plots" in meta

        # Check evaluation metrics are present in metadata
        eval_data = meta["evaluation"]
        assert "confusion_matrix" in eval_data
        assert "prequential_accuracy" in eval_data
        assert "kappa" in eval_data
        assert "windowed_accuracy" in eval_data

    def test_pipeline_produces_valid_accuracy(self, tmp_path):
        """Accuracy should be between 0 and 1."""
        from main import run_pipeline

        config = {
            "csv_path":           DATASET_PATH,
            "max_samples":        200,
            "warmup_size":        30,
            "adwin_delta":        0.1,
            "cooldown_samples":   50,
            "post_drift_settle":  50,
            "model_random_state": 42,
            "n_features":         8,
            "log_dir":            str(tmp_path / "logs"),
            "plots_dir":          str(tmp_path / "plots"),
            "reports_dir":        str(tmp_path / "reports"),
        }

        summary = run_pipeline(config)
        acc = float(summary["Final Accuracy"])
        assert 0.0 <= acc <= 1.0

    def test_detector_comparison_has_all_three(self, tmp_path):
        """Detector comparison should include ADWIN, DDM, and EDDM."""
        from main import run_pipeline

        config = {
            "csv_path":           DATASET_PATH,
            "max_samples":        300,
            "warmup_size":        30,
            "adwin_delta":        0.1,
            "cooldown_samples":   50,
            "post_drift_settle":  50,
            "model_random_state": 42,
            "n_features":         8,
            "log_dir":            str(tmp_path / "logs"),
            "plots_dir":          str(tmp_path / "plots"),
            "reports_dir":        str(tmp_path / "reports"),
        }

        run_pipeline(config)

        meta_files = [f for f in os.listdir(str(tmp_path / "reports"))
                      if f.startswith("meta_")]
        with open(os.path.join(str(tmp_path / "reports"), meta_files[0])) as f:
            meta = json.load(f)

        comp = meta["detector_comparison"]
        assert "ADWIN" in comp
        assert "DDM" in comp
        assert "EDDM" in comp
