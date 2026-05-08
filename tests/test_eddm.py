"""
test_eddm.py
------------
Unit tests for the EDDM (Early Drift Detection Method) detector.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.drift_detectors.eddm_detector import EDDMDetector


class TestEDDMDetector:
    """Tests for the EDDM drift detector."""

    def test_init_defaults(self):
        """EDDM should initialise with correct defaults."""
        eddm = EDDMDetector()
        assert eddm.NAME == "EDDM"
        assert eddm.min_errors == 30
        assert eddm.warning_level == 0.95
        assert eddm.drift_level == 0.90
        assert eddm.drift_indices == []

    def test_no_update_on_correct_prediction(self):
        """EDDM should not update stats on correct predictions."""
        eddm = EDDMDetector()
        for i in range(500):
            result = eddm.update(0.0, i)  # all correct
            assert result is False
        assert eddm._n_errors == 0

    def test_error_count_increments(self):
        """EDDM should count errors correctly."""
        eddm = EDDMDetector()
        for i in range(10):
            eddm.update(1.0, i)  # all errors
        assert eddm._n_errors == 10

    def test_drift_on_frequent_errors(self):
        """EDDM should detect drift when errors become very frequent."""
        eddm = EDDMDetector(min_errors=10)
        # Sparse errors (large gaps) — establishes a high metric baseline
        idx = 0
        for _ in range(20):
            for _ in range(50):
                eddm.update(0.0, idx)
                idx += 1
            eddm.update(1.0, idx)
            idx += 1
        # Now rapid-fire errors (very small gaps) — metric should drop
        detected = False
        for _ in range(100):
            flag = eddm.update(1.0, idx)
            idx += 1
            if flag:
                detected = True
                break
        assert detected, "EDDM should detect drift when error gaps shrink"

    def test_no_drift_before_min_errors(self):
        """EDDM should not signal drift before min_errors are seen."""
        eddm = EDDMDetector(min_errors=50)
        for i in range(50):
            result = eddm.update(1.0, i)
            assert result is False

    def test_reset_clears_stats(self):
        """reset() should clear statistics but preserve drift history."""
        eddm = EDDMDetector()
        for i in range(50):
            eddm.update(1.0, i)
        eddm.reset()
        assert eddm._n_errors == 0
        assert eddm._last_error is None
        assert eddm._max_metric == 0.0

    def test_summary_structure(self):
        """get_summary() should return correct structure."""
        eddm = EDDMDetector()
        s = eddm.get_summary()
        assert s["detector"] == "EDDM"
        assert "total_drifts" in s
        assert "drift_indices" in s
        assert "warnings" in s
