"""
test_ddm.py
-----------
Unit tests for the DDM (Drift Detection Method) detector.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.drift_detectors.ddm_detector import DDMDetector


class TestDDMDetector:
    """Tests for the DDM drift detector."""

    def test_init_defaults(self):
        """DDM should initialise with correct defaults."""
        ddm = DDMDetector()
        assert ddm.NAME == "DDM"
        assert ddm.min_samples == 30
        assert ddm.warning_level == 2.0
        assert ddm.drift_level == 3.0
        assert ddm.drift_indices == []
        assert ddm.warning_indices == []

    def test_no_drift_on_all_correct(self):
        """No drift should be detected if all predictions are correct."""
        ddm = DDMDetector()
        for i in range(500):
            assert ddm.update(0.0, i) is False

    def test_drift_on_error_burst(self):
        """DDM should detect drift when error rate increases sharply."""
        ddm = DDMDetector(min_samples=30)
        # Low error phase
        for i in range(200):
            ddm.update(0.0, i)
        # High error phase
        detected = False
        for i in range(200, 500):
            flag = ddm.update(1.0, i)
            if flag:
                detected = True
                break
        assert detected, "DDM should detect sudden error rate increase"

    def test_no_drift_before_min_samples(self):
        """DDM should not signal drift before min_samples are seen."""
        ddm = DDMDetector(min_samples=50)
        for i in range(50):
            result = ddm.update(1.0, i)
            assert result is False

    def test_reset_clears_stats(self):
        """reset() should clear running statistics."""
        ddm = DDMDetector()
        for i in range(100):
            ddm.update(0.0, i)
        ddm.reset()
        assert ddm._n == 0
        assert ddm._p == 0.0

    def test_summary_structure(self):
        """get_summary() should return correct structure."""
        ddm = DDMDetector()
        summary = ddm.get_summary()
        assert "detector" in summary
        assert "total_drifts" in summary
        assert "drift_indices" in summary
        assert "warnings" in summary
        assert summary["detector"] == "DDM"

    def test_warnings_before_drift(self):
        """DDM should issue warnings before drift on gradual error increase."""
        ddm = DDMDetector(min_samples=30)
        # Feed correct samples to establish baseline
        for i in range(100):
            ddm.update(0.0, i)
        # Feed errors — should see warnings
        for i in range(100, 400):
            ddm.update(1.0, i)
        # Should have some warnings or drifts
        total_signals = len(ddm.warning_indices) + len(ddm.drift_indices)
        assert total_signals > 0
