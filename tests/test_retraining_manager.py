"""
test_retraining_manager.py
--------------------------
Unit tests for the RetrainingManager.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock
from src.retraining_manager import RetrainingManager


class TestRetrainingManager:
    """Tests for the RetrainingManager."""

    def test_init_defaults(self):
        """Manager should initialise with default settings."""
        rm = RetrainingManager()
        assert rm.reset_detector is True
        summary = rm.get_summary()
        assert summary["total_retraining_events"] == 0
        assert summary["retraining_indices"] == []
        assert summary["accuracy_at_each_drift"] == []

    def test_handle_drift_resets_model(self):
        """handle_drift() should call model.reset()."""
        rm = RetrainingManager()
        mock_model = MagicMock()
        mock_detector = MagicMock()

        rm.handle_drift(
            sample_index=1000,
            model=mock_model,
            detector=mock_detector,
            accuracy_at_drift=0.85,
        )

        mock_model.reset.assert_called_once()

    def test_handle_drift_resets_detector_when_enabled(self):
        """handle_drift() should reset detector when reset_detector=True."""
        rm = RetrainingManager(reset_detector=True)
        mock_model = MagicMock()
        mock_detector = MagicMock()

        rm.handle_drift(1000, mock_model, mock_detector, 0.85)
        mock_detector.reset.assert_called_once()

    def test_handle_drift_skips_detector_reset_when_disabled(self):
        """handle_drift() should NOT reset detector when reset_detector=False."""
        rm = RetrainingManager(reset_detector=False)
        mock_model = MagicMock()
        mock_detector = MagicMock()

        rm.handle_drift(1000, mock_model, mock_detector, 0.85)
        mock_detector.reset.assert_not_called()

    def test_handle_drift_with_none_detector(self):
        """handle_drift() should not crash when detector is None (Phase 4)."""
        rm = RetrainingManager(reset_detector=True)
        mock_model = MagicMock()
        # Should not raise
        rm.handle_drift(1000, mock_model, None, 0.85)
        mock_model.reset.assert_called_once()

    def test_events_accumulate(self):
        """Multiple drift events should be recorded correctly."""
        rm = RetrainingManager(reset_detector=False)
        mock_model = MagicMock()

        rm.handle_drift(100, mock_model, None, 0.90)
        rm.handle_drift(500, mock_model, None, 0.80)
        rm.handle_drift(900, mock_model, None, 0.75)

        summary = rm.get_summary()
        assert summary["total_retraining_events"] == 3
        assert summary["retraining_indices"] == [100, 500, 900]
        assert summary["accuracy_at_each_drift"] == [0.9, 0.8, 0.75]

    def test_get_events_returns_copy(self):
        """get_events() should return a copy, not the internal list."""
        rm = RetrainingManager(reset_detector=False)
        mock_model = MagicMock()
        rm.handle_drift(100, mock_model, None, 0.90)

        events = rm.get_events()
        events.clear()
        assert len(rm.get_events()) == 1
