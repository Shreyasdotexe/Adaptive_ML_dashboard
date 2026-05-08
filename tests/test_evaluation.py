"""
test_evaluation.py
------------------
Unit tests for the evaluation module (streaming metrics).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.evaluation import (
    StreamingConfusionMatrix,
    WindowedAccuracyTracker,
    RecoveryTimeAnalyser,
)


class TestStreamingConfusionMatrix:
    """Tests for the streaming confusion matrix."""

    def test_init_zeros(self):
        """Matrix should start with all zeros."""
        cm = StreamingConfusionMatrix()
        assert cm.tp == 0
        assert cm.tn == 0
        assert cm.fp == 0
        assert cm.fn == 0
        assert cm.total == 0

    def test_update_true_positive(self):
        """Update with (1, 1) should increment TP."""
        cm = StreamingConfusionMatrix()
        cm.update(1, 1)
        assert cm.tp == 1

    def test_update_true_negative(self):
        """Update with (0, 0) should increment TN."""
        cm = StreamingConfusionMatrix()
        cm.update(0, 0)
        assert cm.tn == 1

    def test_update_false_positive(self):
        """Update with (0, 1) should increment FP."""
        cm = StreamingConfusionMatrix()
        cm.update(0, 1)
        assert cm.fp == 1

    def test_update_false_negative(self):
        """Update with (1, 0) should increment FN."""
        cm = StreamingConfusionMatrix()
        cm.update(1, 0)
        assert cm.fn == 1

    def test_accuracy(self):
        """Accuracy should be (TP+TN)/total."""
        cm = StreamingConfusionMatrix()
        cm.update(1, 1)  # TP
        cm.update(0, 0)  # TN
        cm.update(0, 1)  # FP
        cm.update(1, 0)  # FN
        assert cm.accuracy == 0.5

    def test_perfect_accuracy(self):
        """All correct predictions should give 1.0 accuracy."""
        cm = StreamingConfusionMatrix()
        for _ in range(50):
            cm.update(1, 1)
            cm.update(0, 0)
        assert cm.accuracy == 1.0

    def test_precision_recall_f1(self):
        """Precision, recall, F1 should be correct for known values."""
        cm = StreamingConfusionMatrix()
        # TP=3, FP=1, FN=2, TN=4
        for _ in range(3):
            cm.update(1, 1)
        cm.update(0, 1)
        for _ in range(2):
            cm.update(1, 0)
        for _ in range(4):
            cm.update(0, 0)
        assert cm.precision == pytest.approx(3 / 4)
        assert cm.recall == pytest.approx(3 / 5)
        expected_f1 = 2 * (3/4) * (3/5) / ((3/4) + (3/5))
        assert cm.f1 == pytest.approx(expected_f1)

    def test_kappa_perfect_agreement(self):
        """Perfect agreement should give kappa close to 1.0."""
        cm = StreamingConfusionMatrix()
        for _ in range(100):
            cm.update(1, 1)
            cm.update(0, 0)
        assert cm.kappa > 0.95

    def test_kappa_random_agreement(self):
        """Random predictions should give kappa near 0."""
        cm = StreamingConfusionMatrix()
        # Equal distribution: all 4 cells equal
        for _ in range(100):
            cm.update(1, 1)
            cm.update(0, 0)
            cm.update(0, 1)
            cm.update(1, 0)
        assert abs(cm.kappa) < 0.05

    def test_reset(self):
        """reset() should clear all counters."""
        cm = StreamingConfusionMatrix()
        cm.update(1, 1)
        cm.update(0, 0)
        cm.reset()
        assert cm.total == 0

    def test_matrix_layout(self):
        """matrix() should return [[TN, FP], [FN, TP]]."""
        cm = StreamingConfusionMatrix()
        cm.update(1, 1)  # TP=1
        cm.update(0, 0)  # TN=1
        cm.update(0, 1)  # FP=1
        cm.update(1, 0)  # FN=1
        m = cm.matrix()
        assert m == [[1, 1], [1, 1]]

    def test_to_dict_structure(self):
        """to_dict() should have all expected keys."""
        cm = StreamingConfusionMatrix()
        cm.update(1, 1)
        d = cm.to_dict()
        expected_keys = {"tp", "tn", "fp", "fn", "accuracy", "precision",
                         "recall", "f1", "specificity", "kappa", "total"}
        assert expected_keys == set(d.keys())

    def test_empty_accuracy(self):
        """Accuracy of empty matrix should be 0."""
        cm = StreamingConfusionMatrix()
        assert cm.accuracy == 0.0

    def test_empty_precision(self):
        """Precision with no positive predictions should be 0."""
        cm = StreamingConfusionMatrix()
        assert cm.precision == 0.0


class TestWindowedAccuracyTracker:
    """Tests for the windowed accuracy tracker."""

    def test_init(self):
        """Tracker should initialise with empty history."""
        tracker = WindowedAccuracyTracker(window_size=100)
        assert len(tracker.history) == 0

    def test_all_correct(self):
        """All correct predictions should give 1.0 windowed accuracy."""
        tracker = WindowedAccuracyTracker(window_size=50)
        for _ in range(100):
            tracker.update(0)  # 0 = correct
        assert tracker.current == 1.0

    def test_all_wrong(self):
        """All wrong predictions should give 0.0 windowed accuracy."""
        tracker = WindowedAccuracyTracker(window_size=50)
        for _ in range(100):
            tracker.update(1)  # 1 = wrong
        assert tracker.current == 0.0

    def test_window_slides(self):
        """Accuracy should reflect only the recent window."""
        tracker = WindowedAccuracyTracker(window_size=100)
        # First 100: all correct (acc=1.0)
        for _ in range(100):
            tracker.update(0)
        assert tracker.current == 1.0
        # Next 100: all wrong
        for _ in range(100):
            tracker.update(1)
        # Now the window should contain only errors
        assert tracker.current == 0.0

    def test_history_length_matches_updates(self):
        """History should have one entry per update."""
        tracker = WindowedAccuracyTracker(window_size=50)
        for _ in range(30):
            tracker.update(0)
        assert len(tracker.history) == 30


class TestRecoveryTimeAnalyser:
    """Tests for the recovery time analyser."""

    def test_no_events(self):
        """Empty analysis should return zero events."""
        analyser = RecoveryTimeAnalyser()
        results = analyser.analyse([], [], [])
        assert len(results) == 0

    def test_recovery_detected(self):
        """Recovery should be detected when accuracy exceeds threshold."""
        errors = [0] * 200 + [1] * 50 + [0] * 300
        indices = list(range(550))

        analyser = RecoveryTimeAnalyser(recovery_threshold=0.85, window_size=50)
        results = analyser.analyse(errors, indices, retraining_indices=[200])

        assert len(results) == 1
        assert results[0]["retrain_at"] == 200
        assert results[0]["recovered"] is True
        assert results[0]["recovery_samples"] is not None

    def test_summary_structure(self):
        """get_summary() should return expected structure."""
        analyser = RecoveryTimeAnalyser()
        analyser.analyse([0] * 100, list(range(100)), [50])
        s = analyser.get_summary()
        assert "events" in s
        assert "details" in s
