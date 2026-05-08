"""
test_preprocessing.py
---------------------
Unit tests for the OnlinePreprocessor (streaming StandardScaler).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from src.preprocessing import OnlinePreprocessor


class TestOnlinePreprocessor:
    """Tests for the streaming online preprocessor."""

    def test_init_defaults(self):
        """Preprocessor should initialise with correct defaults."""
        pp = OnlinePreprocessor(n_features=4, warmup_size=50)
        assert pp.n_features == 4
        assert pp.warmup_size == 50
        assert pp.is_ready is False

    def test_not_ready_during_warmup(self):
        """Preprocessor should not be ready during warmup period."""
        pp = OnlinePreprocessor(n_features=4, warmup_size=10)
        X = np.array([1.0, 2.0, 3.0, 4.0])
        for _ in range(9):
            pp.process(X)
        assert pp.is_ready is False

    def test_ready_after_warmup(self):
        """Preprocessor should become ready after warmup_size samples."""
        pp = OnlinePreprocessor(n_features=4, warmup_size=10)
        X = np.array([1.0, 2.0, 3.0, 4.0])
        for _ in range(10):
            pp.process(X)
        assert pp.is_ready is True

    def test_returns_raw_during_warmup(self):
        """During warmup, process() should return raw features."""
        pp = OnlinePreprocessor(n_features=2, warmup_size=5)
        X = np.array([10.0, 20.0])
        result = pp.process(X)
        np.testing.assert_array_equal(result, X)

    def test_returns_scaled_after_warmup(self):
        """After warmup, process() should return scaled features."""
        pp = OnlinePreprocessor(n_features=2, warmup_size=5)
        rng = np.random.RandomState(42)
        for _ in range(5):
            pp.process(rng.randn(2))
        # After warmup, output should be scaled (different from input)
        X = np.array([100.0, 200.0])
        result = pp.process(X)
        assert not np.allclose(result, X), "Scaled output should differ from raw input"

    def test_get_stats_during_warmup(self):
        """get_stats() should indicate warming_up during warmup."""
        pp = OnlinePreprocessor(n_features=2, warmup_size=10)
        stats = pp.get_stats()
        assert stats["status"] == "warming_up"

    def test_get_stats_after_warmup(self):
        """get_stats() should return fitted stats after warmup."""
        pp = OnlinePreprocessor(n_features=2, warmup_size=5)
        for _ in range(5):
            pp.process(np.array([1.0, 2.0]))
        stats = pp.get_stats()
        assert stats["status"] == "fitted (incremental)"
        assert "feature_means" in stats
        assert "feature_vars" in stats
        assert len(stats["feature_means"]) == 2

    def test_output_shape(self):
        """Output should have same shape as input."""
        pp = OnlinePreprocessor(n_features=4, warmup_size=5)
        X = np.array([1.0, 2.0, 3.0, 4.0])
        for _ in range(10):
            result = pp.process(X)
        assert result.shape == X.shape
