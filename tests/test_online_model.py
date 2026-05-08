"""
test_online_model.py
--------------------
Unit tests for the OnlineModel (SGDClassifier wrapper).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from src.online_model import OnlineModel


class TestOnlineModel:
    """Tests for the OnlineModel (SGD-based online learner)."""

    def test_init_defaults(self):
        """Model should initialise with correct defaults."""
        model = OnlineModel(n_features=4)
        assert model.n_features == 4
        assert model.is_ready is False
        summary = model.get_summary()
        assert summary["model_type"] == "SGDClassifier (log_loss)"
        assert summary["samples_seen"] == 0
        assert summary["retrain_count"] == 0

    def test_not_ready_before_both_classes(self):
        """Model should not be ready until it has seen both classes."""
        model = OnlineModel(n_features=4)
        X = np.array([0.5, 0.5, 0.5, 0.5])
        model.update(X, 0)
        assert model.is_ready is False
        model.update(X, 1)
        assert model.is_ready is True

    def test_predict_returns_none_when_not_ready(self):
        """predict() should return None before model is ready."""
        model = OnlineModel(n_features=4)
        X = np.array([0.5, 0.5, 0.5, 0.5])
        assert model.predict(X) is None

    def test_predict_returns_int_after_ready(self):
        """predict() should return an integer class after training."""
        model = OnlineModel(n_features=4)
        X0 = np.array([0.0, 0.0, 0.0, 0.0])
        X1 = np.array([1.0, 1.0, 1.0, 1.0])
        model.update(X0, 0)
        model.update(X1, 1)
        pred = model.predict(X0)
        assert isinstance(pred, int)
        assert pred in [0, 1]

    def test_reset_creates_new_model(self):
        """reset() should create a fresh model and increment retrain count."""
        model = OnlineModel(n_features=4)
        X = np.array([0.5, 0.5, 0.5, 0.5])
        model.update(X, 0)
        model.update(X, 1)
        assert model.is_ready is True

        model.reset()
        assert model.is_ready is False
        assert model.get_summary()["retrain_count"] == 1

    def test_samples_seen_increments(self):
        """samples_seen should increment with each update."""
        model = OnlineModel(n_features=4)
        X = np.array([0.5, 0.5, 0.5, 0.5])
        for i in range(10):
            model.update(X, i % 2)
        assert model.get_summary()["samples_seen"] == 10

    def test_multiple_resets(self):
        """Multiple resets should increment retrain_count correctly."""
        model = OnlineModel(n_features=4)
        model.reset()
        model.reset()
        model.reset()
        assert model.get_summary()["retrain_count"] == 3

    def test_prediction_on_8_features(self):
        """Model should work with 8 features (ELEC2 dataset)."""
        model = OnlineModel(n_features=8)
        X0 = np.zeros(8)
        X1 = np.ones(8)
        model.update(X0, 0)
        model.update(X1, 1)
        pred = model.predict(X1)
        assert pred in [0, 1]
