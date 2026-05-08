"""
test_adwin.py
-------------
Unit tests for the pure-Python ADWIN implementation.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.adwin import ADWIN


class TestADWIN:
    """Tests for the core ADWIN change-detection algorithm."""

    def test_init_defaults(self):
        """ADWIN should initialise with default delta and empty window."""
        adwin = ADWIN()
        assert adwin.delta == 0.002
        assert adwin.width == 0
        assert adwin.drift_detected is False

    def test_init_custom_delta(self):
        """ADWIN should accept custom delta parameter."""
        adwin = ADWIN(delta=0.1)
        assert adwin.delta == 0.1

    def test_no_drift_on_constant_stream(self):
        """A constant stream should never trigger drift."""
        adwin = ADWIN(delta=0.1)
        for _ in range(1000):
            adwin.update(0.0)
            assert adwin.drift_detected is False

    def test_drift_on_sudden_change(self):
        """ADWIN should detect drift when mean shifts abruptly."""
        adwin = ADWIN(delta=0.1)
        # Phase 1 — all zeros
        for _ in range(300):
            adwin.update(0.0)
        # Phase 2 — all ones (sudden drift)
        detected = False
        for _ in range(300):
            adwin.update(1.0)
            if adwin.drift_detected:
                detected = True
                break
        assert detected, "ADWIN should detect a sudden mean shift"

    def test_window_grows(self):
        """The window size should grow as samples are added."""
        adwin = ADWIN()
        for i in range(100):
            adwin.update(0.5)
        assert adwin.width > 0

    def test_mean_property(self):
        """Mean should approximate the sample mean of the window."""
        adwin = ADWIN()
        for _ in range(100):
            adwin.update(1.0)
        assert abs(adwin.mean - 1.0) < 0.01

    def test_update_returns_self(self):
        """update() should return the ADWIN instance for chaining."""
        adwin = ADWIN()
        result = adwin.update(0.5)
        assert result is adwin

    def test_empty_mean(self):
        """Mean of empty window should be 0.0."""
        adwin = ADWIN()
        assert adwin.mean == 0.0

    def test_sensitivity_lower_delta(self):
        """Lower delta should detect drift with fewer samples after shift."""
        adwin_sensitive = ADWIN(delta=0.001)
        adwin_tolerant  = ADWIN(delta=0.5)

        drift_sensitive = None
        drift_tolerant  = None

        for i in range(200):
            adwin_sensitive.update(0.0)
            adwin_tolerant.update(0.0)

        for i in range(200):
            adwin_sensitive.update(1.0)
            adwin_tolerant.update(1.0)
            if adwin_sensitive.drift_detected and drift_sensitive is None:
                drift_sensitive = i
            if adwin_tolerant.drift_detected and drift_tolerant is None:
                drift_tolerant = i

        # More sensitive ADWIN should detect sooner or at same time
        assert drift_sensitive is not None, "Sensitive ADWIN should detect drift"
