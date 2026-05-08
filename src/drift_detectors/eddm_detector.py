"""
eddm_detector.py
----------------
EDDM (Early Drift Detection Method) — pure Python implementation.

EDDM is an improvement over DDM designed to detect gradual concept drift
earlier. Instead of tracking the running error rate, it tracks the distance
(in number of samples) between consecutive errors. A shortening gap between
errors indicates that the classifier is making mistakes more frequently —
a signature of gradual drift.

Algorithm (Baena-García et al., 2006):
  - Track positions of consecutive errors: i and i-1
  - Compute: mean (μ) and std (σ) of the gap distribution
  - Maintain maximum: (μ + 2σ)_max
  - WARNING  level: current (μ + 2σ)  <  0.95 * max
  - DRIFT    level: current (μ + 2σ)  <  0.90 * max
  - On drift: reset statistics

EDDM is comparison-only in Phase 4. It does NOT trigger retraining.

Reference:
  Baena-García, M., del Campo-Ávila, J., Fidalgo, R., Bifet, A.,
  Gavalda, R., & Morales-Bueno, R. (2006). Early Drift Detection Method.
  4th ECML PKDD International Workshop on Knowledge Discovery from Data Streams.
"""

import math


class EDDMDetector:
    """
    Pure-Python EDDM implementation. Comparison detector only.

    Parameters
    ----------
    min_errors     : Minimum number of errors before drift can be signalled.
    warning_level  : Fraction of max metric for warning  (default 0.95).
    drift_level    : Fraction of max metric for drift    (default 0.90).
    """

    NAME = "EDDM"

    def __init__(
        self,
        min_errors:    int   = 30,
        warning_level: float = 0.95,
        drift_level:   float = 0.90,
    ):
        self.min_errors    = min_errors
        self.warning_level = warning_level
        self.drift_level   = drift_level

        self.drift_indices   = []
        self.warning_indices = []

        self._reset_stats()

    # ── Private ──────────────────────────────────────────────────────────────

    def _reset_stats(self):
        """Resets running statistics but preserves drift/warning history."""
        self._n_errors    = 0       # total errors seen since last reset
        self._last_error  = None    # sample index of previous error
        self._mean        = 0.0     # running mean of inter-error gap
        self._variance    = 0.0     # running variance of inter-error gap (Welford)
        self._max_metric  = 0.0     # max (mean + 2*std) seen so far

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, error: float, index: int) -> bool:
        """
        Feed one binary error value. Returns True if drift detected.

        Parameters
        ----------
        error : float  — 0.0 (correct) or 1.0 (wrong)
        index : int    — current stream sample index
        """
        if error != 1.0:
            return False   # EDDM only updates on actual errors

        self._n_errors += 1

        if self._last_error is None:
            # First error — just record position, no gap yet
            self._last_error = index
            return False

        # Gap between this error and the previous one
        gap = index - self._last_error
        self._last_error = index

        # Welford's online mean and variance update
        self._n_errors_for_stats = getattr(self, "_n_errors_for_stats", 0) + 1
        n   = self._n_errors_for_stats
        old_mean     = self._mean
        self._mean   += (gap - old_mean) / n
        self._variance += (gap - old_mean) * (gap - self._mean)

        std = math.sqrt(self._variance / n) if n > 1 else 0.0

        # Not enough errors to make a reliable estimate
        if self._n_errors < self.min_errors:
            return False

        metric = self._mean + 2.0 * std

        # Update maximum
        if metric > self._max_metric:
            self._max_metric = metric

        if self._max_metric == 0:
            return False

        # Drift check — metric has dropped significantly below its historical max
        if metric < self.drift_level * self._max_metric:
            self.drift_indices.append(index)
            self._reset_stats()
            return True

        # Warning check
        if metric < self.warning_level * self._max_metric:
            self.warning_indices.append(index)

        return False

    def reset(self):
        """Hard reset — called externally if needed."""
        self._reset_stats()

    def get_summary(self) -> dict:
        return {
            "detector":      self.NAME,
            "total_drifts":  len(self.drift_indices),
            "drift_indices": self.drift_indices.copy(),
            "warnings":      len(self.warning_indices),
        }
