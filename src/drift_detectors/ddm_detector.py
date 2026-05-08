"""
ddm_detector.py
---------------
DDM (Drift Detection Method) — pure Python implementation.

DDM monitors the error rate of a classifier over a stream. It tracks the
running mean (p) and standard deviation (s) of the error signal and raises
a warning/drift alarm when the statistics exceed threshold multiples of the
minimum observed p + s seen so far.

Algorithm (Gama et al., 2004):
  - Maintain running estimates: p_i (error rate), s_i = sqrt(p_i(1-p_i)/i)
  - Track minimum: p_min + s_min
  - WARNING  level: p_i + s_i  >=  p_min + 2 * s_min
  - DRIFT    level: p_i + s_i  >=  p_min + 3 * s_min
  - On drift: reset statistics

DDM is comparison-only in Phase 4. It does NOT trigger retraining.

Reference:
  Gama, J., Medas, P., Castillo, G., & Rodrigues, P. (2004).
  Learning with Drift Detection. SBIA 2004, LNAI 3171, pp. 286-295.
"""

import math


class DDMDetector:
    """
    DDM implementation. Comparison detector only.

    Parameters
    ----------
    min_samples    : Minimum samples before drift can be signalled.
                     Avoids false alarms during cold-start.
    warning_level  : Multiplier for warning threshold  (default 2.0).
    drift_level    : Multiplier for drift threshold    (default 3.0).
    """

    NAME = "DDM"

    def __init__(
        self,
        min_samples:   int   = 30,
        warning_level: float = 2.0,
        drift_level:   float = 3.0,
    ):
        self.min_samples   = min_samples
        self.warning_level = warning_level
        self.drift_level   = drift_level

        self.drift_indices   = []
        self.warning_indices = []

        self._reset_stats()

    # ── Private ──────────────────────────────────────────────────────────────

    def _reset_stats(self):
        """Resets running statistics but preserves drift/warning history."""
        self._n     = 0       # samples seen since last reset
        self._p     = 0.0     # running error rate
        self._p_min = float("inf")
        self._s_min = float("inf")

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, error: float, index: int) -> bool:
        """
        Feed one binary error value. Returns True if drift detected.

        Parameters
        ----------
        error : float  — 0.0 (correct) or 1.0 (wrong)
        index : int    — current stream sample index
        """
        self._n += 1

        # Incremental mean update
        self._p += (error - self._p) / self._n

        # Standard deviation of Bernoulli(p): sqrt(p(1-p)/n)
        if self._p > 0 and self._p < 1:
            s = math.sqrt(self._p * (1.0 - self._p) / self._n)
        else:
            s = 0.0

        # Not enough samples yet
        if self._n < self.min_samples:
            return False

        # Update minimum
        metric = self._p + s
        if metric < self._p_min + self._s_min:
            self._p_min = self._p
            self._s_min = s

        # Drift threshold check
        if metric > self._p_min + self.drift_level * self._s_min:
            self.drift_indices.append(index)
            self._reset_stats()   # reset on confirmed drift
            return True

        # Warning threshold check (logged but not a drift event)
        if metric > self._p_min + self.warning_level * self._s_min:
            self.warning_indices.append(index)

        return False

    def reset(self):
        """Hard reset — called externally if needed (e.g. after ADWIN retrain)."""
        self._reset_stats()

    def get_summary(self) -> dict:
        return {
            "detector":      self.NAME,
            "total_drifts":  len(self.drift_indices),
            "drift_indices": self.drift_indices.copy(),
            "warnings":      len(self.warning_indices),
        }
