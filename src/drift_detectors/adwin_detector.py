"""
adwin_detector.py
-----------------
Phase 4 — ADWIN wrapper conforming to the DetectorManager interface.

Wraps the existing DriftDetector (which itself wraps adwin.py) so that
ADWIN participates in the unified DetectorManager alongside DDM and EDDM.

This is the PRIMARY detector — retraining is triggered by ADWIN only.
DDM and EDDM run in parallel for comparison purposes.

Reference:
  Bifet, A. & Gavalda, R. (2007). Learning from Time-Changing Data with
  Adaptive Windowing. SIAM International Conference on Data Mining.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.drift_detection import DriftDetector


class ADWINDetector:
    """
    Thin wrapper that gives DriftDetector the standard detector interface
    expected by DetectorManager.

    Parameters
    ----------
    delta : float
        ADWIN confidence parameter. Lower = more sensitive.
    """

    NAME = "ADWIN"

    def __init__(self, delta: float = 0.1):
        self._detector     = DriftDetector(delta=delta)
        self.drift_indices = []               # indices where drift was flagged

    def update(self, error: float, index: int) -> bool:
        """
        Feed one binary error value. Returns True if drift detected.

        Parameters
        ----------
        error : float  — 0.0 (correct) or 1.0 (wrong)
        index : int    — current stream sample index
        """
        flagged = self._detector.update(error, index)
        if flagged:
            self.drift_indices.append(index)
        return flagged

    def reset(self):
        """Resets the underlying ADWIN window. Called after retraining."""
        self._detector.reset()
        # NOTE: drift_indices is NOT cleared on reset — we keep the full
        # history across retraining events for comparison reporting.

    def get_summary(self) -> dict:
        return {
            "detector":              self.NAME,
            "total_drifts":          len(self.drift_indices),
            "drift_indices":         self.drift_indices.copy(),
        }
