"""
detector_manager.py
-------------------
Runs ADWIN, DDM, and EDDM in parallel on the same error stream.

Role of each detector:
  ADWIN  — PRIMARY detector. Triggers retraining. Window-based.
  DDM    — COMPARISON only. Error-rate based. Gama et al., 2004.
  EDDM   — COMPARISON only. Inter-error gap based. Baena-García et al., 2006.

The manager feeds every binary error signal to all three detectors
simultaneously. Only ADWIN's drift signal is returned for use in the
retraining gate. DDM and EDDM accumulate their detections internally and
are queried at the end for comparison metrics.

This design cleanly separates the control flow (retraining) from the
analysis (comparison) without modifying any existing module.
"""

from src.drift_detectors.adwin_detector import ADWINDetector
from src.drift_detectors.ddm_detector   import DDMDetector
from src.drift_detectors.eddm_detector  import EDDMDetector


class DetectorManager:
    """
    Parallel drift detector manager.

    Parameters
    ----------
    adwin_delta : float — ADWIN confidence parameter.
    """

    def __init__(self, adwin_delta: float = 0.1):
        self.adwin = ADWINDetector(delta=adwin_delta)
        self.ddm   = DDMDetector()
        self.eddm  = EDDMDetector()

        self._detectors = [self.adwin, self.ddm, self.eddm]

    def update(self, error: float, index: int, primary: str = "ADWIN") -> bool:
        """
        Feeds one error value to all three detectors.

        Returns True only if the *primary* detector signals drift.
        The non-primary detectors' results are stored internally for reporting.

        Parameters
        ----------
        error   : float  — 0.0 (correct) or 1.0 (wrong)
        index   : int    — current stream sample index
        primary : str    — "ADWIN" | "DDM" | "EDDM"  (default: "ADWIN")
        """
        adwin_flag = self.adwin.update(error, index)
        ddm_flag   = self.ddm.update(error, index)
        eddm_flag  = self.eddm.update(error, index)

        flag_map = {
            "ADWIN": adwin_flag,
            "DDM":   ddm_flag,
            "EDDM":  eddm_flag,
        }
        return flag_map.get(primary.upper(), adwin_flag)

    def reset_adwin(self):
        """
        Resets ADWIN after a retraining event (kept for backward compatibility).
        """
        self.adwin.reset()

    def reset_primary(self, primary: str = "ADWIN"):
        """
        Resets whichever detector is acting as primary after a retraining event.
        The non-primary detectors continue accumulating history for comparison.

        Parameters
        ----------
        primary : str — "ADWIN" | "DDM" | "EDDM"
        """
        det_map = {
            "ADWIN": self.adwin,
            "DDM":   self.ddm,
            "EDDM":  self.eddm,
        }
        det = det_map.get(primary.upper())
        if det is not None:
            det.reset()

    def get_comparison_metrics(self) -> dict:
        """
        Returns a structured comparison of all three detectors.

        Includes per-detector drift counts, indices, and computed metrics:
          - detection_frequency : drifts per 1000 samples (requires n_samples)
          - avg_inter_drift_gap : mean samples between consecutive detections

        Returns
        -------
        dict with keys: ADWIN, DDM, EDDM — each a summary sub-dict.
        """
        result = {}
        for det in self._detectors:
            s = det.get_summary()
            indices = s["drift_indices"]
            n       = len(indices)

            # Average gap between consecutive drift detections
            if n >= 2:
                gaps = [indices[i+1] - indices[i] for i in range(n - 1)]
                avg_gap = round(sum(gaps) / len(gaps), 1)
            else:
                avg_gap = None

            result[det.NAME] = {
                "total_drifts":       n,
                "drift_indices":      indices,
                "avg_inter_drift_gap": avg_gap,
            }
            # DDM and EDDM also expose warning counts
            if "warnings" in s:
                result[det.NAME]["warnings"] = s["warnings"]

        return result

    def get_adwin_drift_indices(self) -> list:
        """Returns ADWIN drift indices — used by main pipeline for logging."""
        return self.adwin.drift_indices.copy()
