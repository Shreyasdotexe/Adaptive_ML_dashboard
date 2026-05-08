"""
retraining_manager.py
---------------------
Adaptive model retraining on drift detection.

detector parameter is now optional (None).
When DetectorManager is used, it handles its own ADWIN reset directly.
"""

from typing import List

class RetrainingManager:
    def __init__(self, reset_detector: bool = True):
        self.reset_detector = reset_detector
        self._events: List[dict] = []

    def handle_drift(self, sample_index, model, detector, accuracy_at_drift):
        self._events.append({
            "sample_index":      sample_index,
            "accuracy_at_drift": round(accuracy_at_drift, 4),
            "retrain_number":    len(self._events) + 1,
        })
        model.reset()
        # detector may be None in Phase 4 (DetectorManager handles reset)
        if self.reset_detector and detector is not None:
            detector.reset()

    def get_events(self):
        return self._events.copy()

    def get_summary(self):
        if not self._events:
            return {"total_retraining_events": 0,
                    "retraining_indices": [],
                    "accuracy_at_each_drift": []}
        return {
            "total_retraining_events": len(self._events),
            "retraining_indices":      [e["sample_index"] for e in self._events],
            "accuracy_at_each_drift":  [e["accuracy_at_drift"] for e in self._events],
        }
