try:
    from river.drift import ADWIN
except ImportError:
    from src.adwin import ADWIN

class DriftDetector:
    def __init__(self, delta=0.002):
        self.delta = delta
        self._detector = ADWIN(delta=delta)
        self._drift_indices = []
        self._total_updates = 0

    def update(self, error_signal, sample_index):
        self._detector.update(error_signal)
        self._total_updates += 1
        if self._detector.drift_detected:
            self._drift_indices.append(sample_index)
            return True
        return False

    def get_drift_indices(self):
        return self._drift_indices.copy()

    def get_summary(self):
        return {
            "detector": "ADWIN",
            "delta": self.delta,
            "total_updates": self._total_updates,
            "total_drifts_detected": len(self._drift_indices),
            "drift_indices": self._drift_indices,
        }

    def reset(self):
        self._detector = ADWIN(delta=self.delta)
        self._drift_indices = []
        self._total_updates = 0
