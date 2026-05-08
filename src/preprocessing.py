import numpy as np
from sklearn.preprocessing import StandardScaler

class OnlinePreprocessor:
    def __init__(self, n_features=4, warmup_size=50):
        self.n_features = n_features
        self.warmup_size = warmup_size
        self.scaler = StandardScaler()
        self._samples_seen = 0
        self._is_ready = False

    def process(self, features):
        X = features.reshape(1, -1)
        self.scaler.partial_fit(X)
        self._samples_seen += 1
        if not self._is_ready:
            if self._samples_seen >= self.warmup_size:
                self._is_ready = True
            return features
        return self.scaler.transform(X).flatten()

    @property
    def is_ready(self):
        return self._is_ready

    def get_stats(self):
        if not self._is_ready:
            return {"status": "warming_up", "samples_seen": self._samples_seen}
        return {
            "status": "fitted (incremental)",
            "samples_seen": self._samples_seen,
            "feature_means": self.scaler.mean_.tolist(),
            "feature_vars": self.scaler.var_.tolist(),
        }
