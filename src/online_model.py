import numpy as np
from sklearn.linear_model import SGDClassifier

class OnlineModel:
    def __init__(self, n_features=4, random_state=42):
        self.n_features = n_features
        self.random_state = random_state
        self.classes_ = np.array([0, 1])
        self._model = self._build_model()
        self._samples_seen = 0
        self._classes_seen = set()
        self._retrain_count = 0

    def _build_model(self):
        return SGDClassifier(loss="log_loss", learning_rate="optimal",
                             random_state=self.random_state, warm_start=False)

    def predict(self, features):
        if not self.is_ready:
            return None
        return int(self._model.predict(features.reshape(1, -1))[0])

    def update(self, features, label):
        self._model.partial_fit(features.reshape(1, -1), np.array([label]),
                                classes=self.classes_)
        self._samples_seen += 1
        self._classes_seen.add(label)

    def reset(self):
        self._model = self._build_model()
        self._classes_seen = set()
        self._retrain_count += 1

    @property
    def is_ready(self):
        return len(self._classes_seen) == 2

    def get_summary(self):
        return {
            "model_type": "SGDClassifier (log_loss)",
            "samples_seen": self._samples_seen,
            "retrain_count": self._retrain_count,
            "is_ready": self.is_ready,
        }
