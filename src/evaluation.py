"""
evaluation.py
-------------
Streaming evaluation metrics for the Adaptive ML System.

Provides:
  1. Prequential (interleaved test-then-train) accuracy tracking
  2. Windowed accuracy (sliding window of last N predictions)
  3. Recovery time analysis (samples to recover after retraining)
  4. Streaming confusion matrix (TP, TN, FP, FN over time)
  5. Per-class precision, recall, F1 (binary classification)
  6. Kappa statistic (agreement beyond chance)

All metrics are designed for streaming evaluation — no batch assumptions.

References:
  - Gama, J. et al. (2013). Evaluating stream learning algorithms.
    Machine Learning, 90(3), pp. 317–346.
  - Bifet, A. et al. (2015). Efficient Online Evaluation of Big Data
    Stream Classifiers. KDD 2015.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple


class StreamingConfusionMatrix:
    """
    Incrementally maintained confusion matrix for binary classification.

    Updated one sample at a time. Provides precision, recall, F1 and
    Cohen's Kappa at any point during the stream.
    """

    def __init__(self):
        self.tp = 0   # true positives  (predicted 1, actual 1)
        self.tn = 0   # true negatives  (predicted 0, actual 0)
        self.fp = 0   # false positives (predicted 1, actual 0)
        self.fn = 0   # false negatives (predicted 0, actual 1)

    def update(self, y_true: int, y_pred: int):
        """Feed one prediction–truth pair."""
        if y_true == 1 and y_pred == 1:
            self.tp += 1
        elif y_true == 0 and y_pred == 0:
            self.tn += 1
        elif y_true == 0 and y_pred == 1:
            self.fp += 1
        elif y_true == 1 and y_pred == 0:
            self.fn += 1

    def reset(self):
        self.tp = self.tn = self.fp = self.fn = 0

    @property
    def total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def specificity(self) -> float:
        denom = self.tn + self.fp
        return self.tn / denom if denom > 0 else 0.0

    @property
    def kappa(self) -> float:
        """Cohen's Kappa — agreement beyond chance."""
        n = self.total
        if n == 0:
            return 0.0
        p_observed = self.accuracy
        p_yes = ((self.tp + self.fp) * (self.tp + self.fn)) / (n * n)
        p_no  = ((self.tn + self.fn) * (self.tn + self.fp)) / (n * n)
        p_expected = p_yes + p_no
        if p_expected == 1.0:
            return 1.0
        return (p_observed - p_expected) / (1.0 - p_expected)

    def to_dict(self) -> dict:
        return {
            "tp": self.tp, "tn": self.tn, "fp": self.fp, "fn": self.fn,
            "accuracy":    round(self.accuracy, 4),
            "precision":   round(self.precision, 4),
            "recall":      round(self.recall, 4),
            "f1":          round(self.f1, 4),
            "specificity": round(self.specificity, 4),
            "kappa":       round(self.kappa, 4),
            "total":       self.total,
        }

    def matrix(self) -> List[List[int]]:
        """Returns [[TN, FP], [FN, TP]] — standard confusion matrix layout."""
        return [[self.tn, self.fp], [self.fn, self.tp]]


class WindowedAccuracyTracker:
    """
    Tracks accuracy over a sliding window of the last N predictions.

    Unlike cumulative running accuracy, windowed accuracy shows *local*
    model performance — making it easy to see post-drift degradation
    and post-retrain recovery.

    Parameters
    ----------
    window_size : int — number of recent predictions to consider (default 500).
    """

    def __init__(self, window_size: int = 500):
        self.window_size = window_size
        self._errors = []          # full history of 0/1 errors
        self._windowed_acc = []    # windowed accuracy at each step

    def update(self, error: int):
        """Feed one binary error (0 = correct, 1 = wrong)."""
        self._errors.append(error)
        n = len(self._errors)
        start = max(0, n - self.window_size)
        window = self._errors[start:]
        acc = 1.0 - (sum(window) / len(window))
        self._windowed_acc.append(acc)

    @property
    def history(self) -> List[float]:
        return self._windowed_acc.copy()

    @property
    def current(self) -> float:
        return self._windowed_acc[-1] if self._windowed_acc else 0.0


class RecoveryTimeAnalyser:
    """
    Measures how quickly the model recovers after each retraining event.

    Recovery is defined as the number of samples after retraining before
    windowed accuracy returns to a specified threshold (default: 90% of
    the pre-drift accuracy, or an absolute threshold like 0.85).

    Parameters
    ----------
    recovery_threshold : float — accuracy target to declare recovery (default 0.85).
    window_size        : int   — window for computing local accuracy (default 200).
    """

    def __init__(self, recovery_threshold: float = 0.85, window_size: int = 200):
        self.recovery_threshold = recovery_threshold
        self.window_size = window_size
        self._events = []

    def analyse(
        self,
        errors: List[int],
        sample_indices: List[int],
        retraining_indices: List[int],
    ) -> List[dict]:
        """
        Post-hoc analysis of recovery time from logged data.

        Parameters
        ----------
        errors             : list of 0/1 error values (aligned with sample_indices)
        sample_indices     : list of sample indices
        retraining_indices : list of sample indices where retraining occurred

        Returns
        -------
        List of dicts, one per retraining event:
            {retrain_at, recovery_at, recovery_samples, pre_drift_acc, post_recovery_acc}
        """
        idx_to_pos = {idx: pos for pos, idx in enumerate(sample_indices)}
        results = []

        for rt_idx in retraining_indices:
            if rt_idx not in idx_to_pos:
                continue
            pos = idx_to_pos[rt_idx]

            # Pre-drift accuracy (window ending at retrain point)
            pre_start = max(0, pos - self.window_size)
            pre_window = errors[pre_start:pos]
            pre_acc = 1.0 - (sum(pre_window) / len(pre_window)) if pre_window else 0.0

            # Scan forward to find recovery
            recovery_at = None
            recovery_samples = None
            post_recovery_acc = None

            for scan in range(pos + 1, min(pos + 5000, len(errors))):
                w_start = max(pos, scan - self.window_size)
                w = errors[w_start:scan + 1]
                if len(w) < min(50, self.window_size):
                    continue
                local_acc = 1.0 - (sum(w) / len(w))
                if local_acc >= self.recovery_threshold:
                    recovery_at = sample_indices[scan]
                    recovery_samples = scan - pos
                    post_recovery_acc = round(local_acc, 4)
                    break

            results.append({
                "retrain_at":         rt_idx,
                "pre_drift_acc":      round(pre_acc, 4),
                "recovery_at":        recovery_at,
                "recovery_samples":   recovery_samples,
                "post_recovery_acc":  post_recovery_acc,
                "recovered":          recovery_at is not None,
            })

        self._events = results
        return results

    def get_summary(self) -> dict:
        if not self._events:
            return {"events": 0}
        recovered = [e for e in self._events if e["recovered"]]
        avg_recovery = (
            round(np.mean([e["recovery_samples"] for e in recovered]), 1)
            if recovered else None
        )
        return {
            "events":            len(self._events),
            "recovered_count":   len(recovered),
            "avg_recovery_time": avg_recovery,
            "details":           self._events,
        }


def compute_evaluation_from_log(
    log_path: str,
    retraining_indices: List[int],
    window_size: int = 500,
    recovery_threshold: float = 0.85,
) -> dict:
    """
    Reads a stream log CSV and computes all evaluation metrics.

    Returns a dict with:
      - confusion_matrix: overall TP/TN/FP/FN + derived metrics
      - first_half_cm / second_half_cm: split at midpoint
      - windowed_accuracy: list of windowed accuracy values
      - recovery_analysis: per-retraining recovery details
      - prequential_accuracy: overall test-then-train accuracy
      - kappa: Cohen's Kappa

    Parameters
    ----------
    log_path            : path to stream_log CSV
    retraining_indices  : list of sample indices where retraining occurred
    window_size         : sliding window size for windowed accuracy
    recovery_threshold  : accuracy threshold for recovery detection
    """
    df = pd.read_csv(log_path)
    active = df[df["prediction"] != -1].copy()

    if active.empty:
        return {"error": "No active predictions found in log."}

    y_true = active["true_label"].values.astype(int)
    y_pred = active["prediction"].values.astype(int)
    errors = active["error"].values.astype(int)
    indices = active["sample_index"].values.astype(int)
    mid = len(df) // 2

    # 1. Overall confusion matrix
    cm = StreamingConfusionMatrix()
    for yt, yp in zip(y_true, y_pred):
        cm.update(yt, yp)

    # 2. First-half / second-half confusion matrices
    cm_first = StreamingConfusionMatrix()
    cm_second = StreamingConfusionMatrix()
    for idx, yt, yp in zip(indices, y_true, y_pred):
        if idx < mid:
            cm_first.update(yt, yp)
        else:
            cm_second.update(yt, yp)

    # 3. Windowed accuracy
    tracker = WindowedAccuracyTracker(window_size=window_size)
    for e in errors:
        tracker.update(e)

    # 4. Recovery time analysis
    recovery = RecoveryTimeAnalyser(
        recovery_threshold=recovery_threshold,
        window_size=min(200, window_size),
    )
    recovery_results = recovery.analyse(
        errors=errors.tolist(),
        sample_indices=indices.tolist(),
        retraining_indices=retraining_indices,
    )

    return {
        "confusion_matrix":      cm.to_dict(),
        "confusion_matrix_raw":  cm.matrix(),
        "first_half_cm":         cm_first.to_dict(),
        "second_half_cm":        cm_second.to_dict(),
        "windowed_accuracy":     tracker.history,
        "windowed_acc_indices":  indices.tolist(),
        "recovery_analysis":     recovery.get_summary(),
        "prequential_accuracy":  round(cm.accuracy, 4),
        "kappa":                 round(cm.kappa, 4),
        "class_report": {
            "class_UP": {
                "precision": round(cm.precision, 4),
                "recall":    round(cm.recall, 4),
                "f1":        round(cm.f1, 4),
                "support":   cm.tp + cm.fn,
            },
            "class_DOWN": {
                "precision": round(cm.specificity, 4),
                "recall":    round(cm.tn / (cm.tn + cm.fp) if (cm.tn + cm.fp) > 0 else 0, 4),
                "f1":        round(2 * cm.specificity * (cm.tn / (cm.tn + cm.fp) if (cm.tn + cm.fp) > 0 else 0) /
                                   (cm.specificity + (cm.tn / (cm.tn + cm.fp) if (cm.tn + cm.fp) > 0 else 0))
                                   if (cm.specificity + (cm.tn / (cm.tn + cm.fp) if (cm.tn + cm.fp) > 0 else 0)) > 0 else 0, 4),
                "support":   cm.tn + cm.fp,
            },
        },
    }
