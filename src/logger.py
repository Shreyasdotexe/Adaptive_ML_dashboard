import csv, os, numpy as np
from datetime import datetime
from typing import Optional

class StreamLogger:
    def __init__(self, log_dir, n_features=4, run_id=None):
        self.log_dir = log_dir
        self.n_features = n_features
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"stream_log_{self.run_id}.csv")
        self._file = self._writer = None
        self._rows_written = 0

    def open(self):
        self._file = open(self.log_path, mode="w", newline="")
        self._writer = csv.writer(self._file)
        feature_cols = [f"feature_{i}" for i in range(self.n_features)]
        self._writer.writerow(
            ["sample_index"] + feature_cols +
            ["true_label", "detector_input", "drift_detected", "phase",
             "prediction", "error", "running_acc", "retrained"]
        )
        self._file.flush()

    def log(self, index, features, label, detector_input, drift_detected,
            phase, prediction=-1, error=-1, running_acc=0.0, retrained=0):
        if self._writer is None:
            raise RuntimeError("Call logger.open() first.")
        row = ([index] + features.tolist() + [
            label,
            ("" if detector_input is None else round(detector_input, 6)),
            int(drift_detected), phase,
            prediction, error, round(running_acc, 6), retrained,
        ])
        self._writer.writerow(row)
        self._rows_written += 1
        if self._rows_written % 200 == 0:
            self._file.flush()

    def close(self):
        if self._file:
            self._file.flush()
            self._file.close()
        return self.log_path

    def get_path(self): return self.log_path
    def get_rows_written(self): return self._rows_written


class RunReporter:
    def __init__(self, report_dir):
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)

    def save(self, content, run_id):
        path = os.path.join(self.report_dir, f"run_report_{run_id}.txt")
        with open(path, "w") as f:
            f.write(content)
        return path
