"""
real_stream_loader.py
---------------------
Real dataset stream loader for the Electricity Pricing dataset
(ELEC2 / electricity-normalized).

Replaces StreamSimulator .  The interface is identical
to StreamSimulator.stream() so the rest of the pipeline (preprocessing,
model, ADWIN, logger, visualizer) requires zero changes.

Dataset: electricity-normalized.arff / electricity.csv
Source:  Harries, M. (1999). SPLICE-2 Comparative Evaluation: Electricity
         Pricing. Technical Report, University of New South Wales.

Columns used as features (all already normalised 0-1 in the dataset):
    date, day, period, nswprice, nswdemand, vicprice, vicdemand, transfer

Target column:
    class  →  UP = 1,  DOWN = 0

Why ELEC2 is appropriate for this project:
  - It is the standard benchmark for concept drift research.
  - Real concept drift occurs naturally (electricity pricing patterns
    shift with seasons, policy changes, demand growth).
  - No artificial drift injection is needed — ADWIN will detect real drift.
  - Widely cited in concept drift literature (Bifet, Gama, etc.)

Output format (matches StreamSimulator exactly):
    (index, features_array, label, phase)
    phase is always "real_stream" — no artificial pre/post split.
"""

import os
import numpy as np
import pandas as pd
from typing import Generator, Tuple


# Feature columns to extract (order preserved for downstream logging)
FEATURE_COLS = ["date", "day", "period", "nswprice",
                "nswdemand", "vicprice", "vicdemand", "transfer"]
LABEL_COL    = "class"
LABEL_MAP    = {"UP": 1, "DOWN": 0}


class RealStreamLoader:
    """
    Loads the Electricity dataset from CSV and yields samples one at a time,
    simulating a data stream.

    The interface is a drop-in replacement for StreamSimulator:
        for index, features, label, phase in loader.stream():
            ...

    Parameters
    ----------
    csv_path   : Path to electricity.csv (converted from the ARFF file).
    max_samples: Maximum number of samples to stream.  Set to None to use
                 the full dataset.  Useful for quick test runs.
    """

    def __init__(
        self,
        csv_path:    str = os.path.join("data", "electricity.csv"),
        max_samples: int = None,
    ):
        self.csv_path    = csv_path
        self.max_samples = max_samples

        # Load and validate on construction — fail early with a clear message
        self._df = self._load(csv_path)
        if max_samples is not None:
            self._df = self._df.iloc[:max_samples].reset_index(drop=True)

        self.n_samples  = len(self._df)
        self.n_features = len(FEATURE_COLS)

    # ── Private ─────────────────────────────────────────────────────────────

    def _load(self, path: str) -> pd.DataFrame:
        """
        Reads the CSV, validates required columns, maps class labels to 0/1.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dataset not found at: {path}\n"
                f"Convert the ARFF file to CSV and place it at that path.\n"
                f"Expected columns: {FEATURE_COLS + [LABEL_COL]}"
            )

        df = pd.read_csv(path)

        # Validate columns
        missing = [c for c in FEATURE_COLS + [LABEL_COL] if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing expected columns: {missing}")

        # Map class labels
        df[LABEL_COL] = df[LABEL_COL].map(LABEL_MAP)

        unmapped = df[LABEL_COL].isna().sum()
        if unmapped > 0:
            raise ValueError(
                f"{unmapped} rows have unrecognised class values. "
                f"Expected 'UP' or 'DOWN'."
            )

        df[LABEL_COL] = df[LABEL_COL].astype(int)
        return df.reset_index(drop=True)

    # ── Public API ───────────────────────────────────────────────────────────

    def stream(self) -> Generator[Tuple[int, np.ndarray, int, str], None, None]:
        """
        Yields one sample at a time in dataset order.

        Yields
        ------
        index    : int        — row index (0-based)
        features : np.ndarray — shape (n_features,), float64
        label    : int        — 0 (DOWN) or 1 (UP)
        phase    : str        — always "real_stream"
        """
        for i, row in self._df.iterrows():
            features = row[FEATURE_COLS].to_numpy(dtype=np.float64)
            label    = int(row[LABEL_COL])
            yield i, features, label, "real_stream"

    def get_config(self) -> dict:
        """Returns loader configuration for the run report."""
        return {
            "source":     self.csv_path,
            "n_samples":  self.n_samples,
            "n_features": self.n_features,
            "features":   FEATURE_COLS,
            "label_col":  LABEL_COL,
            "label_map":  LABEL_MAP,
        }
