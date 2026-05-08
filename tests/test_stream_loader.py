"""
test_stream_loader.py
---------------------
Unit tests for the RealStreamLoader.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from src.real_stream_loader import RealStreamLoader


# Path to the actual dataset (used for integration-style tests)
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "electricity.csv")
DATASET_EXISTS = os.path.exists(DATASET_PATH)


class TestRealStreamLoader:
    """Tests for the ELEC2 dataset stream loader."""

    @pytest.mark.skipif(not DATASET_EXISTS, reason="electricity.csv not available")
    def test_load_full_dataset(self):
        """Loader should successfully load the full dataset."""
        loader = RealStreamLoader(csv_path=DATASET_PATH)
        config = loader.get_config()
        assert config["n_samples"] == 45312
        assert config["n_features"] == 8

    @pytest.mark.skipif(not DATASET_EXISTS, reason="electricity.csv not available")
    def test_max_samples_limits_stream(self):
        """max_samples should limit the number of streamed samples."""
        loader = RealStreamLoader(csv_path=DATASET_PATH, max_samples=100)
        count = sum(1 for _ in loader.stream())
        assert count == 100

    @pytest.mark.skipif(not DATASET_EXISTS, reason="electricity.csv not available")
    def test_stream_yields_correct_format(self):
        """Each yielded tuple should have (int, ndarray, int, str)."""
        loader = RealStreamLoader(csv_path=DATASET_PATH, max_samples=5)
        for index, features, label, phase in loader.stream():
            assert isinstance(index, (int, np.integer))
            assert isinstance(features, np.ndarray)
            assert features.shape == (8,)
            assert label in [0, 1]
            assert phase == "real_stream"

    @pytest.mark.skipif(not DATASET_EXISTS, reason="electricity.csv not available")
    def test_labels_are_binary(self):
        """All labels should be 0 or 1."""
        loader = RealStreamLoader(csv_path=DATASET_PATH, max_samples=500)
        for _, _, label, _ in loader.stream():
            assert label in [0, 1]

    @pytest.mark.skipif(not DATASET_EXISTS, reason="electricity.csv not available")
    def test_features_are_numeric(self):
        """All features should be finite numbers."""
        loader = RealStreamLoader(csv_path=DATASET_PATH, max_samples=100)
        for _, features, _, _ in loader.stream():
            assert np.all(np.isfinite(features))

    @pytest.mark.skipif(not DATASET_EXISTS, reason="electricity.csv not available")
    def test_get_config_structure(self):
        """get_config() should return expected keys."""
        loader = RealStreamLoader(csv_path=DATASET_PATH, max_samples=10)
        config = loader.get_config()
        assert "source" in config
        assert "n_samples" in config
        assert "n_features" in config
        assert "features" in config
        assert "label_col" in config
        assert "label_map" in config

    def test_missing_file_raises_error(self):
        """Loader should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            RealStreamLoader(csv_path="nonexistent_file.csv")
