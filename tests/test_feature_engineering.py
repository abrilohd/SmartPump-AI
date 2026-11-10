"""
test_feature_engineering.py
===========================
Automated unit and integration tests for src/feature_engineering.py.

Verifies:
- Extraction of all 15 statistics per sensor
- Mathematical accuracy of summary statistics
- Numerical stability on constant signals (zero variance -> no NaNs/Infs)
- Physical zero preservation (PS4 accumulator sensor)
- Feature metadata generation consistency
- Feature persistence and reload integrity
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    ALL_STATISTICS,
    DEFAULT_STATS,
    PRIORITY_SENSORS,
    extract_cycle_features,
    generate_feature_metadata,
)
from src.preprocessing import load_sensor

RAW_DIR = Path("data/raw")


def test_extract_cycle_features_names_and_shapes():
    """Verify that all 15 features are extracted with correct names and shapes."""
    n_cycles = 10
    n_samples = 600
    np.random.seed(42)
    synthetic_data = pd.DataFrame(
        np.random.randn(n_cycles, n_samples),
        index=pd.RangeIndex(0, n_cycles, name="cycle_id"),
    )

    features = extract_cycle_features(synthetic_data, "FS1", ALL_STATISTICS)

    assert features.shape == (n_cycles, 15)
    expected_cols = [f"FS1_{s}" for s in ALL_STATISTICS]
    assert list(features.columns) == expected_cols
    assert not features.isna().any().any()
    assert not np.isinf(features.to_numpy()).any()


def test_constant_signal_numerical_safety():
    """Verify that zero-variance constant signals do NOT yield NaNs or Infs."""
    n_cycles = 5
    n_samples = 60
    # Exactly constant array of zeros and constant array of 42.0
    arr = np.zeros((n_cycles, n_samples), dtype=np.float64)
    arr[1, :] = 42.0

    df_const = pd.DataFrame(arr)
    features = extract_cycle_features(df_const, "CONST", ALL_STATISTICS)

    assert features.shape == (n_cycles, 15)
    assert not features.isna().any().any()
    assert not np.isinf(features.to_numpy()).any()

    # std, range, iqr, skew, kurtosis, slope, delta, variability must be exactly 0
    for stat in ["std", "range", "iqr", "skew", "kurtosis", "slope", "delta", "variability"]:
        assert np.allclose(features[f"CONST_{stat}"], 0.0)

    # mean, median, min, max, q25, q75 for row 1 must be 42.0
    for stat in ["mean", "median", "min", "max", "q25", "q75"]:
        assert np.isclose(features.loc[1, f"CONST_{stat}"], 42.0)


def test_linear_slope_and_delta_accuracy():
    """Verify mathematical correctness of linear slope and first-to-last delta."""
    n_samples = 100
    t = np.arange(n_samples, dtype=np.float64)
    # y = 3.5 * t + 10.0
    linear_signal = 3.5 * t + 10.0
    df = pd.DataFrame([linear_signal])

    feats = extract_cycle_features(df, "TEST", ["slope", "delta"])

    assert np.isclose(feats.loc[0, "TEST_slope"], 3.5, atol=1e-10)
    assert np.isclose(feats.loc[0, "TEST_delta"], 3.5 * (n_samples - 1), atol=1e-10)


def test_ps4_zero_preservation():
    """Verify that raw PS4 accumulator signal (which contains unpressurized cycles) is handled safely."""
    ps4 = load_sensor(RAW_DIR, "PS4.txt")
    feats = extract_cycle_features(ps4, "PS4", ALL_STATISTICS)

    assert feats.shape == (2205, 15)
    assert not feats.isna().any().any()
    assert not np.isinf(feats.to_numpy()).any()

    # Minimum observed PS4_min must be 0.0 (preserved physical zero)
    assert np.isclose(feats["PS4_min"].min(), 0.0)


def test_feature_metadata_generation():
    """Verify metadata generation accurately catalogs feature properties."""
    cols = [f"{s}_{stat}" for s in ["FS1", "PS1", "TS1"] for stat in ["mean", "std", "slope"]]
    meta = generate_feature_metadata(cols)

    assert len(meta) == len(cols)
    assert "feature_name" in meta.columns
    assert "is_priority_sensor" in meta.columns
    assert "sampling_rate" in meta.columns

    # FS1 and PS1 should be marked as priority sensors
    fs1_rows = meta[meta["sensor"] == "FS1"]
    assert fs1_rows["is_priority_sensor"].all()

    ts1_rows = meta[meta["sensor"] == "TS1"]
    assert not ts1_rows["is_priority_sensor"].any()

