"""
test_preprocessing.py
=====================
Automated unit and integration tests for src/preprocessing.py.

Verifies:
- Raw profile and sensor loading and dimension integrity
- Target encoding and stable_flag validity
- Stable cycle filtering (1,449 stable cycles, balanced classes)
- Cycle identifier retention without index loss
- Reproducible, leakage-safe train/val/test splitting (70% / 15% / 15%)
- Absence of cycle overlap between training, validation, and testing sets
- Strict fit-on-train feature scaling transformations
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    TOTAL_CYCLES,
    EXPECTED_SHAPES,
    load_profile,
    load_sensor,
    load_all_sensors,
    filter_stable_cycles,
    create_cycle_metadata,
    split_cycle_data,
    split_cycle_data_chronological,
    fit_transform_scaler,
    validate_raw_data,
)

RAW_DIR = Path("data/raw")


def test_load_profile():
    """Verify profile.txt loads correctly with all required columns and valid target codes."""
    profile = load_profile(RAW_DIR)
    assert profile.shape == (2205, 7)
    assert list(profile.columns) == [
        "cycle_id", "cooler", "valve", "pump_leakage", "accumulator", "stable_flag", "target_name"
    ]
    # Check cycle_id alignment
    assert (profile["cycle_id"].values == np.arange(2205)).all()

    # Target counts in raw dataset
    counts = profile["pump_leakage"].value_counts().to_dict()
    assert counts[0] == 1221  # Normal
    assert counts[1] == 492   # Warning
    assert counts[2] == 492   # Fault

    # Stability flag counts
    stable_counts = profile["stable_flag"].value_counts().to_dict()
    assert stable_counts[0] == 1449
    assert stable_counts[1] == 756


def test_load_sensor():
    """Verify sensor loading, shape checks, and finite numerical values."""
    # Test 10 Hz sensor
    fs1 = load_sensor(RAW_DIR, "FS1.txt")
    assert fs1.shape == (2205, 600)
    assert not fs1.isna().any().any()
    assert not np.isinf(fs1.to_numpy()).any()

    # Test 1 Hz sensor
    se = load_sensor(RAW_DIR, "SE.txt")
    assert se.shape == (2205, 60)
    assert not se.isna().any().any()
    assert not np.isinf(se.to_numpy()).any()


def test_filter_stable_cycles():
    """Verify filtering retains exactly 1,449 stable cycles with balanced target distribution."""
    profile = load_profile(RAW_DIR)
    stable_profile, _ = filter_stable_cycles(profile)

    assert len(stable_profile) == 1449
    assert (stable_profile["stable_flag"] == 0).all()

    # Verify target distribution on stable cycles
    counts = stable_profile["pump_leakage"].value_counts().to_dict()
    assert counts[0] == 489  # Normal
    assert counts[1] == 480  # Warning
    assert counts[2] == 480  # Fault

    # Verify original cycle_ids are preserved
    assert len(stable_profile["cycle_id"].unique()) == 1449
    assert stable_profile["cycle_id"].min() >= 0
    assert stable_profile["cycle_id"].max() < 2205


def test_create_cycle_metadata():
    """Verify metadata extraction isolates operational conditions from predictive features."""
    profile = load_profile(RAW_DIR)
    meta = create_cycle_metadata(profile)

    assert len(meta) == 2205
    assert "target" in meta.columns
    assert "is_stable" in meta.columns
    assert "cooler_desc" in meta.columns
    assert (meta["is_stable"] == (meta["stable_flag"] == 0)).all()


def test_split_cycle_data_proportions_and_disjointness():
    """Verify 70/15/15 train/val/test split on stable cycles with zero overlap and stratification."""
    profile = load_profile(RAW_DIR)
    stable_profile, _ = filter_stable_cycles(profile)
    meta = create_cycle_metadata(stable_profile)

    # Dummy feature matrix with 1,449 rows
    X = pd.DataFrame(
        np.random.randn(len(stable_profile), 10),
        index=stable_profile["cycle_id"],
        columns=[f"feat_{i}" for i in range(10)],
    )
    y = meta["target"]

    splits = split_cycle_data(
        X, y, metadata=meta,
        train_size=0.70, val_size=0.15, test_size=0.15,
        random_state=42, stratify=True,
    )

    X_train, X_val, X_test = splits["X_train"], splits["X_val"], splits["X_test"]
    y_train, y_val, y_test = splits["y_train"], splits["y_val"], splits["y_test"]

    # Verify counts
    assert len(X_train) == 1014
    assert len(X_val) == 217
    assert len(X_test) == 218
    assert len(X_train) + len(X_val) + len(X_test) == 1449

    # Verify zero overlap in cycle indices
    train_idx = set(splits["indices"]["train"])
    val_idx = set(splits["indices"]["val"])
    test_idx = set(splits["indices"]["test"])

    assert len(train_idx.intersection(val_idx)) == 0
    assert len(train_idx.intersection(test_idx)) == 0
    assert len(val_idx.intersection(test_idx)) == 0

    # Verify class stratification
    train_dist = y_train.value_counts(normalize=True).to_dict()
    val_dist = y_val.value_counts(normalize=True).to_dict()
    test_dist = y_test.value_counts(normalize=True).to_dict()

    for c in [0, 1, 2]:
        assert np.isclose(train_dist[c], 1/3, atol=0.02)
        assert np.isclose(val_dist[c], 1/3, atol=0.03)
        assert np.isclose(test_dist[c], 1/3, atol=0.03)


def test_fit_transform_scaler_no_leakage():
    """Verify that feature scaling fits strictly on training data."""
    np.random.seed(42)
    # Train has mean 10, Val has mean 20, Test has mean 30
    X_train = pd.DataFrame(np.random.normal(loc=10.0, scale=2.0, size=(100, 5)))
    X_val = pd.DataFrame(np.random.normal(loc=20.0, scale=2.0, size=(25, 5)))
    X_test = pd.DataFrame(np.random.normal(loc=30.0, scale=2.0, size=(25, 5)))

    X_train_s, X_val_s, X_test_s, scaler = fit_transform_scaler(X_train, X_val, X_test)

    # Scaler mean must match X_train mean
    assert np.allclose(scaler.mean_, X_train.mean().values, atol=1e-5)
    # X_train scaled must have mean ~0 and std ~1
    assert np.allclose(X_train_s.mean().values, 0.0, atol=1e-5)
    assert np.allclose(X_train_s.std(ddof=0).values, 1.0, atol=1e-5)
    # X_val scaled must NOT have mean 0 (since it had different distribution)
    assert not np.allclose(X_val_s.mean().values, 0.0, atol=1.0)

