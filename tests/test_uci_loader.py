"""
test_uci_loader.py
==================
Unit tests for UCI dataset import and verification module.
Uses unittest.mock to test programmatic fetching, error handling,
metadata validation, and canonical raw file verification without
relying on active network availability.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from src.uci_loader import (
    EXPECTED_RAW_FILES,
    UCI_DATASET_ID,
    UCI_DATASET_NAME,
    fetch_hydraulic_dataset,
    verify_canonical_raw_files,
)


def test_invalid_dataset_id():
    """Verify that requesting an incorrect UCI ID returns an invalid_id status."""
    res = fetch_hydraulic_dataset(dataset_id=9999)
    assert res["status"] == "invalid_id"
    assert res["dataset_id"] == 9999
    assert res["features"] is None
    assert "does not match expected SmartPump AI ID" in res["message"]


def test_successful_fetch_simulation():
    """Verify handling when ucimlrepo successfully returns features, targets, and metadata."""
    mock_features = pd.DataFrame({"PS1_mean": [160.0, 161.0], "FS1_mean": [50.0, 48.0]})
    mock_targets = pd.DataFrame({"pump_leakage": [0, 1]})
    mock_metadata = {"uci_id": 447, "name": UCI_DATASET_NAME, "num_instances": 2205}
    mock_variables = pd.DataFrame({"name": ["PS1_mean", "pump_leakage"], "role": ["Feature", "Target"]})

    mock_repo = MagicMock()
    mock_repo.data.features = mock_features
    mock_repo.data.targets = mock_targets
    mock_repo.metadata = mock_metadata
    mock_repo.variables = mock_variables

    with patch("ucimlrepo.fetch_ucirepo", return_value=mock_repo):
        res = fetch_hydraulic_dataset(dataset_id=UCI_DATASET_ID)
        assert res["status"] == "success"
        assert res["dataset_id"] == 447
        assert res["dataset_name"] == UCI_DATASET_NAME
        assert res["features"].shape == (2, 2)
        assert res["targets"].shape == (2, 1)
        assert res["metadata"]["num_instances"] == 2205


def test_unavailable_for_import_handling():
    """
    Verify graceful handling when UCI API raises DatasetNotFoundError (actual behavior for ID 447).
    """
    from ucimlrepo.fetch import DatasetNotFoundError

    with patch("ucimlrepo.fetch_ucirepo", side_effect=DatasetNotFoundError("Dataset not available")):
        res = fetch_hydraulic_dataset(dataset_id=UCI_DATASET_ID)
        assert res["status"] == "unavailable_for_import"
        assert res["dataset_id"] == 447
        assert "unavailable for direct programmatic tabular import" in res["message"]
        assert res["features"] is None
        assert res["targets"] is None


def test_network_connection_error_handling():
    """Verify graceful handling when remote network or DNS failure occurs."""
    with patch("ucimlrepo.fetch_ucirepo", side_effect=ConnectionError("Failed to resolve host")):
        res = fetch_hydraulic_dataset(dataset_id=UCI_DATASET_ID)
        assert res["status"] == "error"
        assert "Network or protocol error" in res["message"]
        assert res["features"] is None


def test_verify_canonical_raw_files_present():
    """Verify that canonical raw file checker reports all 18 files present."""
    summary = verify_canonical_raw_files("data/raw")
    # In the project workspace, data/raw contains all 18 files
    assert summary["all_present"] is True
    assert summary["found_count"] == 18
    assert len(summary["missing_files"]) == 0


def test_verify_canonical_raw_files_missing(tmp_path):
    """Verify that missing files are identified correctly when directory is incomplete."""
    fake_raw = tmp_path / "raw"
    fake_raw.mkdir()
    # Create only 1 file
    (fake_raw / "PS1.txt").write_text("dummy")

    summary = verify_canonical_raw_files(fake_raw)
    assert summary["all_present"] is False
    assert summary["found_count"] == 1
    assert len(summary["missing_files"]) == len(EXPECTED_RAW_FILES) - 1
    assert "profile.txt" in summary["missing_files"]
