"""
uci_loader.py
=============
Module for interacting with the UCI Machine Learning Repository via the
official `ucimlrepo` Python library.

Provides verification, inspection, and diagnostics for:
Dataset: Condition Monitoring of Hydraulic Systems (UCI ID: 447)
URL: https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

UCI_DATASET_ID = 447
UCI_DATASET_NAME = "Condition Monitoring of Hydraulic Systems"
UCI_OFFICIAL_URL = "https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems"

EXPECTED_RAW_FILES = [
    "PS1.txt", "PS2.txt", "PS3.txt", "PS4.txt", "PS5.txt", "PS6.txt",
    "EPS1.txt",
    "FS1.txt", "FS2.txt",
    "TS1.txt", "TS2.txt", "TS3.txt", "TS4.txt",
    "VS1.txt",
    "CE.txt", "CP.txt", "SE.txt",
    "profile.txt",
]


def fetch_hydraulic_dataset(dataset_id: int = UCI_DATASET_ID) -> Dict[str, Any]:
    """
    Attempt to programmatically fetch the dataset from the UCI ML Repository
    using the official `ucimlrepo` library.

    Parameters
    ----------
    dataset_id : int, default=447
        Official UCI repository identifier.

    Returns
    -------
    dict
        Structured diagnostic report containing:
        - 'status': 'success', 'unavailable_for_import', 'error', or 'invalid_id'
        - 'dataset_id': int
        - 'dataset_name': str
        - 'features': DataFrame or None
        - 'targets': DataFrame or None
        - 'metadata': dict
        - 'variables': DataFrame or None
        - 'message': str explanation of outcome
    """
    if dataset_id != UCI_DATASET_ID:
        return {
            "status": "invalid_id",
            "dataset_id": dataset_id,
            "dataset_name": None,
            "features": None,
            "targets": None,
            "metadata": {},
            "variables": None,
            "message": f"Dataset ID {dataset_id} does not match expected SmartPump AI ID ({UCI_DATASET_ID}).",
        }

    try:
        from ucimlrepo import fetch_ucirepo
        from ucimlrepo.fetch import DatasetNotFoundError
    except ImportError as exc:
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "dataset_name": UCI_DATASET_NAME,
            "features": None,
            "targets": None,
            "metadata": {},
            "variables": None,
            "message": f"The 'ucimlrepo' package is not installed: {exc}. Install via 'pip install ucimlrepo'.",
        }

    try:
        repo_data = fetch_ucirepo(id=dataset_id)
        
        # Successfully retrieved programmatic record
        features = getattr(repo_data.data, "features", None)
        targets = getattr(repo_data.data, "targets", None)
        metadata = getattr(repo_data, "metadata", {}) or {}
        variables = getattr(repo_data, "variables", None)

        return {
            "status": "success",
            "dataset_id": dataset_id,
            "dataset_name": metadata.get("name", UCI_DATASET_NAME),
            "features": features,
            "targets": targets,
            "metadata": metadata,
            "variables": variables,
            "message": "Dataset successfully retrieved from UCI Machine Learning Repository.",
        }

    except DatasetNotFoundError as e:
        # UCI Repository API maintains ID 447, but multi-file time-series datasets
        # (>500MB raw tab-delimited matrices) are not provided as single tabular API streams.
        return {
            "status": "unavailable_for_import",
            "dataset_id": dataset_id,
            "dataset_name": UCI_DATASET_NAME,
            "features": None,
            "targets": None,
            "metadata": {"uci_id": dataset_id, "name": UCI_DATASET_NAME, "url": UCI_OFFICIAL_URL},
            "variables": None,
            "message": (
                f"UCI Dataset 447 ('{UCI_DATASET_NAME}') exists in the repository catalog, "
                "but the official UCI web API flags it as unavailable for direct programmatic "
                "tabular import. This is standard for complex multi-rate time-series datasets "
                "(43,680 measurements across 18 raw files). The canonical 18-file archive must "
                f"be downloaded directly from {UCI_OFFICIAL_URL} into data/raw/."
            ),
        }
    except Exception as e:
        # Connection timeouts, DNS failures, or remote HTTP errors
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "dataset_name": UCI_DATASET_NAME,
            "features": None,
            "targets": None,
            "metadata": {},
            "variables": None,
            "message": f"Network or protocol error connecting to UCI repository: {type(e).__name__} - {e}",
        }


def verify_canonical_raw_files(raw_dir: Path | str = "data/raw") -> Dict[str, Any]:
    """
    Verify the presence of the 18 canonical raw sensor and condition profile files.

    Parameters
    ----------
    raw_dir : Path or str, default="data/raw"
        Directory containing raw dataset files.

    Returns
    -------
    dict
        Verification summary with file counts and status.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return {
            "raw_dir": str(raw_path),
            "all_present": False,
            "found_count": 0,
            "missing_files": EXPECTED_RAW_FILES,
        }

    found = []
    missing = []
    for fname in EXPECTED_RAW_FILES:
        fpath = raw_path / fname
        if fpath.exists() and fpath.stat().st_size > 0:
            found.append(fname)
        else:
            missing.append(fname)

    return {
        "raw_dir": str(raw_path),
        "all_present": len(missing) == 0,
        "found_count": len(found),
        "missing_files": missing,
    }


def main() -> None:
    """CLI diagnostics entry point."""
    print("=" * 70)
    print("SmartPump AI — UCI Dataset Import & Verification Utility")
    print(f"Target: {UCI_DATASET_NAME} (UCI ID: {UCI_DATASET_ID})")
    print(f"URL:    {UCI_OFFICIAL_URL}")
    print("=" * 70)

    # 1. Attempt programmatic ucimlrepo fetch
    print("\n[Step 1] Attempting fetch via official 'ucimlrepo' library...")
    result = fetch_hydraulic_dataset(UCI_DATASET_ID)
    print(f"Status:  {result['status']}")
    print(f"Message: {result['message']}")

    if result["status"] == "success":
        print(f"Features Shape: {result['features'].shape if result['features'] is not None else 'N/A'}")
        print(f"Targets Shape:  {result['targets'].shape if result['targets'] is not None else 'N/A'}")

    # 2. Check canonical local raw files
    print("\n[Step 2] Checking canonical raw files in data/raw/...")
    raw_summary = verify_canonical_raw_files("data/raw")
    print(f"Found {raw_summary['found_count']}/18 canonical files.")
    if raw_summary["all_present"]:
        print("✓ All 18 canonical raw sensor and profile files are present and verified.")
    else:
        print(f"⚠ Missing {len(raw_summary['missing_files'])} files: {raw_summary['missing_files']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
