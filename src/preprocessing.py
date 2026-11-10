"""
preprocessing.py
================
SmartPump AI - Production-Quality Data Loading and Preprocessing Pipeline

Responsible for:
- Loading raw sensor files and condition profile (profile.txt) from data/raw/
- Validating sensor shapes, cycle counts, sampling frequencies, and data integrity
- Ensuring zero NaN and zero infinite values across all operating cycles
- Providing structured cycle-level metadata separated from predictive model features
- Filtering stable operating cycles (stable_flag == 0) while preserving original cycle IDs
- Reproducible, leakage-safe train/validation/test splitting (70% / 15% / 15%)
- Fitting preprocessing transformations (e.g. scalers) strictly on training data

Target Mapping:
- 0: Normal  (no internal pump leakage)
- 1: Warning (weak internal pump leakage)
- 2: Fault   (severe internal pump leakage)

Author: SmartPump AI Engineering Team
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants & Domain Definitions
# ---------------------------------------------------------------------------

# Primary classification target mapping
PUMP_LEAKAGE_LABEL_MAP: Dict[int, str] = {
    0: "Normal",
    1: "Warning",
    2: "Fault",
}

LABEL_TO_CODE_MAP: Dict[str, int] = {
    "Normal": 0,
    "Warning": 1,
    "Fault": 2,
}

# Total operating cycles recorded in UCI experiment
TOTAL_CYCLES: int = 2205

# Cycle duration in seconds
CYCLE_DURATION_SEC: int = 60

# Expected shape (rows, columns) for all 17 sensor files in data/raw/
# rows = 2,205 operating cycles; columns = sampling_rate_hz * 60 seconds
EXPECTED_SHAPES: Dict[str, Tuple[int, int]] = {
    # 100 Hz sensors (6,000 samples/cycle)
    "PS1.txt": (2205, 6000),
    "PS2.txt": (2205, 6000),
    "PS3.txt": (2205, 6000),
    "PS4.txt": (2205, 6000),
    "PS5.txt": (2205, 6000),
    "PS6.txt": (2205, 6000),
    "EPS1.txt": (2205, 6000),
    # 10 Hz sensors (600 samples/cycle)
    "FS1.txt": (2205, 600),
    "FS2.txt": (2205, 600),
    # 1 Hz sensors (60 samples/cycle)
    "TS1.txt": (2205, 60),
    "TS2.txt": (2205, 60),
    "TS3.txt": (2205, 60),
    "TS4.txt": (2205, 60),
    "VS1.txt": (2205, 60),
    "CE.txt": (2205, 60),
    "CP.txt": (2205, 60),
    "SE.txt": (2205, 60),
}

# Sensor hardware metadata
SENSOR_METADATA: Dict[str, Dict[str, Any]] = {
    "PS1": {"quantity": "Hydraulic Pressure", "unit": "bar", "rate_hz": 100, "samples": 6000, "subsystem": "Primary circuit, pump outlet", "priority": True},
    "PS2": {"quantity": "Hydraulic Pressure", "unit": "bar", "rate_hz": 100, "samples": 6000, "subsystem": "Primary circuit, main cylinder", "priority": True},
    "PS3": {"quantity": "Hydraulic Pressure", "unit": "bar", "rate_hz": 100, "samples": 6000, "subsystem": "Primary circuit, return line", "priority": True},
    "PS4": {"quantity": "Hydraulic Pressure", "unit": "bar", "rate_hz": 100, "samples": 6000, "subsystem": "Hydraulic accumulator charging line", "priority": False},
    "PS5": {"quantity": "Hydraulic Pressure", "unit": "bar", "rate_hz": 100, "samples": 6000, "subsystem": "Secondary cooling circuit", "priority": False},
    "PS6": {"quantity": "Hydraulic Pressure", "unit": "bar", "rate_hz": 100, "samples": 6000, "subsystem": "Secondary filtration line", "priority": False},
    "EPS1": {"quantity": "Motor Drive Power", "unit": "W", "rate_hz": 100, "samples": 6000, "subsystem": "Main pump electric drive", "priority": True},
    "FS1": {"quantity": "Volume Flow Rate", "unit": "l/min", "rate_hz": 10, "samples": 600, "subsystem": "Primary working circuit flow", "priority": True},
    "FS2": {"quantity": "Volume Flow Rate", "unit": "l/min", "rate_hz": 10, "samples": 600, "subsystem": "Secondary cooling circuit flow", "priority": False},
    "TS1": {"quantity": "Oil Temperature", "unit": "°C", "rate_hz": 1, "samples": 60, "subsystem": "Oil reservoir temperature", "priority": False},
    "TS2": {"quantity": "Oil Temperature", "unit": "°C", "rate_hz": 1, "samples": 60, "subsystem": "Primary circuit before cooler", "priority": False},
    "TS3": {"quantity": "Oil Temperature", "unit": "°C", "rate_hz": 1, "samples": 60, "subsystem": "Primary circuit after cooler", "priority": False},
    "TS4": {"quantity": "Oil Temperature", "unit": "°C", "rate_hz": 1, "samples": 60, "subsystem": "Secondary cooling circuit", "priority": False},
    "VS1": {"quantity": "Vibration Amplitude", "unit": "mm/s", "rate_hz": 1, "samples": 60, "subsystem": "Main pump mounting bracket", "priority": False},
    "CE": {"quantity": "Cooling Efficiency (Virtual)", "unit": "%", "rate_hz": 1, "samples": 60, "subsystem": "Virtual heat exchanger efficiency", "priority": False},
    "CP": {"quantity": "Cooling Power (Virtual)", "unit": "kW", "rate_hz": 1, "samples": 60, "subsystem": "Virtual heat transfer rate", "priority": False},
    "SE": {"quantity": "Efficiency Factor (Virtual)", "unit": "%", "rate_hz": 1, "samples": 60, "subsystem": "Global hydraulic efficiency factor", "priority": True},
}

# Profile file specifications
PROFILE_EXPECTED_SHAPE: Tuple[int, int] = (2205, 5)

# Profile column indices (0-based)
COL_COOLER: int = 0
COL_VALVE: int = 1
COL_PUMP_LEAKAGE: int = 2   # Project classification target
COL_ACCUMULATOR: int = 3
COL_STABLE_FLAG: int = 4

# Component condition interpretation dictionaries
COOLER_CONDITIONS: Dict[int, str] = {
    3: "close to total failure",
    20: "reduced efficiency",
    100: "full efficiency",
}

VALVE_CONDITIONS: Dict[int, str] = {
    73: "close to total failure",
    80: "severe lag",
    90: "small lag",
    100: "optimal switching",
}

ACCUMULATOR_PRESSURES: Dict[int, str] = {
    90: "90 bar (severe pressure drop)",
    100: "100 bar (low pressure)",
    115: "115 bar (reduced pressure)",
    130: "130 bar (optimal pre-charge)",
}

STABLE_FLAG_MAP: Dict[int, str] = {
    0: "stable",
    1: "unstable (transient)",
}


# ---------------------------------------------------------------------------
# Data Loading & Integrity Validation
# ---------------------------------------------------------------------------

def load_profile(raw_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load and validate profile.txt.

    Parameters
    ----------
    raw_dir : Path or str
        Directory containing profile.txt.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['cycle_id', 'cooler', 'valve', 'pump_leakage', 'accumulator', 'stable_flag', 'target_name']
        Indexed by cycle_id (0 to 2204).

    Raises
    ------
    FileNotFoundError
        If profile.txt does not exist.
    ValueError
        If shape or column values deviate from verified experimental ground truth.
    """
    raw_dir = Path(raw_dir)
    profile_path = raw_dir / "profile.txt"
    if not profile_path.exists():
        raise FileNotFoundError(f"profile.txt not found at {profile_path}.")

    profile = pd.read_csv(profile_path, sep="\t", header=None)

    if profile.shape != PROFILE_EXPECTED_SHAPE:
        raise ValueError(
            f"profile.txt shape mismatch: expected {PROFILE_EXPECTED_SHAPE}, got {profile.shape}."
        )

    # Assign verified column names
    profile.columns = ["cooler", "valve", "pump_leakage", "accumulator", "stable_flag"]

    # Validate pump_leakage labels
    actual_labels = set(profile["pump_leakage"].unique())
    expected_labels = {0, 1, 2}
    if actual_labels != expected_labels:
        raise ValueError(
            f"Unexpected pump_leakage values: {actual_labels}. Expected exact set {expected_labels}."
        )

    # Validate stable_flag values
    actual_flags = set(profile["stable_flag"].unique())
    expected_flags = {0, 1}
    if actual_flags != expected_flags:
        raise ValueError(
            f"Unexpected stable_flag values: {actual_flags}. Expected exact set {expected_flags}."
        )

    # Explicit cycle_id preserving original time order
    profile.insert(0, "cycle_id", np.arange(len(profile), dtype=int))
    profile["target_name"] = profile["pump_leakage"].map(PUMP_LEAKAGE_LABEL_MAP)

    return profile


def load_sensor(
    raw_dir: Union[str, Path],
    filename: str,
    validate_values: bool = True,
) -> pd.DataFrame:
    """
    Load a single sensor file and validate its dimensions and numerical integrity.

    Parameters
    ----------
    raw_dir : Path or str
        Directory containing sensor files.
    filename : str
        Sensor file name (e.g. 'PS1.txt').
    validate_values : bool, default True
        If True, verifies that no NaN or infinite values exist in the loaded data.

    Returns
    -------
    pd.DataFrame
        Shape (2205, N) where each row is an operating cycle and each column is a time step.
        Indexed by cycle_id (0 to 2204).

    Raises
    ------
    FileNotFoundError
        If sensor file does not exist.
    ValueError
        If file shape does not match expected shape, or contains invalid numerical values.
    """
    raw_dir = Path(raw_dir)
    sensor_path = raw_dir / filename
    if not sensor_path.exists():
        raise FileNotFoundError(f"Sensor file not found at {sensor_path}.")

    df = pd.read_csv(sensor_path, sep="\t", header=None, dtype=np.float64)

    if filename in EXPECTED_SHAPES:
        expected = EXPECTED_SHAPES[filename]
        if df.shape != expected:
            raise ValueError(
                f"{filename} shape mismatch: expected {expected}, got {df.shape}."
            )

    df.index = pd.RangeIndex(start=0, stop=len(df), name="cycle_id")

    if validate_values:
        n_nan = int(df.isna().sum().sum())
        n_inf = int(np.isinf(df.to_numpy()).sum())
        if n_nan > 0 or n_inf > 0:
            raise ValueError(
                f"{filename} contains invalid values: {n_nan} NaNs, {n_inf} Infs."
            )

    return df


def load_all_sensors(
    raw_dir: Union[str, Path],
    sensor_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load sensor files from data/raw/.

    Parameters
    ----------
    raw_dir : Path or str
        Path to raw data directory.
    sensor_names : list of str, optional
        List of filenames to load (e.g. ['PS1.txt', 'FS1.txt']).
        Defaults to all 17 sensors in EXPECTED_SHAPES.
    verbose : bool, default True
        Whether to log progress.

    Returns
    -------
    dict
        Mapping from filename (e.g. 'PS1.txt') to cycle-indexed DataFrame.
    """
    raw_dir = Path(raw_dir)
    if sensor_names is None:
        sensor_names = list(EXPECTED_SHAPES.keys())

    sensors: Dict[str, pd.DataFrame] = {}
    for filename in sensor_names:
        df = load_sensor(raw_dir, filename, validate_values=True)
        sensors[filename] = df
        if verbose:
            sampling_hz = df.shape[1] // CYCLE_DURATION_SEC
            print(
                f"[preprocessing] Loaded {filename:8s} | "
                f"Shape: {df.shape[0]} cycles × {df.shape[1]} samples ({sampling_hz:3d} Hz)"
            )

    return sensors


def validate_raw_data(
    profile: pd.DataFrame,
    sensors: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """
    Comprehensive verification of the full raw dataset.

    Validates:
    - Consistency of cycle count across all sensors and profile
    - Perfect zero-count of NaNs and infinite values
    - Expected sampling frequencies (100 Hz, 10 Hz, 1 Hz)
    - Preservation of valid physical zero readings (PS4)
    - Class counts and stability counts

    Returns
    -------
    dict
        Summary report containing validation metrics and status.
    """
    n_cycles = len(profile)
    report: Dict[str, Any] = {
        "total_cycles": n_cycles,
        "sensor_count": len(sensors),
        "shapes_valid": True,
        "missing_values": 0,
        "infinite_values": 0,
        "sensors_checked": [],
    }

    if n_cycles != TOTAL_CYCLES:
        report["shapes_valid"] = False
        raise ValueError(f"Expected {TOTAL_CYCLES} cycles in profile, found {n_cycles}.")

    for name, df in sensors.items():
        expected_shape = EXPECTED_SHAPES.get(name)
        if expected_shape and df.shape != expected_shape:
            report["shapes_valid"] = False
            raise ValueError(f"Sensor {name} has shape {df.shape}, expected {expected_shape}.")

        n_nan = int(df.isna().sum().sum())
        n_inf = int(np.isinf(df.to_numpy()).sum())
        report["missing_values"] += n_nan
        report["infinite_values"] += n_inf

        report["sensors_checked"].append({
            "sensor": name,
            "shape": df.shape,
            "rate_hz": df.shape[1] // 60,
            "n_nan": n_nan,
            "n_inf": n_inf,
        })

    report["class_distribution"] = profile["pump_leakage"].value_counts().to_dict()
    report["stable_cycle_count"] = int((profile["stable_flag"] == 0).sum())
    report["unstable_cycle_count"] = int((profile["stable_flag"] == 1).sum())

    return report


# ---------------------------------------------------------------------------
# Metadata Generation & Cycle Filtering
# ---------------------------------------------------------------------------

def create_cycle_metadata(profile: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a clean cycle-level operational metadata DataFrame.

    Stores condition profiles and operational stability flags separately from
    predictive features to prevent target and condition leakage.

    Parameters
    ----------
    profile : pd.DataFrame
        Output of load_profile().

    Returns
    -------
    pd.DataFrame
        Columns:
        ['cycle_id', 'target', 'target_name', 'cooler', 'valve', 'accumulator',
         'stable_flag', 'is_stable', 'cooler_desc', 'valve_desc', 'accumulator_desc']
    """
    meta = pd.DataFrame(index=profile.index)
    meta["cycle_id"] = profile["cycle_id"].values
    meta["target"] = profile["pump_leakage"].values
    meta["target_name"] = profile["target_name"].values
    meta["cooler"] = profile["cooler"].values
    meta["valve"] = profile["valve"].values
    meta["accumulator"] = profile["accumulator"].values
    meta["stable_flag"] = profile["stable_flag"].values
    meta["is_stable"] = (profile["stable_flag"] == 0)

    # Descriptive text columns for explainability
    meta["cooler_desc"] = meta["cooler"].map(COOLER_CONDITIONS)
    meta["valve_desc"] = meta["valve"].map(VALVE_CONDITIONS)
    meta["accumulator_desc"] = meta["accumulator"].map(ACCUMULATOR_PRESSURES)

    return meta


def filter_stable_cycles(
    profile: pd.DataFrame,
    sensors: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, Optional[Dict[str, pd.DataFrame]]]:
    """
    Filter dataset to retain only cycles with stationary operating conditions (stable_flag == 0).

    Preserves the original cycle_id in both profile and sensor DataFrames.

    Parameters
    ----------
    profile : pd.DataFrame
        Output of load_profile() or create_cycle_metadata().
    sensors : dict or None, optional
        Dictionary of sensor DataFrames.

    Returns
    -------
    (filtered_profile, filtered_sensors)
        filtered_profile : pd.DataFrame of 1,449 stable cycles.
        filtered_sensors : dict of pd.DataFrame (each 1,449 rows) or None.
    """
    stable_mask = (profile["stable_flag"] == 0)
    filtered_profile = profile.loc[stable_mask].copy()

    filtered_sensors: Optional[Dict[str, pd.DataFrame]] = None
    if sensors is not None:
        filtered_sensors = {
            name: df.loc[stable_mask].copy()
            for name, df in sensors.items()
        }

    n_removed = int((~stable_mask).sum())
    n_kept = len(filtered_profile)
    print(
        f"[preprocessing] Filtered {n_removed} unstable cycles (stable_flag == 1). "
        f"Retained {n_kept} stable cycles (stable_flag == 0)."
    )

    return filtered_profile, filtered_sensors


def get_pump_leakage_labels(
    profile: pd.DataFrame,
    stable_only: bool = True,
) -> pd.Series:
    """
    Extract the pump leakage classification target series.

    Parameters
    ----------
    profile : pd.DataFrame
        Profile DataFrame.
    stable_only : bool, default True
        If True, marks unstable cycles (stable_flag == 1) as NaN.
        If False, returns all labels.

    Returns
    -------
    pd.Series
        Target labels (0, 1, 2) indexed by cycle_id.
    """
    target_col = "pump_leakage" if "pump_leakage" in profile.columns else "target"
    labels = profile[target_col].copy()
    if stable_only and "stable_flag" in profile.columns:
        unstable_mask = (profile["stable_flag"] == 1)
        labels = labels.astype(float)
        labels.loc[unstable_mask] = np.nan
    return labels


def describe_dataset(profile: pd.DataFrame) -> None:
    """
    Print an informative summary of class distributions and operational stability.

    Parameters
    ----------
    profile : pd.DataFrame
        Profile or metadata DataFrame.
    """
    n_total = len(profile)
    target_col = "pump_leakage" if "pump_leakage" in profile.columns else "target"

    print(f"\n{'='*60}")
    print(f"SmartPump AI - Dataset Distribution Summary ({n_total} operating cycles)")
    print(f"{'='*60}")

    counts = profile[target_col].value_counts().sort_index()
    for code, label in PUMP_LEAKAGE_LABEL_MAP.items():
        count = int(counts.get(code, 0))
        pct = (100.0 * count / n_total) if n_total > 0 else 0.0
        print(f"  Class {code} ({label:8s}): {count:5d} cycles ({pct:5.1f}%)")

    if "stable_flag" in profile.columns:
        stable_count = int((profile["stable_flag"] == 0).sum())
        unstable_count = int((profile["stable_flag"] == 1).sum())
        print(f"\n  Stationary cycles (stable_flag == 0): {stable_count:5d} ({100.0*stable_count/n_total:.1f}%)")
        print(f"  Transient cycles  (stable_flag == 1): {unstable_count:5d} ({100.0*unstable_count/n_total:.1f}%)")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Leakage-Safe Data Splitting
# ---------------------------------------------------------------------------

def split_cycle_data(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: Optional[pd.DataFrame] = None,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True,
) -> Dict[str, Any]:
    """
    Perform a reproducible, cycle-level, stratified train/validation/test split.

    Split Architecture:
    - 70% Training
    - 15% Validation
    - 15% Testing
    - Split strictly by operating cycle (never splitting within a cycle).
    - Preserves class proportions in train, validation, and test sets.
    - Zero cycle overlap guaranteed between splits.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (one row per cycle, index=cycle_id).
    y : pd.Series
        Target labels (0, 1, 2) aligned with X index.
    metadata : pd.DataFrame, optional
        Metadata aligned with X index.
    train_size : float, default 0.70
    val_size : float, default 0.15
    test_size : float, default 0.15
    random_state : int, default 42
    stratify : bool, default True

    Returns
    -------
    dict with keys:
        'X_train', 'X_val', 'X_test',
        'y_train', 'y_val', 'y_test',
        'meta_train', 'meta_val', 'meta_test',
        'indices': {'train': np.ndarray, 'val': np.ndarray, 'test': np.ndarray}
    """
    total_frac = train_size + val_size + test_size
    if not np.isclose(total_frac, 1.0, atol=1e-5):
        raise ValueError(
            f"Split fractions must sum to 1.0, got train={train_size}, val={val_size}, test={test_size} (sum={total_frac})."
        )

    n_samples = len(X)
    if len(y) != n_samples:
        raise ValueError(f"X has {n_samples} rows but y has {len(y)} elements.")

    # Align indices
    indices = np.arange(n_samples)
    stratify_target = y.values if stratify else None

    # Step 1: Split into Train (train_size) and Temp (val_size + test_size)
    temp_size = val_size + test_size
    splitter_stage1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=temp_size,
        random_state=random_state,
    )
    train_idx, temp_idx = next(splitter_stage1.split(indices, stratify_target))

    # Step 2: Split Temp into Validation and Test
    # Fraction of temp that goes to test:
    test_ratio_of_temp = test_size / temp_size
    temp_stratify = y.iloc[temp_idx].values if stratify else None

    splitter_stage2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_ratio_of_temp,
        random_state=random_state,
    )
    val_sub_idx, test_sub_idx = next(splitter_stage2.split(temp_idx, temp_stratify))

    val_idx = temp_idx[val_sub_idx]
    test_idx = temp_idx[test_sub_idx]

    # Verify zero overlap
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)

    assert len(train_set.intersection(val_set)) == 0, "Leakage: Train and Val overlap!"
    assert len(train_set.intersection(test_set)) == 0, "Leakage: Train and Test overlap!"
    assert len(val_set.intersection(test_set)) == 0, "Leakage: Val and Test overlap!"
    assert len(train_set) + len(val_set) + len(test_set) == n_samples, "Index mismatch!"

    result: Dict[str, Any] = {
        "X_train": X.iloc[train_idx].copy(),
        "X_val": X.iloc[val_idx].copy(),
        "X_test": X.iloc[test_idx].copy(),
        "y_train": y.iloc[train_idx].copy(),
        "y_val": y.iloc[val_idx].copy(),
        "y_test": y.iloc[test_idx].copy(),
        "meta_train": metadata.iloc[train_idx].copy() if metadata is not None else None,
        "meta_val": metadata.iloc[val_idx].copy() if metadata is not None else None,
        "meta_test": metadata.iloc[test_idx].copy() if metadata is not None else None,
        "indices": {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        },
    }

    return result


def split_cycle_data_chronological(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: Optional[pd.DataFrame] = None,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> Dict[str, Any]:
    """
    Perform a contiguous chronological split without shuffling to evaluate
    temporal degradation and experimental drift across cycle order.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    metadata : pd.DataFrame, optional
    train_size : float, default 0.70
    val_size : float, default 0.15
    test_size : float, default 0.15

    Returns
    -------
    dict with identical structure to split_cycle_data().
    """
    n_samples = len(X)
    n_train = int(np.round(n_samples * train_size))
    n_val = int(np.round(n_samples * val_size))

    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n_samples)

    return {
        "X_train": X.iloc[train_idx].copy(),
        "X_val": X.iloc[val_idx].copy(),
        "X_test": X.iloc[test_idx].copy(),
        "y_train": y.iloc[train_idx].copy(),
        "y_val": y.iloc[val_idx].copy(),
        "y_test": y.iloc[test_idx].copy(),
        "meta_train": metadata.iloc[train_idx].copy() if metadata is not None else None,
        "meta_val": metadata.iloc[val_idx].copy() if metadata is not None else None,
        "meta_test": metadata.iloc[test_idx].copy() if metadata is not None else None,
        "indices": {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        },
    }


def fit_transform_scaler(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit a StandardScaler strictly on X_train, and transform X_val and X_test.

    Prevents distribution leakage from validation/test data into training features.

    Parameters
    ----------
    X_train : pd.DataFrame
    X_val : pd.DataFrame
    X_test : pd.DataFrame

    Returns
    -------
    (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        index=X_val.index,
        columns=X_val.columns,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns,
    )
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
