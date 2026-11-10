"""
feature_engineering.py
======================
SmartPump AI - Production-Quality Feature Extraction & Engineering Pipeline

Responsible for:
- Extracting cycle-level statistical, distribution, and temporal summary features
  from high-frequency sensor time-series (1 Hz to 100 Hz, 60 to 6,000 samples per cycle).
- Eliminating within-cycle time-step data leakage by reducing each 60-second cycle
  to a single fixed-width feature vector.
- Generating 15 domain-justified features per sensor (255 features across 17 sensors):
  * Statistical: mean, std, min, max, median, range, q25, q75, iqr, rms
  * Distribution: skewness, kurtosis (with numerical safety guards for zero-variance signals)
  * Temporal: linear slope, first-to-last delta, consecutive variability
- Preserving true physical zeros (particularly PS4 accumulator gauge pressure).
- Guaranteeing zero NaN and zero infinite feature values across all operating cycles.
- Generating detailed feature metadata cataloging physical descriptions, units, and priority status.
- Saving standardized production datasets to data/processed/.

Author: SmartPump AI Engineering Team
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Supported Feature Definitions
# ---------------------------------------------------------------------------

# All 15 domain-justified cycle-level statistics
ALL_STATISTICS: List[str] = [
    "mean",
    "std",
    "min",
    "max",
    "median",
    "range",
    "q25",
    "q75",
    "iqr",
    "rms",
    "skew",
    "kurtosis",
    "slope",
    "delta",
    "variability",
]

# Default statistics set (computes all 15 features per sensor)
DEFAULT_STATS: List[str] = list(ALL_STATISTICS)

# Minimal statistical baseline (used in initial exploratory baseline)
BASIC_STATS: List[str] = ["mean", "std", "min", "max", "median"]

# Priority sensors identified in exploratory analysis for hydraulic pump leakage
PRIORITY_SENSORS: List[str] = ["FS1", "SE", "EPS1", "PS1", "PS2", "PS3"]

# Sensor descriptions and physical units dictionary
SENSOR_INFO: Dict[str, Dict[str, str]] = {
    "PS1": {"desc": "Primary circuit pressure at pump outlet", "unit": "bar", "rate": "100 Hz"},
    "PS2": {"desc": "Primary circuit pressure at main cylinder", "unit": "bar", "rate": "100 Hz"},
    "PS3": {"desc": "Primary circuit return line pressure", "unit": "bar", "rate": "100 Hz"},
    "PS4": {"desc": "Hydraulic accumulator charging pressure", "unit": "bar", "rate": "100 Hz"},
    "PS5": {"desc": "Secondary cooling circuit pressure", "unit": "bar", "rate": "100 Hz"},
    "PS6": {"desc": "Secondary filtration circuit pressure", "unit": "bar", "rate": "100 Hz"},
    "EPS1": {"desc": "Main pump motor electric power", "unit": "W", "rate": "100 Hz"},
    "FS1": {"desc": "Primary circuit volume flow rate", "unit": "l/min", "rate": "10 Hz"},
    "FS2": {"desc": "Secondary cooling circuit volume flow rate", "unit": "l/min", "rate": "10 Hz"},
    "TS1": {"desc": "Oil reservoir temperature", "unit": "°C", "rate": "1 Hz"},
    "TS2": {"desc": "Primary circuit temperature before cooler", "unit": "°C", "rate": "1 Hz"},
    "TS3": {"desc": "Primary circuit temperature after cooler", "unit": "°C", "rate": "1 Hz"},
    "TS4": {"desc": "Secondary cooling circuit temperature", "unit": "°C", "rate": "1 Hz"},
    "VS1": {"desc": "Main pump bracket vibration velocity amplitude", "unit": "mm/s", "rate": "1 Hz"},
    "CE": {"desc": "Virtual cooling efficiency factor", "unit": "%", "rate": "1 Hz"},
    "CP": {"desc": "Virtual cooling power transfer rate", "unit": "kW", "rate": "1 Hz"},
    "SE": {"desc": "Global hydraulic efficiency factor", "unit": "%", "rate": "1 Hz"},
}


# ---------------------------------------------------------------------------
# Single-Sensor Feature Extraction
# ---------------------------------------------------------------------------

def extract_cycle_features(
    sensor_df: pd.DataFrame,
    sensor_name: str,
    statistics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Extract vector of statistical, distribution, and temporal summary features
    from a sensor time-series matrix for each operating cycle.

    Parameters
    ----------
    sensor_df : pd.DataFrame
        Shape (n_cycles, n_time_samples). Each row is one 60-second cycle.
    sensor_name : str
        Human-readable prefix, e.g. 'FS1', 'PS1'.
    statistics : list of str, optional
        List of statistics to compute. Defaults to DEFAULT_STATS (all 15 features).

    Returns
    -------
    pd.DataFrame
        Shape (n_cycles, len(statistics)).
        Columns named '<sensor_name>_<statistic>', preserving sensor_df index.
    """
    if statistics is None:
        statistics = DEFAULT_STATS

    arr = np.asarray(sensor_df, dtype=np.float64)
    n_rows, n_cols = arr.shape
    features: Dict[str, np.ndarray] = {}

    # Precompute common statistics to optimize vectorized operations
    mean_val: Optional[np.ndarray] = None
    std_val: Optional[np.ndarray] = None
    min_val: Optional[np.ndarray] = None
    max_val: Optional[np.ndarray] = None
    q25_val: Optional[np.ndarray] = None
    q75_val: Optional[np.ndarray] = None

    def get_mean() -> np.ndarray:
        nonlocal mean_val
        if mean_val is None:
            mean_val = arr.mean(axis=1)
        return mean_val

    def get_std() -> np.ndarray:
        nonlocal std_val
        if std_val is None:
            std_val = arr.std(axis=1, ddof=1) if n_cols > 1 else np.zeros(n_rows, dtype=np.float64)
            std_val = np.nan_to_num(std_val, nan=0.0)
        return std_val

    def get_min() -> np.ndarray:
        nonlocal min_val
        if min_val is None:
            min_val = arr.min(axis=1)
        return min_val

    def get_max() -> np.ndarray:
        nonlocal max_val
        if max_val is None:
            max_val = arr.max(axis=1)
        return max_val

    def get_q25() -> np.ndarray:
        nonlocal q25_val
        if q25_val is None:
            q25_val = np.percentile(arr, 25, axis=1)
        return q25_val

    def get_q75() -> np.ndarray:
        nonlocal q75_val
        if q75_val is None:
            q75_val = np.percentile(arr, 75, axis=1)
        return q75_val

    for stat in statistics:
        col = f"{sensor_name}_{stat}"
        if stat == "mean":
            features[col] = get_mean()
        elif stat == "std":
            features[col] = get_std()
        elif stat == "min":
            features[col] = get_min()
        elif stat == "max":
            features[col] = get_max()
        elif stat == "median":
            features[col] = np.median(arr, axis=1)
        elif stat == "range":
            features[col] = get_max() - get_min()
        elif stat == "q25":
            features[col] = get_q25()
        elif stat == "q75":
            features[col] = get_q75()
        elif stat == "iqr":
            features[col] = get_q75() - get_q25()
        elif stat == "rms":
            features[col] = np.sqrt(np.mean(arr ** 2, axis=1))
        elif stat == "skew":
            # Guard against zero-variance constant signals (e.g. unpressurized PS4)
            s = get_std()
            non_const_idx = np.where(s >= 1e-12)[0]
            sk = np.zeros(n_rows, dtype=np.float64)
            if len(non_const_idx) > 0:
                sk[non_const_idx] = scipy_stats.skew(arr[non_const_idx], axis=1, bias=True)
            features[col] = np.nan_to_num(sk, nan=0.0)
        elif stat == "kurtosis":
            # Guard against zero-variance constant signals
            s = get_std()
            non_const_idx = np.where(s >= 1e-12)[0]
            kt = np.zeros(n_rows, dtype=np.float64)
            if len(non_const_idx) > 0:
                kt[non_const_idx] = scipy_stats.kurtosis(arr[non_const_idx], axis=1, bias=True)
            features[col] = np.nan_to_num(kt, nan=0.0)
        elif stat == "slope":
            # Closed-form vectorized linear trend: arr @ ((t - mean(t)) / sum((t - mean(t))^2))
            t = np.arange(n_cols, dtype=np.float64)
            t_centered = t - t.mean()
            denom = np.sum(t_centered ** 2)
            weights = t_centered / (denom if denom > 0 else 1.0)
            features[col] = arr @ weights
        elif stat == "delta":
            # First-to-last change across the operating cycle
            features[col] = arr[:, -1] - arr[:, 0]
        elif stat == "variability":
            # Consecutive point-to-point variability (mean absolute first difference)
            if n_cols > 1:
                features[col] = np.mean(np.abs(np.diff(arr, axis=1)), axis=1)
            else:
                features[col] = np.zeros(n_rows, dtype=np.float64)
        else:
            raise ValueError(
                f"Unknown statistic '{stat}'. Supported: {ALL_STATISTICS}."
            )

    out_df = pd.DataFrame(features, index=sensor_df.index)

    # Sanity verify no NaNs or Infs
    n_nan = int(out_df.isna().sum().sum())
    n_inf = int(np.isinf(out_df.to_numpy()).sum())
    if n_nan > 0 or n_inf > 0:
        raise ValueError(
            f"Feature extraction generated invalid values for {sensor_name}: "
            f"{n_nan} NaNs, {n_inf} Infs."
        )

    return out_df


# ---------------------------------------------------------------------------
# Full Feature Matrix Assembly
# ---------------------------------------------------------------------------

def build_feature_matrix(
    sensors: Dict[str, pd.DataFrame],
    statistics: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Extract statistical features from all sensor DataFrames and assemble
    a unified cycle-level feature matrix.

    Parameters
    ----------
    sensors : dict
        Mapping from filename (e.g. 'PS1.txt') to sensor DataFrame.
    statistics : list of str, optional
        Statistics to extract for each sensor. Defaults to DEFAULT_STATS (15 stats).
    verbose : bool, default True
        Whether to log extraction progress.

    Returns
    -------
    pd.DataFrame
        Shape (n_cycles, n_sensors * len(statistics)).
        Preserves cycle_id index from sensor DataFrames.
    """
    if statistics is None:
        statistics = DEFAULT_STATS

    feature_dfs: List[pd.DataFrame] = []

    for filename, sensor_df in sensors.items():
        sensor_name = Path(filename).stem
        sensor_features = extract_cycle_features(sensor_df, sensor_name, statistics)
        feature_dfs.append(sensor_features)
        if verbose:
            print(
                f"[feature_engineering] {sensor_name:5s} ({filename:8s}) -> "
                f"Extracted {sensor_features.shape[1]} features across {len(sensor_features)} cycles."
            )

    feature_matrix = pd.concat(feature_dfs, axis=1)

    if verbose:
        print(
            f"\n[feature_engineering] Assembled feature matrix: "
            f"{feature_matrix.shape[0]} cycles × {feature_matrix.shape[1]} features."
        )

    return feature_matrix


# ---------------------------------------------------------------------------
# Feature Metadata Generation
# ---------------------------------------------------------------------------

def generate_feature_metadata(feature_columns: List[str]) -> pd.DataFrame:
    """
    Create a structured metadata DataFrame documenting all engineered features.

    Parameters
    ----------
    feature_columns : list of str
        List of feature column names (e.g. ['FS1_mean', 'PS1_slope']).

    Returns
    -------
    pd.DataFrame
        Table with columns:
        ['feature_name', 'sensor', 'statistic', 'category', 'unit', 'sampling_rate',
         'sensor_subsystem', 'is_priority_sensor', 'description']
    """
    records: List[Dict[str, Any]] = []

    category_map: Dict[str, str] = {
        "mean": "statistical_central_tendency",
        "median": "statistical_central_tendency",
        "std": "statistical_dispersion",
        "range": "statistical_dispersion",
        "iqr": "statistical_dispersion",
        "min": "statistical_extremes",
        "max": "statistical_extremes",
        "q25": "statistical_percentile",
        "q75": "statistical_percentile",
        "rms": "signal_energy",
        "skew": "distribution_shape",
        "kurtosis": "distribution_shape",
        "slope": "temporal_trend",
        "delta": "temporal_displacement",
        "variability": "temporal_variability",
    }

    stat_descriptions: Dict[str, str] = {
        "mean": "Arithmetic mean of signal over 60-second cycle",
        "std": "Sample standard deviation (ddof=1) over 60-second cycle",
        "min": "Minimum value observed during cycle",
        "max": "Maximum value observed during cycle",
        "median": "50th percentile value during cycle",
        "range": "Peak-to-peak amplitude difference (max - min)",
        "q25": "First quartile (25th percentile) value",
        "q75": "Third quartile (75th percentile) value",
        "iqr": "Interquartile range (q75 - q25)",
        "rms": "Root mean square signal power (sqrt(mean(x^2)))",
        "skew": "Fisher-Pearson coefficient of skewness (zero-safe)",
        "kurtosis": "Fisher excess kurtosis (zero-safe)",
        "slope": "Closed-form least-squares linear trend rate per sample",
        "delta": "Net signal displacement between final and initial sample (last - first)",
        "variability": "Mean absolute difference between consecutive time samples",
    }

    for col in feature_columns:
        if "_" in col:
            sensor_name, stat = col.split("_", 1)
        else:
            sensor_name, stat = col, "unknown"

        info = SENSOR_INFO.get(sensor_name, {
            "desc": f"Sensor {sensor_name}",
            "unit": "unknown",
            "rate": "unknown",
        })

        is_priority = sensor_name in PRIORITY_SENSORS
        category = category_map.get(stat, "other")
        stat_desc = stat_descriptions.get(stat, stat)

        records.append({
            "feature_name": col,
            "sensor": sensor_name,
            "statistic": stat,
            "category": category,
            "unit": info.get("unit", ""),
            "sampling_rate": info.get("rate", ""),
            "sensor_subsystem": info.get("desc", ""),
            "is_priority_sensor": is_priority,
            "description": f"{info.get('desc', '')} - {stat_desc}",
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Dataset Persistence & Loading
# ---------------------------------------------------------------------------

def save_processed_datasets(
    stable_features: pd.DataFrame,
    full_features: pd.DataFrame,
    stable_metadata: pd.DataFrame,
    full_metadata: pd.DataFrame,
    processed_dir: Union[str, Path],
    feature_metadata: Optional[pd.DataFrame] = None,
) -> Dict[str, Path]:
    """
    Save the production datasets to data/processed/.

    Outputs:
    1. stable_cycle_features.csv : 1,449 stable cycles × 255 features + cycle_id + target
    2. full_cycle_features.csv   : 2,205 full cycles × 255 features + cycle_id + target
    3. stable_metadata.csv       : 1,449 rows operational metadata
    4. full_metadata.csv         : 2,205 rows operational metadata
    5. feature_metadata.csv      : 255 rows feature definitions catalog
    6. features_per_cycle.csv    : Backward-compatible link/copy of stable features

    Parameters
    ----------
    stable_features : pd.DataFrame
    full_features : pd.DataFrame
    stable_metadata : pd.DataFrame
    full_metadata : pd.DataFrame
    processed_dir : Path or str
    feature_metadata : pd.DataFrame, optional

    Returns
    -------
    dict of saved file paths
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: Dict[str, Path] = {}

    # Helper to assemble feature file with cycle_id and target
    def format_feature_file(feats: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
        df = feats.copy()
        if "cycle_id" not in df.columns:
            df.insert(0, "cycle_id", meta["cycle_id"].values)
        if "target" not in df.columns:
            df["target"] = meta["target"].values
        return df

    stable_out = format_feature_file(stable_features, stable_metadata)
    full_out = format_feature_file(full_features, full_metadata)

    # 1. stable_cycle_features.csv
    p_stable = processed_dir / "stable_cycle_features.csv"
    stable_out.to_csv(p_stable, index=False)
    saved_paths["stable_cycle_features"] = p_stable
    print(f"[feature_engineering] Saved: {p_stable} ({stable_out.shape})")

    # 2. full_cycle_features.csv
    p_full = processed_dir / "full_cycle_features.csv"
    full_out.to_csv(p_full, index=False)
    saved_paths["full_cycle_features"] = p_full
    print(f"[feature_engineering] Saved: {p_full} ({full_out.shape})")

    # 3. stable_metadata.csv
    p_stable_meta = processed_dir / "stable_metadata.csv"
    stable_metadata.to_csv(p_stable_meta, index=False)
    saved_paths["stable_metadata"] = p_stable_meta
    print(f"[feature_engineering] Saved: {p_stable_meta} ({stable_metadata.shape})")

    # 4. full_metadata.csv
    p_full_meta = processed_dir / "full_metadata.csv"
    full_metadata.to_csv(p_full_meta, index=False)
    saved_paths["full_metadata"] = p_full_meta
    print(f"[feature_engineering] Saved: {p_full_meta} ({full_metadata.shape})")

    # 5. feature_metadata.csv
    if feature_metadata is None:
        feat_cols = [c for c in stable_features.columns if c not in ("cycle_id", "target", "pump_leakage")]
        feature_metadata = generate_feature_metadata(feat_cols)
    p_feat_meta = processed_dir / "feature_metadata.csv"
    feature_metadata.to_csv(p_feat_meta, index=False)
    saved_paths["feature_metadata"] = p_feat_meta
    print(f"[feature_engineering] Saved: {p_feat_meta} ({feature_metadata.shape})")

    # 6. Backward-compatible features_per_cycle.csv for train.py
    # Includes 'pump_leakage' column name expected by train.py
    p_legacy = processed_dir / "features_per_cycle.csv"
    legacy_df = stable_features.copy()
    legacy_df["pump_leakage"] = stable_metadata["target"].values
    legacy_df.to_csv(p_legacy, index=False)
    saved_paths["features_per_cycle"] = p_legacy
    print(f"[feature_engineering] Saved legacy compatibility file: {p_legacy} ({legacy_df.shape})")

    return saved_paths


def load_processed_dataset(
    processed_dir: Union[str, Path],
    dataset_type: str = "stable",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load a processed dataset (features, target, and metadata).

    Parameters
    ----------
    processed_dir : Path or str
        Path to data/processed/.
    dataset_type : str, default 'stable'
        'stable' or 'full'.

    Returns
    -------
    (X, y, metadata)
        X : pd.DataFrame of predictive features (no metadata, no target, no cycle_id)
        y : pd.Series of integer target labels (0, 1, 2)
        metadata : pd.DataFrame of cycle-level operational metadata
    """
    processed_dir = Path(processed_dir)
    features_file = processed_dir / f"{dataset_type}_cycle_features.csv"
    metadata_file = processed_dir / f"{dataset_type}_metadata.csv"

    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found at {features_file}.")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_file}.")

    df_feats = pd.read_csv(features_file)
    metadata = pd.read_csv(metadata_file)

    # Isolate predictive features from identifiers and target
    drop_cols = [c for c in ["cycle_id", "target", "pump_leakage"] if c in df_feats.columns]
    X = df_feats.drop(columns=drop_cols)
    y = metadata["target"].astype(int)

    # Set cycle_id as index
    if "cycle_id" in df_feats.columns:
        X.index = df_feats["cycle_id"]
        y.index = df_feats["cycle_id"]
        metadata.index = df_feats["cycle_id"]

    return X, y, metadata


def load_features(
    processed_dir: Union[str, Path],
    filename: str = "features_per_cycle.csv",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Backward-compatible feature loader for existing train.py script.

    Parameters
    ----------
    processed_dir : Path or str
    filename : str

    Returns
    -------
    (X, y) : (pd.DataFrame, pd.Series)
    """
    path = Path(processed_dir) / filename
    if not path.exists():
        # Fall back to stable_cycle_features.csv if legacy file is missing
        alt_path = Path(processed_dir) / "stable_cycle_features.csv"
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Feature file not found at {path} or {alt_path}.")

    df = pd.read_csv(path)
    target_col = "pump_leakage" if "pump_leakage" in df.columns else "target"
    if target_col not in df.columns:
        raise KeyError(f"Neither 'pump_leakage' nor 'target' column found in {path}.")

    y = df[target_col].astype(int)
    drop_cols = [c for c in [target_col, "cycle_id"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    return X, y


# Backward compatibility alias
def save_features(
    feature_matrix: pd.DataFrame,
    labels: pd.Series,
    processed_dir: Union[str, Path],
    filename: str = "features_per_cycle.csv",
) -> Path:
    """Legacy helper preserving existing API signatures."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out = feature_matrix.copy()
    out["pump_leakage"] = labels.values
    out = out.dropna(subset=["pump_leakage"])
    out["pump_leakage"] = out["pump_leakage"].astype(int)
    out_path = processed_dir / filename
    out.to_csv(out_path, index=False)
    return out_path
