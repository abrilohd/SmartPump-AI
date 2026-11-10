"""
train.py
========
SmartPump AI - Machine Learning Model Development, Cross-Validation & Evaluation

Responsible for:
- Building Scikit-learn Pipelines with zero-variance filtering and scale-appropriate transformers
- Training six candidate classifiers:
    1. DummyClassifier (Baseline)
    2. DecisionTreeClassifier
    3. RandomForestClassifier
    4. SupportVectorClassifier (SVC RBF)
    5. GaussianNaïveBayes
    6. AdaBoostClassifier
- 5-Fold Stratified & Group-Aware Cross-Validation on training data only
- Hyperparameter tuning using GridSearchCV/RandomizedSearchCV
- Multi-metric evaluation prioritizing Fault Recall, Fault Precision, and Macro F1
- Feature importance analysis (Gini importances & Permutation importance on validation data)
- Secondary Experiment: Full-cycle classification (2,205 cycles with transient data)
- Unstable-Cycle Sensitivity Analysis: Evaluating stable-trained models on 756 transient cycles
- Model serialization (smartpump_best_model.joblib) and comprehensive metadata logging (model_metadata.json)

Author: SmartPump AI Engineering Team
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Configure headless matplotlib for automated reporting
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", "/tmp")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

RANDOM_STATE: int = 42

# Target class mappings
CLASS_NAMES: List[str] = ["Normal", "Warning", "Fault"]
CLASS_CODES: List[int] = [0, 1, 2]

# Directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
EVALUATION_DIR = RESULTS_DIR / "evaluation"
VISUALIZATIONS_DIR = RESULTS_DIR / "visualizations"


# ---------------------------------------------------------------------------
# Pipeline Construction
# ---------------------------------------------------------------------------

def build_model_pipelines(
    random_state: int = RANDOM_STATE,
) -> Dict[str, Pipeline]:
    """
    Define candidate classifiers wrapped in Scikit-learn Pipelines.

    Pipeline Design Principles:
    - Zero-variance features (e.g. PS2_min, SE_min) are filtered via VarianceThreshold.
    - Scale-sensitive models (SVM, GaussianNB) include StandardScaler.
    - Tree-based models (DecisionTree, RandomForest) preserve original feature scales.
    - Class weights are set to 'balanced' where appropriate.

    Parameters
    ----------
    random_state : int, default 42

    Returns
    -------
    dict
        Mapping of model names to unfitted Pipeline objects.
    """
    models: Dict[str, Pipeline] = {
        "Dummy": Pipeline([
            ("clf", DummyClassifier(strategy="stratified", random_state=random_state)),
        ]),
        "DecisionTree": Pipeline([
            ("selector", VarianceThreshold(threshold=0.0)),
            ("clf", DecisionTreeClassifier(
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
        "RandomForest": Pipeline([
            ("selector", VarianceThreshold(threshold=0.0)),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
        "SVM": Pipeline([
            ("selector", VarianceThreshold(threshold=0.0)),
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=10.0,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
        "GaussianNB": Pipeline([
            ("selector", VarianceThreshold(threshold=0.0)),
            ("scaler", StandardScaler()),
            ("clf", GaussianNB(var_smoothing=1e-2)),
        ]),
        "AdaBoost": Pipeline([
            ("selector", VarianceThreshold(threshold=0.0)),
            ("clf", AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=2, random_state=random_state),
                n_estimators=100,
                learning_rate=0.5,
                random_state=random_state,
            )),
        ]),
    }
    return models


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

def evaluate_predictions(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: np.ndarray,
    model_name: str = "",
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics prioritizing fault detection.

    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        Ground-truth labels (0, 1, 2).
    y_pred : np.ndarray
        Model predicted class labels (0, 1, 2).
    model_name : str, optional

    Returns
    -------
    dict
        Scalar metrics, confusion matrix, and detailed classification reports.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    acc = accuracy_score(y_true_arr, y_pred_arr)
    macro_f1 = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    macro_prec = precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    macro_rec = recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)

    # Per-class metrics: 0: Normal, 1: Warning, 2: Fault
    per_class_rec = recall_score(y_true_arr, y_pred_arr, labels=CLASS_CODES, average=None, zero_division=0)
    per_class_prec = precision_score(y_true_arr, y_pred_arr, labels=CLASS_CODES, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true_arr, y_pred_arr, labels=CLASS_CODES, average=None, zero_division=0)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=CLASS_CODES)
    report_dict = classification_report(
        y_true_arr, y_pred_arr, labels=CLASS_CODES, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true_arr, y_pred_arr, labels=CLASS_CODES, target_names=CLASS_NAMES, zero_division=0
    )

    metrics = {
        "model": model_name,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "weighted_f1": float(weighted_f1),
        # Priority fault metrics (class index 2)
        "fault_recall": float(per_class_rec[2]),
        "fault_precision": float(per_class_prec[2]),
        "fault_f1": float(per_class_f1[2]),
        # Warning metrics (class index 1)
        "warning_recall": float(per_class_rec[1]),
        "warning_precision": float(per_class_prec[1]),
        "warning_f1": float(per_class_f1[1]),
        # Normal metrics (class index 0)
        "normal_recall": float(per_class_rec[0]),
        "normal_precision": float(per_class_prec[0]),
        "normal_f1": float(per_class_f1[0]),
        "confusion_matrix": cm,
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
    }
    return metrics


# ---------------------------------------------------------------------------
# Cross-Validation & Hyperparameter Tuning
# ---------------------------------------------------------------------------

def run_cross_validation(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    groups: Optional[np.ndarray] = None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Perform 5-fold cross-validation on training data strictly.

    Supports both StratifiedKFold (standard) and StratifiedGroupKFold
    (when experimental block identifiers are supplied).

    Parameters
    ----------
    models : dict of Pipeline
    X_train : pd.DataFrame
    y_train : pd.Series
    n_splits : int, default 5
    groups : np.ndarray, optional
    random_state : int, default 42

    Returns
    -------
    pd.DataFrame
        Table summarizing CV mean and standard deviation for each model.
    """
    scoring = {
        "macro_f1": "f1_macro",
        "accuracy": "accuracy",
    }

    if groups is not None:
        cv_splitter = StratifiedGroupKFold(n_splits=n_splits)
        cv_name = "StratifiedGroupKFold"
    else:
        cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_name = "StratifiedKFold"

    results_list: List[Dict[str, Any]] = []

    print(f"\n[train] Running {n_splits}-Fold Cross-Validation ({cv_name}) on training set ({len(X_train)} cycles)...")

    for name, pipeline in models.items():
        t0 = time.time()
        cv_res = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv_splitter,
            groups=groups,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        elapsed = time.time() - t0

        f1_scores = cv_res["test_macro_f1"]
        acc_scores = cv_res["test_accuracy"]

        results_list.append({
            "model": name,
            "cv_macro_f1_mean": float(f1_scores.mean()),
            "cv_macro_f1_std": float(f1_scores.std()),
            "cv_accuracy_mean": float(acc_scores.mean()),
            "cv_accuracy_std": float(acc_scores.std()),
            "cv_time_sec": round(elapsed, 2),
        })

        print(
            f"  {name:14s} | CV Macro F1: {f1_scores.mean():.4f} (±{f1_scores.std():.4f}) | "
            f"CV Acc: {acc_scores.mean():.4f} | Time: {elapsed:.2f}s"
        )

    df_cv = pd.DataFrame(results_list).sort_values("cv_macro_f1_mean", ascending=False).reset_index(drop=True)
    return df_cv


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> Tuple[Dict[str, Pipeline], pd.DataFrame]:
    """
    Perform controlled grid search on candidate models using 5-fold Stratified CV.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series
    random_state : int, default 42

    Returns
    -------
    (best_pipelines, tuning_summary_df)
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    param_grids: Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]] = {
        "RandomForest": (
            Pipeline([
                ("selector", VarianceThreshold(threshold=0.0)),
                ("clf", RandomForestClassifier(random_state=random_state, class_weight="balanced", n_jobs=-1)),
            ]),
            {
                "clf__n_estimators": [100, 200, 300],
                "clf__max_depth": [None, 15, 25],
                "clf__min_samples_split": [2, 5],
            },
        ),
        "SVM": (
            Pipeline([
                ("selector", VarianceThreshold(threshold=0.0)),
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=random_state)),
            ]),
            {
                "clf__C": [1.0, 10.0, 100.0],
                "clf__gamma": ["scale", "auto", 0.01],
            },
        ),
        "DecisionTree": (
            Pipeline([
                ("selector", VarianceThreshold(threshold=0.0)),
                ("clf", DecisionTreeClassifier(random_state=random_state, class_weight="balanced")),
            ]),
            {
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_split": [2, 5],
                "clf__min_samples_leaf": [1, 2],
            },
        ),
        "AdaBoost": (
            Pipeline([
                ("selector", VarianceThreshold(threshold=0.0)),
                ("clf", AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(random_state=random_state),
                    random_state=random_state,
                )),
            ]),
            {
                "clf__n_estimators": [50, 100],
                "clf__learning_rate": [0.1, 0.5, 1.0],
                "clf__estimator__max_depth": [1, 2],
            },
        ),
    }

    best_pipelines: Dict[str, Pipeline] = {}
    tuning_records: List[Dict[str, Any]] = []

    print("\n[train] Executing Systematic Hyperparameter Grid Search (5-Fold CV)...")

    for name, (base_pipe, grid) in param_grids.items():
        t0 = time.time()
        search = GridSearchCV(
            estimator=base_pipe,
            param_grid=grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        elapsed = time.time() - t0

        best_pipelines[name] = search.best_estimator_
        tuning_records.append({
            "model": name,
            "best_macro_f1": float(search.best_score_),
            "best_params": json.dumps(search.best_params_),
            "tuning_time_sec": round(elapsed, 2),
        })

        print(
            f"  {name:14s} | Best Macro F1: {search.best_score_:.4f} | "
            f"Best Params: {search.best_params_} | Time: {elapsed:.2f}s"
        )

    # Include default pipelines for Dummy and GaussianNB
    defaults = build_model_pipelines(random_state=random_state)
    best_pipelines["Dummy"] = defaults["Dummy"]
    best_pipelines["GaussianNB"] = defaults["GaussianNB"]

    tuning_df = pd.DataFrame(tuning_records).sort_values("best_macro_f1", ascending=False)
    return best_pipelines, tuning_df


# ---------------------------------------------------------------------------
# Training & Evaluation Runners
# ---------------------------------------------------------------------------

def train_and_evaluate_models(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    split_name: str = "Validation",
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Pipeline]]:
    """
    Train each model pipeline on training data and evaluate on a specified partition.

    Parameters
    ----------
    models : dict of Pipeline
    X_train : pd.DataFrame
    y_train : pd.Series
    X_eval : pd.DataFrame
    y_eval : pd.Series
    split_name : str, default 'Validation'

    Returns
    -------
    (comparison_df, detailed_reports, fitted_pipelines)
    """
    fitted_models: Dict[str, Pipeline] = {}
    detailed_reports: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []

    print(f"\n[train] Training and evaluating {len(models)} models on {split_name} Set ({len(X_eval)} cycles)...")

    for name, pipeline in models.items():
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        fit_time = time.time() - t0

        t1 = time.time()
        y_pred = pipeline.predict(X_eval)
        pred_time = time.time() - t1

        metrics = evaluate_predictions(y_eval, y_pred, model_name=name)
        metrics["fit_time_sec"] = round(fit_time, 3)
        metrics["pred_time_sec"] = round(pred_time, 4)

        fitted_models[name] = pipeline
        detailed_reports[name] = metrics

        summary_rows.append({
            "model": name,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "fault_recall": metrics["fault_recall"],
            "fault_precision": metrics["fault_precision"],
            "fault_f1": metrics["fault_f1"],
            "warning_f1": metrics["warning_f1"],
            "normal_f1": metrics["normal_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "fit_time_sec": metrics["fit_time_sec"],
        })

        print(
            f"  {name:14s} | Acc: {metrics['accuracy']:.4f} | "
            f"Macro F1: {metrics['macro_f1']:.4f} | "
            f"Fault Recall: {metrics['fault_recall']:.4f} | "
            f"Fault Prec: {metrics['fault_precision']:.4f}"
        )

    comparison_df = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    return comparison_df, detailed_reports, fitted_models


# ---------------------------------------------------------------------------
# Feature Importance & Interpretability
# ---------------------------------------------------------------------------

def compute_feature_importance(
    model: Pipeline,
    feature_names: List[str],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    top_n: int = 25,
) -> pd.DataFrame:
    """
    Compute Gini impurity importances (if tree-based) and Permutation Importance
    on held-out validation data.

    Parameters
    ----------
    model : Pipeline
        Fitted model pipeline.
    feature_names : list of str
    X_val : pd.DataFrame
    y_val : pd.Series
    top_n : int, default 25

    Returns
    -------
    pd.DataFrame
        Ranked table of feature importances with domain descriptions.
    """
    clf = model.named_steps.get("clf")
    selector = model.named_steps.get("selector")

    # Get retained feature names after VarianceThreshold
    if selector is not None and hasattr(selector, "get_support"):
        retained_mask = selector.get_support()
        retained_features = [f for f, keep in zip(feature_names, retained_mask) if keep]
    else:
        retained_features = list(feature_names)

    records: List[Dict[str, Any]] = []

    # Tree Gini importance
    gini_dict: Dict[str, float] = {}
    if hasattr(clf, "feature_importances_"):
        for f, g in zip(retained_features, clf.feature_importances_):
            gini_dict[f] = float(g)

    # Permutation importance on validation data
    print(f"[train] Computing permutation importance on validation set ({len(X_val)} cycles)...")
    perm_res = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="f1_macro",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    for i, col in enumerate(feature_names):
        sensor_name = col.split("_")[0] if "_" in col else col
        is_priority = sensor_name in ["FS1", "SE", "EPS1", "PS1", "PS2", "PS3"]

        records.append({
            "feature_name": col,
            "sensor": sensor_name,
            "is_priority_sensor": is_priority,
            "perm_importance_mean": float(perm_res.importances_mean[i]),
            "perm_importance_std": float(perm_res.importances_std[i]),
            "gini_importance": gini_dict.get(col, 0.0),
        })

    df_imp = pd.DataFrame(records).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)
    return df_imp


# ---------------------------------------------------------------------------
# Unstable-Cycle Sensitivity Analysis
# ---------------------------------------------------------------------------

def run_unstable_sensitivity_test(
    model: Pipeline,
    processed_dir: Path = PROCESSED_DIR,
) -> Dict[str, Any]:
    """
    Evaluate the stable-trained primary model on the 756 unstable cycles.

    This measures how the classifier behaves when exposed to warm-up and
    non-stationary operating regimes (operational distribution shift).

    Parameters
    ----------
    model : Pipeline
        Fitted on stable cycles.
    processed_dir : Path

    Returns
    -------
    dict
        Evaluation report for unstable cycles.
    """
    full_features_path = processed_dir / "full_cycle_features.csv"
    full_metadata_path = processed_dir / "full_metadata.csv"

    if not full_features_path.exists() or not full_metadata_path.exists():
        raise FileNotFoundError("Processed full cycle files not found.")

    full_features = pd.read_csv(full_features_path)
    full_metadata = pd.read_csv(full_metadata_path)

    unstable_mask = (full_metadata["stable_flag"] == 1)
    X_unstable = full_features.loc[unstable_mask].drop(columns=["cycle_id", "target"])
    y_unstable = full_features.loc[unstable_mask, "target"].astype(int)

    n_unstable = len(X_unstable)
    true_counts = y_unstable.value_counts().to_dict()

    y_pred_unstable = model.predict(X_unstable)
    pred_counts = pd.Series(y_pred_unstable).value_counts().to_dict()

    metrics = evaluate_predictions(y_unstable, y_pred_unstable, model_name="StableModel_on_UnstableCycles")

    result = {
        "n_unstable_cycles": n_unstable,
        "true_distribution": true_counts,
        "predicted_distribution": pred_counts,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "fault_recall": metrics["fault_recall"],
        "fault_precision": metrics["fault_precision"],
        "warning_f1": metrics["warning_f1"],
        "normal_f1": metrics["normal_f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report_text": metrics["classification_report_text"],
    }
    return result


# ---------------------------------------------------------------------------
# Secondary Experiment: Full-Cycle Classification
# ---------------------------------------------------------------------------

def run_secondary_full_cycle_experiment(
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Train and evaluate all candidate classifiers on the full 2,205 cycle dataset
    to analyze the effect of class imbalance and transient cycles.

    Returns
    -------
    pd.DataFrame
        Evaluation results table on the full cycle test set.
    """
    from preprocessing import split_cycle_data
    from feature_engineering import load_processed_dataset

    print("\n[train] Running Secondary Experiment: Full-Cycle Classification (2,205 cycles)...")
    X_full, y_full, meta_full = load_processed_dataset(PROCESSED_DIR, dataset_type="full")

    splits = split_cycle_data(
        X_full,
        y_full,
        metadata=meta_full,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        random_state=random_state,
        stratify=True,
    )

    models = build_model_pipelines(random_state=random_state)
    comparison_df, detailed_reports, _ = train_and_evaluate_models(
        models,
        splits["X_train"],
        splits["y_train"],
        splits["X_test"],
        splits["y_test"],
        split_name="Full-Cycle Test",
    )
    return comparison_df


# ---------------------------------------------------------------------------
# Artifact Serialization
# ---------------------------------------------------------------------------

def save_model_artifacts(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    models_dir: Path = MODELS_DIR,
    model_name: str = "smartpump_best_model",
) -> Tuple[Path, Path]:
    """
    Serialize the trained pipeline and save detailed model metadata JSON.

    Parameters
    ----------
    pipeline : Pipeline
    metadata : dict
    models_dir : Path
    model_name : str

    Returns
    -------
    (model_path, metadata_path)
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{model_name}.joblib"
    meta_path = models_dir / "model_metadata.json"

    joblib.dump(pipeline, model_path)
    print(f"[train] Serialized best model to: {model_path}")

    # Convert non-serializable objects (numpy arrays, sets)
    def sanitize(obj: Any) -> Any:
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [sanitize(v) for v in obj]
        return str(obj)

    sanitized_meta = sanitize(metadata)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(sanitized_meta, f, indent=2)
    print(f"[train] Saved model metadata to: {meta_path}")

    return model_path, meta_path


# ---------------------------------------------------------------------------
# Visualization Generation
# ---------------------------------------------------------------------------

def generate_visualizations(
    model_comparison_df: pd.DataFrame,
    best_cm: np.ndarray,
    best_model_name: str,
    feature_importance_df: pd.DataFrame,
    unstable_cm: np.ndarray,
    out_dir: Path = VISUALIZATIONS_DIR,
) -> List[Path]:
    """
    Generate professional figures comparing model performance, confusion matrices,
    and feature importances.

    Returns
    -------
    list of saved figure Paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_figs: List[Path] = []

    # 1. Model comparison bar plot (Macro F1 vs Fault Recall)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(model_comparison_df))
    width = 0.35

    ax.bar(x - width/2, model_comparison_df["macro_f1"], width, label="Macro F1", color="#2980b9")
    ax.bar(x + width/2, model_comparison_df["fault_recall"], width, label="Fault Recall", color="#c0392b")

    ax.set_xticks(x)
    ax.set_xticklabels(model_comparison_df["model"], rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Comparison: Macro F1 vs. Fault Recall (Validation Set)")
    ax.legend(loc="lower right")
    ax.axhline(y=0.95, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    p1 = out_dir / "model_comparison.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=150)
    plt.close(fig)
    saved_figs.append(p1)

    # 2. Confusion matrix for best model (Held-out Test set)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        best_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix: {best_model_name} (Held-out Test Set)")
    ax.set_xlabel("Predicted Condition")
    ax.set_ylabel("Actual Ground Truth")

    p2 = out_dir / "confusion_matrix_best_model.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=150)
    plt.close(fig)
    saved_figs.append(p2)

    # 3. Top 20 Feature Importances (Gini Impurity from Random Forest)
    fig, ax = plt.subplots(figsize=(10, 8))
    top_feats = feature_importance_df.sort_values("gini_importance", ascending=False).head(20).sort_values("gini_importance", ascending=True)
    colors = ["#e74c3c" if p else "#3498db" for p in top_feats["is_priority_sensor"]]

    ax.barh(top_feats["feature_name"], top_feats["gini_importance"], color=colors)
    ax.set_xlabel("Gini Impurity Importance (Random Forest)")
    ax.set_title("Top 20 Predictive Features (Red = Priority Hydraulic Channel: SE, FS1, EPS1, PS3)")
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    p3 = out_dir / "feature_importance.png"
    plt.tight_layout()
    plt.savefig(p3, dpi=150)
    plt.close(fig)
    saved_figs.append(p3)

    # 4. Unstable Cycles Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        unstable_cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_title(f"Sensitivity Matrix: {best_model_name} on 756 Unstable Cycles")
    ax.set_xlabel("Predicted Condition")
    ax.set_ylabel("Actual Ground Truth")

    p4 = out_dir / "stable_vs_unstable_sensitivity.png"
    plt.tight_layout()
    plt.savefig(p4, dpi=150)
    plt.close(fig)
    saved_figs.append(p4)

    print(f"[train] Generated {len(saved_figs)} visualization figures in: {out_dir}")
    return saved_figs


# ---------------------------------------------------------------------------
# End-to-End Orchestrator
# ---------------------------------------------------------------------------

def run_full_experiment_suite(
    processed_dir: Path = PROCESSED_DIR,
    results_dir: Path = RESULTS_DIR,
    models_dir: Path = MODELS_DIR,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Orchestrate the complete Step 4 Machine Learning experiment suite:
    1. Primary Experiment: Stable-cycle 70/15/15 classification
    2. 5-Fold Stratified & Group-Aware Cross-Validation
    3. Hyperparameter tuning & model selection
    4. Held-out test set evaluation (strictly once)
    5. Feature importance computation
    6. Secondary Experiment: Full-cycle classification
    7. Unstable-cycle sensitivity test
    8. Artifact serialization & results export

    Returns
    -------
    dict
        Complete summary of experimental results.
    """
    from preprocessing import split_cycle_data
    from feature_engineering import load_processed_dataset

    eval_dir = results_dir / "evaluation"
    viz_dir = results_dir / "visualizations"
    eval_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SmartPump AI - Step 4 Machine Learning Engine Execution")
    print("=" * 70)

    # 1. Load Primary Stable Dataset
    X_stable, y_stable, meta_stable = load_processed_dataset(processed_dir, dataset_type="stable")
    feature_names = list(X_stable.columns)

    # 2. Cycle-Level 70/15/15 Split
    splits = split_cycle_data(
        X_stable,
        y_stable,
        metadata=meta_stable,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        random_state=random_state,
        stratify=True,
    )
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    print(f"[train] Dataset Partitions: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Compute experimental block IDs for Group-Aware validation
    full_meta = pd.read_csv(processed_dir / "full_metadata.csv")
    leak_blocks_full = (full_meta["target"] != full_meta["target"].shift()).cumsum()
    stable_blocks = leak_blocks_full.loc[full_meta["is_stable"]].values
    train_blocks = stable_blocks[splits["indices"]["train"]]

    # 3. Baseline Model Pipelines & 5-Fold Cross-Validation on Train
    base_models = build_model_pipelines(random_state=random_state)
    cv_stratified = run_cross_validation(base_models, X_train, y_train, n_splits=5, random_state=random_state)
    cv_stratified.to_csv(eval_dir / "cross_validation_results.csv", index=False)

    cv_grouped = run_cross_validation(base_models, X_train, y_train, n_splits=5, groups=train_blocks, random_state=random_state)
    cv_grouped.to_csv(eval_dir / "cross_validation_grouped_results.csv", index=False)

    # 4. Hyperparameter Tuning
    tuned_models, hyperparam_df = tune_hyperparameters(X_train, y_train, random_state=random_state)
    hyperparam_df.to_csv(eval_dir / "hyperparameter_results.csv", index=False)

    # 5. Validation Set Evaluation
    val_comparison, val_reports, fitted_pipelines = train_and_evaluate_models(
        tuned_models, X_train, y_train, X_val, y_val, split_name="Validation"
    )
    val_comparison.to_csv(eval_dir / "model_comparison.csv", index=False)

    # 6. Model Selection
    # Select candidate with highest Macro F1 and highest Fault Recall on Validation
    best_model_name = val_comparison.sort_values(
        by=["fault_recall", "macro_f1", "fault_precision"], ascending=False
    ).iloc[0]["model"]
    print(f"\n[train] *** SELECTED BEST MODEL: {best_model_name} ***")

    # 7. Final Held-Out Test Evaluation (Refit on Train + Val, Test exactly once)
    print(f"[train] Refitting {best_model_name} on combined Train + Validation ({len(X_train) + len(X_val)} cycles)...")
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)

    final_best_pipeline = tuned_models[best_model_name]
    final_best_pipeline.fit(X_train_val, y_train_val)

    y_test_pred = final_best_pipeline.predict(X_test)
    test_metrics = evaluate_predictions(y_test, y_test_pred, model_name=best_model_name)

    print(f"\n=== FINAL HELD-OUT TEST EVALUATION ({best_model_name}) ===")
    print(test_metrics["classification_report_text"])
    print("Confusion Matrix:\n", test_metrics["confusion_matrix"])

    # 8. Feature Importance
    feat_imp_df = compute_feature_importance(final_best_pipeline, feature_names, X_val, y_val, top_n=30)
    feat_imp_df.to_csv(eval_dir / "feature_importance.csv", index=False)

    # 9. Secondary Experiment: Full-Cycle Classification
    full_cycle_df = run_secondary_full_cycle_experiment(random_state=random_state)
    full_cycle_df.to_csv(eval_dir / "full_cycle_comparison.csv", index=False)

    # 10. Unstable-Cycle Sensitivity Test
    unstable_res = run_unstable_sensitivity_test(final_best_pipeline, processed_dir=processed_dir)
    pd.DataFrame([{
        "n_unstable_cycles": unstable_res["n_unstable_cycles"],
        "accuracy": unstable_res["accuracy"],
        "macro_f1": unstable_res["macro_f1"],
        "fault_recall": unstable_res["fault_recall"],
        "fault_precision": unstable_res["fault_precision"],
        "warning_f1": unstable_res["warning_f1"],
        "normal_f1": unstable_res["normal_f1"],
    }]).to_csv(eval_dir / "unstable_sensitivity_results.csv", index=False)

    # 11. Save Visualizations
    generate_visualizations(
        model_comparison_df=val_comparison,
        best_cm=test_metrics["confusion_matrix"],
        best_model_name=best_model_name,
        feature_importance_df=feat_imp_df,
        unstable_cm=unstable_res["confusion_matrix"],
        out_dir=viz_dir,
    )

    # 12. Save Model Artifacts & Detailed Metadata
    model_metadata = {
        "model_name": best_model_name,
        "algorithm": str(final_best_pipeline.named_steps["clf"].__class__.__name__),
        "target_mapping": {0: "Normal", 1: "Warning", 2: "Fault"},
        "random_state": random_state,
        "training_cycles": len(X_train_val),
        "test_cycles": len(X_test),
        "num_features_input": len(feature_names),
        "validation_macro_f1": float(val_comparison.loc[val_comparison["model"] == best_model_name, "macro_f1"].iloc[0]),
        "validation_fault_recall": float(val_comparison.loc[val_comparison["model"] == best_model_name, "fault_recall"].iloc[0]),
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_fault_recall": test_metrics["fault_recall"],
        "test_fault_precision": test_metrics["fault_precision"],
        "test_fault_f1": test_metrics["fault_f1"],
        "unstable_macro_f1": unstable_res["macro_f1"],
        "unstable_fault_recall": unstable_res["fault_recall"],
        "sklearn_version": sklearn_version_str(),
        "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    save_model_artifacts(final_best_pipeline, model_metadata, models_dir=models_dir)

    print("\n[train] Full Experiment Suite Finished Successfully!")
    return {
        "best_model_name": best_model_name,
        "val_comparison": val_comparison,
        "test_metrics": test_metrics,
        "unstable_metrics": unstable_res,
        "feature_importance": feat_imp_df,
    }


def sklearn_version_str() -> str:
    import sklearn
    return sklearn.__version__


# ---------------------------------------------------------------------------
# Legacy Backward-Compatibility API
# ---------------------------------------------------------------------------

def build_models() -> Dict[str, Pipeline]:
    """Legacy alias preserved for notebook 03 import."""
    return build_model_pipelines(random_state=RANDOM_STATE)


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Legacy cycle-level split preserved for backward compatibility."""
    from preprocessing import split_cycle_data
    val_ratio = test_size / 2.0
    splits = split_cycle_data(
        X, y, train_size=(1.0 - test_size), val_size=val_ratio, test_size=val_ratio, random_state=random_state
    )
    X_test_combined = pd.concat([splits["X_val"], splits["X_test"]], axis=0)
    y_test_combined = pd.concat([splits["y_val"], splits["y_test"]], axis=0)
    return splits["X_train"], X_test_combined, splits["y_train"], y_test_combined


def evaluate_model(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Legacy evaluate_model wrapper."""
    res = evaluate_predictions(y_true, y_pred, model_name=model_name)
    return {
        "model": model_name,
        "accuracy": res["accuracy"],
        "macro_f1": res["macro_f1"],
        "fault_recall": res["fault_recall"],
        "fault_precision": res["fault_precision"],
        "fault_f1": res["fault_f1"],
        "warning_f1": res["warning_f1"],
    }


def save_model(pipeline: Pipeline, model_name: str) -> Path:
    """Legacy save_model wrapper."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / f"{model_name}.joblib"
    joblib.dump(pipeline, out)
    return out


def save_results(results: List[Dict[str, float]]) -> Path:
    """Legacy save_results wrapper."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "model_comparison.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    return out


if __name__ == "__main__":
    run_full_experiment_suite()
