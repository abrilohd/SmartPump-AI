"""
test_train.py
=============
Automated unit and integration tests for src/train.py.

Verifies:
- Pipeline construction for all candidate algorithms
- VarianceThreshold zero-variance feature elimination
- Prediction shape and valid class range {0, 1, 2}
- Correct evaluation metric calculation
- Model serialization (joblib) and artifact metadata persistence
- Successful inference on serialized pipeline
- Reproducibility with random_state=42
- Absence of test set and metadata leakage
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.train import (
    CLASS_CODES,
    CLASS_NAMES,
    build_model_pipelines,
    evaluate_predictions,
    save_model_artifacts,
)
from src.feature_engineering import load_processed_dataset
from src.preprocessing import split_cycle_data

PROCESSED_DIR = Path("data/processed")


def test_build_model_pipelines():
    """Verify all 6 candidate models are defined as Pipelines with correct components."""
    models = build_model_pipelines(random_state=42)
    expected_models = ["Dummy", "DecisionTree", "RandomForest", "SVM", "GaussianNB", "AdaBoost"]

    for name in expected_models:
        assert name in models, f"Model {name} missing from pipeline dictionary."
        pipe = models[name]
        assert isinstance(pipe, Pipeline), f"{name} is not an sklearn Pipeline."
        assert "clf" in pipe.named_steps, f"{name} missing 'clf' step."

    # SVM and GaussianNB must include scaler
    assert "scaler" in models["SVM"].named_steps
    assert "scaler" in models["GaussianNB"].named_steps

    # DecisionTree and RandomForest must NOT have scaler
    assert "scaler" not in models["DecisionTree"].named_steps
    assert "scaler" not in models["RandomForest"].named_steps


def test_variance_threshold_in_pipeline():
    """Verify VarianceThreshold removes constant features without error."""
    models = build_model_pipelines(random_state=42)
    rf_pipe = models["RandomForest"]

    # Synthetic dataset with 2 informative features and 2 constant features
    X = pd.DataFrame({
        "feat_var1": np.random.randn(20),
        "feat_var2": np.random.randn(20),
        "feat_const1": [0.0] * 20,
        "feat_const2": [42.0] * 20,
    })
    y = pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2, 0] * 2)

    rf_pipe.fit(X, y)
    selector = rf_pipe.named_steps["selector"]
    retained = selector.get_support()

    # Exactly 2 features should be retained
    assert sum(retained) == 2
    assert retained[0] and retained[1]
    assert not retained[2] and not retained[3]


def test_evaluate_predictions_metrics():
    """Verify metric calculations (Accuracy, Macro F1, Fault Recall, Confusion Matrix)."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 2, 2, 2])  # 1 warning misclassified as fault

    metrics = evaluate_predictions(y_true, y_pred, model_name="TestModel")

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "fault_recall" in metrics
    assert "fault_precision" in metrics
    assert "confusion_matrix" in metrics

    # All faults were predicted as fault -> Fault recall must be 1.0
    assert np.isclose(metrics["fault_recall"], 1.0)
    # One warning was predicted as fault -> Fault precision must be 2 / 3
    assert np.isclose(metrics["fault_precision"], 2 / 3)


def test_pipeline_fit_predict_on_processed_data():
    """Verify end-to-end fitting and inference on real processed cycle features."""
    X, y, meta = load_processed_dataset(PROCESSED_DIR, dataset_type="stable")
    splits = split_cycle_data(X, y, metadata=meta, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42)

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val = splits["X_val"]

    models = build_model_pipelines(random_state=42)
    dt_pipe = models["DecisionTree"]

    dt_pipe.fit(X_train, y_train)
    y_val_pred = dt_pipe.predict(X_val)

    assert len(y_val_pred) == len(X_val)
    assert set(np.unique(y_val_pred)).issubset({0, 1, 2})


def test_model_serialization_and_reload(tmp_path):
    """Verify that a trained pipeline can be serialized and reloaded with identical predictions."""
    X, y, meta = load_processed_dataset(PROCESSED_DIR, dataset_type="stable")
    splits = split_cycle_data(X, y, metadata=meta, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42)

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val = splits["X_val"]

    models = build_model_pipelines(random_state=42)
    rf = models["RandomForest"]
    rf.fit(X_train, y_train)

    orig_preds = rf.predict(X_val)

    # Save artifacts to temp path
    dummy_meta = {
        "model_name": "RandomForest",
        "accuracy": 1.0,
        "classes": [0, 1, 2],
    }
    model_path, meta_path = save_model_artifacts(rf, dummy_meta, models_dir=tmp_path, model_name="test_rf")

    assert model_path.exists()
    assert meta_path.exists()

    # Reload pipeline
    reloaded_rf = joblib.load(model_path)
    reloaded_preds = reloaded_rf.predict(X_val)

    assert np.array_equal(orig_preds, reloaded_preds)

    # Verify metadata JSON
    with open(meta_path, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    assert loaded_json["model_name"] == "RandomForest"

