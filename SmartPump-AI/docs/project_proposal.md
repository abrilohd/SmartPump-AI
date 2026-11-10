# SmartPump AI — Technical Project Proposal

**Project Title:** Intelligent Hydraulic Pump Condition Monitoring and Fault Classification Using Machine Learning  
**Version:** 1.0  
**Date:** November 2026  
**Status:** Foundation Phase — Awaiting Dataset Download

---

## 1. Executive Summary

SmartPump AI applies supervised machine learning to automatically classify the internal pump leakage condition of an industrial hydraulic system into three severity classes — **Normal**, **Warning**, and **Fault** — using real sensor time-series data from the UCI Condition Monitoring of Hydraulic Systems dataset.

The dataset was collected by Helwig, Pignanelli, and Schütze at ZeMA gGmbH, Saarbrücken, Germany, and published on the UCI Machine Learning Repository under a CC BY 4.0 license. It contains 2,205 experimental 60-second cycles with measurements from 17 sensor channels at sampling rates between 1 Hz and 100 Hz.

The project is designed for a three-person engineering team and emphasizes correctness, reproducibility, and leakage-free evaluation.

---

## 2. Problem Statement

Internal pump leakage is a progressive degradation mode in hydraulic systems that reduces system efficiency and, if undetected, leads to catastrophic failure. Early and automatic detection of this condition enables:

- Predictive maintenance instead of fixed-schedule or reactive repair.
- Reduced unplanned downtime.
- Improved safety in industrial operations.

The technical challenge is building a classification pipeline that:
1. Correctly identifies which target column in the dataset corresponds to pump leakage.
2. Handles the multi-rate, high-volume time-series structure of the sensor data.
3. Avoids temporal data leakage between training and testing cycles.
4. Reports evaluation metrics that are meaningful for an imbalanced three-class problem.

---

## 3. Dataset Description

### 3.1 Source

| Property | Detail |
|----------|--------|
| Repository | UCI Machine Learning Repository |
| Dataset ID | 447 |
| DOI | https://doi.org/10.24432/C5CW21 |
| URL | https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems |
| Authors | Helwig, N., Pignanelli, E., Schütze, A. |
| Year | 2015 |
| License | CC BY 4.0 |

### 3.2 Structure

The dataset contains **17 raw sensor text files** plus **one target file** (`profile.txt`), all in tab-delimited format.

- **Rows:** 2,205 (one per experimental cycle)
- **Each cycle:** 60 seconds of continuous hydraulic operation under constant load
- **Columns per file:** Determined by the sensor's sampling rate × 60 seconds

### 3.3 Sensor Channels

| File(s) | Quantity | Unit | Rate | Cols/Cycle |
|---------|----------|------|------|------------|
| PS1–PS6 | Pressure | bar | 100 Hz | 6,000 |
| EPS1 | Motor power | W | 100 Hz | 6,000 |
| FS1, FS2 | Volume flow | l/min | 10 Hz | 600 |
| TS1–TS4 | Temperature | °C | 1 Hz | 60 |
| VS1 | Vibration | mm/s | 1 Hz | 60 |
| CE | Cooling efficiency (virtual) | % | 1 Hz | 60 |
| CP | Cooling power (virtual) | kW | 1 Hz | 60 |
| SE | Efficiency factor (virtual) | % | 1 Hz | 60 |

### 3.4 Target Labels — `profile.txt`

`profile.txt` has 2,205 rows and 5 columns. **Column indices are 0-based in code.**

| Column | Component | Values |
|--------|-----------|--------|
| 0 | Cooler condition (%) | 3, 20, 100 |
| 1 | Valve condition (%) | 73, 80, 90, 100 |
| **2** | **Internal pump leakage** ← **our target** | **0, 1, 2** |
| 3 | Hydraulic accumulator (bar) | 90, 100, 115, 130 |
| 4 | Stable flag | 0 (stable), 1 (not fully stable) |

### 3.5 Pump Leakage Class Definitions

| Raw Value | Condition | SmartPump Label |
|-----------|-----------|-----------------|
| `0` | No internal leakage | **Normal** |
| `1` | Weak internal leakage (reduced efficiency) | **Warning** |
| `2` | Severe internal leakage | **Fault** |

> **Note:** The other four columns describe the cooler, valve, accumulator, and system stability — not pump leakage. They must not be conflated with our classification target.

---

## 4. Classification Task

### 4.1 Task Type

Supervised, multi-class classification (3 classes) over cycle-level feature vectors derived from raw sensor time-series.

### 4.2 Input Feature Space

The raw data is **not** directly usable as a feature matrix because:
- Different sensors have different sampling rates (1 Hz to 100 Hz).
- Each cycle produces tens of thousands of data points, not a fixed-width feature vector.
- Treating raw samples as independent observations would cause severe data leakage.

**Solution:** Per-cycle statistical feature extraction — reduce each 60-second sensor time series to a small set of summary statistics:
- Mean, standard deviation, minimum, maximum, median
- Optionally: RMS, kurtosis, skewness, interquartile range

This yields a **single feature row per cycle**, ensuring each of the 2,205 cycles is an independent observation.

### 4.3 Output

A three-class label: `0` (Normal), `1` (Warning), `2` (Fault).

---

## 5. Candidate Algorithms

| Algorithm | Library | Key Hyperparameter(s) |
|-----------|---------|----------------------|
| Decision Tree | `sklearn.tree.DecisionTreeClassifier` | `max_depth`, `min_samples_leaf` |
| Random Forest | `sklearn.ensemble.RandomForestClassifier` | `n_estimators`, `max_depth`, `class_weight` |
| Support Vector Machine | `sklearn.svm.SVC` | `C`, `kernel`, `gamma`, `class_weight` |
| Gaussian Naïve Bayes | `sklearn.naive_bayes.GaussianNB` | `var_smoothing` |
| AdaBoost | `sklearn.ensemble.AdaBoostClassifier` | `n_estimators`, `learning_rate` |

All models use `random_state=42` for reproducibility.

---

## 6. Validation Strategy

### 6.1 Train/Test Split

- **Split unit:** Cycle (row in `profile.txt`), never individual sensor samples.
- **Ratio:** 80% train / 20% test using `StratifiedShuffleSplit` to preserve class proportions.
- **Stratification:** On the pump leakage label (column 2 of `profile.txt`).
- **No cross-contamination:** A cycle used in test is never represented in train.

### 6.2 Stable Flag Handling

Cycles where `profile.txt` column 4 equals `1` indicate the system had not yet reached steady state. Strategy:
- **Primary evaluation:** Filter these cycles out (use only `stable_flag == 0`).
- **Sensitivity analysis:** Repeat evaluation with all cycles included and report the difference.

### 6.3 Cross-Validation

For hyperparameter search, use `StratifiedKFold(n_splits=5)` applied to the training set only.

---

## 7. Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| **Fault-class Recall** | Critical: how many real faults are caught |
| **Fault-class Precision** | Critical: how many fault alerts are real |
| **Warning-class F1** | Tracks performance on the hardest minority class |
| **Macro F1-score** | Overall quality across all three classes equally |
| **Confusion Matrix** | Full error analysis across all class pairs |
| **Classification Report** | Per-class precision, recall, F1 |

**Overall accuracy is reported but is NOT the primary metric.**

---

## 8. Technical Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **Data leakage** — mixing same-cycle samples in train/test | 🔴 High | Feature extraction aggregates whole cycles; split at cycle level |
| **Label confusion** — using wrong `profile.txt` column | 🔴 High | Explicitly load column 2 (0-indexed); verify unique values are {0, 1, 2} |
| **Class imbalance** — Warning class underrepresented | 🟡 Medium | Report macro F1; use `class_weight='balanced'` for SVM/RF; evaluate per-class |
| **Stable flag** — transient cycles may distort features | 🟡 Medium | Filter `stable_flag==1` for primary run; include in sensitivity analysis |
| **Multi-rate features** — naïve concatenation of different-rate sensors | 🟡 Medium | Extract statistics per sensor independently; align at cycle level |
| **Overfitting** — deep Decision Trees or complex SVMs | 🟢 Low | Cross-validate; tune `max_depth` and regularization |

---

## 9. Project Deliverables

| Phase | Deliverable | Assigned To |
|-------|-------------|-------------|
| Phase 1 | Project foundation — README, proposal, data docs | Member C |
| Phase 2 | Dataset download, EDA notebook, label verification | Member A |
| Phase 3 | Feature extraction pipeline — `feature_engineering.py` | Member A |
| Phase 4 | Preprocessing pipeline — `preprocessing.py` | Member A |
| Phase 5 | Model training — `train.py`, `03_model_training.ipynb` | Member B |
| Phase 6 | Evaluation report — confusion matrices, feature importance | Member B |
| Phase 7 | Final integration, reproducibility audit, documentation | Member C |

---

## 10. Reproducibility Requirements

- Python version: 3.9+
- All random states: `random_state=42`
- All dependencies pinned in `requirements.txt`
- Raw data files stored in `data/raw/` and excluded from Git via `.gitignore`
- Processed features stored in `data/processed/` and excluded from Git
- Notebooks committed with **cleared outputs**
- One `README.md` documents the full reproduction procedure

---

## 11. References

1. Helwig, N., Pignanelli, E., & Schütze, A. (2015). *Condition monitoring of hydraulic systems* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CW21

2. Helwig, N., Pignanelli, E., & Schütze, A. (2015). Condition monitoring of a complex hydraulic system using multivariate statistics. *2015 IEEE International Instrumentation and Measurement Technology Conference (I2MTC)*, Pisa, Italy, pp. 210–215. doi:10.1109/I2MTC.2015.7151267

---

*This proposal is a living document. Update it as technical decisions are made during implementation.*
