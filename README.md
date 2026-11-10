# SmartPump AI 🔧

**Intelligent Hydraulic Pump Condition Monitoring and Internal Leakage Classification Using Machine Learning**

*Classwork 3.12: Group Machine Learning Project*
## Team Members

| Full Name | Student ID | GitHub Username |
|---|---|---|
| Abrham Gebremedhin | WOUR/0082/16 | [@abrilohd](https://github.com/abrilohd) |
| Israel Abera | WOUR/4167/16 | [@Isru96](https://github.com/Isru96) |
| Biruk Ayitenew | WOUR/0483/16 | [@Biruk1997](https://github.com/Biruk1997) |

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/Dataset%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Dataset Source & Description](#4-dataset-source--description)
5. [Condition Target Labels](#5-condition-target-labels)
6. [Methodology & Pipeline Architecture](#6-methodology--pipeline-architecture)
7. [Required Algorithms](#7-required-algorithms)
8. [Evaluation Strategy & Metrics](#8-evaluation-strategy--metrics)
9. [Repository Structure](#9-repository-structure)
10. [Installation & Execution Instructions](#10-installation--execution-instructions)
11. [Actual Experimental Results](#11-actual-experimental-results)
12. [Team Collaboration & Workflow](#12-team-collaboration--workflow)
13. [Limitations & Engineering Safety Disclaimer](#13-limitations--engineering-safety-disclaimer)

---

## 1. Project Overview

**SmartPump AI** is a collaborative machine learning project for **Classwork 3.12** that analyzes high-frequency multivariate sensor streams from an industrial hydraulic system to automatically classify the **internal pump leakage condition** into three distinct operational states: **Normal**, **Warning**, and **Fault**.

The project is built on the benchmark **UCI Condition Monitoring of Hydraulic Systems** dataset, collected from a physical industrial hydraulic test rig at ZeMA gGmbH (Saarbrücken, Germany). The system demonstrates a rigorous end-to-end machine learning engineering workflow: from multi-rate time-series ingestion and leak-free feature extraction to multi-algorithm comparison, hyperparameter optimization, and generalization auditing.

---

## 2. Problem Statement

Hydraulic pumps are mission-critical components in industrial manufacturing, heavy machinery, and aerospace systems. Internal pump leakage caused by mechanical wear between piston and cylinder surfaces leads to severe volumetric efficiency loss, oil overheating, pressure drops, and eventual catastrophic pump seizure.

Traditional industrial maintenance strategies rely on reactive repair (after failure has occurred) or rigid calendar-based maintenance schedules. **SmartPump AI** provides condition-based monitoring by:
- Ingesting high-frequency time series (up to 100 Hz across 17 sensor channels) over recurring 60-second operational cycles.
- Transforming 43,680 raw measurements per cycle into structured, physically meaningful statistical features.
- Distinguishing subtle incipient leakage (**Warning**) from severe failure (**Fault**) without data leakage between training and evaluation cycles.

---

## 3. Objectives

| # | Objective | Deliverable |
|---|---|---|
| **1** | **Target Extraction & Label Mapping** | Extract pump leakage states from `profile.txt` and map to Normal (0), Warning (1), and Fault (2). |
| **2** | **Exploratory Data Analysis (EDA)** | Analyze multi-rate sensor distributions, stable/unstable operational flags, and inter-sensor correlations. |
| **3** | **Cycle-Level Feature Engineering** | Extract 15 statistical, distribution, and temporal features per sensor (255 total per cycle) with zero-variance safety. |
| **4** | **Leakage-Free Cross-Validation** | Implement cycle-level stratified and group-aware cross-validation across experimental blocks. |
| **5** | **Multi-Algorithm Comparison** | Implement, tune, and compare the 5 required algorithms (Decision Tree, Random Forest, SVM, Naïve Bayes, AdaBoost) plus a baseline. |
| **6** | **Fault Recall Optimization** | Prioritize the detection of severe internal leakage (`Fault` recall) to prevent unmonitored equipment failure. |
| **7** | **Team Collaboration** | Maintain a clean, reproducible repository with clear Git workflows for a three-student team. |

---

## 4. Dataset Source & Description

| Property | Value |
|---|---|
| **Dataset Name** | Condition Monitoring of Hydraulic Systems |
| **Source** | UCI Machine Learning Repository (ID: 447) |
| **DOI** | [10.24432/C5CW21](https://doi.org/10.24432/C5CW21) |
| **Origin** | ZeMA gGmbH, Saarbrücken, Germany (Helwig et al., 2015) |
| **License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Cycle Duration** | 60 seconds per cycle (repetitive constant load profile) |
| **Total Observations** | 2,205 operating cycles (1,449 stable cycles, 756 unstable cycles) |
| **Raw Attributes** | 43,680 data points per cycle across 17 sensor channels |

### 17 Sensor Channels by Sampling Rate

| Sensor Group | Sensors | Sampling Rate | Points / Cycle | Quantity & Unit |
|---|---|---|---|---|
| **Pressure** | `PS1` – `PS6` | 100 Hz | 6,000 | Hydraulic pressure (bar) |
| **Motor Power** | `EPS1` | 100 Hz | 6,000 | Electrical motor power (W) |
| **Volume Flow** | `FS1`, `FS2` | 10 Hz | 600 | Volume flow rate (l/min) |
| **Temperature** | `TS1` – `TS4` | 1 Hz | 60 | Oil temperature (°C) |
| **Vibration** | `VS1` | 1 Hz | 60 | Mechanical vibration (mm/s) |
| **Virtual Sensors** | `CE`, `CP`, `SE` | 1 Hz | 60 | Cooling efficiency (%), Cooling power (kW), Efficiency factor (%) |

The ground-truth conditions are annotated in `profile.txt` (column 3: internal pump leakage; column 5: system stability flag).

---

## 5. Condition Target Labels

The dataset's physical pump leakage states are mapped into three integer target classes:

| Class Code | Label | Physical Condition | Raw Profile Value | Stable Population Count | Full Population Count |
|:---:|:---:|:---|:---:|:---:|:---:|
| **0** | **Normal** | No internal pump leakage (full efficiency) | `0` | 489 cycles (33.7%) | 1,221 cycles (55.4%) |
| **1** | **Warning** | Weak internal pump leakage (reduced efficiency) | `1` | 480 cycles (33.1%) | 492 cycles (22.3%) |
| **2** | **Fault** | Severe internal pump leakage (critical degradation) | `2` | 480 cycles (33.1%) | 492 cycles (22.3%) |
| **Total** | | | | **1,449 cycles** | **2,205 cycles** |

---

## 6. Methodology & Pipeline Architecture

```text
Raw Multi-Rate Sensors (17 files, 43,680 pts/cycle)
                        │
                        ▼
       [ Multi-Rate Alignment & Stable Filtering ]
           (1,449 stable cycles, profile.txt)
                        │
                        ▼
       [ 255 Cycle-Level Feature Extraction ]
      (15 statistical/distribution/temporal features per sensor)
                        │
                        ▼
       [ Cycle-Aware Stratified Train/Val/Test Split ]
             Train: 1,014 | Val: 217 | Test: 218
                        │
                        ▼
      [ Scikit-Learn Leakage-Free Pipeline ]
          Step 1: VarianceThreshold(0.0)
          Step 2: StandardScaler (fitted on train only)
          Step 3: Classifier (5 algorithms evaluated)
                        │
                        ▼
       [ 5-Fold Stratified & 36-Block Grouped CV ]
                        │
                        ▼
  [ Final Evaluation on Held-Out Test & 756 Unstable Cycles ]
```

### Feature Engineering (255 Features)
For each 60-second operating cycle, 15 statistical features are extracted per sensor channel:
1. **Central Tendency:** Mean, Median
2. **Dispersion:** Standard Deviation, Interquartile Range (IQR), Min, Max
3. **Distribution Shape:** Skewness, Kurtosis, 25th Percentile, 75th Percentile
4. **Energy & Power:** Energy, Power, Absolute Energy
5. **Dynamics & Drift:** Linear Slope, Net Delta ($\text{end} - \text{start}$)

---

## 7. Required Algorithms

In compliance with Classwork 3.12 requirements, five distinct supervised classification algorithms were implemented and rigorously compared alongside a baseline:

1. **Dummy Classifier (Baseline):** Stratified random guessing baseline.
2. **Decision Tree Classifier:** Non-parametric recursive splitting model (`max_depth=5`, `min_samples_split=5`).
3. **Random Forest Classifier:** Bagged ensemble of 100 decorrelated decision trees (`n_estimators=100`, `max_depth=10`).
4. **Support Vector Machine (SVM):** Kernel-based maximum margin classifier with Radial Basis Function (`C=10.0`, `gamma='scale'`).
5. **Gaussian Naïve Bayes:** Probabilistic classifier modeling normal class-conditional feature distributions.
6. **AdaBoost Classifier:** Sequential adaptive boosting ensemble of decision stumps (`n_estimators=100`, `learning_rate=0.5`).

---

## 8. Evaluation Strategy & Metrics

To prioritize industrial safety, the evaluation framework emphasizes early and accurate detection of critical pump degradation:

- **Fault Recall ($\text{Recall}_2$):** $\frac{\text{TP}_2}{\text{TP}_2 + \text{FN}_2}$ — critical metric measuring the proportion of severe leakage cycles correctly identified (minimizing catastrophic false negatives).
- **Fault Precision ($\text{Precision}_2$):** $\frac{\text{TP}_2}{\text{TP}_2 + \text{FP}_2}$ — proportion of declared faults that are genuine.
- **Macro-Averaged F1-Score:** Unweighted mean of F1-scores across all three classes, giving equal importance to Normal, Warning, and Fault.
- **Multi-Class Confusion Matrix:** Cycle-by-cycle distribution of true versus predicted classes.
- **Classification Report:** Detailed per-class precision, recall, F1-score, and support.
- **Group-Aware Cross-Validation:** Stratified grouped 5-fold CV across 36 contiguous experimental blocks to detect thermal and sensor drift across separate test runs.

---

## 9. Repository Structure

```text
SmartPump-AI/
│
├── data/
│   ├── raw/                    ← Original sensor matrices (*.txt, excluded from git)
│   ├── processed/              ← Engineered cycle features (stable & full datasets)
│   └── README.md               ← Dataset acquisition and schema documentation
│
├── docs/
│   ├── project_proposal.md     ← Detailed technical proposal and objectives
│   ├── data_exploration_report.md ← Comprehensive EDA report with sensor statistics
│   ├── preprocessing_report.md ← Preprocessing and feature engineering specifications
│   ├── model_training_report.md← Detailed evaluation report across all 5 algorithms
│   ├── technical_risks.md      ← Risk assessment register and mitigation strategies
│   └── team_workflow.md        ← Team organization, branch rules, and PR guidelines
│
├── notebooks/
│   ├── 01_data_exploration.ipynb ← EDA, distributions, correlation, stability analysis
│   ├── 02_preprocessing.ipynb    ← Feature extraction, scaling, and dataset creation
│   └── 03_model_training.ipynb   ← Pipeline building, cross-validation, and evaluation
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        ← Loading, validation, filtering, and splitters
│   ├── feature_engineering.py  ← 255 statistical feature extraction engine
│   └── train.py                ← Pipeline definitions, CV, grid search, and metrics
│
├── tests/
│   ├── __init__.py
│   ├── test_feature_engineering.py ← Tests for statistical calculations and safety
│   ├── test_preprocessing.py       ← Tests for loading, disjoint splits, and scaling
│   └── test_train.py               ← Tests for pipeline fit/predict and metrics
│
├── models/
│   ├── .gitkeep
│   ├── smartpump_best_model.joblib ← Serialized champion Random Forest pipeline
│   └── model_metadata.json         ← Model parameters, feature ordering, and scores
│
├── results/
│   ├── eda/                    ← EDA statistical tables and correlation matrices
│   ├── evaluation/             ← Model comparison CSVs, CV scores, and feature importances
│   └── visualizations/         ← Confusion matrices, ROC curves, and importance charts
│
├── .github/
│   └── PULL_REQUEST_TEMPLATE.md← Standardized PR checklist for peer review
│
├── .gitignore                  ← Git exclusion rules (raw data, joblib models, caches)
├── CONTRIBUTING.md             ← Step-by-step Git workflow instructions for students
├── README.md                   ← Master project documentation
└── requirements.txt            ← Pinned Python dependencies
```

---

## 10. Installation & Execution Instructions

### Prerequisites
- Python 3.9+ (tested on Python 3.9 – 3.14)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/abrilohd/SmartPump-AI.git
cd SmartPump-AI
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Raw Dataset
Follow instructions in [`data/README.md`](data/README.md) to place raw sensor text files into `data/raw/`.

### 5. Run the End-to-End Pipeline
The project can be executed sequentially via Jupyter Notebooks:
1. `notebooks/01_data_exploration.ipynb` — Visualizes raw sensor dynamics and condition profiles.
2. `notebooks/02_preprocessing.ipynb` — Extracts 255 features and generates processed CSVs.
3. `notebooks/03_model_training.ipynb` — Trains all 5 classifiers and produces comparison tables.

### 6. Run the Automated Test Suite
```bash
python -m pytest tests/ -v
```
*(All 16 core unit and integration tests passing: 5 feature engineering, 6 preprocessing, 5 model training).*

---

## 11. Actual Experimental Results

### Held-Out Test Set Performance (218 Cycles, Stable Population)

All models were evaluated on the exact same held-out test partition (218 cycles: 74 Normal, 72 Warning, 72 Fault):

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Fault Precision | Fault Recall | Fault F1-Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (Champion)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **SVM (RBF Kernel)** | 0.9908 | 0.9912 | 0.9907 | 0.9908 | 0.9863 | **1.0000** | 0.9931 |
| **Decision Tree** | 0.9862 | 0.9866 | 0.9861 | 0.9862 | 0.9861 | 0.9861 | 0.9861 |
| **AdaBoost** | 0.8303 | 0.8354 | 0.8282 | 0.8267 | 0.8806 | 0.8194 | 0.8489 |
| **Gaussian Naïve Bayes** | 0.8257 | 0.8351 | 0.8239 | 0.8248 | 0.9492 | 0.7778 | 0.8550 |
| **Dummy (Baseline)** | 0.3440 | 0.3456 | 0.3447 | 0.1706 | 0.0000 | 0.0000 | 0.0000 |

### Top Physical Predictors
Permutation importance and Gini impurity analysis revealed that internal leakage is governed primarily by four physical sensor channels:
1. **`SE` (Efficiency Factor):** Strongest drop during severe internal slippage ($\approx 35\%$ drop).
2. **`FS1` (Volume Flow):** Immediate volumetric delivery reduction during leakage cycles.
3. **`EPS1` (Motor Power):** Electrical motor increases power draw to compensate for pressure loss.
4. **`PS3` (Return Line Pressure):** Elevated return line backpressure from leakage oil returning directly to the tank.

### Group-Aware Validation Across 36 Experimental Blocks
When complete contiguous test blocks are held out using `StratifiedGroupKFold(n_splits=5)`, the Random Forest achieves a Macro F1 of **0.9168 ± 0.0769**, confirming strong generalization while highlighting realistic ambient and thermal drift across separate days of operation.

---

## 12. Team Collaboration & Workflow

SmartPump AI is developed by a team of three students. Responsibilities are partitioned to ensure genuine collaborative contributions across the full machine learning lifecycle:

| Member | Domain | Key Files & Responsibilities |
|---|---|---|
| **Member 1** | Data Engineering & Exploration | Ingestion of 17 raw sensor channels, multi-rate alignment, stable filtering, cycle-level feature extraction (255 features), `01_data_exploration.ipynb`, `02_preprocessing.ipynb`, `src/preprocessing.py`, `src/feature_engineering.py`. |
| **Member 2** | Machine Learning & Evaluation | Implementation of 5 ML algorithms, 5-fold stratified and grouped cross-validation, hyperparameter tuning, metrics evaluation, `03_model_training.ipynb`, `src/train.py`, `results/`. |
| **Member 3** | Integration, QA & Documentation | Project architecture, automated test suite (`tests/`), risk register, technical proposal, root `README.md`, `CONTRIBUTING.md`, `.github/`, peer review coordination. |

For detailed instructions on Git commands, branch conventions, and Pull Requests, see [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`docs/team_workflow.md`](docs/team_workflow.md).

---

## 13. Limitations & Engineering Safety Disclaimer

### Limitations
1. **Physical Test Rig Context:** The dataset was collected under controlled cyclic test-rig conditions with a constant 60-second repetitive cycle. Real-world hydraulic systems may experience variable duty cycles, unmetered external loads, and varying hydraulic fluid viscosities.
2. **Thermal & Day-to-Day Drift:** As demonstrated by the 36-block group cross-validation analysis (Macro F1 = 0.9168), ambient temperature and continuous thermal accumulation introduce minor drift across separate operating days.
3. **Fixed Operational Window:** The model expects complete 60-second operational cycles. Continuous sub-second anomaly detection requires streaming windowing architectures.

### Safety Disclaimer
> [!IMPORTANT]
> **Academic Engineering Disclaimer:** SmartPump AI is an academic machine learning study developed for Classwork 3.12. Model predictions are intended to support engineering analysis and condition-based maintenance scheduling. They do **not** replace physical safety inspections, OEM operational manuals, relief valves, or automated plant emergency shutdown systems.
