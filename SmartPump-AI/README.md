# SmartPump AI 🔧

**Intelligent Hydraulic Pump Condition Monitoring and Fault Classification Using Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/Dataset%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Status](https://img.shields.io/badge/Status-Foundation-orange.svg)]()

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Dataset Source](#4-dataset-source)
5. [Classification Targets](#5-classification-targets)
6. [Label Mapping: Raw Dataset → SmartPump Classes](#6-label-mapping-raw-dataset--smartpump-classes)
7. [Proposed Algorithms](#7-proposed-algorithms)
8. [Evaluation Strategy](#8-evaluation-strategy)
9. [Project Architecture](#9-project-architecture)
10. [Team Collaboration](#10-team-collaboration)
11. [Getting Started](#11-getting-started)
12. [Future Development Roadmap](#12-future-development-roadmap)
13. [Citation and Licensing](#13-citation-and-licensing)

---

## 1. Project Overview

SmartPump AI is a supervised machine learning project that analyzes real industrial hydraulic system sensor measurements to automatically classify the **internal pump leakage condition** into three operational states:

| Class | Label | Meaning |
|-------|-------|---------|
| 0 | **Normal** | No internal pump leakage detected |
| 1 | **Warning** | Weak internal pump leakage — reduced efficiency |
| 2 | **Fault** | Severe internal pump leakage — critical degradation |

The system is built on the **UCI Condition Monitoring of Hydraulic Systems** dataset — a real experimental dataset collected from an industrial hydraulic test rig at ZeMA gGmbH, Saarbrücken, Germany.

---

## 2. Problem Statement

Hydraulic pump failures cause significant industrial downtime and safety risks. Traditional maintenance approaches — either fixed schedules or reactive repairs — are costly and unreliable. 

**SmartPump AI addresses this by:**
- Analyzing raw sensor time-series from real hydraulic machinery.
- Detecting progressive pump degradation (no leakage → weak leakage → severe leakage) before catastrophic failure.
- Providing a reproducible, explainable ML pipeline that engineers can trust and audit.

The core challenge is extracting meaningful per-cycle features from **high-frequency multivariate sensor streams** (up to 100 Hz, 6,000 data points per 60-second cycle), without introducing data leakage between train and test cycles.

---

## 3. Objectives

| # | Objective |
|---|-----------|
| 1 | Correctly identify the pump leakage target column from `profile.txt` and map it to Normal / Warning / Fault |
| 2 | Perform cycle-aware feature extraction — aggregate each 60-second sensor time series into per-cycle statistical features |
| 3 | Train and compare five ML classifiers using a leakage-free, cycle-aware train/test split |
| 4 | Maximize **Fault-class recall** to minimize missed critical faults |
| 5 | Report macro F1-score, per-class precision/recall, and confusion matrices |
| 6 | Maintain a reproducible, well-documented codebase for a three-person team |

---

## 4. Dataset Source

| Property | Value |
|----------|-------|
| **Name** | Condition Monitoring of Hydraulic Systems |
| **Source** | UCI Machine Learning Repository |
| **DOI** | [10.24432/C5CW21](https://doi.org/10.24432/C5CW21) |
| **URL** | https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems |
| **Origin** | ZeMA gGmbH, Saarbrücken, Germany |
| **Authors** | Helwig, N., Pignanelli, E., & Schütze, A. (2015) |
| **License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Total Cycles** | 2,205 experimental 60-second cycles |
| **Sensor Streams** | 17 channels (pressure, flow, temperature, power, vibration, virtual) |

### Sensor Files

Each `.txt` file is a **tab-delimited matrix** with 2,205 rows (cycles) and columns equal to `sampling_rate × 60 seconds`.

| File | Quantity | Unit | Sampling Rate | Columns per Cycle |
|------|----------|------|---------------|-------------------|
| `PS1.txt` – `PS6.txt` | Pressure (6 sensors) | bar | 100 Hz | 6,000 |
| `EPS1.txt` | Motor power | W | 100 Hz | 6,000 |
| `FS1.txt`, `FS2.txt` | Volume flow (2 sensors) | l/min | 10 Hz | 600 |
| `TS1.txt` – `TS4.txt` | Temperature (4 sensors) | °C | 1 Hz | 60 |
| `VS1.txt` | Vibration | mm/s | 1 Hz | 60 |
| `CE.txt` | Cooling efficiency (virtual) | % | 1 Hz | 60 |
| `CP.txt` | Cooling power (virtual) | kW | 1 Hz | 60 |
| `SE.txt` | Efficiency factor (virtual) | % | 1 Hz | 60 |

### Target File: `profile.txt`

Tab-delimited, 2,205 rows × 5 columns. Each row matches the corresponding cycle row in sensor files.

| Column | Component | Possible Values |
|--------|-----------|----------------|
| 1 | Cooler condition (%) | 3 (near failure), 20 (reduced), 100 (full) |
| 2 | Valve condition (%) | 73 (near failure), 80 (severe lag), 90 (small lag), 100 (optimal) |
| **3** | **Internal pump leakage** ← our target | **0** (no leakage), **1** (weak leakage), **2** (severe leakage) |
| 4 | Hydraulic accumulator (bar) | 90 (near failure), 100 (severely reduced), 115 (slightly reduced), 130 (optimal) |
| 5 | Stable flag | 0 (stable), 1 (not yet stable — may need filtering) |

---

## 5. Classification Targets

> ⚠️ **Important:** The dataset does **not** ship with pre-labeled "Normal / Warning / Fault" columns. These are derived from Column 3 of `profile.txt` (Internal pump leakage). All other columns describe different hydraulic components and are **not** used as classification targets in this project.

Our three-class problem uses only Column 3 of `profile.txt`:

```
0  →  Normal   (no internal pump leakage)
1  →  Warning  (weak internal pump leakage)
2  →  Fault    (severe internal pump leakage)
```

---

## 6. Label Mapping: Raw Dataset → SmartPump Classes

| `profile.txt` Col 3 Value | Condition | SmartPump Label | Action Priority |
|---------------------------|-----------|-----------------|-----------------|
| `0` | No leakage | **Normal** | Routine monitoring |
| `1` | Weak leakage | **Warning** | Schedule inspection |
| `2` | Severe leakage | **Fault** | Immediate intervention |

**Class imbalance note:** Class 1 (Warning) is typically the minority and hardest-to-detect class because its physical sensor signature closely overlaps with Normal operation. Class 2 (Fault) tends to show clear separation in flow sensors (FS1, FS2). Our evaluation strategy accounts for this explicitly.

---

## 7. Proposed Algorithms

| Algorithm | Rationale |
|-----------|-----------|
| **Decision Tree** | Interpretable baseline; reveals which features are most discriminative |
| **Random Forest** | Robust ensemble; handles non-linear boundaries; produces feature importances |
| **Support Vector Machine (SVM)** | Effective in high-dimensional feature spaces; good boundary separation |
| **Gaussian Naïve Bayes** | Fast probabilistic baseline; reveals feature independence assumptions |
| **AdaBoost** | Boosting baseline; emphasizes hard-to-classify Warning cycles |

All algorithms use **per-cycle statistical features** (mean, std, min, max, etc.) extracted from the raw time-series — not raw 6,000-column sensor vectors.

---

## 8. Evaluation Strategy

### Leakage-Aware Validation

> ⚠️ **Data Leakage Risk:** The 2,205 cycles are time-ordered operating cycles from a continuous experiment. A naïve random 80/20 split can place temporally adjacent cycles in both train and test sets, inflating scores.

**Mitigation strategy:**
- Use `StratifiedShuffleSplit` or block-based train/test split at the **cycle level** (not at the sensor sample level).
- Each cycle row is treated as one independent observation after feature extraction.
- Never mix samples from the same cycle across train and test.
- Consider filtering cycles where `stable_flag == 1` for primary evaluation; include them in a sensitivity analysis.

### Primary Metrics

| Metric | Why |
|--------|-----|
| **Fault-class Recall** | Minimize false negatives — a missed fault is dangerous |
| **Fault-class Precision** | Minimize false alarms — unnecessary shutdowns are costly |
| **Macro F1-score** | Treats all three classes equally; penalizes poor minority performance |
| **Confusion Matrix** | Full picture of class-wise errors |
| **Classification Report** | Per-class precision, recall, F1 |

**Do not report only overall accuracy.** With a class-imbalanced dataset, accuracy is misleading.

---

## 9. Project Architecture

```
SmartPump-AI/
├── data/
│   ├── raw/                    ← Downloaded UCI dataset files (not committed to git)
│   │   ├── PS1.txt … PS6.txt
│   │   ├── EPS1.txt
│   │   ├── FS1.txt, FS2.txt
│   │   ├── TS1.txt … TS4.txt
│   │   ├── VS1.txt
│   │   ├── CE.txt, CP.txt, SE.txt
│   │   └── profile.txt
│   ├── processed/              ← Feature-extracted cycle-level DataFrames
│   └── README.md               ← Dataset instructions and schema
├── docs/
│   ├── project_proposal.md     ← Technical project proposal
│   └── technical_risks.md      ← Risk register (future)
├── models/                     ← Serialized trained models (.pkl / .joblib)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
├── results/                    ← Evaluation outputs, plots, reports
├── src/
│   ├── preprocessing.py        ← Data loading and stable-flag handling
│   ├── feature_engineering.py  ← Statistical feature extraction from time series
│   └── train.py                ← Model training, evaluation and comparison
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 10. Team Collaboration

### Team Roles

| Member | Primary Responsibility | Key Files |
|--------|------------------------|-----------|
| **Member A** | Data Engineering & Exploration | `src/preprocessing.py`, `src/feature_engineering.py`, `notebooks/01_data_exploration.ipynb`, `notebooks/02_preprocessing.ipynb`, `data/` |
| **Member B** | Machine Learning & Evaluation | `src/train.py`, `notebooks/03_model_training.ipynb`, `results/` |
| **Member C** | Integration, Documentation & Coordination | `README.md`, `docs/`, `requirements.txt`, `.gitignore`, code review |

### Collaboration Rules

- **Never force-push** to `main`. All changes via pull requests.
- **Never delete or overwrite** another member's completed files without review.
- Each PR must include a brief description of what changed and why.
- Raw data files are **not committed** — see `data/README.md` for download instructions.
- All notebooks must be **cleared of outputs** before committing.
- Use `random_state=42` everywhere for reproducibility.

### Branch Strategy

```
main              ← stable, reviewed code only
├── data-engineering   ← Member A's branch
├── ml-models          ← Member B's branch
└── integration        ← Member C's branch
```

---

## 11. Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd SmartPump-AI

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Download the Dataset

See [`data/README.md`](data/README.md) for step-by-step download instructions.

### Project Workflow

```
Step 1: Download dataset → data/raw/
Step 2: Run notebook 01_data_exploration.ipynb
Step 3: Run notebook 02_preprocessing.ipynb
Step 4: Run notebook 03_model_training.ipynb
```

---

## 12. Future Development Roadmap

| Phase | Milestone | Status |
|-------|-----------|--------|
| ✅ **Phase 1** | Project foundation, documentation, dataset verification | Complete |
| 🔄 **Phase 2** | Dataset download, exploration, label mapping verification | Planned |
| 🔄 **Phase 3** | Feature extraction pipeline (statistical aggregation per cycle) | Planned |
| 🔄 **Phase 4** | Model training, hyperparameter search, comparison | Planned |
| 🔄 **Phase 5** | Evaluation reporting — confusion matrices, feature importance | Planned |
| 🔄 **Phase 6** | Final documentation, results, reproducibility audit | Planned |

---

## 13. Citation and Licensing

### Dataset Citation

```
Helwig, N., Pignanelli, E., & Schütze, A. (2015).
Condition monitoring of hydraulic systems [Dataset].
UCI Machine Learning Repository.
https://doi.org/10.24432/C5CW21
```

### Original Research Paper

```
N. Helwig, E. Pignanelli, and A. Schütze,
"Condition Monitoring of a Complex Hydraulic System Using Multivariate Statistics,"
2015 IEEE International Instrumentation and Measurement Technology Conference
(I2MTC) Proceedings, Pisa, Italy, 2015, pp. 210–215.
doi: 10.1109/I2MTC.2015.7151267
```

### Dataset License

The dataset is published under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.  
You may share, adapt, and use the data for any purpose, provided appropriate credit is given.

---

*SmartPump AI — Built by a three-member engineering team as a professional machine learning project.*
