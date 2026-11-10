# SmartPump AI — Technical Project Proposal

**Project Title:** Intelligent Hydraulic Pump Condition Monitoring and Internal Leakage Classification Using Machine Learning
**Assignment:** Classwork 3.12 — Group Machine Learning Project
**Version:** 2.0 (Final Academic Submission)
**Date:** November 2026
**Status:** Complete — Fully Implemented, Evaluated, and Verified for Academic Submission

## Team Members

| Full Name | Student ID | GitHub Username | Assigned Role |
|---|---|---|---|
| **Biruk Ayitenew** | WOUR/0483/16 | [@Biruk1997](https://github.com/Biruk1997) | **Member 1:** Data Engineering & Exploration |
| **Abrham Gebremedhin** | WOUR/0082/16 | [@abrilohd](https://github.com/abrilohd) | **Member 2:** Machine Learning & Evaluation |
| **Israel Abera** | WOUR/4167/16 | [@Isru96](https://github.com/Isru96) | **Member 3:** Integration, Documentation & Testing Lead |

---

## 1. Executive Summary

**SmartPump AI** is a collaborative machine learning study developed for **Classwork 3.12** that automatically diagnoses the internal pump leakage condition of an industrial hydraulic system. The system classifies pump operation into three distinct degradation states — **Normal** (no leakage), **Warning** (weak leakage), and **Fault** (severe leakage) — by analyzing multi-rate sensor time series from the benchmark **UCI Condition Monitoring of Hydraulic Systems** dataset.

Collected by Helwig, Pignanelli, and Schütze at ZeMA gGmbH (Saarbrücken, Germany), the dataset provides 2,205 sixty-second operating cycles across 17 sensor channels (sampling frequencies between 1 Hz and 100 Hz). The project emphasizes engineering rigor, cycle-level feature extraction without data leakage, multi-algorithm benchmarking across five required classification techniques, and group-aware validation across separate experimental runs.

---

## 2. Problem Statement & Motivation

Hydraulic pumps are prime movers in industrial automation, material handling, injection molding, and aerospace systems. Internal pump leakage is a progressive mechanical failure mode caused by wear on internal pistons, swashplates, and seals. This leakage allows pressurized hydraulic fluid to slip from high-pressure delivery chambers back into low-pressure return lines or the reservoir.

Unchecked internal leakage causes:
1. **Volumetric Efficiency Loss:** Severe drops in actuator speed and system pressure.
2. **Thermal Runaway:** Mechanical energy converted directly into fluid heat, degrading oil viscosity.
3. **Catastrophic Seizure:** Metal-on-metal friction resulting in pump destruction and unmonitored plant downtime.

Traditional maintenance approaches rely on reactive repair or rigid schedule-based overhaul. **SmartPump AI** develops a non-intrusive, data-driven condition monitoring pipeline capable of detecting weak internal leakage (**Warning**) before catastrophic breakdown occurs (**Fault**).

---

## 3. Dataset Selection & Specifications

### 3.1 Source and Provenance
- **Repository:** UCI Machine Learning Repository (Dataset ID: 447)
- **DOI:** [10.24432/C5CW21](https://doi.org/10.24432/C5CW21)
- **Official URL:** `https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems`
- **Authors:** Nikolai Helwig, Eliseo Pignanelli, Andreas Schütze (2015)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

### 3.2 Dataset Architecture
The raw dataset contains **18 tab-delimited text files** representing 2,205 sixty-second operational cycles:
- **17 Sensor Matrices:** Raw sensor readings where rows represent cycles and columns represent timestamps within each 60-second cycle.
- **1 Target Annotation File:** `profile.txt` (2,205 rows × 5 columns).

### 3.3 Sensor Channels by Acquisition Frequency
1. **100 Hz Channels (6,000 values/cycle):** Hydraulic pressures `PS1` through `PS6` (bar), Electrical motor power `EPS1` (W).
2. **10 Hz Channels (600 values/cycle):** Volume flow rates `FS1` and `FS2` (l/min).
3. **1 Hz Channels (60 values/cycle):** Oil temperatures `TS1` through `TS4` (°C), Mechanical vibration `VS1` (mm/s), Virtual cooling efficiency `CE` (%), Virtual cooling power `CP` (kW), Efficiency factor `SE` (%).
- **Total raw measurements per cycle:** $(7 \times 6,000) + (2 \times 600) + (8 \times 60) = 43,680$ data points.

### 3.4 Target Label Formulation
Ground truth component wear states are recorded in `profile.txt`:
- Column 0: Hydraulic cooler condition
- Column 1: Directional control valve condition
- **Column 2: Internal pump leakage (Primary Target)**
- Column 3: Hydraulic accumulator condition
- Column 4: System stability flag (`0` = stable steady state; `1` = transient warm-up state)

The three operational classes are mapped as:
- **`0` → Normal:** No internal pump leakage (full volumetric efficiency).
- **`1` → Warning:** Weak internal pump leakage (incipient wear and efficiency loss).
- **`2` → Fault:** Severe internal pump leakage (critical volumetric slippage).

---

## 4. Methodology & Scope

### 4.1 Unit of Observation & Cycle-Level Feature Extraction
Treating individual time steps as independent observations would cause catastrophic data leakage and inflate performance artificially. In SmartPump AI:
- Each 60-second operating cycle is treated as a **single, discrete observation**.
- 15 statistical, dispersion, distribution shape, and temporal features are extracted per sensor channel:
  - Central tendency: Mean, Median
  - Dispersion: Standard Deviation, IQR, Min, Max
  - Distribution: Skewness, Kurtosis, 25th Percentile, 75th Percentile
  - Energy: Energy, Power, Absolute Energy
  - Dynamics: Linear Slope, Net Delta ($\text{end} - \text{start}$)
- Yields a structured matrix of **255 features per cycle** across all 17 sensors.

### 4.2 Data Partitioning & Leakage Protections
- **Primary Population:** 1,449 stable operating cycles (`stable_flag == 0`), which are naturally balanced (489 Normal, 480 Warning, 480 Fault).
- **Cycle-Level Stratified Split:** 70% Training (1,014 cycles), 15% Validation (217 cycles), and 15% Held-Out Testing (218 cycles).
- **Transformation Hygiene:** `VarianceThreshold(0.0)` and `StandardScaler` are wrapped within scikit-learn `Pipeline` objects and fitted strictly on `X_train`.

---

## 5. Required Algorithms & Experimental Matrix

In full alignment with Classwork 3.12, five classical supervised algorithms were implemented, tuned, and compared:

1. **Dummy Classifier (Baseline):** Stratified random guessing baseline.
2. **Decision Tree:** Non-parametric recursive splitting (`max_depth=5`, `min_samples_split=5`).
3. **Random Forest:** Bagged ensemble of 100 decorrelated decision trees (`n_estimators=100`, `max_depth=10`).
4. **Support Vector Machine (SVM):** Radial Basis Function kernel with balanced class weights (`C=10.0`, `gamma='scale'`).
5. **Gaussian Naïve Bayes:** Probabilistic classifier modeling class-conditional Gaussian distributions.
6. **AdaBoost:** Sequential adaptive boosting of decision stumps (`n_estimators=100`, `learning_rate=0.5`).

---

## 6. Evaluation Framework

To prioritize industrial safety, model selection is driven by:
- **Fault Recall ($\text{Recall}_2$):** Proportion of severe leakage cycles correctly flagged ($\frac{\text{TP}_2}{\text{TP}_2 + \text{FN}_2}$).
- **Fault Precision ($\text{Precision}_2$):** Proportion of declared faults that are genuine.
- **Macro F1-Score:** Unweighted arithmetic mean of F1-scores across all three classes.
- **Multi-Class Confusion Matrix & Classification Report:** Comprehensive analysis of misclassification patterns.
- **Group-Aware Validation:** Stratified grouped 5-fold CV across 36 contiguous experimental blocks to detect thermal and ambient drift.
- **Operational Sensitivity:** Zero-retraining evaluation on the 756 unstable cycles.

---

## 7. Summary of Results & Model Selection

Evaluated on the 218 held-out test cycles:
- **Champion Model:** **Random Forest** achieved **1.0000 Macro F1** and **1.0000 Fault Recall**, followed closely by SVM (0.9908 Macro F1) and Decision Tree (0.9862 Macro F1).
- **Group-Aware Generalization:** Across 36 experimental blocks, Random Forest achieved **0.9168 ± 0.0769 Macro F1**, reflecting realistic ambient laboratory drift.
- **Top Physical Predictors:** Permutation and Gini importance identified `SE` (Efficiency Factor), `FS1` (Volume Flow), `EPS1` (Motor Power), and `PS3` (Return Line Pressure) as the primary indicators of internal pump leakage.

---

## 8. Academic Deliverables & Work Partition

| Phase | Milestone / Deliverable | Responsible Member | Deliverable Location |
|:---:|:---|:---:|:---|
| **Phase 1** | Foundation, dataset documentation & proposal | Israel Abera (Member 3) | `README.md`, `docs/project_proposal.md`, `data/README.md` |
| **Phase 2** | Exploratory Data Analysis & distribution study | Biruk Ayitenew (Member 1) | `notebooks/01_data_exploration.ipynb`, `docs/data_exploration_report.md` |
| **Phase 3** | Preprocessing, feature engineering & scaling | Biruk Ayitenew (Member 1) | `src/preprocessing.py`, `src/feature_engineering.py`, `notebooks/02_*.ipynb` |
| **Phase 4** | 5 ML algorithms, CV, tuning & evaluation | Abrham Gebremedhin (Member 2) | `src/train.py`, `notebooks/03_model_training.ipynb`, `docs/model_training_report.md` |
| **Phase 5** | Model verification & technical risk analysis | Israel Abera (Member 3) | `docs/technical_risks.md`, `results/evaluation/*.csv` |
| **Phase 6** | UCI programmatic loader, test suite & QA | Israel Abera (Member 3) | `src/uci_loader.py`, `tests/test_*.py` |
| **Phase 7** | Final integration, peer review & submission | Israel Abera (Member 3) | `CONTRIBUTING.md`, `docs/team_workflow.md`, `.github/` |

---

## 9. References

1. Helwig, N., Pignanelli, E., & Schütze, A. (2015). *Condition monitoring of hydraulic systems* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CW21
2. Helwig, N., Pignanelli, E., & Schütze, A. (2015). Condition monitoring of a complex hydraulic system using multivariate statistics. *2015 IEEE International Instrumentation and Measurement Technology Conference (I2MTC)*, Pisa, Italy, pp. 210–215. https://doi.org/10.1109/I2MTC.2015.7151267
