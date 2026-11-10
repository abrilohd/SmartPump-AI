# SmartPump AI  - Technical Risk Register

**Version:** 1.0  
**Date:** November 2026  
**Status:** Active  - update as risks are resolved or new ones are discovered

This document is maintained by Member C (Integration & Coordination) and updated as the team progresses through project phases.

---

## Risk Summary

| ID | Risk | Severity | Likelihood | Status |
|----|------|----------|------------|--------|
| R1 | Data leakage  - wrong split unit | 🔴 High | Medium |   Mitigated |
| R2 | Label confusion  - wrong profile column | 🔴 High | Medium |   Mitigated |
| R3 | Class imbalance  - biased evaluation | 🟡 Medium | High |   Mitigated |
| R4 | Stable flag  - transient cycles distort features | 🟡 Medium | Medium |   Mitigated |
| R5 | Multi-rate sensors  - naive feature concatenation | 🟡 Medium | Medium |   Mitigated |
| R6 | Overfitting  - complex models on small dataset | 🟢 Low | Medium | 🔄 Monitor |
| R7 | Dataset size  - 2,205 cycles may be small for deep models | 🟢 Low | Low | 🔄 Monitor |
| R8 | Reproducibility  - different random states across runs | 🟢 Low | Medium |   Mitigated |

---

## Detailed Risk Descriptions

### R1  - Data Leakage: Wrong Train/Test Split Unit

**Description:**  
The 2,205 rows in each sensor file are time-ordered operating cycles. Each cycle has up to 6,000 time samples (at 100 Hz). A naïve random split applied at the **sample level** (treating each of the 6,000 time steps as an independent observation) would:
- Produce a feature matrix with 2,205 × 6,000 = 13.2 million rows.
- Place time steps from the **same cycle** into both train and test.
- Massively inflate test scores (models would be "predicting" time steps from cycles they already saw).

**Severity:** 🔴 High  - Would invalidate all results.

**Mitigation (implemented):**  
Statistical features are extracted at the **cycle level** in `feature_engineering.py`. The resulting feature matrix has exactly 2,205 rows (one per cycle). The train/test split in `train.py` uses `StratifiedShuffleSplit` on this 2,205-row matrix  - **never** on individual time samples.

**Verification:** Each cycle row in `features_per_cycle.csv` corresponds to exactly one row in `profile.txt`.

---

### R2  - Label Confusion: Wrong `profile.txt` Column

**Description:**  
`profile.txt` contains five target columns describing four different hydraulic components and one stability flag. The columns are **not labeled** in the raw file. Using the wrong column would train a model on a completely different classification task (e.g., cooler condition instead of pump leakage).

**Severity:** 🔴 High  - Model would solve the wrong problem entirely.

**Verified column mapping (confirmed against `docs/description.txt`):**

| Column (0-indexed) | Component | Our target? |
|--------------------|-----------|-------------|
| 0 | Cooler condition (%) |   No |
| 1 | Valve condition (%) |   No |
| **2** | **Internal pump leakage** | **  Yes** |
| 3 | Hydraulic accumulator (bar) |   No |
| 4 | Stable flag |   No (used for filtering) |

**Verified unique values of column 2:** `{0, 1, 2}`   (confirmed by running against actual `profile.txt`)

**Mitigation (implemented):** `preprocessing.py` explicitly selects column `pump_leakage` (index 2) by name after assigning column names during load. It validates unique values against `{0, 1, 2}` and raises `ValueError` if unexpected values appear.

---

### R3  - Class Imbalance: Biased Evaluation

**Description:**  
The pump leakage class distribution is imbalanced:

| Label | Count | Share |
|-------|-------|-------|
| Normal (0) | 1,221 | 55.4% |
| Warning (1) | 492 | 22.3% |
| Fault (2) | 492 | 22.3% |

A model that predicts "Normal" for every cycle achieves ~55% accuracy. Overall accuracy is therefore misleading.

**Severity:** 🟡 Medium  - Affects evaluation but not data correctness.

**Mitigation (implemented):**
- Primary metric is **macro F1-score** (treats all classes equally).
- **Fault-class recall** and **Fault-class precision** are explicitly tracked.
- SVM and Random Forest use `class_weight="balanced"`.
- Stratified splits ensure class proportions are preserved in both train and test.

---

### R4  - Stable Flag: Transient Cycles May Distort Features

**Description:**  
756 of 2,205 cycles (34.3%) are marked `stable_flag = 1`, meaning the hydraulic system had not yet reached steady state when those cycles were recorded. Features extracted from transient cycles may not be representative of normal operating conditions.

**Severity:** 🟡 Medium  - May inflate or deflate performance metrics.

**Mitigation (implemented):**
- **Primary evaluation:** Filter out all cycles where `stable_flag == 1` before training. Implemented in `preprocessing.filter_stable_cycles()`.
- **Sensitivity analysis:** Repeat the full pipeline with all 2,205 cycles and compare results. Document the difference.
- Both runs should be compared in `results/`.

---

### R5  - Multi-Rate Sensors: Naive Feature Concatenation

**Description:**  
Sensors operate at 1 Hz, 10 Hz, and 100 Hz, producing 60, 600, or 6,000 columns per cycle. Naïve concatenation would produce a feature matrix where high-frequency sensors (PS1–PS6, EPS1) dominate with thousands of columns, while low-frequency sensors (TS1–TS4, etc.) contribute only 60 columns each.

**Severity:** 🟡 Medium  - Distorts feature importance; may confuse tree-based models.

**Mitigation (implemented):**  
`feature_engineering.py` extracts the **same set of statistics** from each sensor independently (e.g., mean, std, min, max, median), regardless of its sampling rate. This yields the same number of features per sensor and eliminates sampling-rate bias.

---

### R6  - Overfitting: Complex Models on Small Dataset

**Description:**  
With ~1,449 training cycles (after filtering unstable), deep Decision Trees and large SVMs with complex kernels can overfit.

**Severity:** 🟢 Low  - Detectable and correctable during evaluation.

**Mitigation (ongoing):**
- Use `StratifiedKFold(n_splits=5)` cross-validation on training set for hyperparameter tuning.
- Apply `max_depth` limits and `min_samples_leaf` constraints to Decision Trees.
- Monitor train vs test performance gap; flag models with >10% gap.

---

### R7  - Dataset Size: 2,205 Cycles

**Description:**  
After filtering unstable cycles (~34%), the effective dataset is ~1,449 cycles. This is adequate for the five proposed classical ML algorithms but would be insufficient for deep learning approaches.

**Severity:** 🟢 Low  - Within scope; deep learning not targeted.

**Decision:** The five classical algorithms (Decision Tree, Random Forest, SVM, Gaussian NB, AdaBoost) are well-suited to datasets of this size. No neural networks are planned for the initial scope.

---

### R8  - Reproducibility: Random State Consistency

**Description:**  
Any result that uses randomness (train/test split, ensemble training, cross-validation) must use a fixed seed to be reproducible.

**Severity:** 🟢 Low  - Easy to control.

**Mitigation (implemented):**  
`random_state=42` is used everywhere in `train.py` and `preprocessing.py`. This constant is defined once in `train.py` as `RANDOM_STATE = 42`.

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| Nov 2026 | Use column 2 of profile.txt as pump leakage target | Verified against official dataset documentation |
| Nov 2026 | Extract statistical features (mean/std/min/max/median) per cycle | Prevents data leakage; handles multi-rate sensors uniformly |
| Nov 2026 | Filter stable_flag==1 for primary run | Removes transient cycles; sensitivity analysis with all cycles |
| Nov 2026 | Use StratifiedShuffleSplit at cycle level | Prevents temporal data leakage; preserves class balance |
| Nov 2026 | Primary metric: Macro F1 + Fault recall | Overall accuracy is misleading with imbalanced classes |

---

*Last updated: November 2026 | Member C  - SmartPump AI*

