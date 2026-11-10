# SmartPump AI — Technical Risk Register & Limitations

**Project:** SmartPump AI — Intelligent Hydraulic Pump Condition Monitoring
**Assignment:** Classwork 3.12 (Group Machine Learning Project)
**Maintained by:** Abrham Gebremedhin (Member 3 — Integration, Documentation and Testing Lead)
**Version:** 2.0 (Post-Model Evaluation & Final Integration)
**Date:** November 2026  
**Status:** Verified & Closed for Final Submission

---

## 1. Executive Summary & Risk Matrix

This document tracks technical risks, dataset limitations, modeling constraints, and real-world industrial discrepancies identified during the SmartPump AI lifecycle.

| ID | Technical Risk / Limitation Area | Severity | Likelihood | Resolution / Status |
|:---:|:---|:---:|:---:|:---:|
| **R1** | **Data Leakage:** Sub-cycle sample splitting | 🔴 High | Low | **Mitigated:** Aggregated to 2,205 cycle-level observations; split exclusively by `cycle_id`. |
| **R2** | **Label Conflation:** Mixing component conditions | 🔴 High | Low | **Mitigated:** Explicitly isolated Column 2 (`pump_leakage`) in `profile.txt`; validated `{0, 1, 2}`. |
| **R3** | **Class Imbalance Distortion:** Misleading accuracy | 🟡 Medium | High | **Mitigated:** Macro F1 and Fault Recall prioritized; class weights balanced. |
| **R4** | **Operational Stability Drift:** Transient warm-up | 🟡 Medium | High | **Mitigated:** Filtered 756 unstable cycles (`stable_flag == 1`) for primary development; audited via sensitivity test. |
| **R5** | **Multi-Rate Concatenation Bias:** Sampling disparity | 🟡 Medium | Medium | **Mitigated:** 15 uniform statistical features extracted per channel regardless of rate (255 total). |
| **R6** | **Group-Aware Generalization & Thermal Drift** | 🟡 Medium | High | **Quantified:** StratifiedGroupKFold on 36 blocks reveals drift: Macro F1 drops from 1.000 to 0.9168. |
| **R7** | **Experimental vs Industrial Environment Mismatch** | 🟡 Medium | High | **Documented:** Test-rig constant 60s load cycles differ from variable industrial duty cycles. |
| **R8** | **High Sensor Collinearity & Redundancy** | 🟢 Low | High | **Managed:** `VarianceThreshold(0.0)` removes constants; Random Forest handles correlated pressure features. |
| **R9** | **Sensor Availability & Critical Channel Loss** | 🟡 Medium | Medium | **Analyzed:** Permutation importance identifies `SE`, `FS1`, `EPS1`, and `PS3` as primary single-point dependencies. |
| **R10** | **Model Interpretability vs Physical Telemetry** | 🟢 Low | Medium | **Documented:** Tree feature importances confirm physical drop in volumetric flow, but do not replace physical inspection. |

---

## 2. Detailed Technical Risk Assessments & Mitigations

### R1 — Data Leakage via Intra-Cycle Time-Sample Splitting
- **Vulnerability:** Raw sensor matrices contain up to 6,000 points per cycle. Splitting individual time-series points into train/test would result in points from the same 60-second cycle appearing in both partitions, creating artificial 100% memorization.
- **Mitigation Implemented:** Feature engineering aggregates each 60-second cycle into 15 statistical features. Splitting occurs strictly at the operating cycle level via `split_cycle_data()` in `src/preprocessing.py`, guaranteeing $\text{Train} \cap \text{Val} \cap \text{Test} = \emptyset$.

### R2 — Target Column Misidentification in `profile.txt`
- **Vulnerability:** `profile.txt` contains five unlabeled integer columns representing four different hydraulic components and one stability flag. Confusing column indices would train a classifier on cooler condition (Column 0) or valve condition (Column 1) rather than pump leakage.
- **Mitigation Implemented:** `src/preprocessing.py` explicitly indexes Column 2 (`pump_leakage`), validates that unique labels match `{0, 1, 2}`, and checks that Column 4 represents `stable_flag`.

### R3 — Class Imbalance & Evaluation Skew
- **Vulnerability:** The raw 2,205-cycle dataset contains 1,221 Normal cycles (55.4%), 492 Warning cycles (22.3%), and 492 Fault cycles (22.3%). A trivial baseline predicting "Normal" achieves 55.4% accuracy with 0% Fault recall.
- **Mitigation Implemented:** Standard accuracy was demoted. Primary metrics are **Macro F1-Score** and **Fault-Class Recall** ($\frac{\text{TP}_2}{\text{TP}_2 + \text{FN}_2}$). Furthermore, within the 1,449 stable cycles, classes are naturally balanced (33.7% Normal, 33.1% Warning, 33.1% Fault).

### R4 — Operational Stability & Warm-Up Transients
- **Vulnerability:** 756 of the 2,205 cycles (34.3%) are recorded before the hydraulic test rig reached thermal equilibrium (`stable_flag == 1`). Crucially, 96.8% of unstable cycles are "Normal", with cycles 0–209 (Block 1) representing cold rig startup.
- **Mitigation Implemented:** Primary model training is conducted exclusively on the 1,449 stable cycles. The 756 unstable cycles were evaluated as an operational sensitivity test (`results/evaluation/unstable_sensitivity_results.csv`), demonstrating that the Random Forest generalizes across thermal transients without false alarms.

### R5 — Multi-Rate Sensor Dimensionality Imbalance
- **Vulnerability:** Ingesting raw values would cause 100 Hz pressure sensors (6,000 values) to overwhelm 1 Hz temperature and efficiency sensors (60 values) by a ratio of 100:1.
- **Mitigation Implemented:** `src/feature_engineering.py` calculates 15 statistical, distribution, and temporal features per channel regardless of acquisition frequency, creating a uniform 255-feature matrix where low-frequency efficiency channels (`SE`, `CE`) compete equally with pressure channels.

---

## 3. Generalization Limitations & Environmental Factors

### R6 — Group-Aware Generalization across Experimental Blocks
The 1,449 stable cycles were collected across **36 contiguous experimental blocks** representing separate test runs.
- In-distribution stratified 5-fold CV achieves **Macro F1 = 1.0000**.
- Group-aware cross-validation (`StratifiedGroupKFold(n_splits=5)`) holding out complete experimental blocks drops performance to **Macro F1 = 0.9168 ± 0.0769**.
- **Engineering Finding:** While the model learns robust physical boundaries, real-world deployment across separate days experiences slight performance degradation due to ambient laboratory temperature changes and hydraulic fluid viscosity drift.

### R7 — Test-Rig vs Industrial Field Conditions
- **Constant Duty Profile:** The ZeMA test rig repeated a fixed 60-second rectangular load cycle. Industrial machinery experiences variable cycle durations, fluctuating work loads, and unpredictable idle periods.
- **Controlled Contamination:** Wear was induced experimentally. In real manufacturing plants, internal pump leakage is compounded by particulate contamination, aeration, cavitation, and seal hardening.
- **Mitigation:** SmartPump AI predictions are explicitly documented as decision-support alerts for maintenance engineers rather than automatic override controls.

### R8 — High Inter-Sensor Collinearity
- Multiple pressure sensors (`PS1` through `PS6`) monitor adjacent points along the main hydraulic line, resulting in pairwise Pearson correlations exceeding $r = 0.90$.
- Linear models and Naïve Bayes suffer from multi-collinearity (Gaussian Naïve Bayes achieves Macro F1 = 0.8248 due to the violated independence assumption).
- Bagged ensembles (Random Forest) randomly sub-sample features (`max_features='sqrt'`), effectively decorrelating individual trees and thriving in high-collinearity regimes.

### R9 — Sensor Availability & Critical Path Dependencies
Feature importance analysis revealed that pump condition classification is heavily concentrated in four physical channels:
1. `SE` (Efficiency Factor — virtual sensor derived from mechanical power and flow)
2. `FS1` (High-pressure volume flow)
3. `EPS1` (Electrical motor power)
4. `PS3` (Return line backpressure)
- **Risk:** If volume flow meter `FS1` or virtual sensor `SE` suffers hardware failure or decalibration, model performance will degrade significantly.
- **Mitigation:** Future iterations should evaluate sensor-dropout training to ensure redundancy across secondary pressure sensors (`PS1`, `PS2`).

---

## 4. Engineering Safety Disclaimer

> [!IMPORTANT]
> **Safety and Operational Disclaimer:**
> SmartPump AI was developed as an academic machine learning research project for Classwork 3.12. Model predictions indicate statistical probabilities of component degradation based on historical sensor data. They do **not** replace physical visual inspection, fluid sampling analysis, manufacturer service manuals, calibrated pressure relief valves, or plant emergency shutdown systems.
