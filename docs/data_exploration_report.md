# SmartPump AI — Dataset Exploration & Statistical Analysis Report (Step 2)

**Project:** SmartPump AI — Industrial Hydraulic Pump Condition Classification  
**Target:** Supervised multiclass classification of internal pump leakage (`0: Normal`, `1: Warning`, `2: Fault`)  
**Dataset:** UCI Machine Learning Repository — *Condition Monitoring of Hydraulic Systems* (ID 447)  
**Authors/Origin:** Helwig, N., Pignanelli, E., & Schütze, A. (ZeMA gGmbH / Saarland University, Germany)  
**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)  
**Document Status:** Complete & Verified  

---

## Executive Summary

This report documents the comprehensive exploratory data analysis (EDA) and verification performed for Step 2 of the SmartPump AI project. All 18 raw files (17 sensor channels and `profile.txt`) were audited for structural integrity, missingness, noise, and statistical distributions across 2,205 operating cycles.

Key conclusions from this exploratory phase:
1. **Target Verification:** Target labels in `profile.txt` (column 2, 0-indexed) were verified against ZeMA specifications: `0: Normal` (1,221 cycles), `1: Warning` (492 cycles), and `2: Fault` (492 cycles).
2. **Class Imbalance & Operational Stability:** While the raw dataset has a 55.4% / 22.3% / 22.3% distribution, the 1,449 **stable cycles** (`stable_flag == 0`) are almost perfectly balanced (33.7% Normal, 33.1% Warning, 33.1% Fault). Conversely, the 756 **unstable cycles** are 96.8% Normal, identifying a critical data skew risk.
3. **Sensor Discriminability:** Primary volume flow (`FS1`), efficiency factor (`SE`), motor power (`EPS1`), and primary pressures (`PS1`, `PS2`) exhibit marked, statistically significant shifts with pump leakage. Temperature sensors (`TS1`–`TS4`) reflect cooler degradation and long-term experiment thermal drift rather than pump leakage.
4. **Data Integrity:** Exactly 0 missing values and 0 infinite values across all 43,680 attributes per cycle. All files conform to expected matrix shapes.

---

## 1. Dataset Overview

The dataset was collected from an industrial hydraulic test rig comprising a primary working circuit and a secondary cooling-filtration circuit connected via an oil reservoir. The system cyclically executes 60-second load cycles while varying the degradation state of four components: cooler, valve, internal pump leakage, and hydraulic accumulator.

```
Total Operating Cycles : 2,205 cycles
Cycle Duration         : 60 seconds
Total Sensor Channels  : 17 channels (14 physical, 3 virtual)
Total Attributes/Cycle : 43,680 time samples
Target Profile File    : profile.txt (2,205 rows × 5 condition attributes)
Missing Values         : 0 (verified across all 18 files)
Infinite / NaN Values  : 0
```

### Table 1: Raw Sensor Matrix Specifications

| Sensor ID | Measured Physical Quantity | Physical Unit | Sampling Frequency | Samples per Cycle (Columns) | Circuit / System Subsystem |
|:---|:---|:---:|:---:|:---:|:---|
| **PS1** | Hydraulic Pressure | bar | 100 Hz | 6,000 | Primary circuit, pump outlet |
| **PS2** | Hydraulic Pressure | bar | 100 Hz | 6,000 | Primary circuit, main cylinder |
| **PS3** | Hydraulic Pressure | bar | 100 Hz | 6,000 | Primary circuit, return line |
| **PS4** | Hydraulic Pressure | bar | 100 Hz | 6,000 | Accumulator pressure |
| **PS5** | Hydraulic Pressure | bar | 100 Hz | 6,000 | Secondary cooling circuit |
| **PS6** | Hydraulic Pressure | bar | 100 Hz | 6,000 | Secondary filtration line |
| **EPS1** | Motor Power | W | 100 Hz | 6,000 | Main pump electric drive |
| **FS1** | Volume Flow Rate | l/min | 10 Hz | 600 | Primary circuit volume flow |
| **FS2** | Volume Flow Rate | l/min | 10 Hz | 600 | Secondary circuit volume flow |
| **TS1** | Temperature | °C | 1 Hz | 60 | Primary circuit, oil reservoir |
| **TS2** | Temperature | °C | 1 Hz | 60 | Primary circuit, before cooler |
| **TS3** | Temperature | °C | 1 Hz | 60 | Primary circuit, after cooler |
| **TS4** | Temperature | °C | 1 Hz | 60 | Secondary cooling circuit |
| **VS1** | Vibration Amplitude | mm/s | 1 Hz | 60 | Test rig main pump bracket |
| **CE** | Cooling Efficiency (Virtual) | % | 1 Hz | 60 | Virtual heat exchanger metric |
| **CP** | Cooling Power (Virtual) | kW | 1 Hz | 60 | Virtual heat transfer rate |
| **SE** | Efficiency Factor (Virtual) | % | 1 Hz | 60 | Global hydraulic efficiency |

---

## 2. Verified Target Labels & Profile Architecture

The file `data/raw/profile.txt` contains 2,205 rows and 5 tab-delimited columns. The column structure and condition values were validated against ZeMA reference documentation:

- **Column 0 (`cooler`):** Cooler condition [%] (3: close to total failure, 20: reduced efficiency, 100: full efficiency)
- **Column 1 (`valve`):** Valve switching behavior [%] (73: close to total failure, 80: severe lag, 90: small lag, 100: optimal)
- **Column 2 (`pump_leakage`):** **SmartPump AI Target**  
  - `0` → **Normal** (no internal pump leakage)
  - `1` → **Warning** (weak internal pump leakage — early degradation)
  - `2` → **Fault** (severe internal pump leakage — critical failure state)
- **Column 3 (`accumulator`):** Hydraulic accumulator pre-charge pressure [bar] (90, 100, 115, 130 bar)
- **Column 4 (`stable_flag`):** Operational stabilization state  
  - `0`: Process condition was stationary and fully stabilized
  - `1`: Quasi-static condition; transient or non-stabilized state

---

## 3. Class Distribution & Operational Stability Analysis

### 3.1 Overall Class Distribution (All 2,205 Cycles)

| Class Code | SmartPump AI Label | Operational Severity | Number of Cycles | Percentage of Dataset |
|:---:|:---|:---|:---:|:---:|
| **0** | **Normal** | No internal pump leakage | 1,221 | **55.37%** |
| **1** | **Warning** | Weak internal pump leakage | 492 | **22.31%** |
| **2** | **Fault** | Severe internal pump leakage | 492 | **22.31%** |
| **Total** | — | — | **2,205** | **100.0%** |

### 3.2 Breakdown by Operational Stability Flag

Cross-tabulating `pump_leakage` with `stable_flag` reveals a crucial operational insight:

| Operational Subgroup | Total Cycles | Normal (0) | Warning (1) | Fault (2) | Imbalance Ratio (N:W:F) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **All Cycles** | 2,205 | 1,221 (55.4%) | 492 (22.3%) | 492 (22.3%) | 2.48 : 1.00 : 1.00 |
| **Stable Cycles (`stable_flag == 0`)** | **1,449** (65.7%) | **489** (33.7%) | **480** (33.1%) | **480** (33.1%) | **1.02 : 1.00 : 1.00** |
| **Unstable Cycles (`stable_flag == 1`)** | **756** (34.3%) | **732** (96.8%) | **12** (1.6%) | **12** (1.6%) | **61.0 : 1.00 : 1.00** |

```
Key Finding:
Inside the stable operating regime (1,449 cycles), the three pump condition classes
are almost identically balanced (~480 cycles per class).
The entire dataset-level imbalance is caused by 756 unstable cycles, 96.8% of which
belong to the Normal class.
```

---

## 4. Statistical Findings: Sensor Measurements Across Pump Conditions

To evaluate signal discriminability, cycle-level summary statistics were computed across all 2,205 cycles. Table 2 summarizes cycle-mean statistics across the three target classes.

### Table 2: Cycle-Mean Sensor Statistics by Class

| Sensor | Unit | Overall Mean ± Std | Normal (0) Mean ± Std | Warning (1) Mean ± Std | Fault (2) Mean ± Std | Cohen's d (Norm vs Fault) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **FS1** | l/min | 6.20 ± 1.03 | **6.60 ± 0.38** | **5.79 ± 1.30** | **5.61 ± 1.35** | **1.01** (Large) |
| **SE** | % | 55.29 ± 8.96 | **59.16 ± 3.31** | **51.47 ± 10.92** | **49.50 ± 11.33** | **1.15** (Large) |
| **EPS1** | W | 2495.56 ± 73.84 | **2467.55 ± 62.95** | **2519.36 ± 69.76** | **2541.29 ± 71.74** | **1.09** (Large) |
| **PS1** | bar | 160.49 ± 4.70 | 159.09 ± 2.44 | 162.05 ± 5.86 | 162.38 ± 6.25 | **0.69** (Medium) |
| **PS2** | bar | 109.38 ± 4.99 | 107.92 ± 2.36 | 110.99 ± 6.35 | 111.40 ± 6.74 | **0.68** (Medium) |
| **PS3** | bar | 1.75 ± 0.25 | 1.84 ± 0.14 | 1.67 ± 0.31 | 1.63 ± 0.32 | **0.83** (Large) |
| **PS4** | bar | 2.60 ± 4.28 | 1.95 ± 3.74 | 3.39 ± 4.75 | 3.42 ± 4.74 | 0.34 (Small) |
| **PS5** | bar | 9.16 ± 0.58 | 9.17 ± 0.57 | 9.15 ± 0.59 | 9.15 ± 0.58 | 0.03 (Negligible) |
| **PS6** | bar | 9.08 ± 0.55 | 9.09 ± 0.54 | 9.06 ± 0.56 | 9.07 ± 0.55 | 0.03 (Negligible) |
| **FS2** | l/min | 9.65 ± 0.45 | 9.67 ± 0.42 | 9.62 ± 0.49 | 9.64 ± 0.47 | 0.07 (Negligible) |
| **TS1** | °C | 45.42 ± 7.99 | 45.24 ± 7.73 | 45.74 ± 8.45 | 45.55 ± 8.16 | 0.04 (Negligible) |
| **TS2** | °C | 50.37 ± 7.40 | 50.31 ± 7.28 | 50.54 ± 7.69 | 50.32 ± 7.39 | 0.00 (Negligible) |
| **TS3** | °C | 47.66 ± 7.45 | 47.48 ± 7.17 | 47.97 ± 7.91 | 47.81 ± 7.65 | 0.04 (Negligible) |
| **TS4** | °C | 40.74 ± 8.11 | 40.54 ± 7.85 | 41.05 ± 8.56 | 40.89 ± 8.27 | 0.04 (Negligible) |
| **VS1** | mm/s | 0.61 ± 0.06 | 0.60 ± 0.05 | 0.62 ± 0.07 | 0.63 ± 0.08 | 0.34 (Small) |
| **CE** | % | 31.30 ± 11.58 | 31.32 ± 11.58 | 31.22 ± 11.63 | 31.33 ± 11.53 | 0.00 (Negligible) |
| **CP** | kW | 1.81 ± 0.28 | 1.81 ± 0.29 | 1.80 ± 0.27 | 1.80 ± 0.26 | 0.03 (Negligible) |

---

## 5. Visualization Insights

Nine high-resolution visualizations were generated and saved to `results/eda/visualizations/`:

1. **`01_class_distribution.png` (Class Distribution):** Visualizes the 55.4% majority of Normal vs. equal 22.3% Warning and Fault classes, benchmarked against a hypothetical balanced baseline (735 cycles/class).
2. **`02_stable_flag_analysis.png` (Operational Stability Breakdown):** Displays side-by-side bar plots showing that the 1,449 stable cycles are virtually balanced, while 96.8% of unstable cycles are concentrated in the Normal class.
3. **`03_boxplots_pressure_power.png` (Pressure & Motor Power):** Documents that `PS1`, `PS2`, and `EPS1` increase systematically with leakage, while `PS3` decreases. Motor power (`EPS1`) rises because the hydraulic pump must expend more electrical energy to sustain hydraulic delivery against internal bypass leakage.
4. **`04_boxplots_flow_vibration.png` (Flow & Vibration):** Clearly illustrates the sharp reduction in `FS1` volume flow and its variance expansion under Warning and Fault. `VS1` shows a subtle upward shift in extreme vibration peaks.
5. **`05_boxplots_temperature_virtual.png` (Temperature & Efficiency):** Highlights that oil temperatures (`TS1`–`TS4`) and cooling parameters (`CE`, `CP`) have virtually identical inter-class medians, whereas hydraulic efficiency (`SE`) degrades precipitously from 59.2% (Normal) down to 49.5% (Fault).
6. **`06_timeseries_comparison.png` (Raw Time-Series Comparison):** Plots 600 time samples of `FS1` (10 Hz) and 60 samples of `TS1` (1 Hz) for individual cycles across Normal, Warning, and Fault. The `FS1` trace reveals that leakage impairs the pump's capability to maintain flow during high-load switching phases (seconds 10 to 40).
7. **`07_correlation_heatmap.png` (Sensor Correlation Matrix):** Illustrates dense collinearity blocks among the 17 channels, identifying redundant sensor pairs.
8. **`08_sensor_histograms.png` (Histograms of Key Channels):** Depicts multimodal, skewed distributions in `FS1`, `SE`, `EPS1`, and `PS1` across conditions.
9. **`09_all_sensors_overview.png` (Comprehensive 17-Sensor Matrix):** A 4×5 panel of box plots establishing an immediate visual benchmark across every sensor channel in the test rig.

---

## 6. Sensor Relationships & Physical Mechanisms

### 6.1 Top Sensor-to-Sensor Correlations

Analysis of the Pearson correlation matrix (`results/eda/correlation_matrix.csv`) reveals four distinct physical coupling clusters:

1. **Main Hydraulic Line Pressure Collinearity:**
   - `PS1` ↔ `PS2`: $r = +0.995$ (almost identical physical trajectory between pump outlet and cylinder port)
   - `PS1` ↔ `SE`: $r = -0.944$
   - `PS2` ↔ `SE`: $r = -0.958$
2. **Pressure–Flow Hydraulic Coupling:**
   - `PS3` (return line pressure) ↔ `FS1` (volume flow): $r = +0.921$
   - Higher internal pump bypass reduces return flow, simultaneously dropping return pressure.
3. **Electromechanical Coupling:**
   - `EPS1` (motor power) ↔ `PS1` (outlet pressure): $r = +0.831$
   - Drive motor consumes more power to compensate for bypass volumetric losses.
4. **Thermal System Collinearity:**
   - `TS1` ↔ `TS2`: $r = +0.984$
   - `TS1` ↔ `TS3`: $r = +0.989$
   - `TS1` ↔ `TS4`: $r = +0.992$
   - All four temperature sensors move together, governed by long-term rig heating and cooler operations.

### 6.2 Observed vs. Proven Predictive Channels

- **Primary Diagnostic Drivers (Observed):** `FS1`, `SE`, `EPS1`, `PS1`, `PS2`, `PS3`. These align directly with fluid-power conservation laws: internal leakage bypasses fluid from high-pressure chamber to low-pressure chamber inside the pump casing, reducing external discharge flow (`FS1`), degrading efficiency (`SE`), and demanding excess motor power (`EPS1`).
- **Uninformative / Orthogonal Channels (for Pump Leakage):** `TS1`–`TS4`, `CE`, `CP`, `PS5`, `PS6`, `FS2`. These channels monitor cooler health, secondary filtration flow, and thermal equilibrium. While vital for multi-target rig diagnostics, they offer negligible discriminatory signal for internal pump leakage.
- **Important Distinction:** High Cohen's $d$ on cycle-means is an exploratory indicator. Final feature importance must be proven via multivariable model training and permutation/impurity analysis in subsequent steps.

---

## 7. Data Quality Findings

| Quality Dimension | Inspection Result | Technical Detail / Assessment |
|:---|:---:|:---|
| **Missing Values** | **0** across all files | Full data completeness. No imputation required. |
| **Infinite / NaN** | **0** | Clean numeric float data throughout. |
| **Matrix Shapes** | **17/17 Match** | Every sensor file has 2,205 rows and $F_s \times 60$ columns. |
| **Constant Channels** | None globally | No sensor is dead/flatline across all cycles. |
| **Constant Within-Cycle** | **PS4** (1,238 cycles) | PS4 (accumulator pressure) flatlines at 0.0 bar for 1,238 cycles. This is expected physical behavior when the accumulator is discharged or isolated. |
| **Scale Heterogeneity** | 4 Orders of Magnitude | Values range from ~0.5 mm/s (`VS1`) to ~2,700 W (`EPS1`). Standardized scaling is mandatory prior to distance-based or linear modeling. |

---

## 8. Identified Engineering Risks & Mitigations

### Risk 1: Temporal Autocorrelation & Inappropriate Data Splitting (CRITICAL)
- **Problem:** Each 60-second cycle contains 60 to 6,000 time-series measurements. These points are auto-correlated and share a single machine operational state. Randomly partitioning individual time samples into train and test sets constitutes catastrophic data leakage.
- **Mitigation:** The unit of modeling must strictly be **one full cycle (one row)**. Splitting must occur at the cycle level using `StratifiedShuffleSplit`.

### Risk 2: Unstable Cycles Inducing Majority-Class Bias
- **Problem:** 756 cycles (34.3% of the dataset) are flagged as non-stationary (`stable_flag == 1`). Critically, 96.8% (732 cycles) of these unstable cycles are labeled `Normal`. Including unstable cycles without adjustment biases classifiers toward predicting `Normal`.
- **Mitigation:**
  1. Primary training and benchmark evaluation should be conducted strictly on the 1,449 **stable cycles** (`stable_flag == 0`), which exhibit near-perfect class balance (33.7% / 33.1% / 33.1%).
  2. Unstable cycles should be retained as an out-of-distribution robustness test set in subsequent evaluations.

### Risk 3: Warning vs. Fault Class Overlap
- **Problem:** Weak leakage (`Warning`) represents an incipient failure stage. While `Fault` (severe leakage) separates cleanly from `Normal` in flow and power sensors, `Warning` distributions partially overlap with both adjacent classes.
- **Mitigation:** Evaluate using **Macro F1-score** and class-specific recall rather than global accuracy. Baseline classifiers should be configured with class weighting or boosting to penalize missed `Warning` and `Fault` transitions.

### Risk 4: Extreme Collinearity Among Predictors
- **Problem:** `PS1` and `PS2` correlate at $r = 0.995$; temperature channels correlate at $r > 0.98$. Unregularized linear classifiers or inverse-covariance models will suffer numeric instability.
- **Mitigation:** Tree-based ensembles (Random Forest, Extra Trees, Gradient Boosting) are naturally resilient to collinearity. For linear models (SVM, Logistic Regression), collinear channel pruning or PCA must precede training.

### Risk 5: Cumulative Experimental Thermal Drift
- **Problem:** Cycles were acquired sequentially. As the hydraulic rig ran over several days, oil temperatures (`TS1`–`TS4`) exhibited an upward secular trend due to ambient conditions.
- **Mitigation:** Avoid raw absolute temperature thresholds as classification rules; focus on differential or within-cycle dynamic features if temperature channels are utilized.

---

## 9. Recommendations for Step 3 (Preprocessing & Feature Engineering)

1. **Cycle-Level Statistical Feature Extraction:**
   - Aggregate each raw 60-second sensor time series into fixed-width statistical descriptors:
     - Central tendency: `mean`, `median`
     - Dispersion: `std`, `IQR`
     - Extremes: `min`, `max`
     - Signal dynamics / Higher-order: `rms`, `skewness`, `kurtosis`
   - This transforms the raw $2,205 \times 43,680$ array into an engineered feature matrix of shape $2,205 \times 153$ (17 sensors × 9 features).
2. **Feature Matrix Storage:**
   - Save the extracted features and aligned target labels into `data/processed/features_per_cycle.csv`.
3. **Data Splitting Protocol:**
   - Use `StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)` on the stable cycle subset.
   - Fit feature scalers (`StandardScaler`) **strictly on training fold cycles**, subsequently transforming test fold cycles without target or test distribution leakage.
4. **Target Alignment:**
   - Encode `pump_leakage` as an integer series (`0, 1, 2`) with direct mapping to `Normal`, `Warning`, `Fault`.
5. **No ML Model Training in Step 2:**
   - Maintain strict separation of concerns: exploratory analysis and data auditing conclude with this report.

---

## 10. Audit Sign-Off & Generated Deliverables

The following artifacts have been generated, validated, and persisted:

- **Outputs in `results/eda/`:**
  - `dataset_summary.csv` (18 properties verified)
  - `class_distribution.csv` (overall, stable, and unstable class breakdowns)
  - `sensor_statistics.csv` (descriptive statistics across 17 sensors)
  - `sensor_class_statistics.csv` (per-sensor, per-class statistical breakdown)
  - `correlation_matrix.csv` (17×17 Pearson correlation matrix)
  - `data_quality_report.csv` (integrity audit, shapes, missingness, constant cycles)
- **Outputs in `results/eda/visualizations/`:**
  - `01_class_distribution.png`
  - `02_stable_flag_analysis.png`
  - `03_boxplots_pressure_power.png`
  - `04_boxplots_flow_vibration.png`
  - `05_boxplots_temperature_virtual.png`
  - `06_timeseries_comparison.png`
  - `07_correlation_heatmap.png`
  - `08_sensor_histograms.png`
  - `09_all_sensors_overview.png`
- **Notebook:**
  - `notebooks/01_data_exploration.ipynb` (fully documented, standardized relative paths, ready for top-to-bottom execution)

**Readiness for Step 3:** Approved. The dataset structure, class mappings, operational stability characteristics, and signal dynamics are thoroughly quantified.
