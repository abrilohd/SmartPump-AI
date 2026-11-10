# SmartPump AI — Dataset Guide

**Dataset:** Condition Monitoring of Hydraulic Systems  
**Source:** UCI Machine Learning Repository (ID 447)  
**DOI:** https://doi.org/10.24432/C5CW21  
**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## 1. Overview

This dataset was collected from an industrial hydraulic test rig at ZeMA gGmbH, Saarbrücken, Germany by Nikolai Helwig, Eliseo Pignanelli, and Andreas Schütze (2015). It contains real experimental measurements from a hydraulic system under varied component degradation conditions.

| Property | Value |
|----------|-------|
| Total cycles | 2,205 |
| Cycle duration | 60 seconds each |
| Sensor streams | 17 (pressure, flow, temperature, power, vibration, virtual) |
| Target file | `profile.txt` (5 component condition columns) |
| Our target column | Column index 2 — Internal pump leakage |

---

## 2. Expected Files After Download

After downloading and extracting the dataset, your `data/raw/` folder must contain these files:

### Sensor Data Files (tab-delimited .txt matrices)

| File | Quantity Measured | Unit | Sampling Rate | Columns per Row |
|------|-------------------|------|---------------|-----------------|
| `PS1.txt` | Pressure sensor 1 | bar | 100 Hz | 6,000 |
| `PS2.txt` | Pressure sensor 2 | bar | 100 Hz | 6,000 |
| `PS3.txt` | Pressure sensor 3 | bar | 100 Hz | 6,000 |
| `PS4.txt` | Pressure sensor 4 | bar | 100 Hz | 6,000 |
| `PS5.txt` | Pressure sensor 5 | bar | 100 Hz | 6,000 |
| `PS6.txt` | Pressure sensor 6 | bar | 100 Hz | 6,000 |
| `EPS1.txt` | Motor power | W | 100 Hz | 6,000 |
| `FS1.txt` | Volume flow sensor 1 | l/min | 10 Hz | 600 |
| `FS2.txt` | Volume flow sensor 2 | l/min | 10 Hz | 600 |
| `TS1.txt` | Temperature sensor 1 | °C | 1 Hz | 60 |
| `TS2.txt` | Temperature sensor 2 | °C | 1 Hz | 60 |
| `TS3.txt` | Temperature sensor 3 | °C | 1 Hz | 60 |
| `TS4.txt` | Temperature sensor 4 | °C | 1 Hz | 60 |
| `VS1.txt` | Vibration | mm/s | 1 Hz | 60 |
| `CE.txt` | Cooling efficiency (virtual) | % | 1 Hz | 60 |
| `CP.txt` | Cooling power (virtual) | kW | 1 Hz | 60 |
| `SE.txt` | Efficiency factor (virtual) | % | 1 Hz | 60 |

### Target File

| File | Description |
|------|-------------|
| `profile.txt` | 2,205 rows × 5 columns — component condition labels per cycle |

**Total files: 18** (17 sensor files + 1 target file)

---

## 3. File Format

All files are **tab-delimited text matrices** with no header row.

### Sensor Files
```
<value1_t1>  <value1_t2>  ...  <value1_tN>   ← Cycle 1 (row 1)
<value2_t1>  <value2_t2>  ...  <value2_tN>   ← Cycle 2 (row 2)
...
<value2205_t1>  ...  <value2205_tN>           ← Cycle 2205 (row 2205)
```
- Each row = one 60-second experimental cycle.
- Each column = one time point within that cycle (at the sensor's sampling rate).

### Target File (`profile.txt`)
```
<cooler>  <valve>  <pump_leakage>  <accumulator>  <stable_flag>   ← Cycle 1
...
```

| Column (0-indexed) | Meaning | Possible Values |
|-------------------|---------|-----------------|
| 0 | Cooler condition (%) | 3, 20, 100 |
| 1 | Valve condition (%) | 73, 80, 90, 100 |
| **2** | **Internal pump leakage ← PROJECT TARGET** | **0, 1, 2** |
| 3 | Hydraulic accumulator pressure (bar) | 90, 100, 115, 130 |
| 4 | Stable flag | 0, 1 |

---

## 4. Classification Target — Pump Leakage (Column 2)

This project uses **only column 2** of `profile.txt` as the classification target:

| Raw Value | Physical Condition | SmartPump Label |
|-----------|-------------------|-----------------|
| `0` | No internal pump leakage | **Normal** |
| `1` | Weak internal pump leakage (reduced efficiency) | **Warning** |
| `2` | Severe internal pump leakage | **Fault** |

> ⚠️ **Do not use columns 0, 1, 3, or 4 as classification targets.** They describe other hydraulic components (cooler, valve, accumulator) and system stability — not pump leakage.

> ⚠️ **The stable flag (column 4).** Cycles where `stable_flag == 1` indicate the system had not yet reached steady state. For primary model training and evaluation, consider filtering these cycles. See `docs/project_proposal.md` for details.

---

## 5. Data Download Instructions

> ⚠️ **Raw data files must NOT be committed to the Git repository.** The `data/raw/` directory is listed in `.gitignore`. Each team member downloads the dataset independently.

### Option A — UCI Repository Direct Download (Recommended)

1. Navigate to: https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems
2. Click the **Download** button to download the ZIP archive.
3. Extract the contents.
4. Move all `.txt` files into `data/raw/` within this repository.

### Option B — Manual File Verification

After download, verify that all 18 files are present in `data/raw/`:

```
data/
└── raw/
    ├── PS1.txt
    ├── PS2.txt
    ├── PS3.txt
    ├── PS4.txt
    ├── PS5.txt
    ├── PS6.txt
    ├── EPS1.txt
    ├── FS1.txt
    ├── FS2.txt
    ├── TS1.txt
    ├── TS2.txt
    ├── TS3.txt
    ├── TS4.txt
    ├── VS1.txt
    ├── CE.txt
    ├── CP.txt
    ├── SE.txt
    └── profile.txt
```

### Quick Verification (Python)

After placing files in `data/raw/`, run this snippet to verify shapes:

```python
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")

# Verify sensor file shapes
expected = {
    "PS1.txt": (2205, 6000), "PS2.txt": (2205, 6000), "PS3.txt": (2205, 6000),
    "PS4.txt": (2205, 6000), "PS5.txt": (2205, 6000), "PS6.txt": (2205, 6000),
    "EPS1.txt": (2205, 6000),
    "FS1.txt": (2205, 600), "FS2.txt": (2205, 600),
    "TS1.txt": (2205, 60), "TS2.txt": (2205, 60),
    "TS3.txt": (2205, 60), "TS4.txt": (2205, 60),
    "VS1.txt": (2205, 60),
    "CE.txt": (2205, 60), "CP.txt": (2205, 60), "SE.txt": (2205, 60),
}

for fname, expected_shape in expected.items():
    df = pd.read_csv(RAW / fname, sep="\t", header=None)
    status = "✅ OK" if df.shape == expected_shape else f"❌ Expected {expected_shape}, got {df.shape}"
    print(f"{fname}: {status}")

# Verify profile.txt
profile = pd.read_csv(RAW / "profile.txt", sep="\t", header=None)
print(f"\nprofile.txt shape: {profile.shape}")
print(f"Column 2 (pump leakage) unique values: {sorted(profile[2].unique())}")
print("Expected: [0, 1, 2]")
```

---

## 6. Data Organization

```
data/
├── raw/                 ← UCI dataset files (NOT committed to git)
│   ├── PS1.txt … PS6.txt
│   ├── EPS1.txt
│   ├── FS1.txt, FS2.txt
│   ├── TS1.txt … TS4.txt
│   ├── VS1.txt
│   ├── CE.txt, CP.txt, SE.txt
│   └── profile.txt
├── processed/           ← Feature-extracted DataFrames (NOT committed to git)
│   └── features_per_cycle.csv   ← One row per cycle, generated by feature_engineering.py
└── README.md            ← This file
```

Both `data/raw/` and `data/processed/` are excluded from Git version control. Only documentation and code are committed.

---

## 7. Important Technical Notes

### Time-Series Structure
Each sensor file row is a **time series**, not a set of independent features. The 6,000 columns of a 100 Hz pressure file are 6,000 consecutive time points from one 60-second cycle. They must be treated as a single observation and reduced to summary statistics before training.

### Cycle-Level Independence
The 2,205 rows (cycles) are the **unit of observation** for classification. Train/test splits must be performed at the cycle level, not at the individual sample level.

### No Pre-Existing Normal/Warning/Fault Labels
The dataset does not ship with these labels. They are derived from column 2 of `profile.txt` using the mapping defined in Section 4 above.

### Approximate File Sizes (Uncompressed)
| Group | Files | Approx. Size |
|-------|-------|-------------|
| 100 Hz sensors (PS1–PS6, EPS1) | 7 files | ~700–800 MB |
| 10 Hz sensors (FS1, FS2) | 2 files | ~20 MB |
| 1 Hz sensors (TS1–TS4, VS1, CE, CP, SE) | 8 files | ~5 MB |
| profile.txt | 1 file | < 1 MB |

Ensure at least **1 GB of free disk space** before downloading.

---

## 8. License and Attribution

The dataset is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

### Required Citation

```
Helwig, N., Pignanelli, E., & Schütze, A. (2015).
Condition monitoring of hydraulic systems [Dataset].
UCI Machine Learning Repository.
https://doi.org/10.24432/C5CW21
```

### Original Research Paper

```
N. Helwig, E. Pignanelli, and A. Schütze,
"Condition monitoring of a complex hydraulic system using multivariate statistics,"
2015 IEEE I2MTC, Pisa, Italy, pp. 210–215.
doi: 10.1109/I2MTC.2015.7151267
```

---

*Last updated: November 2026 | SmartPump AI Team*
