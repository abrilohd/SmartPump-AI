# UCI Dataset Reference: Condition Monitoring of Hydraulic Systems

**Dataset:** Condition Monitoring of Hydraulic Systems
**UCI Machine Learning Repository ID:** [447](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)
**DOI:** [10.24432/C5CW21](https://doi.org/10.24432/C5CW21)
**Origin:** ZeMA gGmbH, Saarbrücken, Germany (Helwig et al., 2015)
**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## 1. Overview and Physical Context

This benchmark dataset was experimentally obtained using an industrial hydraulic test rig consisting of a **primary working circuit** and a **secondary cooling-filtration circuit** connected via a common oil reservoir. The system executes repetitive constant load cycles, each exactly **60 seconds** in duration, while process sensors monitor pressures, volume flows, temperatures, electrical power, and mechanical vibration.

During experimental data acquisition, the physical wear states of four major hydraulic components were quantitatively varied to simulate industrial aging and faults:
1. Hydraulic cooler condition
2. Directional control valve switching behavior
3. **Internal pump leakage (the target of SmartPump AI)**
4. Hydraulic accumulator pre-charge pressure

---

## 2. Dataset Structure and Raw Sensor Channels

The raw dataset comprises **18 tab-delimited text matrices** (2,205 rows each, where each row corresponds to one complete 60-second cycle). Because sensors operate at different acquisition rates, the number of column values varies by channel:

| File Name | Measured Physical Quantity | Engineering Unit | Sampling Frequency | Measurements per 60s Cycle | Total Matrix Dimensions (Rows × Cols) |
|---|---|---|---|---|---|
| `PS1.txt` | Pressure sensor 1 | bar | 100 Hz | 6,000 | 2,205 × 6,000 |
| `PS2.txt` | Pressure sensor 2 | bar | 100 Hz | 6,000 | 2,205 × 6,000 |
| `PS3.txt` | Pressure sensor 3 | bar | 100 Hz | 6,000 | 2,205 × 6,000 |
| `PS4.txt` | Pressure sensor 4 | bar | 100 Hz | 6,000 | 2,205 × 6,000 |
| `PS5.txt` | Pressure sensor 5 | bar | 100 Hz | 6,000 | 2,205 × 6,000 |
| `PS6.txt` | Pressure sensor 6 | bar | 100 Hz | 6,000 | 2,205 × 6,000 |
| `EPS1.txt` | Electrical motor power | W | 100 Hz | 6,000 | 2,205 × 6,000 |
| `FS1.txt` | Volume flow rate 1 | l/min | 10 Hz | 600 | 2,205 × 600 |
| `FS2.txt` | Volume flow rate 2 | l/min | 10 Hz | 600 | 2,205 × 600 |
| `TS1.txt` | Oil temperature 1 | °C | 1 Hz | 60 | 2,205 × 60 |
| `TS2.txt` | Oil temperature 2 | °C | 1 Hz | 60 | 2,205 × 60 |
| `TS3.txt` | Oil temperature 3 | °C | 1 Hz | 60 | 2,205 × 60 |
| `TS4.txt` | Oil temperature 4 | °C | 1 Hz | 60 | 2,205 × 60 |
| `VS1.txt` | Mechanical vibration | mm/s | 1 Hz | 60 | 2,205 × 60 |
| `CE.txt` | Cooling efficiency (virtual) | % | 1 Hz | 60 | 2,205 × 60 |
| `CP.txt` | Cooling power (virtual) | kW | 1 Hz | 60 | 2,205 × 60 |
| `SE.txt` | Efficiency factor (virtual) | % | 1 Hz | 60 | 2,205 × 60 |

- **Attributes per cycle:** $(7 \times 6,000) + (2 \times 600) + (8 \times 60) = 42,000 + 1,200 + 480 = 43,680$ raw sensor measurements.
- **Total dataset observations:** $2,205 \times 43,680 = 96,314,400$ sensor readings.

---

## 3. Target Profile Annotations (`profile.txt`)

Ground-truth component states are recorded in `data/raw/profile.txt` (2,205 rows × 5 tab-delimited columns):

| Column Index | System Attribute | Discrete Ground Truth Values | SmartPump AI Role |
|:---:|:---|:---|:---|
| `0` | Cooler Condition | `3` (close to failure), `20` (reduced), `100` (full efficiency) | Context / Confounding factor |
| `1` | Valve Condition | `73` (close to failure), `80` (severe lag), `90` (small lag), `100` (optimal) | Context / Confounding factor |
| **`2`** | **Internal Pump Leakage** | **`0` (None), `1` (Weak leakage), `2` (Severe leakage)** | **Primary Classification Target** |
| `3` | Accumulator Pressure | `90` (close to failure), `100` (severely reduced), `115` (reduced), `130` (optimal) | Context / Confounding factor |
| `4` | Stability Flag | `0` (stable conditions), `1` (unstable / static thermal drift) | Partitioning & Filtering |

---

## 4. SmartPump AI Target Mapping

The primary prediction target is Column 2 (`pump_leakage`):

```text
Physical Profile Value (Col 2)    →    SmartPump Class    →    Condition Label
               0                  →           0           →        Normal
               1                  →           1           →        Warning
               2                  →           2           →        Fault
```

### Operational Class Counts

| Class | Label | Physical Description | Stable Cycles (`col4 == 0`) | Full Cycles (All 2,205) |
|:---:|:---:|:---|:---:|:---:|
| **0** | **Normal** | No internal leakage; nominal volumetric output | 489 cycles (33.7%) | 1,221 cycles (55.4%) |
| **1** | **Warning** | Weak internal leakage; incipient volumetric loss | 480 cycles (33.1%) | 492 cycles (22.3%) |
| **2** | **Fault** | Severe internal leakage; critical pressure/flow slippage | 480 cycles (33.1%) | 492 cycles (22.3%) |
| **Total** | | | **1,449 cycles** | **2,205 cycles** |

---

## 5. Official Python Import & API Notes (`ucimlrepo`)

The official `ucimlrepo` library facilitates programmatic dataset querying from the UCI repository.

```python
from ucimlrepo import fetch_ucirepo

# Querying UCI Dataset ID 447
try:
    dataset = fetch_ucirepo(id=447)
except Exception as e:
    print(f"Outcome: {e}")
```

### Scientific and Technical Finding
When requesting dataset ID 447 via `fetch_ucirepo(id=447)`, the UCI API raises:
```text
DatasetNotFoundError: "Condition monitoring of hydraulic systems" dataset (id=447) exists in the repository, but is not available for import.
```
**Reason:** The UCI web API indexes standard tabular single-file datasets (`Python=true`). Complex multi-rate time-series datasets containing multiple independent matrix files (18 separate `.txt` files totaling >500 MB) are hosted as direct downloadable archives rather than flattened tabular API streams.

**Canonical Source Policy:** The canonical source for SmartPump AI feature engineering is the official 18-file archive extracted into `data/raw/`. The repository provides `src/uci_loader.py` to inspect and verify dataset availability.

---

## 6. Official Academic Citation

```bibtex
@misc{helwig2015condition,
  author       = {Helwig, Nikolai and Pignanelli, Eliseo and Sch{\"u}tze, Andreas},
  title        = {{Condition Monitoring of Hydraulic Systems}},
  year         = {2015},
  howpublished = {UCI Machine Learning Repository},
  doi          = {10.24432/C5CW21}
}

@inproceedings{helwig2015conference,
  author    = {Helwig, Nikolai and Pignanelli, Eliseo and Sch{\"u}tze, Andreas},
  title     = {Condition Monitoring of a Complex Hydraulic System Using Multivariate Statistics},
  booktitle = {2015 IEEE International Instrumentation and Measurement Technology Conference (I2MTC) Proceedings},
  year      = {2015},
  pages     = {210--215},
  doi       = {10.1109/I2MTC.2015.7151267}
}
```
