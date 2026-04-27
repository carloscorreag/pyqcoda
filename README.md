# pyqcoda

**pyqcoda** is a Python library for temporal disaggregation of daily precipitation into hourly time series using a combination of **comonotonicity transformation** and an **iterative adjusted k-nearest neighbors (KNN)** algorithm. It is tailored for hydrological and climate data processing tasks where hourly data is required but only daily observations are available.

---

## 🌧️ Overview

- **Input:**
  - `train_data.csv`: Hourly precipitation data with columns `datetime` (hourly resolution) and `precipitation` (mm).
  - `test_data.csv`: Daily precipitation data with the same column names but daily resolution (`datetime` at 00:00:00 for each day).
  - *(Optional)* `params.csv`: Parameters for the semi-parametric Bernoulli-Gamma mode.
  - *(Optional)* `seasons.csv`: User-defined climatological seasons because default seasons are DJF, MAM, JJA, SON.

- **Output:**
  - A pandas DataFrame (or CSV) with hourly precipitation disaggregated from the daily values in `test_data`, using statistical patterns learned from `train_data`.

---

## ✨ Features

- Disaggregates daily totals into 24-hour precipitation series.
- Preserves sub-daily maxima in reconstructed data.
- Season-aware (DJF, MAM, JJA, SON) to capture seasonal variability.
- Combines **comonotonicity** with **KNN-based iterative adjustments**.
- Suitable for hydrological modeling and climate studies.
- Optional **semi-parametric Bernoulli-Gamma mode**.
- Optional **enhanced autocorrelation refinement** via permutations.

---

## 📦 Installation

### From PyPI (recommended)

```bash
pip install pyqcoda
```
### From Github

```bash
git clone https://github.com/carloscorreag/pyqcoda.git
cd pyqcoda
pip install .
```

---

## 🚀 Usage examples

### 🔹 1. Standard mode (default)

```python
import pandas as pd
from pyqcoda import pyqcoda

# 1. Load your training (hourly) and testing (daily) datasets
df_train = pd.read_csv("train_data.csv", index_col=0, parse_dates=True)
df_test = pd.read_csv("test_data.csv", index_col=0, parse_dates=True)

# 2. Instantiate pyqcoda and disaggregate
qc = pyqcoda()
simulated_series = qc.disaggregate(df_train, df_test)

# 3. Convert results to hourly DataFrame
df_hourly = qc.get_hourly_dataframe(simulated_series)

# 4. Save output
df_hourly.to_csv("disaggregated_output.csv")
print("Hourly disaggregated precipitation saved to disaggregated_output.csv")
```

### 🔹 2. Semi-parametric mode (Load Bernoulli-Gamma params with CSV)

This mode uses fitted Bernoulli-Gamma distributions instead of the empirical transformation.

```python
import pandas as pd
from pyqcoda import pyqcoda

params_df = pd.read_csv("params.csv")

# Convert to dictionary required by pyqcoda
params = {}
for _, row in params_df.iterrows():
    season = row["season"]
    duration = int(row["duration"])

    params.setdefault(season, {})
    params[season][duration] = {
        "p0": row["p0"],
        "shape": row["shape"],
        "scale": row["scale"]
    }

qc = pyqcoda()
simulated_series = qc.disaggregate(
    df_train,
    df_test,
    semi_parametrical_mode=params
)

df_hourly = qc.get_hourly_dataframe(simulated_series)

df_hourly.to_csv("disaggregated_output.csv")
print("Hourly disaggregated precipitation saved to disaggregated_output.csv")

```
📄 Format of params.csv

The file must contain one row per:

season (DJF, MAM, JJA, SON)
duration (1, 2, 6, 12, 24)

Example params.csv

```csv
season,duration,p0,shape,scale
DJF,24,0.3,2.1,5.0
DJF,1,0.5,1.2,2.0
DJF,2,0.45,1.5,2.5
DJF,6,0.4,2.0,3.0
DJF,12,0.35,2.3,4.0
MAM,24,0.25,2.5,4.5
MAM,1,0.4,1.8,2.2
```

### 🔹 3.  Custom seasons (user-defined climatological seasons)

By default, **pyqcoda** uses standard climatological seasons:

- DJF (Dec–Jan–Feb)
- MAM (Mar–Apr–May)
- JJA (Jun–Jul–Aug)
- SON (Sep–Oct–Nov)

However, users can define **custom seasonal partitions** using a CSV file, in a way fully consistent with the `params.csv` workflow.

```python
import pandas as pd
from pyqcoda import pyqcoda

seasons_df = pd.read_csv("seasons.csv")

# Seasons mapping
seasons = {}
for _, row in seasons_df.iterrows():
    season = row["season"]
    month = int(row["month"])

    seasons.setdefault(season, []).append(month)


qc = pyqcoda()
simulated_series = qc.disaggregate(
    df_train,
    df_test,
    seasons_dict=seasons  
)

df_hourly = qc.get_hourly_dataframe(simulated_series)

df_hourly.to_csv("disaggregated_output.csv")
print("Hourly disaggregated precipitation saved to disaggregated_output.csv")
```

📄 `seasons.csv` format

The file must define a mapping between:

- `season` → custom season name
- `month` → month number (1–12)

Each month must belong to exactly one season.

Example

```csv
season,month
WET,10
WET,11
WET,12
WET,1
WET,2
WET,3
DRY,4
DRY,5
DRY,6
DRY,7
DRY,8
DRY,9
```
- All 12 months (1–12) must be assigned exactly once.
- Season names in seasons.csv must match those used in params.csv if using semi-parametric mode. For example, if user define WET and DRY seasons, params.csv must contain them:

```csv
season,duration,p0,shape,scale
WET,24,0.5,2.1,5.0
WET,1,0.5,1.2,2.0
WET,2,0.5,1.5,2.5
WET,6,0.5,2.0,3.0
WET,12,0.5,2.3,4.0
DRY,24,0.25,2.5,4.5
DRY,1,0.25,1.3,2.5
DRY,2,0.25,1.7,3
DRY,6,0.25,1.9,4
DRY,12,0.25,2.2,4.2
```
- Overlapping or missing months will raise an error.
- This feature is fully optional: if seasons_dict=None, default climatological seasons are used.


###  🔹 4. Enhanced autocorrelation refinement (permutations mode)

**pyqcoda** includes an optional advanced refinement step designed to improve the **temporal structure** of the reconstructed hourly precipitation series, specifically targeting **lag-1 autocorrelation**.

This mode applies a **local permutation-based optimization** over short hourly windows while preserving:

- Daily totals (`P24`)
- Sub-daily maxima constraints (`PMAX1H`, `PMAX2H`, `PMAX6H`, `PMAX12H`)
- Physical consistency rules

What this mode does

When enabled, the algorithm:

1. Selects short rolling windows (typically 3–5 hours)
2. Generates permutations of values within each window
3. Evaluates each candidate series using:
   - Sub-daily maxima preservation
   - Constraint consistency
   - Lag-1 autocorrelation improvement
4. Keeps the configuration that maximizes temporal coherence

How to use

Enable the mode by setting `use_permutations=True` in `disaggregate`:

```python
from pyqcoda import pyqcoda

qc = pyqcoda()

simulated_series = qc.disaggregate(
    df_train,
    df_test,
    use_permutations=True
)

df_hourly = qc.get_hourly_dataframe(simulated_series)

df_hourly.to_csv("disaggregated_output.csv")
print("Hourly disaggregated precipitation saved to disaggregated_output.csv")
```

---

## 🔧 Requirements

- Python 3.7+
- pandas ≥ 1.2.4
- numpy ≥ 1.21.6
- scikit-learn ≥ 1.0.2

---

## 📄 License
This project is licensed under the MIT License — see the LICENSE file for details.

---

## 📖 Citation
Correa, C. (2025). pyqcoda: Temporal disaggregation of daily precipitation into hourly using Q-CODA. [![DOI](https://zenodo.org/badge/1024745705.svg)](https://doi.org/10.5281/zenodo.16364100)
