# pyqcoda

**pyqcoda** is a Python library for temporal disaggregation of daily precipitation into hourly time series using a combination of **comonotonicity transformation** and an **iterative adjusted k-nearest neighbors (KNN)** algorithm. It is tailored for hydrological and climate data processing tasks where hourly data is required but only daily observations are available.

---

## 🌧️ Overview

- **Input:**
  - `train_data.csv`: Hourly precipitation data with columns `datetime` (hourly resolution) and `precipitation` (mm).
  - `test_data.csv`: Daily precipitation data with the same column names but daily resolution (`datetime` at 00:00:00 for each day).

- **Output:**
  - A pandas DataFrame (or CSV) with hourly precipitation disaggregated from the daily values in `test_data`, using statistical patterns learned from `train_data`.

---

## ✨ Features

- Disaggregates daily totals into 24-hour precipitation series.
- Preserves sub-daily maxima in reconstructed data.
- Season-aware (DJF, MAM, JJA, SON) to capture seasonal variability.
- Combines **comonotonicity** with **KNN-based iterative adjustments**.
- Suitable for hydrological modeling and climate studies.

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

## 🚀 Usage example

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
Correa Guinea, C. (2025). pyqcoda: Temporal disaggregation of daily precipitation into hourly using Q-CODA. DOI: 
