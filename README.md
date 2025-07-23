# pyqcoda

**pyqcoda** is a Python library for temporal disaggregation of daily precipitation into hourly time series using a combination of comonotonicity transformation and iterative adjusted k-nearest neighbors (KNN). It is tailored for hydrological and climate data processing tasks.

---

## 🌧️ Overview

- **Input**:
  - `train_data.csv`: A CSV file containing hourly precipitation data with two columns: `datetime` (in hourly resolution) and `precipitation` (in mm).
  - `test_data.csv`: A CSV file containing daily precipitation data with the same column names but with daily resolution (`datetime` at 00:00:00 for each day).

- **Output**:
  - A DataFrame (or CSV file) with hourly precipitation disaggregated from daily values in `test_data`, using patterns learned from `train_data`.

---

## 📦 Installation

```bash
pip install .

