# example.py

import pandas as pd
from pyqcoda.core import pyqcoda

# 1. Load your training and test data
# These should be preprocessed with appropriate columns:
# - df_train should have datetime index with hourly resolution and a columns: Precipitation
# - df_test should have datetime index with daily resolution and a column: Precipitation

# Example: Load from CSV
df_train = pd.read_csv("train_data.csv", index_col=0, parse_dates=True)
df_test = pd.read_csv("test_data.csv", index_col=0, parse_dates=True)

# 2. Instantiate pyqcoda and disaggregate
qc = pyqcoda()
simulated_series = qc.disaggregate(df_train, df_test)

# 3. Convert results to hourly DataFrame
df_hourly = qc.get_hourly_dataframe(simulated_series)

# 4. Save output if needed
df_hourly.to_csv("disaggregated_output.csv")
print("Saved hourly disaggregated precipitation to disaggregated_output.csv")
 
