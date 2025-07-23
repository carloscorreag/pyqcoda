# example.py

import pandas as pd
from pyqcoda import pyqcoda  # Simple import thanks to __init__.py

# 1. Load your training and test data
# - df_train: hourly precipitation data (datetime index, column "precipitation")
# - df_test: daily precipitation data (datetime index, column "precipitation")

df_train = pd.read_csv("train_data.csv", index_col=0, parse_dates=True)
df_test = pd.read_csv("test_data.csv", index_col=0, parse_dates=True)

# 2. Instantiate the pyqcoda class
qc = pyqcoda()

# 3. Perform temporal disaggregation
simulated_series = qc.disaggregate(df_train, df_test)

# 4. Convert results to hourly DataFrame
df_hourly = qc.get_hourly_dataframe(simulated_series)

# 5. Save results to CSV
df_hourly.to_csv("disaggregated_output.csv")
print("Hourly disaggregated precipitation saved to disaggregated_output.csv")


 
