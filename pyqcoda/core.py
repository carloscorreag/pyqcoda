import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from itertools import permutations
import re

# ---------------------- Configuration ----------------------
HOURS = [f"PH{str(i).zfill(2)}" for i in range(1, 25)]
DURATION_MAP = {
    "PMAX1H": 1,
    "PMAX2H": 2,
    "PMAX6H": 6,
    "PMAX12H": 12
}

# ---------------------- Utility Functions ----------------------
def get_season(month):
    return "DJF" if month in [12, 1, 2] else "MAM" if month in [3, 4, 5] else "JJA" if month in [6, 7, 8] else "SON"

def calculate_subdaily_maxima(hourly_values):
    hourly_values = np.array(hourly_values, dtype=np.float32)
    valid = ~np.isnan(hourly_values) & (hourly_values != -999.0)
    filtered = hourly_values[valid]
    if len(filtered) == 0:
        return {k: -999.0 for k in DURATION_MAP.keys() | {"P24"}}
    valid_values = np.where(valid, hourly_values, 0.0)
    return {
        "PMAX1H": np.max(valid_values),
        "PMAX2H": max(np.sum(valid_values[i:i+2]) for i in range(23)),
        "PMAX6H": max(np.sum(valid_values[i:i+6]) for i in range(19)),
        "PMAX12H": max(np.sum(valid_values[i:i+12]) for i in range(13)),
        "P24": np.sum(filtered),
    }

def is_consistent(hourly):
    if pd.isnull(hourly).any():
        return False
    hourly = np.nan_to_num(hourly, nan=0.0)
    p1h = np.max(hourly)
    p2h = max(np.sum(hourly[i:i+2]) for i in range(23))
    p6h = max(np.sum(hourly[i:i+6]) for i in range(19))
    p12h = max(np.sum(hourly[i:i+12]) for i in range(13))
    p24 = np.sum(hourly)
    return p1h <= p2h <= p6h <= p12h <= p24

def apply_comonotonicity_transformation(p24_test, p24_train, pmax_train):
    qbcd_train = np.sort(p24_train)
    p24_percentiles = np.searchsorted(qbcd_train, p24_test) / len(qbcd_train)
    p24_percentiles = np.clip(p24_percentiles, 0, 1)
    pmax_test = np.array([
        np.interp(p24_percentiles, np.linspace(0, 1, len(pmax_train)), np.sort(pmax_train[:, i]))
        for i in range(pmax_train.shape[1])
    ]).T
    return pmax_test

def adjust_hourly_to_constraints(ph_base, p24_target, pmax_target, max_iter=20, p24_tolerance=0.04):
    ph = ph_base.copy()
    ph = np.maximum(ph, 0)
    hard_constraints = ["PMAX1H", "PMAX2H", "PMAX6H", "PMAX12H"]

    def dynamic_maxima(ph):
        return {
            "PMAX1H": np.max(ph),
            "PMAX2H": max(np.sum(ph[i:i+2]) for i in range(23)),
            "PMAX6H": max(np.sum(ph[i:i+6]) for i in range(19)),
            "PMAX12H": max(np.sum(ph[i:i+12]) for i in range(13)),
        }

    for _ in range(max_iter):
        for pmax_key in hard_constraints:
            d = int(re.findall(r'\d+', pmax_key)[0])
            max_sum = -np.inf
            max_idx = 0
            for i in range(25 - d):
                s = ph[i:i + d].sum()
                if s > max_sum:
                    max_sum = s
                    max_idx = i
            diff = pmax_target.get(pmax_key, 0) - max_sum
            if abs(diff) > 0.01:
                ph[max_idx:max_idx + d] += diff / d
                ph = np.maximum(ph, 0)

        total = ph.sum()
        if total > 0:
            ph *= p24_target / total
        else:
            ph = np.zeros(24)
            ph[0] = p24_target

        calculated = dynamic_maxima(ph)
        if all(abs(calculated[k] - pmax_target.get(k, 0)) < 0.1 for k in hard_constraints):
            break

    final_total = ph.sum()
    if abs(final_total - p24_target) > p24_tolerance and final_total > 0:
        ph *= p24_target / final_total

    return np.round(ph, 1)

def autocorrelation_lag1(series):
    if len(series) < 2:
        return np.nan
    if np.std(series[:-1]) == 0 or np.std(series[1:]) == 0:
        return np.nan
    return np.corrcoef(series[:-1], series[1:])[0, 1]

def refine_hourly_distribution(ph, max_jump=10, window_min=3, window_max=5):
    refined = ph.copy()
    original_maxima = calculate_subdaily_maxima(ph)
    best = refined.copy()
    best_autocorr = autocorrelation_lag1(refined)

    for i in range(len(refined) - 1):
        if refined[i] != 0.0 and refined[i] < 5.0 and refined[i + 1] > 0.0:
            temp = refined.copy()
            aux = temp[i]
            temp[i] = 0.0
            temp[i + 1] += aux
            if abs(temp[i + 1] - temp[i]) <= max_jump:
                maxima_temp = calculate_subdaily_maxima(temp)
                if all(abs(maxima_temp[k] - original_maxima[k]) < 0.04 for k in DURATION_MAP.keys() | {"P24"}):
                    autocorr_temp = autocorrelation_lag1(temp)
                    if autocorr_temp >= best_autocorr:
                        best_autocorr = autocorr_temp
                        best = temp.copy()

    for window in range(window_min, window_max + 1):
        for start in range(len(best) - window + 1):
            segment = best[start:start + window]
            if 0.0 in segment:
                continue
            diffs = np.abs(np.diff(segment))
            if np.all(diffs <= max_jump):
                for perm in permutations(segment):
                    if list(perm) == list(segment):
                        continue
                    temp = best.copy()
                    temp[start:start + window] = perm
                    maxima_temp = calculate_subdaily_maxima(temp)
                    if all(abs(maxima_temp[k] - original_maxima[k]) < 0.04 for k in DURATION_MAP.keys() | {"P24"}):
                        autocorr_temp = autocorrelation_lag1(temp)
                        if autocorr_temp > best_autocorr:
                            best_autocorr = autocorr_temp
                            best = temp.copy()
    return best

# ---------------------- Main Class ----------------------
class pyqcoda:
    def disaggregate(self, df_train_hourly, df_test_daily):
        # Preprocess train data: convert from hourly format to daily format with PH01–PH24 + P24 + PMAX
        df_train_hourly = df_train_hourly.copy()
        df_train_hourly = df_train_hourly[df_train_hourly["precipitation"] >= 0]

        df_train_hourly["date"] = df_train_hourly.index.floor("D")
        grouped = df_train_hourly.groupby("date")["precipitation"].agg(list).reset_index()
        grouped = grouped[grouped["precipitation"].apply(lambda x: len(x) == 24)]

        for i in range(24):
            grouped[f"PH{str(i+1).zfill(2)}"] = grouped["precipitation"].apply(lambda x: x[i])
        grouped = grouped.drop(columns=["precipitation"]).set_index("date")
        grouped[["PMAX1H", "PMAX2H", "PMAX6H", "PMAX12H", "P24"]] = grouped[HOURS].apply(calculate_subdaily_maxima, axis=1, result_type="expand")
        grouped["season"] = grouped.index.month.map(get_season)
        df_train = grouped

        df_test = df_test_daily.copy()
        df_test = df_test[df_test["precipitation"] >= 0]

        df_test["P24"] = df_test["precipitation"]
        df_test["season"] = df_test.index.month.map(get_season)


        simulations = {}

        for date, row in df_test.iterrows():
            if pd.isnull(row["P24"]) or row["P24"] <= 0:
                continue

            season = row["season"]
            df_train_season = df_train[df_train["season"] == season]
            if df_train_season.empty:
                continue

            p24_train = df_train_season["P24"].dropna().values
            pmax_train = df_train_season.dropna(subset=["PMAX1H", "PMAX2H", "PMAX6H", "PMAX12H"])
            pmax_train = pmax_train[["PMAX1H", "PMAX2H", "PMAX6H", "PMAX12H"]].values

            if len(p24_train) < 10 or len(pmax_train) < 10:
                continue

            p24_max_train = p24_train.max()
            df_train_valid = df_train_season.dropna(subset=["P24"] + HOURS + ["PMAX1H", "PMAX2H", "PMAX6H", "PMAX12H"])
            p24_train_knn = df_train_valid["P24"].values.reshape(-1, 1)

            if len(p24_train_knn) < 10:
                continue

            nn = NearestNeighbors(n_neighbors=10).fit(p24_train_knn)
            _, idxs = nn.kneighbors([[row["P24"]]])

            if row["P24"] <= p24_max_train:
                pmax_estimated = apply_comonotonicity_transformation(np.array([row["P24"]]), p24_train, pmax_train)
                pmax_dict = dict(zip(["PMAX1H", "PMAX2H", "PMAX6H", "PMAX12H"], pmax_estimated.flatten()))

                for idx in np.random.permutation(idxs[0]):
                    serie = df_train_valid.iloc[idx][HOURS].fillna(0.0).values
                    if not is_consistent(serie):
                        continue
                    original_sum = np.sum(serie)
                    if original_sum == 0:
                        continue

                    adjusted = serie * (row["P24"] / original_sum)
                    refined = adjust_hourly_to_constraints(adjusted, row["P24"], pmax_dict)
                    final_sum = np.sum(refined)
                    refined = refined * (row["P24"] / final_sum)
                    refined = refine_hourly_distribution(refined)

                    if is_consistent(refined):
                        simulations[date] = refined
                        break
            else:
                for idx in np.random.permutation(idxs[0]):
                    serie = df_train_valid.iloc[idx][HOURS].fillna(0.0).values
                    if not is_consistent(serie):
                        continue
                    original_sum = np.sum(serie)
                    if original_sum == 0:
                        continue

                    refined = serie * (row["P24"] / original_sum)

                    if is_consistent(refined):
                        simulations[date] = refined
                        break

        return simulations

    def get_hourly_dataframe(self, simulations):
        records = []
        for date, hourly_values in simulations.items():
            for i, value in enumerate(hourly_values):
                records.append({
                    "datetime": date + pd.Timedelta(hours=i),
                    "precipitation": round(value, 1)
                })
        return pd.DataFrame.from_records(records).set_index("datetime").sort_index()
