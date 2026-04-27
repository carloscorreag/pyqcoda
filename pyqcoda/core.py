import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from itertools import permutations
import re
from scipy.stats import gamma

# ---------------------- Configuration ----------------------
HOURS = [f"PH{str(i).zfill(2)}" for i in range(1, 25)]
DURATION_MAP = {
    "PMAX1H": 1,
    "PMAX2H": 2,
    "PMAX6H": 6,
    "PMAX12H": 12
}

# ---------------------- Utility Functions ----------------------
def get_season(month, seasons_dict=None):
    if seasons_dict is None:
        return (
            "DJF" if month in [12, 1, 2]
            else "MAM" if month in [3, 4, 5]
            else "JJA" if month in [6, 7, 8]
            else "SON"
        )

    for season, months in seasons_dict.items():
        if month in months:
            return season

    raise ValueError(f"Month {month} not assigned to any season.")

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

def fix_flat_percentiles(percentiles):
    percentiles = np.array(percentiles)
    _, idx = np.unique(percentiles, return_index=True)
    return percentiles[np.sort(idx)]

def apply_comonotonicity_transformation(
    p24_test,
    p24_train,
    pmax_train,
    semi_parametrical_mode=None,
    season=None
):

    if semi_parametrical_mode is None:
        qbcd_train = np.sort(p24_train)
        p24_percentiles = np.searchsorted(qbcd_train, p24_test) / len(qbcd_train)
        p24_percentiles = np.clip(p24_percentiles, 0, 1)

        pmax_test = np.array([
            np.interp(p24_percentiles, np.linspace(0, 1, len(pmax_train)), np.sort(pmax_train[:, i]))
            for i in range(pmax_train.shape[1])
        ]).T
        return pmax_test

    else:
        dist_params_dict = semi_parametrical_mode
        if season not in dist_params_dict:
            raise ValueError(f"Season '{season}' not in dist_params_dict.")

        season_dict = dist_params_dict[season]

        if 24 not in season_dict:
            raise ValueError(f"No parameters for 24h window in '{season}'.")

        windows = sorted([w for w in season_dict.keys() if w != 24])
        params_24 = season_dict[24]
        p0_24 = params_24['p0']
        shape_24 = params_24['shape']
        scale_24 = params_24['scale']

        def mixed_cdf(x, p0, shape, scale):
            x = np.asarray(x)
            F = np.zeros_like(x, dtype=float)
            mask = x > 0
            F[~mask] = p0
            if np.any(mask):
                F[mask] = p0 + (1 - p0) * gamma.cdf(x[mask], a=shape, scale=scale)
            return F

        p24_val = float(np.asarray(p24_test).ravel()[0])
        upper = max(1.5 * p24_val, 1e-6)
        x_vals = np.linspace(0.0, upper, 1200)
        cdf_vals = mixed_cdf(x_vals, p0_24, shape_24, scale_24)
        q = float(np.interp(p24_val, x_vals, cdf_vals))
        q = float(np.clip(q, 0, 1))

        pmax_list = []
        for w in windows:
            params = season_dict[w]
            p0 = params['p0']
            shape = params['shape']
            scale = params['scale'] 

            if q <= p0:
                pmax = 0.0
            else:
                p_q = (q - p0) / (1 - p0)
                pmax = gamma.ppf(p_q, a=shape, scale=scale)

            pmax_list.append(float(pmax))

        return np.array(pmax_list)


def adjust_hourly_to_constraints(hourly_base, p24_target, pmax_target, max_iter=20, p24_tolerance=0.04):
    hourly = hourly_base.copy()
    hourly = np.maximum(hourly, 0)
    hard_constraints = ["PMAX1H", "PMAX2H", "PMAX6H", "PMAX12H"]

    def dynamic_maxima(hourly):
        return {
            "PMAX1H": np.max(hourly),
            "PMAX2H": max(np.sum(hourly[i:i+2]) for i in range(23)),
            "PMAX6H": max(np.sum(hourly[i:i+6]) for i in range(19)),
            "PMAX12H": max(np.sum(hourly[i:i+12]) for i in range(13)),
        }

    for _ in range(max_iter):
        for pmax_key in hard_constraints:
            window_size = int(re.findall(r'\d+', pmax_key)[0])
            max_sum = -np.inf
            max_idx = 0
            for i in range(25 - window_size):
                s = hourly[i:i + window_size].sum()
                if s > max_sum:
                    max_sum = s
                    max_idx = i
            diff = pmax_target.get(pmax_key, 0) - max_sum
            if abs(diff) > 0.01:
                hourly[max_idx:max_idx + window_size] += diff / window_size
                hourly = np.maximum(hourly, 0)

        total = hourly.sum()
        if total > 0:
            hourly *= p24_target / total
        else:
            hourly = np.zeros(24)
            hourly[0] = p24_target

        calculated = dynamic_maxima(hourly)
        if all(abs(calculated[k] - pmax_target.get(k, 0)) < 0.1 for k in hard_constraints):
            break

    final_total = hourly.sum()
    if abs(final_total - p24_target) > p24_tolerance and final_total > 0:
        hourly *= p24_target / final_total

    return np.round(hourly, 1)


def autocorrelation_lag1(series):
    if len(series) < 2:
        return np.nan
    if np.std(series[:-1]) == 0 or np.std(series[1:]) == 0:
        return np.nan
    return np.corrcoef(series[:-1], series[1:])[0, 1]


def refine_hourly_distribution(hourly, max_jump=10, window_min=3, window_max=5, use_permutations=False):
    refined = hourly.copy()
    original_maxima = calculate_subdaily_maxima(hourly)
    best = refined.copy()
    best_autocorr = autocorrelation_lag1(refined)

    for i in range(len(refined) - 1):
        if refined[i] != 0.0 and refined[i] < 5.0 and refined[i + 1] > 0.0:
            temp_hourly = refined.copy()
            aux = temp_hourly[i]
            temp_hourly[i] = 0.0
            temp_hourly[i + 1] += aux
            if abs(temp_hourly[i + 1] - temp_hourly[i]) <= max_jump:
                maxima_temp = calculate_subdaily_maxima(temp_hourly)
                if all(abs(maxima_temp[k] - original_maxima[k]) < 0.04 for k in DURATION_MAP.keys() | {"P24"}):
                    autocorr_temp = autocorrelation_lag1(temp_hourly)
                    if autocorr_temp >= best_autocorr:
                        best_autocorr = autocorr_temp
                        best = temp_hourly.copy()

    if use_permutations:
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
                        temp_hourly = best.copy()
                        temp_hourly[start:start + window] = perm
                        maxima_temp = calculate_subdaily_maxima(temp_hourly)
                        if all(abs(maxima_temp[k] - original_maxima[k]) < 0.04 for k in DURATION_MAP.keys() | {"P24"}):
                            autocorr_temp = autocorrelation_lag1(temp_hourly)
                            if autocorr_temp > best_autocorr:
                                best_autocorr = autocorr_temp
                                best = temp_hourly.copy()

    return best


# ---------------------- Main Class ----------------------
class pyqcoda:
    def disaggregate(self, df_train_hourly, df_test_daily, semi_parametrical_mode=None, use_permutations=False, seasons_dict=None):
        df_train_hourly = df_train_hourly.copy()
        df_train_hourly = df_train_hourly[df_train_hourly["precipitation"] >= 0]

        df_train_hourly["date"] = df_train_hourly.index.floor("D")
        grouped = df_train_hourly.groupby("date")["precipitation"].agg(list).reset_index()
        grouped = grouped[grouped["precipitation"].apply(lambda x: len(x) == 24)]

        for i in range(24):
            grouped[f"PH{str(i+1).zfill(2)}"] = grouped["precipitation"].apply(lambda x: x[i])
        grouped = grouped.drop(columns=["precipitation"]).set_index("date")
        grouped[["PMAX1H", "PMAX2H", "PMAX6H", "PMAX12H", "P24"]] = grouped[HOURS].apply(calculate_subdaily_maxima, axis=1, result_type="expand")
        grouped["season"] = grouped.index.month.map(lambda m: get_season(m, seasons_dict))
        df_train = grouped

        df_test = df_test_daily.copy()
        df_test = df_test[df_test["precipitation"] >= 0]

        df_test["P24"] = df_test["precipitation"]
        df_test["season"] = df_test.index.month.map(lambda m: get_season(m, seasons_dict))

        simulations = {}

        for date, row in df_test.iterrows():
            if pd.isnull(row["P24"]):
                continue
            if row["P24"] == 0:
                simulations[date] = np.zeros(24)
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
                if semi_parametrical_mode is not None:
                    pmax_estimated = apply_comonotonicity_transformation(
                        np.array([row["P24"]]),
                        p24_train=None,
                        pmax_train=None,
                        semi_parametrical_mode=semi_parametrical_mode,
                        season=season
                    )
                else:
                    pmax_estimated = apply_comonotonicity_transformation(np.array([row["P24"]]), p24_train, pmax_train)
                pmax_dict = dict(zip(["PMAX1H", "PMAX2H", "PMAX6H", "PMAX12H"], pmax_estimated.flatten()))

                for idx in np.random.permutation(idxs[0]):
                    hourly_series = df_train_valid.iloc[idx][HOURS].fillna(0.0).values
                    if not is_consistent(hourly_series):
                        continue
                    original_sum = np.sum(hourly_series)
                    if original_sum == 0:
                        continue

                    adjusted = hourly_series * (row["P24"] / original_sum)
                    refined = adjust_hourly_to_constraints(adjusted, row["P24"], pmax_dict)
                    final_sum = np.sum(refined)
                    refined = refined * (row["P24"] / final_sum)
                    refined = refine_hourly_distribution(refined, use_permutations=use_permutations)

                    if is_consistent(refined):
                        simulations[date] = refined
                        break
            else:
                for idx in np.random.permutation(idxs[0]):
                    hourly_series = df_train_valid.iloc[idx][HOURS].fillna(0.0).values
                    if not is_consistent(hourly_series):
                        continue
                    original_sum = np.sum(hourly_series)
                    if original_sum == 0:
                        continue

                    refined = hourly_series * (row["P24"] / original_sum)

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
