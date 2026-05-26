import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, medfilt, find_peaks
from scipy import interpolate
import os
import warnings
import time
import itertools

warnings.filterwarnings("ignore")

FS = 125
DT = 1 / FS


def interpolate_nans(arr):
    mask = np.isnan(arr)
    if not mask.any():
        return arr
    good = np.where(~mask)[0]
    bad = np.where(mask)[0]
    if len(good) < 2:
        arr[mask] = 0.0
        return arr
    f = interpolate.interp1d(good, arr[good], kind="linear", bounds_error=False, fill_value="extrapolate")
    arr[mask] = f(bad)
    return arr


def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=FS, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def notch_filter(data, f0=50.0, Q=30.0, fs=FS):
    b, a = iirnotch(f0 / (0.5 * fs), Q)
    return filtfilt(b, a, data)


def mean_feature(signal):
    return np.mean(signal)


def shape_factor(signal):
    xrms = np.sqrt(np.mean(signal ** 2))
    mean_sqrt_abs = np.mean(np.sqrt(np.abs(signal)))
    return xrms / mean_sqrt_abs if mean_sqrt_abs != 0 else 0.0


def hjorth_mobility(signal, fs=FS):
    var_y = np.var(signal, ddof=1)
    if var_y == 0:
        return 0.0
    dy_dt = np.diff(signal) * fs
    return np.sqrt(np.var(dy_dt, ddof=1) / var_y)


def skewness(x):
    n = len(x)
    mean_x = np.mean(x)
    std_x = np.std(x, ddof=1)
    if std_x == 0:
        return 0.0
    return np.sum((x - mean_x) ** 3) / ((n - 1) * (std_x ** 3))


def coefficient_of_variation(x):
    mean_x = np.mean(x)
    std_x = np.std(x, ddof=1)
    if mean_x == 0:
        return 0.0
    return (std_x / mean_x) * 100


def hjorth_complexity(signal, fs=FS):
    mob_y = hjorth_mobility(signal, fs)
    if mob_y == 0:
        return 0.0
    dy_dt = np.diff(signal) * fs
    d2y_dt2 = np.diff(dy_dt) * fs
    var_dy = np.var(dy_dt, ddof=1)
    var_d2y = np.var(d2y_dt2, ddof=1)
    if var_dy == 0:
        return 0.0
    mob_dy = np.sqrt(var_d2y / var_dy)
    return mob_dy / mob_y


def central_moment_10(x):
    mean_x = np.mean(x)
    return np.mean((x - mean_x) ** 10)


all_feature_columns = [
    "ecg_mean",
    "ecg_sf",
    "ecg_mobility",
    "ecg_skewness",
    "ecg_cv",
    "ecg_complexity",
    "ecg_cm10",
]

target_sbp = "systolic"


def process_dataset(filepath):
    df = pd.read_csv(filepath)
    fs = FS
    dt = DT

    ecg_raw = interpolate_nans(df["ECG"].values.astype(float))
    ecg_clean = bandpass_filter(ecg_raw)
    ecg_clean = notch_filter(ecg_clean)

    ecg_mean_val, ecg_std_val = ecg_clean.mean(), ecg_clean.std()
    if ecg_std_val == 0:
        ecg_std_val = 1.0
    ecg_norm = (ecg_clean - ecg_mean_val) / ecg_std_val

    y_der = np.gradient(ecg_norm, dt)
    y_sq = y_der ** 2
    mwi_win = int(0.150 * fs)
    y_mwi = np.convolve(y_sq, np.ones(mwi_win) / mwi_win, mode="same")

    epoch_sec = 20
    samples_per_epoch = fs * epoch_sec
    kernel = int(0.2 * fs)
    if kernel % 2 == 0:
        kernel += 1

    epoch_indices = []
    epoch_sbp, epoch_dbp = [], []
    num_epochs = len(df) // samples_per_epoch

    for i in range(num_epochs):
        start = i * samples_per_epoch
        end = start + samples_per_epoch
        epoch_indices.append((start, end))

        abp_raw = df["ABP"].iloc[start:end].values.astype(float)
        abp_seg = interpolate_nans(abp_raw)
        abp_smooth = medfilt(abp_seg, kernel_size=kernel)

        locs_s, _ = find_peaks(abp_smooth, distance=int(0.333 * fs))
        vals_s = abp_seg[locs_s]

        locs_d = []
        for j in range(len(locs_s) - 1):
            seg = abp_smooth[locs_s[j]: locs_s[j + 1]]
            if seg.size:
                trough = np.argmin(seg)
                locs_d.append(locs_s[j] + trough)
        vals_d = abp_seg[locs_d]

        sbp = np.median(np.sort(vals_s)[-epoch_sec:]) if vals_s.size else np.nan
        dbp = np.median(np.sort(vals_d)[-epoch_sec:]) if vals_d.size else np.nan

        epoch_sbp.append(sbp)
        epoch_dbp.append(dbp)

    epoch_df = pd.DataFrame({"systolic": epoch_sbp, "diastolic": epoch_dbp})

    all_features = []
    for start, end in epoch_indices:
        ecg_epoch = ecg_clean[start:end]

        feature_dict = {
            "ecg_mean": mean_feature(ecg_epoch),
            "ecg_sf": shape_factor(ecg_epoch),
            "ecg_mobility": hjorth_mobility(ecg_epoch, fs=fs),
            "ecg_skewness": skewness(ecg_epoch),
            "ecg_cv": coefficient_of_variation(ecg_epoch),
            "ecg_complexity": hjorth_complexity(ecg_epoch, fs=fs),
            "ecg_cm10": central_moment_10(ecg_epoch),
        }
        all_features.append(feature_dict)

    features_df = pd.DataFrame(all_features)
    epoch_df = epoch_df.reset_index(drop=True)
    features_df = features_df.reset_index(drop=True)
    epoch_df = pd.concat([epoch_df, features_df], axis=1)

    epoch_df.dropna(inplace=True)

    return epoch_df, {"mean": ecg_mean_val, "std": ecg_std_val}


def main():
    dataset_dir = "BloodPressureDataset"
    all_epoch_dfs = []
    all_ecg_stats = []

    csv_files = sorted(
        [f for f in os.listdir(dataset_dir) if f.endswith(".csv") and f != "DataIndices.csv"],
        key=lambda x: int(x.replace(".csv", "")),
    )

    print(f"{'Processing datasets...':-^60}")
    total_start = time.time()

    for fname in csv_files:
        filepath = os.path.join(dataset_dir, fname)
        label = fname.replace(".csv", "")
        t0 = time.time()
        print(f"  {label:<7}", end=" ", flush=True)

        try:
            epoch_df, stats = process_dataset(filepath)
            if len(epoch_df) < 10:
                print(f"SKIP (only {len(epoch_df)} epochs)")
                continue

            epoch_df["dataset"] = int(label)
            all_epoch_dfs.append(epoch_df)
            all_ecg_stats.append(stats)

            elapsed = time.time() - t0
            print(f"{len(epoch_df):>6} epochs  ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"ERROR: {e}")

    combined_df = pd.concat(all_epoch_dfs, axis=0, ignore_index=True)
    n_datasets = len(all_epoch_dfs)

    print(f"\nCombined: {len(combined_df)} epochs across {n_datasets} datasets")
    print(f"SBP: {combined_df[target_sbp].min():.0f} – {combined_df[target_sbp].max():.0f} mmHg")

    y_sbp = combined_df[target_sbp].values

    total_combinations = sum(
        1 for r in range(1, len(all_feature_columns) + 1)
        for _ in itertools.combinations(all_feature_columns, r)
    )

    print(f"\nTesting {total_combinations} feature combinations...")
    print("=" * 100)
    print(f"{' ':<6} {'#F':<3} {'Features':<60} {'MAE':<10} {'MAPE':<10} {'R2':<10}")
    print("-" * 100)

    results = []
    best_model = None
    best_scaler = None
    best_features = None
    best_r2 = -np.inf
    combo_start = time.time()
    combo_idx = 0

    for r in range(1, len(all_feature_columns) + 1):
        for combo in itertools.combinations(all_feature_columns, r):
            combo_idx += 1
            feature_list = list(combo)
            X = combined_df[feature_list]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y_sbp, test_size=0.2, random_state=42
            )

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            rf = RandomForestRegressor(
                n_estimators=500,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(X_train_scaled, y_train)
            y_pred = rf.predict(X_test_scaled)

            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            r2 = r2_score(y_test, y_pred)

            combo_name = ", ".join(feature_list)
            results.append({
                "fitur_aktif": combo_name,
                "jumlah_fitur": len(feature_list),
                "MAE": mae,
                "MAPE": mape,
                "R2": r2,
            })

            marker = ""
            if r2 > best_r2:
                best_r2 = r2
                best_model = rf
                best_scaler = scaler
                best_features = feature_list
                marker = " ***"

            elapsed = time.time() - combo_start
            pct = combo_idx / total_combinations * 100
            eta = elapsed / combo_idx * (total_combinations - combo_idx)
            print(f"{combo_idx:<6} {len(feature_list):<3} {combo_name:<60} {mae:<10.2f} {mape:<10.2f} {r2:<10.4f}{marker}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="R2", ascending=False).reset_index(drop=True)

    print(f"\n{'=' * 100}")
    print("TOP 10 FEATURE COMBINATIONS (Multi-Dataset):")
    print("-" * 100)
    print(f"{'Rank':<5} {'#F':<3} {'MAE':<10} {'MAPE':<10} {'R2':<10}  Features")
    print("-" * 100)
    for i, row in results_df.head(10).iterrows():
        rank = i + 1
        print(f"{rank:<5} {row['jumlah_fitur']:<3} {row['MAE']:<10.2f} {row['MAPE']:<10.2f} {row['R2']:<10.4f}  {row['fitur_aktif']}")

    print(f"\nBEST:")
    print(f"  Features : {best_features}")
    print(f"  Count    : {len(best_features)}")
    print(f"  R2       : {best_r2:.4f}")
    print(f"  MAE      : {results_df['MAE'].iloc[0]:.4f} mmHg")
    print(f"  MAPE     : {results_df['MAPE'].iloc[0]:.4f} %")

    results_df.to_csv("hasil_kombinasi_fitur_sbp_multi.csv", index=False)
    joblib.dump(best_model, "best_rf_sbp_multi.pkl")
    joblib.dump(best_scaler, "best_feat_scaler_multi.pkl")
    joblib.dump(best_features, "best_feature_columns_multi.pkl")

    joblib.dump(best_model, "rf2_sbp.pkl")
    joblib.dump(best_scaler, "feat_scaler.pkl")

    mean_std = float(np.mean([s["std"] for s in all_ecg_stats]))
    joblib.dump({"mean": 0.0, "std": mean_std}, "params.pkl")

    print(f"\nSaved:")
    print(f"  rf2_sbp.pkl, feat_scaler.pkl, params.pkl    (production)")
    print(f"  best_feature_columns_multi.pkl               (feature list)")

    total_elapsed = time.time() - total_start
    print(f"\nDone in {total_elapsed:.1f}s ({total_combinations} combinations tested)")


if __name__ == "__main__":
    main()
