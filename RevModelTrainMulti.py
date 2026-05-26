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

warnings.filterwarnings("ignore")


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

FS = 125
DT = 1 / FS


def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=FS, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def notch_filter(data, f0=50.0, Q=30.0, fs=FS):
    b, a = iirnotch(f0 / (0.5 * fs), Q)
    return filtfilt(b, a, data)


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


feature_columns = [
    "ecg_sf",
    "ecg_mobility",
    "ecg_skewness",
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

    ecg_mean, ecg_std = ecg_clean.mean(), ecg_clean.std()
    if ecg_std == 0:
        ecg_std = 1.0
    ecg_norm = (ecg_clean - ecg_mean) / ecg_std

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
            "ecg_sf": shape_factor(ecg_epoch),
            "ecg_mobility": hjorth_mobility(ecg_epoch, fs=fs),
            "ecg_skewness": skewness(ecg_epoch),
            "ecg_complexity": hjorth_complexity(ecg_epoch, fs=fs),
            "ecg_cm10": central_moment_10(ecg_epoch),
        }
        all_features.append(feature_dict)

    features_df = pd.DataFrame(all_features)
    epoch_df = epoch_df.reset_index(drop=True)
    features_df = features_df.reset_index(drop=True)
    epoch_df = pd.concat([epoch_df, features_df], axis=1)

    epoch_df.dropna(inplace=True)

    return epoch_df, {"mean": ecg_mean, "std": ecg_std}


def main():
    dataset_dir = "BloodPressureDataset"
    all_epoch_dfs = []
    all_ecg_stats = []

    csv_files = sorted(
        [f for f in os.listdir(dataset_dir) if f.endswith(".csv") and f != "DataIndices.csv"],
        key=lambda x: int(x.replace(".csv", "")),
    )

    print(f"{'Dataset':<10} {'Epochs':<8} {'SBP range':<22} {'Time':<12}")
    print("=" * 52)
    total_start = time.time()

    for fname in csv_files:
        filepath = os.path.join(dataset_dir, fname)
        label = fname.replace(".csv", "")
        t0 = time.time()
        print(f">>> {label:<7}", end=" ", flush=True)

        try:
            epoch_df, stats = process_dataset(filepath)
            if len(epoch_df) < 10:
                print(f"SKIP (only {len(epoch_df)} epochs)")
                continue

            epoch_df["dataset"] = int(label)
            all_epoch_dfs.append(epoch_df)
            all_ecg_stats.append(stats)

            elapsed = time.time() - t0
            print(f"{len(epoch_df):<8} {epoch_df[target_sbp].min():.0f}-{epoch_df[target_sbp].max():.0f} mmHg{'':>8} {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"ERROR: {e} ({elapsed:.1f}s)")

    if not all_epoch_dfs:
        print("\nNo datasets produced valid results.")
        return

    combined_df = pd.concat(all_epoch_dfs, axis=0, ignore_index=True)
    label_counts = combined_df["dataset"].value_counts().sort_index()

    print(f"\n{'=' * 60}")
    print(f"Combined: {len(combined_df)} epochs across {len(label_counts)} datasets")
    print(f"SBP range: {combined_df[target_sbp].min():.0f} - {combined_df[target_sbp].max():.0f} mmHg")
    print(f"SBP mean +/- std: {combined_df[target_sbp].mean():.1f} +/- {combined_df[target_sbp].std():.1f} mmHg")
    print()
    print("Per-dataset epoch counts:")
    for ds, cnt in label_counts.items():
        ds_df = combined_df[combined_df["dataset"] == ds]
        print(f"  Dataset {ds:.0f}: {cnt} epochs (SBP {ds_df[target_sbp].min():.0f}-{ds_df[target_sbp].max():.0f} mmHg)")

    X_all = combined_df[feature_columns]
    y_all = combined_df[target_sbp].values
    dataset_labels = combined_df["dataset"].values

    X_train, X_test, y_train, y_test, ds_train, ds_test = train_test_split(
        X_all, y_all, dataset_labels, test_size=0.2, random_state=42
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_sbp = RandomForestRegressor(n_estimators=500, max_features="sqrt", random_state=42, n_jobs=-1)
    print(f"\nTraining Random Forest (500 trees) on {len(X_train)} samples...")
    train_t0 = time.time()
    rf_sbp.fit(X_train_scaled, y_train)
    train_elapsed = time.time() - train_t0
    print(f"Training done in {train_elapsed:.1f}s")

    joblib.dump(rf_sbp, "rf2_sbp.pkl")
    joblib.dump(scaler, "feat_scaler.pkl")

    mean_std = float(np.mean([s["std"] for s in all_ecg_stats]))
    joblib.dump({"mean": 0.0, "std": mean_std}, "params.pkl")
    print(f"\nModels saved: rf2_sbp.pkl, feat_scaler.pkl, params.pkl")
    print(f"params.pkl: mean=0.0, std={mean_std:.6f}")

    y_pred = rf_sbp.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2 = r2_score(y_test, y_pred)

    print(f"\n{'=' * 60}")
    print("GLOBAL EVALUATION (all datasets combined test set):")
    print(f"  MAE  : {mae:.2f} mmHg")
    print(f"  MAPE : {mape:.2f} %")
    print(f"  R2   : {r2:.4f}")
    print(f"  Test samples: {len(y_test)}")

    print(f"\n{'=' * 80}")
    print(f"{'Dataset':<10} {'Samples':<8} {'SBP range':<18} {'MAE':<10} {'MAPE':<10} {'R2':<10}")
    print("-" * 66)

    per_ds_results = []
    for ds_id in sorted(label_counts.index):
        mask = ds_test == ds_id
        if mask.sum() < 3:
            continue
        yt = y_test[mask]
        yp = y_pred[mask]
        mae_ds = mean_absolute_error(yt, yp)
        mape_ds = mean_absolute_percentage_error(yt, yp) * 100
        r2_ds = r2_score(yt, yp)
        per_ds_results.append({
            "label": f"{ds_id:.0f}",
            "mae": mae_ds,
            "mape": mape_ds,
            "r2": r2_ds,
            "samples": mask.sum(),
        })
        print(f"{ds_id:<10.0f} {mask.sum():<8} {yt.min():.0f}-{yt.max():.0f} mmHg{'':>8} {mae_ds:<10.2f} {mape_ds:<10.2f} {r2_ds:<10.4f}")

    results_df = pd.DataFrame(per_ds_results)
    results_df.to_csv("multi_dataset_results.csv", index=False)
    print(f"\nPer-dataset results saved: multi_dataset_results.csv")

    importances = rf_sbp.feature_importances_
    indices = np.argsort(importances)

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Multi-Dataset Model)")
    plt.barh(range(len(indices)), importances[indices], color="#2563eb", align="center")
    plt.yticks(range(len(indices)), [feature_columns[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig("feature_importances_multi.png", dpi=150)
    print("Chart saved: feature_importances_multi.png")

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.15, s=8, c="#2563eb")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", lw=2)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Actual SBP (mmHg)")
    plt.ylabel("Predicted SBP (mmHg)")
    plt.title(f"SBP Regression (Multi-Dataset)\nMAE={mae:.2f} mmHg, MAPE={mape:.2f}%, R2={r2:.4f}")
    plt.tight_layout()
    plt.savefig("actual_vs_predicted_multi.png", dpi=150)
    print("Chart saved: actual_vs_predicted_multi.png")

    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
