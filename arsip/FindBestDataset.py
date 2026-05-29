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
import os
import warnings
warnings.filterwarnings("ignore")


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


def mean_feature(signal):
    return np.mean(signal)


def shape_factor(signal):
    xrms = np.sqrt(np.mean(signal**2))
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
    return np.sum((x - mean_x) ** 3) / ((n - 1) * (std_x**3))


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


def svd_feature(signal_matrix):
    try:
        _, S, _ = np.linalg.svd(signal_matrix, full_matrices=False)
        return S[0]
    except np.linalg.LinAlgError:
        return 0.0


def hjorth_activity(signal):
    return np.var(signal, ddof=1)


def rms_feature(signal):
    return np.sqrt(np.mean(signal**2))


def average_energy(signal):
    return np.mean(signal**2)


def standard_deviation(signal):
    return np.std(signal, ddof=1)


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

    df["time"] = np.arange(len(df)) * dt

    ecg_raw = df["ECG"].values
    ecg_clean = bandpass_filter(ecg_raw)
    ecg_clean = notch_filter(ecg_clean)

    ecg_mean, ecg_std = ecg_clean.mean(), ecg_clean.std()
    if ecg_std == 0:
        ecg_std = 1.0
    ecg_norm = (ecg_clean - ecg_mean) / ecg_std

    y_der = np.gradient(ecg_norm, dt)
    y_sq = y_der**2
    mwi_win = int(0.150 * fs)
    y_mwi = np.convolve(y_sq, np.ones(mwi_win) / mwi_win, mode="same")
    df["ecg_preproc"] = y_mwi

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

        abp_seg = df["ABP"].iloc[start:end].values
        abp_smooth = medfilt(abp_seg, kernel_size=kernel)

        locs_s, _ = find_peaks(abp_smooth, distance=int(0.333 * fs))
        vals_s = abp_seg[locs_s]

        locs_d = []
        for j in range(len(locs_s) - 1):
            seg = abp_smooth[locs_s[j] : locs_s[j + 1]]
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
        ecg_2d = ecg_epoch.reshape(-1, 1)

        feature_dict = {
            "ecg_sf": shape_factor(ecg_epoch),
            "ecg_mobility": hjorth_mobility(ecg_epoch, fs=fs),
            "ecg_skewness": skewness(ecg_epoch),
            "ecg_complexity": hjorth_complexity(ecg_epoch, fs=fs),
            "ecg_cm10": central_moment_10(ecg_epoch),
            "ecg_rms": rms_feature(ecg_epoch),
            "ecg_energy": average_energy(ecg_epoch),
            "ecg_svd": svd_feature(ecg_2d),
            "ecg_activity": hjorth_activity(ecg_epoch),
            "ecg_std": standard_deviation(ecg_epoch),
        }
        all_features.append(feature_dict)

    features_df = pd.DataFrame(all_features)
    epoch_df = epoch_df.reset_index(drop=True)
    features_df = features_df.reset_index(drop=True)
    epoch_df = pd.concat([epoch_df, features_df], axis=1)

    epoch_df.dropna(inplace=True)
    if len(epoch_df) < 10:
        return None

    X = epoch_df[feature_columns]
    y_sbp = epoch_df[target_sbp].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_sbp, test_size=0.2, random_state=42
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_sbp = RandomForestRegressor(
        n_estimators=500, max_features="sqrt", random_state=42
    )
    rf_sbp.fit(X_train_scaled, y_train)

    y_pred = rf_sbp.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2 = r2_score(y_test, y_pred)

    return {
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "num_epochs": len(epoch_df),
        "y_test": y_test,
        "y_pred": y_pred,
    }


def main():
    dataset_dir = "BloodPressureDataset"
    results = []

    csv_files = sorted(
        [f for f in os.listdir(dataset_dir) if f.endswith(".csv") and f != "DataIndices.csv"],
        key=lambda x: int(x.replace(".csv", "")),
    )

    print(f"{'Dataset':<10} {'MAE (mmHg)':<12} {'MAPE (%)':<12} {'R2 Score':<10} {'Epochs':<8} {'Rank (MAE)'}")
    print("=" * 70)

    for fname in csv_files:
        filepath = os.path.join(dataset_dir, fname)
        label = fname.replace(".csv", "")
        print(f"\n>>> Processing dataset {label} ...", end=" ", flush=True)

        try:
            result = process_dataset(filepath)
            if result is None:
                print(f"SKIP (too few epochs)")
                continue
            print(f"OK")
            results.append(
                {
                    "label": label,
                    "mae": result["mae"],
                    "mape": result["mape"],
                    "r2": result["r2"],
                    "num_epochs": result["num_epochs"],
                }
            )
        except Exception as e:
            print(f"ERROR: {e}")

    if not results:
        print("\nNo datasets produced valid results.")
        return

    results_sorted = sorted(results, key=lambda r: r["mae"])

    print(f"\n{'=' * 70}")
    print(f"{'Rank':<6} {'Dataset':<10} {'MAE (mmHg)':<12} {'MAPE (%)':<12} {'R2 Score':<10} {'Epochs':<8}")
    print(f"{'-' * 58}")
    for rank, r in enumerate(results_sorted, 1):
        print(
            f"{rank:<6} {r['label']:<10} {r['mae']:<12.2f} {r['mape']:<12.2f} {r['r2']:<10.4f} {r['num_epochs']:<8}"
        )

    best = results_sorted[0]
    print(f"\n{'=' * 70}")
    print(f"BEST DATASET: {best['label']}.csv")
    print(f"  MAE  : {best['mae']:.2f} mmHg")
    print(f"  MAPE : {best['mape']:.2f} %")
    print(f"  R2   : {best['r2']:.4f}")

    plt.figure(figsize=(10, 6))
    labels = [r["label"] for r in results_sorted]
    mae_vals = [r["mae"] for r in results_sorted]

    colors = ["#2ecc71" if i == 0 else "#e74c3c" for i in range(len(results_sorted))]
    bars = plt.bar(labels, mae_vals, color=colors, edgecolor="black", linewidth=0.8)
    plt.xlabel("Dataset")
    plt.ylabel("MAE (mmHg)")
    plt.title("Perbandingan MAE antar Dataset")
    plt.xticks(rotation=45)

    for bar, val in zip(bars, mae_vals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig("dataset_comparison_mae.png", dpi=150)
    print(f"\nChart saved: dataset_comparison_mae.png")

    plt.figure(figsize=(10, 6))
    r2_vals = [r["r2"] for r in results_sorted]
    colors_r2 = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(results_sorted))]
    bars2 = plt.bar(labels, r2_vals, color=colors_r2, edgecolor="black", linewidth=0.8)
    plt.xlabel("Dataset")
    plt.ylabel("R2 Score")
    plt.title("Perbandingan R2 Score antar Dataset")
    plt.xticks(rotation=45)

    for bar, val in zip(bars2, r2_vals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig("dataset_comparison_r2.png", dpi=150)
    print(f"Chart saved: dataset_comparison_r2.png")

    results_df = pd.DataFrame(results_sorted)
    results_df.to_csv("dataset_comparison_results.csv", index=False)
    print(f"Results saved: dataset_comparison_results.csv")


if __name__ == "__main__":
    main()
