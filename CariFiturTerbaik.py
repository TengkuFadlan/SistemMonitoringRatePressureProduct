import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, medfilt, find_peaks
import joblib
import itertools
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

############################
##### 1. Preprocessing #####
############################

# ── 1) LOAD DATA
df = pd.read_csv("BloodPressureDataset/11.csv")
fs = 125
dt = 1 / fs

df["time"] = np.arange(len(df)) * dt


# ── 2) DEFINISI FILTER STANDAR
def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")  # type: ignore
    return filtfilt(b, a, data)


def notch_filter(data, f0=50.0, Q=30.0, fs=fs):
    b, a = iirnotch(f0 / (0.5 * fs), Q)
    return filtfilt(b, a, data)


# ── 3) ECG PREPROCESSING
ecg_raw = df["ECG"].values
ecg_clean = bandpass_filter(ecg_raw)
ecg_clean = notch_filter(ecg_clean)

ecg_mean_all, ecg_std_all = ecg_clean.mean(), ecg_clean.std()
ecg_norm = (ecg_clean - ecg_mean_all) / ecg_std_all
joblib.dump({"mean": ecg_mean_all, "std": ecg_std_all}, "params.pkl")

y_der = np.gradient(ecg_norm, dt)
y_sq = y_der**2

mwi_win = int(0.150 * fs)
y_mwi = np.convolve(y_sq, np.ones(mwi_win) / mwi_win, mode="same")
df["ecg_preproc"] = y_mwi

# ── 4) EPOCH-LEVEL ABP EXTRACTION
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

##############################
##### 2. Definisi Fungsi #####
##############################


def mean_feature(signal):
    return np.mean(signal)


def shape_factor(signal):
    xrms = np.sqrt(np.mean(signal**2))
    mean_sqrt_abs = np.mean(np.sqrt(np.abs(signal)))
    return xrms / mean_sqrt_abs if mean_sqrt_abs != 0 else 0.0


def hjorth_mobility(signal, fs=125):
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


def hjorth_complexity(signal, fs=125):
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


##############################
##### 3. Ekstraksi Fitur #####
##############################

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

print("Fitur berhasil diekstrak. Ukuran data:", epoch_df.shape)

##############################
##### 4. Pelatihan Model #####
##############################

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

epoch_df.dropna(inplace=True)
y_sbp = epoch_df[target_sbp].values

results = []
best_model = None
best_scaler = None
best_features = None
best_r2 = -np.inf

for r in range(1, len(all_feature_columns) + 1):
    for combo in itertools.combinations(all_feature_columns, r):
        feature_columns = list(combo)
        X = epoch_df[feature_columns]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_sbp, test_size=0.2, random_state=42
        )

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        rf_sbp = RandomForestRegressor(
            n_estimators=500,
            max_features="sqrt",  # type: ignore
            random_state=42,
            n_jobs=-1,  # type: ignore
        )
        rf_sbp.fit(X_train_scaled, y_train)

        y_pred = rf_sbp.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        r2 = r2_score(y_test, y_pred)

        results.append(
            {
                "fitur_aktif": ", ".join(feature_columns),
                "jumlah_fitur": len(feature_columns),
                "MAE": mae,
                "MAPE": mape,
                "R2": r2,
            }
        )

        if r2 > best_r2:
            best_r2 = r2
            best_model = rf_sbp
            best_scaler = scaler
            best_features = feature_columns

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="R2", ascending=False).reset_index(drop=True)

print("\n===== TOP 10 KOMBINASI FITUR TERBAIK =====")
print(results_df.head(10).to_string(index=False))

print("\n===== HASIL TERBAIK =====")
print("Fitur terbaik :", best_features)
print("Jumlah fitur  :", len(best_features))  # type: ignore
print(f"R2 terbaik    : {results_df.loc[0, 'R2']:.4f}")
print(f"MAE terbaik   : {results_df.loc[0, 'MAE']:.4f} mmHg")
print(f"MAPE terbaik  : {results_df.loc[0, 'MAPE']:.4f} %")

results_df.to_csv("hasil_kombinasi_fitur_sbp.csv", index=False)
joblib.dump(best_model, "best_rf_sbp.pkl")
joblib.dump(best_scaler, "best_feat_scaler.pkl")
joblib.dump(best_features, "best_feature_columns.pkl")

print("\nHasil kombinasi fitur disimpan ke: hasil_kombinasi_fitur_sbp.csv")
print("Model terbaik disimpan ke: best_rf_sbp.pkl")
print("Scaler terbaik disimpan ke: best_feat_scaler.pkl")
print("Daftar fitur terbaik disimpan ke: best_feature_columns.pkl")
