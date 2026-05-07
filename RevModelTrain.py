############################
##### 1. Preprocessing #####
############################
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, medfilt, find_peaks
import joblib

# ── 1) LOAD DATA
df = pd.read_csv('df.csv')
fs = 125 # Hz sampling rate
dt = 1/fs # 0.008 detik (interval antar sampel)

# Tambahkan kolom waktu agar secara data kita punya sumbu X detik (Domain Waktu)
df['time'] = np.arange(len(df)) * dt

# ── 2) DEFINISI FILTER STANDAR
def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def notch_filter(data, f0=50.0, Q=30.0, fs=fs):
    b, a = iirnotch(f0 / (0.5 * fs), Q)
    return filtfilt(b, a, data)

# ── 3) ECG PREPROCESSING ALUR BARU (Domain Voltase & Waktu)
ecg_raw = df['ECG'].values # Ini adalah Amplitudo (Voltase)

# Step 1: Pembersihan Utama
# Hasilnya tetap dalam satuan Voltase yang sudah bersih
ecg_clean = bandpass_filter(ecg_raw)
ecg_clean = notch_filter(ecg_clean)

# Step 2: Normalisasi
# Dinormalisasi dengan Z-Score Normalization (Standardization) agar variasi amplitudo antar subjek tidak membuat model ML bias
# Mean µ = 0
# Standar Deviasi σ = 1
ecg_mean, ecg_std = ecg_clean.mean(), ecg_clean.std()
ecg_norm = (ecg_clean - ecg_mean) / ecg_std
joblib.dump({'mean': ecg_mean, 'std': ecg_std}, 'params.pkl')

# Step 3: Pan-Tompkins Tahap Lanjut (Domain Waktu)
# Ini menghitung dV/dt (Laju perubahan Voltase terhadap Detik)
y_der = np.gradient(ecg_norm, dt) 

# Squaring: Menonjolkan energi puncak
y_sq = y_der ** 2

# Moving Window Integration (MWI): 
# Menghitung area di bawah kurva dalam jendela 150ms
mwi_win = int(0.150 * fs)
y_mwi = np.convolve(y_sq, np.ones(mwi_win)/mwi_win, mode='same')

# Simpan hasil envelope untuk deteksi R-Peak
df['ecg_preproc'] = y_mwi

# ── 4) EPOCH-LEVEL ABP EXTRACTION
# ABP (mmHg) diolah per waktu dalam detik
epoch_sec = 20
samples_per_epoch = fs * epoch_sec
kernel = int(0.2 * fs)
if kernel % 2 == 0: kernel += 1

epoch_indices = []
epoch_sbp, epoch_dbp = [], []
num_epochs = len(df) // samples_per_epoch

for i in range(num_epochs):
    start = i * samples_per_epoch
    end = start + samples_per_epoch
    epoch_indices.append((start, end))
    
    abp_seg = df['ABP'].iloc[start:end].values # Domain: mmHg
    abp_smooth = medfilt(abp_seg, kernel_size=kernel)
    
    # Deteksi Systolic (Mencari puncak tekanan darah)
    # t = 0.333 detik (Batas maks ~180 BPM)
    locs_s, _ = find_peaks(abp_smooth, distance=int(0.333 * fs))
    vals_s = abp_seg[locs_s]
    
    # Deteksi Diastolic (Mencari lembah tekanan darah)
    locs_d = []
    for j in range(len(locs_s) - 1):
        seg = abp_smooth[locs_s[j]:locs_s[j+1]]
        if seg.size:
            trough = np.argmin(seg)
            locs_d.append(locs_s[j] + trough)
    vals_d = abp_seg[locs_d]
    
    # Ambil median dari SBP dan DBP per epoch (window 20 detik)
    sbp = np.median(np.sort(vals_s)[-epoch_sec:]) if vals_s.size else np.nan
    dbp = np.median(np.sort(vals_d)[-epoch_sec:]) if vals_d.size else np.nan
    
    epoch_sbp.append(sbp)
    epoch_dbp.append(dbp)

epoch_df = pd.DataFrame({'systolic': epoch_sbp, 'diastolic': epoch_dbp})

##############################
##### 2. Definisi Fungsi #####
##############################

def mean_feature(signal):
    """1. MEAN: Rata-rata amplitudo sinyal (Volt)"""
    return np.mean(signal)

def shape_factor(signal):
    """2. SHAPE FACTOR: Mengukur bentuk distribusi sinyal"""
    xrms = np.sqrt(np.mean(signal**2))
    mean_sqrt_abs = np.mean(np.sqrt(np.abs(signal)))
    return xrms / mean_sqrt_abs if mean_sqrt_abs != 0 else 0.0

def hjorth_mobility(signal, fs=125):
    """
    3. MOBILITY: Estimasi frekuensi rata-rata.
    Satuan: Hz. Rumus: sqrt(Var(dy/dt) / Var(y))
    """
    var_y = np.var(signal, ddof=1)
    if var_y == 0: return 0.0
    # Turunan pertama terhadap waktu (dV/dt)
    dy_dt = np.diff(signal) * fs 
    return np.sqrt(np.var(dy_dt, ddof=1) / var_y)

def skewness(x):
    """4. SKEWNESS: Derajat asimetri sinyal"""
    n = len(x)
    mean_x = np.mean(x)
    std_x = np.std(x, ddof=1)
    if std_x == 0: return 0.0
    return np.sum((x - mean_x)**3) / ((n - 1) * (std_x**3))

def coefficient_of_variation(x):
    """5. COEFFICIENT OF VARIATION (CV): Variabilitas relatif (%)"""
    mean_x = np.mean(x)
    std_x = np.std(x, ddof=1)
    if mean_x == 0: return 0.0
    return (std_x / mean_x) * 100

def hjorth_complexity(signal, fs=125):
    """
    6. COMPLEXITY: Mengukur perubahan bandwidth sinyal.
    Rumus melibatkan turunan kedua (d^2y/dt^2).
    """
    mob_y = hjorth_mobility(signal, fs)
    if mob_y == 0: return 0.0
    
    # Turunan pertama dan kedua terhadap waktu
    dy_dt = np.diff(signal) * fs
    d2y_dt2 = np.diff(dy_dt) * fs
    
    var_dy = np.var(dy_dt, ddof=1)
    var_d2y = np.var(d2y_dt2, ddof=1)
    
    if var_dy == 0: return 0.0
    
    # Mobility dari turunan pertama
    mob_dy = np.sqrt(var_d2y / var_dy)
    return mob_dy / mob_y

def central_moment_10(x):
    """7. 10th CENTRAL MOMENT: Statistik momen tingkat tinggi"""
    mean_x = np.mean(x)
    return np.mean((x - mean_x)**10)

# --- FUNGSI TAMBAHAN  ---

def svd_feature(signal_matrix):
    """Singular Value Decomposition: Mengambil nilai singular terbesar"""
    U, S, Vt = np.linalg.svd(signal_matrix, full_matrices=False)
    return S[0]

def hjorth_activity(signal):
    """Hjorth Activity: Identik dengan varians sinyal"""
    return np.var(signal, ddof=1)

def rms_feature(signal):
    """Root Mean Square"""
    return np.sqrt(np.mean(signal**2))

def average_energy(signal):
    """Average Energy (RMS^2)"""
    return np.mean(signal**2)

def standard_deviation(signal):
    """Standard Deviation"""
    return np.std(signal, ddof=1)

##############################
##### 3. Ekstraksi Fitur #####
##############################

# 1. Inisialisasi list untuk menampung semua fitur dalam bentuk dictionary
all_features = []

# 2. Loop melalui setiap epoch (window 20 detik)
for (start, end) in epoch_indices:
    # PENTING: Fitur statistik diekstrak dari ecg_clean (Domain Voltase & Waktu)
    # ecg_clean adalah sinyal setelah Bandpass + Notch, sebelum dikuadratkan/MWI
    ecg_epoch = ecg_clean[start:end]
    
    # Khusus untuk SVD, kita butuh bentuk matriks 2D
    ecg_2d = ecg_epoch.reshape(-1, 1)
    
    # Hitung semua fitur menggunakan fungsi yang sudah diperbaiki di Tahap 2
    feature_dict = {
        # --- 7 Fitur Utama ---
        #'ecg_mean': mean_feature(ecg_epoch),
        'ecg_sf': shape_factor(ecg_epoch),
        'ecg_mobility': hjorth_mobility(ecg_epoch, fs=fs),
        'ecg_skewness': skewness(ecg_epoch),
        #'ecg_cv': coefficient_of_variation(ecg_epoch),
        'ecg_complexity': hjorth_complexity(ecg_epoch, fs=fs),
        'ecg_cm10': central_moment_10(ecg_epoch),
        
        # --- Fitur Tambahan & Pendukung Model ---
        'ecg_rms': rms_feature(ecg_epoch),
        'ecg_energy': average_energy(ecg_epoch),
        'ecg_svd': svd_feature(ecg_2d),
        'ecg_activity': hjorth_activity(ecg_epoch),
        'ecg_std': standard_deviation(ecg_epoch)
    }
    
    all_features.append(feature_dict)

# 3. Gabungkan hasil ekstraksi dengan epoch_df awal (yang berisi SBP/DBP)
# Kita buat DataFrame baru dari list dictionary
features_df = pd.DataFrame(all_features)

# Reset index agar penggabungan tidak berantakan
epoch_df = epoch_df.reset_index(drop=True)
features_df = features_df.reset_index(drop=True)

# Gabungkan secara horizontal
epoch_df = pd.concat([epoch_df, features_df], axis=1)

# Tampilkan hasil akhir untuk pengecekan
print("Fitur berhasil diekstrak. Ukuran data:", epoch_df.shape)
epoch_df.tail()

##############################
##### 4. Pelatihan Model #####
##############################

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1) Definisi Fitur & Target
#feature_columns = ["ecg_mean", "ecg_sf", "ecg_mobility", "ecg_skewness", "ecg_cv", "ecg_complexity", "ecg_cm10"]
feature_columns = ["ecg_sf", "ecg_mobility", "ecg_skewness", "ecg_complexity", "ecg_cm10"]
target_sbp = "systolic"

epoch_df.dropna(inplace=True)
X = epoch_df[feature_columns]
y_sbp = epoch_df[target_sbp].values

# 2) Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y_sbp, test_size=0.2, random_state=42)

# 3) Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'feat_scaler.pkl')

# 4) Training Random Forest
rf_sbp = RandomForestRegressor(n_estimators=500, max_features='sqrt', random_state=42)
rf_sbp.fit(X_train_scaled, y_train)
joblib.dump(rf_sbp, "rf2_sbp.pkl")

# 5) EVALUASI & ANALISIS
y_pred = rf_sbp.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r2 = r2_score(y_test, y_pred)

print(f"Hasil Evaluasi SBP:")
print(f"MAE  : {mae:.2f} mmHg")
print(f"MAPE : {mape:.2f} %")
print(f"R2 Score : {r2:.2f} (Korelasi Prediksi)")

# A. FEATURE IMPORTANCE: Menjelaskan Fitur Mana yang Paling Berpengaruh
importances = rf_sbp.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [feature_columns[i] for i in indices])
plt.xlabel('Relative Importance')

# B. ACTUAL VS PREDICTED PLOT: "Persamaan" Visual Regresi
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual SBP (mmHg)')
plt.ylabel('Predicted SBP (mmHg)')
plt.title('Regresi Estimasi SBP')

plt.show()