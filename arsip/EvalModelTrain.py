############################
##### 1. Preprocessing #####
############################

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, lfilter, medfilt, find_peaks
import joblib

# ── 1) LOAD YOUR DATA
df = pd.read_csv('BloodPressureDataset/8.csv')

# ── 2) SETUP FILTERS & PARAMETERS
fs = 125 # Hz sampling rate

# (a) Baseline-wander high-pass
def highpass_filter(x, cutoff=0.5, fs=fs, order=2):
    b, a = butter(order, cutoff/(0.5*fs), btype='high')
    return filtfilt(b, a, x)
    
# (b) Notch at mains frequency
def notch_filter(x, fs=fs, f0=50.0, Q=30.0):
    b, a = iirnotch(f0/(fs/2), Q)
    return filtfilt(b, a, x)
    
# (c) Pan–Tompkins low-pass: H_lp(z) = (1 - z^-6)^2 / (1 - z^-1)^2
b_lp = np.zeros(13)
b_lp[0] = 1
b_lp[6] = -2
b_lp[12] = 1
# denominator: 1 - 2z^-1 + z^-2
a_lp = np.array([1, -2, 1])

# (d) Pan–Tompkins high-pass: H_hp(z) = z^-16 - (1 - z^-32)/(1 - z^-1)
# Expand numerator: -1 for z^0..z^31, +1 at z^16
b_hp = -np.ones(32)
b_hp[16] += 1
# denominator: 1 - z^-1
a_hp = np.array([1, -1])

# (e) Derivative (5-point) kernel: y[n] = (1/8)[2x(n) + x(n-1) - x(n-3) - 2x(n-4)]
deriv_kernel = np.array([2, 1, 0, -1, -2]) / 8.0

# (f) Moving-window integration (150 ms)
mwi_win = int(0.150 * fs)
mwi_kernel = np.ones(mwi_win) / mwi_win

# ── 3) ECG PREPROCESSING
ecg_raw = df['ECG'].values

# 1) high-pass → notch
ecg_hp = highpass_filter(ecg_raw)
ecg_notch = notch_filter(ecg_hp)

# 2) compute & save global normalization params
ecg_mean = ecg_notch.mean()
ecg_std = ecg_notch.std()
ecg_stats = {
    'mean': ecg_mean,
    'std': ecg_std,
    'min': ecg_notch.min(),
    'max': ecg_notch.max()
}
joblib.dump(ecg_stats, 'params.pkl')

# 3) z-score normalize
ecg_norm = (ecg_notch - ecg_mean) / ecg_std

# 4) Pan–Tompkins filters
# a) low-pass
y_lp = lfilter(b_lp, a_lp, ecg_norm)
# b) high-pass
y_hp = lfilter(b_hp, a_hp, y_lp)
# c) derivative
y_der = np.convolve(y_hp, deriv_kernel, mode='same')
# d) squaring
y_sq = y_der ** 2
# e) moving-window integration
y_mwi = np.convolve(y_sq, mwi_kernel, mode='same')

# 5) store envelope (for QRS detection / features)
df['ecg_preproc'] = y_mwi

# ── 4) EPOCH-LEVEL ABP EXTRACTION
epoch_sec = 20
samples_per_epoch = fs * epoch_sec
# median filter kernel for ABP (0.2 s)
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
    
    abp_seg = df['ABP'].iloc[start:end].values
    abp_smooth = medfilt(abp_seg, kernel_size=kernel)
    
    # systolic peaks
    locs_s, _ = find_peaks(abp_smooth, distance=int(0.25 * fs))
    vals_s = abp_seg[locs_s]
    
    # diastolic troughs between peaks
    locs_d = []
    for j in range(len(locs_s) - 1):
        seg = abp_smooth[locs_s[j]:locs_s[j+1]]
        if seg.size:
            trough = np.argmin(seg)
            locs_d.append(locs_s[j] + trough)
    vals_d = abp_seg[locs_d]
    
    # median of top-beats
    sbp = np.median(np.sort(vals_s)[-epoch_sec:]) if vals_s.size else np.nan
    dbp = np.median(np.sort(vals_d)[-epoch_sec:]) if vals_d.size else np.nan
    
    epoch_sbp.append(sbp)
    epoch_dbp.append(dbp)
    
# assemble results
epoch_df = pd.DataFrame({'systolic': epoch_sbp, 'diastolic': epoch_dbp})
epoch_df.tail()

##############################
##### 2. Definisi Fungsi #####
##############################

def shape_factor(signal):
    """
    Compute the Shape Factor of a 1D signal array.
    SF = X_rms / ( (1/n) * sum of sqrt(|x_i|) )
    """
    n = len(signal)
    xrms = np.sqrt(np.mean(signal**2))
    # compute mean of sqrt(|x_i|)
    mean_sqrt_abs = np.mean(np.sqrt(np.abs(signal)))
    sf = xrms / mean_sqrt_abs
    return sf

def svd_feature(signal_matrix):
    """
    Compute a feature from the SVD of a 2D 'signal_matrix'.
    For instance, we might return the largest singular value (sigma_1).
    """
    # Perform SVD: X = U * S * V^T
    U, S, Vt = np.linalg.svd(signal_matrix, full_matrices=False)
    # Example: return the largest singular value
    return S[0] # sigma_1

def mean_feature(signal):
    """
    Compute the mean of a 1D signal array.
    Formula:
    mean = (1 / n) * sum(x_i), for i in [1..n]
    """
    return np.mean(signal)

def rms_feature(signal):
    """
    Compute the RMS (root mean square) of a 1D signal array.
    Formula:
    RMS = sqrt( (1/n) * sum(x_i^2) ), for i in [1..n]
    """
    return np.sqrt(np.mean(signal**2))

def average_energy(signal):
    """
    Compute the average energy of a 1D signal array.
    Formula:
    Energy = (1/n) * sum(x_i^2), for i in [1..n]
    Note: This is equivalent to RMS^2.
    """
    return np.mean(signal**2)

def skewness(x):
    """
    Computes the sample skewness of x.
    Formula (for sample skewness):
    skew = [ Σ (x_i - mean_x)^3 ] / [ (n-1) * (std_x^3) ]
    """
    n = len(x)
    mean_x = np.mean(x)
    std_x = np.std(x, ddof=1) # sample standard deviation
    if std_x == 0:
        return 0.0
    skew = np.sum((x - mean_x)**3) / ((n - 1) * (std_x**3))
    return skew

def coefficient_of_variation(x):
    """
    Computes the Coefficient of Variation (CV) of x, in percentage.
    Formula:
    CV = (std_x / mean_x) * 100
    """
    mean_x = np.mean(x)
    std_x = np.std(x, ddof=1)
    if mean_x == 0:
        return 0.0
    return (std_x / mean_x) * 100

def standard_deviation(signal):
    """
    Computes the sample standard deviation of a 1D signal.
    """
    return np.std(signal, ddof=1)

def hjorth_activity(signal):
    """
    Hjorth Activity is the variance of the signal.
    """
    return np.var(signal, ddof=1)

def hjorth_mobility(signal):
    """
    Hjorth Mobility = sqrt( Var(d/dt(signal)) / Var(signal) )
    """
    activity = hjorth_activity(signal)
    if activity == 0:
        return 0.0
    first_deriv = np.diff(signal)
    return np.sqrt(np.var(first_deriv, ddof=1) / activity)

def hjorth_complexity(signal):
    """
    Hjorth Complexity = (Mobility of first derivative) / (Mobility of original signal).
    Equivalent to:
    sqrt(
    [Var(d^2/dt^2(signal)) / Var(d/dt(signal))] /
    [Var(d/dt(signal)) / Var(signal)]
    )
    """
    activity = hjorth_activity(signal)
    mob = hjorth_mobility(signal)
    if mob == 0:
        return 0.0
        
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)
    var_first = np.var(first_deriv, ddof=1)
    var_second = np.var(second_deriv, ddof=1)
    
    if var_first == 0:
        return 0.0
        
    return np.sqrt((var_second / var_first) / (var_first / activity))

def central_moment_10(x):
    """
    Computes the 10th central moment of x.
    Formula (general nth central moment):
    CM_n = (1/n) * Σ (x_i - mean_x)^n
    Here we use n=10 by default.
    """
    mean_x = np.mean(x)
    return np.mean((x - mean_x)**10)

##############################
##### 3. Ekstraksi Fitur #####
##############################

# 1. Create lists to store features for each epoch
ecg_means = []
ecg_rms_vals = []
ecg_energy_vals = []
ecg_sf_vals = []
ecg_svd_vals = []

# -- New features --
ecg_skewness_vals = []
ecg_cv_vals = []
ecg_activity_vals = []
ecg_mobility_vals = []
ecg_cm10_vals = []
ecg_complexity_vals = []
ecg_std_vals = []

# 2. Loop over each epoch, extract features
for (start, end) in epoch_indices:
    ecg_epoch = df['ecg_preproc'].iloc[start:end].values
    # Existing features
    val_mean = mean_feature(ecg_epoch)
    val_rms = rms_feature(ecg_epoch)
    val_energy = average_energy(ecg_epoch)
    val_sf = shape_factor(ecg_epoch)
    ecg_2d = ecg_epoch.reshape(-1, 1)
    val_svd = svd_feature(ecg_2d)
    # New features
    val_skew = skewness(ecg_epoch)
    val_cv = coefficient_of_variation(ecg_epoch)
    val_activity = hjorth_activity(ecg_epoch)
    val_mobility = hjorth_mobility(ecg_epoch)
    val_cm10 = central_moment_10(ecg_epoch)
    val_complexity = hjorth_complexity(ecg_epoch)
    val_std = standard_deviation(ecg_epoch)
    # Append to lists
    ecg_means.append(val_mean)
    ecg_rms_vals.append(val_rms)
    ecg_energy_vals.append(val_energy)
    ecg_sf_vals.append(val_sf)
    ecg_svd_vals.append(val_svd)
    ecg_skewness_vals.append(val_skew)
    ecg_cv_vals.append(val_cv)
    ecg_activity_vals.append(val_activity)
    ecg_mobility_vals.append(val_mobility)
    ecg_cm10_vals.append(val_cm10)
    ecg_complexity_vals.append(val_complexity)
    ecg_std_vals.append(val_std)

# 3. Rebuild epoch_df to include epoch_start, epoch_end, BPs, and new features
epoch_df = pd.DataFrame({
    'systolic': epoch_df['systolic'],
    'diastolic': epoch_df['diastolic'],
    'ecg_mean': ecg_means,
    'ecg_rms': ecg_rms_vals,
    'ecg_energy': ecg_energy_vals,
    'ecg_sf': ecg_sf_vals,
    'ecg_svd': ecg_svd_vals,
    'ecg_skewness': ecg_skewness_vals,
    'ecg_cv': ecg_cv_vals,
    'ecg_activity': ecg_activity_vals,
    'ecg_mobility': ecg_mobility_vals,
    'ecg_cm10': ecg_cm10_vals,
    'ecg_complexity': ecg_complexity_vals,
    'ecg_std': ecg_std_vals
})

epoch_df.tail()

##############################
##### 4. Pelatihan Model #####
##############################

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

# ----------------------------------
# 1) Define feature & target columns
# ----------------------------------
# Suppose you have these columns in epoch_df (including the ones you want to scale)
feature_columns = [
"ecg_mean", "ecg_sf", "ecg_mobility","ecg_skewness", "ecg_cv", "ecg_complexity", "ecg_cm10"
]

# Targets
target_sbp = "systolic"
target_dbp = "diastolic"

# Bersihkan
epoch_df.dropna(inplace=True)

# ----------------------------------
# 2) Split into X, y
# ----------------------------------
X = epoch_df[feature_columns].copy() # copy so we can safely modify
y_sbp = epoch_df[target_sbp].values
y_dbp = epoch_df[target_dbp].values

# ----------------------------------
# 3) Train-Test Split
# ----------------------------------
X_train, X_test, y_sbp_train, y_sbp_test = train_test_split(X, y_sbp, test_size=0.2, random_state=42)

# For DBP, use the same split indices:
_, _, y_dbp_train, y_dbp_test = train_test_split(
    X, y_dbp, test_size=0.2, random_state=42
)

from sklearn.preprocessing import MinMaxScaler
import joblib

# ─── 4) SCALE FEATURES
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train) # fit on train only
X_test_scaled = scaler.transform(X_test) # same transform on test

# save for live inference
joblib.dump(scaler, 'feat_scaler.pkl')

from sklearn.ensemble import RandomForestRegressor

# Define the hyperparameters
params = {
    'max_depth': None,
    'max_features': 'sqrt',
    'min_samples_split': 2,
    'n_estimators': 500
}

# Train Random Forest Regression for Systolic BP
rf_sbp = RandomForestRegressor(random_state=42)
rf_sbp.fit(X_train_scaled, y_sbp_train)

# Train Random Forest Regression for Diastolic BP
rf_dbp = RandomForestRegressor(random_state=42)
rf_dbp.fit(X_train_scaled, y_dbp_train)

joblib.dump(rf_sbp, "rf2_sbp.pkl") 
joblib.dump(rf_dbp, "rf2_dbp.pkl")

print("Model SBP dan DBP berhasil disimpan!")

# Prediksi menggunakan data testing yang disisihkan
y_pred_sbp = rf_sbp.predict(X_test_scaled)

# Hitung metrik evaluasi
mae_sbp = mean_absolute_error(y_sbp_test, y_pred_sbp)
mape_sbp = mean_absolute_percentage_error(y_sbp_test, y_pred_sbp) * 100

print(f"Hasil Evaluasi Model SBP:")
print(f"MAE  : {mae_sbp:.2f} mmHg")
print(f"MAPE : {mape_sbp:.2f}%")

# Prediksi data testing DBP
y_pred_dbp = rf_dbp.predict(X_test_scaled)

# Hitung metrik DBP
mae_dbp = mean_absolute_error(y_dbp_test, y_pred_dbp)
mape_dbp = mean_absolute_percentage_error(y_dbp_test, y_pred_dbp) * 100

print(f"Hasil Evaluasi Model DBP:")
print(f"MAE  : {mae_dbp:.2f} mmHg")
print(f"MAPE : {mape_dbp:.2f}%")