# Libraries
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, lfilter, medfilt, find_peaks
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import time
from collections import deque
import warnings
import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

##################################
##### 1. Inisialisasi Sistem #####
##################################

import sys
import time
from collections import deque
import warnings
import numpy as np
import pandas as pd
import serial
#from serial import SerialException
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import joblib
#from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType
# ————— SUPPRESS SCALER WARNING —————
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names"
)
# ————— CONFIG —————
PORT = 'COM3' # port serial
FS = 125 # sampling rate [Hz]
BUF_SEC = 20 # detik untuk buffer
BUF_SZ = FS * BUF_SEC
V_REF = 2.42 # referensi tegangan [V]
GAIN = 4 # gain amplifier
SENS_MV = V_REF / (GAIN * (2**23 - 1)) * 1000 # mV per ADC count
#CH_ECG = EChannelType.EXG_ADS1292R_1_CH1_24BIT # channel ECG
# ————— LOAD MODELS & SCALERS —————
model_sbp = joblib.load("rf2_sbp.pkl") # trained RF for SBP

model_dbp = joblib.load("rf2_dbp.pkl") # trained RF for DBP
feat_scaler = joblib.load("feat_scaler.pkl")# feature-level scaler
ecg_stats = joblib.load("params.pkl") # global envelope stats (if needed)
# ————— BUFFER & COUNTER —————
buf = deque(maxlen=BUF_SZ)
sample_count = 0
def to_signed24(x):
    return x - 0x1000000 if (x & 0x800000) else x



# SIMULASI
df_sim = pd.read_csv('df2.csv')
sim_data = df_sim['ECG'].values # Ambil kolom ECG
sim_idx = 0

# Tambahkan mwi_kernel
mwi_win = int(0.150 * FS)
mwi_kernel = np.ones(mwi_win) / mwi_win

# Buffer tetap sama
buf = deque(maxlen=BUF_SZ)
sample_count = 0

##################################
##### 2. Callback Datastream #####
##################################

# GANTI SELURUH KODE 2 DENGAN INI:
def simulate_stream():
    global sample_count, sim_idx
    # Simulasi data datang 1 sampel
    if sim_idx < len(sim_data):
        val = sim_data[sim_idx]
        buf.append(val)
        sample_count += 1
        sim_idx += 1

##############################################
##### 3. Definisi Fungsi Ekstraksi Fitur #####
##############################################

def mean_feature(x):
    return np.mean(x)
    
def shape_factor(x):
    xrms = np.sqrt(np.mean(x**2))
    msa = np.mean(np.sqrt(np.abs(x)))
    return xrms/msa if msa else 0.0

def mobility(x):
    vs = np.var(x, ddof=1)
    vd = np.var(np.diff(x), ddof=1) if x.size > 1 else 0.0
    return np.sqrt(vd/vs) if vs else 0.0
    
def complexity(x):
    vs = np.var(x, ddof=1)
    d1 = np.diff(x)
    v1 = np.var(d1, ddof=1) if d1.size > 1 else 0.0
    d2 = np.diff(d1)
    v2 = np.var(d2, ddof=1) if d2.size > 1 else 0.0
    return np.sqrt((v2/v1)/(v1/vs)) if vs and v1 else 0.0
    
def skewness(x):
    n = x.size
    mu = np.mean(x)
    s = np.std(x, ddof=1)
    return np.sum((x-mu)**3)/((n-1)*s**3) if s and n > 1 else 0.0
    
def coef_var(x):
    mu = np.mean(x)
    s = np.std(x, ddof=1)
    return (s/mu)*100 if mu else 0.0

def cm10(x):
    mu = np.mean(x)
    return np.mean((x-mu)**10)

#################################
##### 4. Visualisasi Sistem #####
#################################
import matplotlib.style as style
style.use('dark_background') 

# --- SETUP FIGURE ---
# Kita buat grid yang lebih kompleks: 
# Atas: ECG, Tengah: RPP Trend, Bawah: Text Metrics
fig = plt.figure(figsize=(12, 10))
grid = fig.add_gridspec(4, 2, height_ratios=[3, 2, 1, 1])

# 1. Plot ECG (Full Width)
ax_ecg = fig.add_subplot(grid[0, :])
line_ecg, = ax_ecg.plot([], [], lw=1.5, color='#00ff41')
ax_ecg.set_title("Live ECG Stream", fontsize=12, color='gray')
ax_ecg.set_ylim(-0.1, 1.1)
ax_ecg.set_xlim(0, BUF_SZ)

# 2. Plot Tren RPP (Full Width)
ax_rpp = fig.add_subplot(grid[1, :])
rpp_history = deque(maxlen=100)
line_rpp, = ax_rpp.plot([], [], color='#ff3333', lw=2)
ax_rpp.set_title("RPP (Rate Pressure Product) Trend", fontsize=10)
ax_rpp.set_ylim(5000, 25000) 
ax_rpp.set_xlim(0, 100)

# 3. Text Metrics Display (Besar dan Jelas)
# SBP Display
ax_sbp_txt = fig.add_subplot(grid[2, 0])
ax_sbp_txt.axis('off')
sbp_text = ax_sbp_txt.text(0.5, 0.5, "SBP: --", fontsize=30, weight='bold', 
                           color='#3399ff', ha='center', va='center')

# HR Display
ax_hr_txt = fig.add_subplot(grid[2, 1])
ax_hr_txt.axis('off')
hr_text = ax_hr_txt.text(0.5, 0.5, "HR: --", fontsize=30, weight='bold', 
                         color='#ffcc00', ha='center', va='center')

# RPP Display (Di paling bawah)
ax_rpp_txt = fig.add_subplot(grid[3, :])
ax_rpp_txt.axis('off')
rpp_text = ax_rpp_txt.text(0.5, 0.5, "Current RPP: --", fontsize=20, weight='bold', 
                           color="#ff3535", ha='center', va='center')


# Komputasi Waktu
total_comp_times = []
preprocess_feat_times = []
predict_times = []

def update(frame):
    global sim_idx, sample_count
    try:
        # Simulasi data masuk
        for _ in range(5): 
            simulate_stream()
        
        # Update ECG Plot
        current_data = list(buf)
        if len(current_data) > 0:
            raw_np = np.array(current_data)
            norm_view = (raw_np - raw_np.min())/(raw_np.max() - raw_np.min() + 1e-9)
            line_ecg.set_data(range(len(norm_view)), norm_view)
        
        # Benchmark & Prediksi
        if sample_count >= BUF_SZ and sim_idx % 60 == 0:
            start_total = time.perf_counter()
            
            # --- PHASE 1: PREPROCESSING & FEATURES ---
            start_pf = time.perf_counter()
            raw = np.array(buf)
            mn_r, mx_r = raw.min(), raw.max()
            norm = (raw - mn_r)/(mx_r - mn_r) if mx_r > mn_r else np.zeros_like(raw)
            env = np.convolve(norm**2, mwi_kernel, mode="same")
            
            feats = np.array([
                mean_feature(env), shape_factor(env), mobility(env),
                skewness(env), coef_var(env), complexity(env), cm10(env)
            ]).reshape(1, -1)
            end_pf = time.perf_counter()
            
            # --- PHASE 2: SCALING & PREDICTION ---
            start_pred = time.perf_counter()
            feats_scaled = feat_scaler.transform(feats)
            sbp_pred = model_sbp.predict(feats_scaled)[0]
            
            peaks, _ = find_peaks(env, distance=int(0.6 * FS), height=np.mean(env))
            hr_val = (len(peaks) / BUF_SEC) * 60
            rpp_val = hr_val * sbp_pred
            end_pred = time.perf_counter()
            
            # Record Benchmark
            total_comp_times.append(time.perf_counter() - start_total)
            preprocess_feat_times.append(end_pf - start_pf)
            predict_times.append(end_pred - start_pred)
            
            # Update GUI Text
            sbp_text.set_text(f"SBP: {sbp_pred:.0f} mmHg")
            hr_text.set_text(f"HR: {hr_val:.0f} BPM")
            rpp_text.set_text(f"Current RPP: {rpp_val:.0f}")
            
            rpp_history.append(rpp_val)
            line_rpp.set_data(range(len(rpp_history)), list(rpp_history))

    except KeyboardInterrupt:
        plt.close(fig) # Tutup grafik jika Ctrl+C ditekan saat update
    
    return line_ecg, line_rpp, sbp_text, hr_text, rpp_text

# --- Tambahkan print report saat GUI ditutup ---
def on_close(event):
    if not total_comp_times:
        print("\nData benchmark tidak mencukupi.")
        return

    print("\n" + "="*30)
    print("      HASIL BENCHMARK")
    print("="*30)
    print(f"Window valid terproses : {len(total_comp_times)}")
    print("\n[TOTAL PER WINDOW]")
    print(f"Mean   : {np.mean(total_comp_times):.6f} detik")
    print(f"Median : {np.median(total_comp_times):.6f} detik")
    print(f"Std    : {np.std(total_comp_times):.6f} detik")
    print(f"Min    : {np.min(total_comp_times):.6f} detik")
    print(f"Max    : {np.max(total_comp_times):.6f} detik")

    print("\n[PREPROCESSING + FEATURE EXTRACTION]")
    print(f"Mean   : {np.mean(preprocess_feat_times):.6f} detik")

    print("\n[SCALING + PREDICTION]")
    print(f"Mean   : {np.mean(predict_times):.6f} detik")

    avg_total = np.mean(total_comp_times)
    ratio = (avg_total / BUF_SEC) * 100
    print("\n" + "-"*30)
    print(f"Rata-rata waktu komputasi = {avg_total:.4f} detik")
    print(f"Rasio terhadap window {BUF_SEC}s = {ratio:.4f}%")
    print("="*30)

# Sambungkan fungsi close event
fig.canvas.mpl_connect('close_event', on_close)

plt.tight_layout()
ani = FuncAnimation(fig, update, interval=20, blit=False, cache_frame_data=False)

print("Memulai simulasi... Tekan Ctrl+C di terminal untuk berhenti.")

try:
    plt.show()
except KeyboardInterrupt:
    print("\nInterupsi terdeteksi, menutup sistem...")
    plt.close('all') # Memaksa semua window Matplotlib tertutup
    on_close(None)   # Panggil fungsi report secara manual
finally:
    print("Selesai.")