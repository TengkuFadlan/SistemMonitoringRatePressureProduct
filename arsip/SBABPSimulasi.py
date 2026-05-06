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
from serial import SerialException
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import joblib
from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType
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
CH_ECG = EChannelType.EXG_ADS1292R_1_CH1_24BIT # channel ECG
MWI_WINDOW = int(0.15 * FS) 
mwi_kernel = np.ones(MWI_WINDOW) / MWI_WINDOW
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

###################################
##### 2. Callback Data Stream #####
###################################

import threading

# --- MOCKING SHIMMER UNTUK SIMULASI ---
class ShimmerSimulator:
    def __init__(self, file_path, callback):
        self.file_path = file_path
        self.callback = callback
        self.running = False

    def start_streaming(self):
        self.running = True
        self.thread = threading.Thread(target=self._stream_data)
        self.thread.daemon = True
        self.thread.start()

    def _stream_data(self):
        try:
            # Pastikan sep='\t' dan skiprows tepat untuk file Shimmer-mu
            df = pd.read_csv(self.file_path, skiprows=[0, 2], sep='\t')
            column_name = 'Shimmer_B64E_ECG_LA-RA_24BIT_CAL'
            
            # Bersihkan data dari NaN agar tidak error saat processing
            data = df[column_name].dropna().values 
            
            print(f"Thread Started: Mengirim {len(data)} sampel...")
            
            for val in data:
                if not self.running: break
                
                # Mocking paket Shimmer
                mock_pkt = {CH_ECG: val} 
                
                # PANGGIL CALLBACK
                self.callback(mock_pkt)
                
                # Gunakan interval yang tepat (1/125 = 0.008s)
                time.sleep(1/FS)
                
            print("Thread Finished: Semua data terkirim.")
        except Exception as e:
            print(f"Error di Simulator Thread: {e}")

    def stop_streaming(self):
        self.running = False

def handler(pkt: DataPacket):
    global sample_count
    try:
        raw_counts = pkt[CH_ECG]
        # Jika dataset sudah dalam mV, jangan dikali SENS_MV lagi
        # Jika dataset masih raw ADC, gunakan to_signed24
        signed = to_signed24(raw_counts)
        buf.append(signed * SENS_MV)
        sample_count += 1
    except Exception:
        pass

# --- REPLACEMENT DI MAIN LOOP ---
# ser = serial.Serial(PORT, ...) # Matikan ini saat simulasi
# shim = ShimmerBluetooth(ser)   # Matikan ini saat simulasi

# Ganti dengan ini:
shim = ShimmerSimulator("SampleECGShimmer.csv", handler)
shim.start_streaming()

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

####################################
##### 4. Visualisasi dan Tabel #####
####################################p

# ————— PLOT SETUP —————
t = np.linspace(-BUF_SEC, 0, BUF_SZ)
fig, ax = plt.subplots(figsize=(12, 4))
fig.subplots_adjust(bottom=0.25)

line_norm, = ax.plot(t, np.zeros(BUF_SZ), lw=1, label="Normalized ECG")
ax.set_title("Live ECG (0–1) with BP Prediction")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Normalized Amplitude (0–1)")
ax.set_xlim(-BUF_SEC, 0)
ax.set_ylim(0, 1)
ax.legend(loc="upper right")
ax.grid(True)

# ————— ADD TABLE UNDER GRAPH —————
table = ax.table(
    cellText=[["--", "--"]],
    colLabels=["SBP", "DBP"],
    cellLoc="center",
    loc="bottom",
    bbox=[0, -0.3, 1, 0.2]
)

######################################
##### 5. Loop Utama dan Prediksi #####
######################################

# ————— UPDATE FUNCTION —————
def update(frame):
    global sample_count
    if sample_count < BUF_SZ:
        print("WAITING: ", sample_count/BUF_SZ)
        return line_norm, table
        
    # --- NORMALIZE SIGNAL ---
    raw = np.array(buf)
    mn_r, mx_r = raw.min(), raw.max()
    norm = (raw - mn_r)/(mx_r - mn_r) if mx_r > mn_r else np.zeros_like(raw)
    line_norm.set_ydata(norm)
    
    # --- ENVELOPE (MWI) & FEATURE EXTRACTION ---
    env = np.convolve(norm**2, mwi_kernel, mode="same")
    feats = np.array([
        mean_feature(env),
        shape_factor(env),
        mobility(env),
        skewness(env),
        coef_var(env),
        complexity(env),
        cm10(env)
    ]).reshape(1, -1)
    
    # --- SCALE & PREDICT ---
    cols = (feat_scaler.feature_names_in_ if hasattr(feat_scaler, "feature_names_in_") else [f"f{i}" for i in range(feats.shape[1])])
    feats_df = pd.DataFrame(feats, columns=cols)
    feats_scaled = feat_scaler.transform(feats_df)
    sbp, dbp = model_sbp.predict(feats_scaled)[0],model_dbp.predict(feats_scaled)[0]
    
    # --- UPDATE TABLE & TERMINAL OUTPUT ---
    table._cells[(1, 0)].get_text().set_text(f"{sbp:.1f}")
    table._cells[(1, 1)].get_text().set_text(f"{dbp:.1f}")
    print(f"SBP={sbp:.1f} DBP={dbp:.1f}", end="\r")
    
    return line_norm, table
    
# ————— RUN LOOP —————

print("LAGI JALAN")

ani = FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)
plt.ion()
plt.show(block=False)

try:
    while plt.fignum_exists(fig.number):
        plt.pause(0.1)
except KeyboardInterrupt:
    print("\nInterrupted by user. Cleaning up…")
except SerialException:
    print("\nSerial exception occurred. Cleaning up…")
finally:
    #try: shim.stop_streaming()
    #except: pass
    time.sleep(0.5)
    #try: ser.close()
    #except: pass
    plt.close("all")
    sys.exit(0)