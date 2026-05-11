# ==========================================
# RPP Monitoring Real-Time with Shimmer3 ECG
# Main Menu Phase Selection + GUI + Logging
# ==========================================

import sys
import time
import warnings
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.style as style

from scipy.signal import butter, filtfilt, iirnotch, find_peaks
import joblib
import serial
from serial import SerialException

from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType

style.use("dark_background")

##################################
##### 1. Inisialisasi Sistem #####
##################################

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names",
)

# ===== SERIAL / SHIMMER CONFIG =====
PORT = "/dev/rfcomm0"  # ganti sesuai port Shimmer
FS = 125
BUF_SEC = 20
BUF_SZ = FS * BUF_SEC

V_REF = 2.42
GAIN = 4
SENS_MV = V_REF / (GAIN * (2**23 - 1)) * 1000
CH_ECG = EChannelType.EXG_ADS1292R_1_CH1_24BIT

# ===== KLASIFIKASI CONFIG =====
LOW_MULT = 1.20
MED_MULT = 1.50
HIGH_MULT = 1.80

GOOD_RECOVERY_PCT = 50.0
MOD_RECOVERY_PCT = 25.0

RPP_REST_SAFE_LOW = 7000
RPP_REST_SAFE_HIGH = 9000
RPP_REST_RISK = 10000
RPP_HIGH_ALERT = 22000

# fallback absolut jika baseline belum ada
ABS_LIGHT = 10000
ABS_MOD = 15000

# ===== LOAD MODEL =====
model_sbp = joblib.load("rf2_sbp.pkl")
feat_scaler = joblib.load("feat_scaler.pkl")
ecg_stats = joblib.load("params.pkl")

# ===== BUFFER =====
buf = deque(maxlen=BUF_SZ)
sample_count = 0


def to_signed24(x):
    return x - 0x1000000 if (x & 0x800000) else x


mwi_win = int(0.150 * FS)
mwi_kernel = np.ones(mwi_win) / mwi_win

##################################
##### 2. State Aplikasi ##########
##################################

app_mode = "MENU"  # MENU / MONITOR
selected_phase = None
session_id = 0
session_start_time = None

# status progres fase agar berurutan
rest_done = False
post_done = False

# logging
all_window_logs = []
current_session_logs = []

# histori analitik global antar fase
rest_rpp_values = []
post_rpp_values = []
recovery_rpp_values = []

baseline_rpp = None
post_peak_rpp = None

# nilai display terakhir
last_hr_val = None
last_sbp_val = None
last_rpp_val = None
last_load_label = "--"
last_load_color = "#ffffff"
last_recovery_label = "--"
last_recovery_color = "#ffffff"
last_delta_rpp = None
last_recovery_pct = None
last_info_message = "Pilih fase untuk memulai monitoring"

##################################
##### 3. Koneksi Shimmer #########
##################################

ser = None
shim = None


def handler(pkt: DataPacket):
    global sample_count
    try:
        raw_counts = pkt[CH_ECG]
        signed = to_signed24(raw_counts)
        ecg_mv = signed * SENS_MV
        buf.append(ecg_mv)
        sample_count += 1
    except Exception:
        pass


def init_shimmer():
    global ser, shim
    print("Membuka serial...")
    ser = serial.Serial(PORT, baudrate=DEFAULT_BAUDRATE, timeout=None)
    time.sleep(2)

    print("Membuat objek ShimmerBluetooth...")
    shim = ShimmerBluetooth(ser)

    print("Memanggil initialize()...")
    shim.initialize()
    print("initialize() selesai")

    data_types = shim.get_data_types()
    print("Data types aktif:", data_types)

    if CH_ECG not in data_types:
        raise RuntimeError(
            "ECG channel tidak aktif di Shimmer. Check konfigurasi sensor."
        )

    shim.add_stream_callback(handler)
    shim.start_streaming()
    print("Streaming dimulai.")


def close_shimmer():
    global ser, shim
    try:
        if shim is not None:
            shim.stop_streaming()
    except:
        pass

    time.sleep(0.5)

    try:
        if ser is not None:
            ser.close()
    except:
        pass


##############################################
##### 4. Definisi Fungsi Ekstraksi Fitur #####
##############################################


def shape_factor(x):
    xrms = np.sqrt(np.mean(x**2))
    msa = np.mean(np.sqrt(np.abs(x)))
    return xrms / msa if msa else 0.0


def mobility(x):
    vs = np.var(x, ddof=1)
    vd = np.var(np.diff(x), ddof=1) if x.size > 1 else 0.0
    return np.sqrt(vd / vs) if vs else 0.0


def complexity(x):
    vs = np.var(x, ddof=1)
    d1 = np.diff(x)
    v1 = np.var(d1, ddof=1) if d1.size > 1 else 0.0
    d2 = np.diff(d1)
    v2 = np.var(d2, ddof=1) if d2.size > 1 else 0.0
    return np.sqrt((v2 / v1) / (v1 / vs)) if vs and v1 else 0.0


def skewness(x):
    n = x.size
    mu = np.mean(x)
    s = np.std(x, ddof=1)
    return np.sum((x - mu) ** 3) / ((n - 1) * s**3) if s and n > 1 else 0.0


def cm10(x):
    mu = np.mean(x)
    return np.mean((x - mu) ** 10)


#########################################
##### 5. Fungsi Klasifikasi #############
##########################################


def classify_load(rpp, baseline_rpp, phase):
    # fallback absolut jika baseline belum tersedia
    if baseline_rpp is None:
        if rpp >= RPP_HIGH_ALERT:
            return "Beban sangat tinggi", "#ff2222"
        elif rpp >= ABS_MOD:
            return "Beban tinggi", "#ff6600"
        elif rpp >= ABS_LIGHT:
            return "Beban sedang", "#ffcc00"
        else:
            return "Beban ringan", "#33cc66"

    if phase == "REST":
        if rpp <= RPP_REST_SAFE_HIGH:
            return "Rest aman", "#33cc66"
        elif rpp <= RPP_REST_RISK:
            return "Rest meningkat", "#ffcc00"
        else:
            return "Rest waspada", "#ff3333"

    ratio = rpp / (baseline_rpp + 1e-9)

    if rpp >= RPP_HIGH_ALERT:
        return "Beban sangat tinggi", "#ff2222"
    elif ratio >= HIGH_MULT:
        return "Beban tinggi", "#ff6600"
    elif ratio >= MED_MULT:
        return "Beban sedang", "#ffcc00"
    elif ratio >= LOW_MULT:
        return "Beban meningkat", "#66ccff"
    else:
        return "Mendekati baseline", "#33cc66"


def classify_recovery(rpp, baseline_rpp, post_peak_rpp):
    if baseline_rpp is None:
        return "Baseline REST belum tersedia", "#aaaaaa", None, None

    if post_peak_rpp is None:
        return "Peak POST-EXERCISE belum tersedia", "#aaaaaa", None, None

    denom = post_peak_rpp - baseline_rpp
    if denom <= 1e-9:
        return "Recovery belum stabil", "#aaaaaa", None, None

    recovery_drop = post_peak_rpp - rpp
    recovery_pct = (recovery_drop / denom) * 100.0

    if recovery_pct >= GOOD_RECOVERY_PCT:
        label = "Recovery baik"
        color = "#33cc66"
    elif recovery_pct >= MOD_RECOVERY_PCT:
        label = "Recovery sedang"
        color = "#ffcc00"
    else:
        label = "Recovery lambat"
        color = "#ff3333"

    return label, color, recovery_drop, recovery_pct


#########################################
##### 6. Filter #########################
##########################################


def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=FS, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def notch_filter(data, f0=50.0, Q=30.0, fs=FS):
    b, a = iirnotch(f0 / (0.5 * fs), Q)
    return filtfilt(b, a, data)


#########################################
##### 7. Session Management #############
##########################################


def reset_session_display_only():
    global current_session_logs
    global last_hr_val, last_sbp_val, last_rpp_val
    global last_load_label, last_load_color
    global last_recovery_label, last_recovery_color
    global last_delta_rpp, last_recovery_pct

    current_session_logs = []

    last_hr_val = None
    last_sbp_val = None
    last_rpp_val = None
    last_load_label = "--"
    last_load_color = "#ffffff"
    last_recovery_label = "--"
    last_recovery_color = "#ffffff"
    last_delta_rpp = None
    last_recovery_pct = None


def save_current_session():
    global current_session_logs, all_window_logs, session_id
    if not current_session_logs:
        print("Tidak ada data sesi untuk disimpan.")
        return

    df_session = pd.DataFrame(current_session_logs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"rpp_session_{session_id}_{timestamp}.csv"
    df_session.to_csv(fname, index=False)
    print(f"Sesi disimpan ke: {fname}")

    all_window_logs.extend(current_session_logs)
    current_session_logs = []


def save_all_logs():
    if not all_window_logs:
        print("Belum ada log global untuk disimpan.")
        return

    df_all = pd.DataFrame(all_window_logs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"rpp_monitoring_all_{timestamp}.csv"
    df_all.to_csv(fname, index=False)
    print(f"Seluruh log disimpan ke: {fname}")


def start_phase_session(phase_name):
    global app_mode, selected_phase, session_id, session_start_time, last_info_message

    if phase_name == "POST-EXERCISE" and not rest_done:
        last_info_message = "Jalankan fase REST terlebih dahulu."
        print(last_info_message)
        return

    if phase_name == "RECOVERY" and not post_done:
        last_info_message = "Jalankan fase POST-EXERCISE terlebih dahulu."
        print(last_info_message)
        return

    session_id += 1
    selected_phase = phase_name
    session_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    reset_session_display_only()
    app_mode = "MONITOR"
    last_info_message = f"Monitoring fase {selected_phase} dimulai"
    print(f"Mulai sesi {session_id} | Phase = {selected_phase}")


def back_to_menu():
    global app_mode, selected_phase
    global rest_done, post_done, last_info_message

    if selected_phase == "REST":
        rest_done = True
    elif selected_phase == "POST-EXERCISE":
        post_done = True

    save_current_session()
    selected_phase = None
    app_mode = "MENU"
    last_info_message = "Kembali ke Main Menu"
    print(last_info_message)


#########################################
##### 8. Visualisasi ####################
##########################################

fig = plt.figure(figsize=(14, 12))
grid = fig.add_gridspec(5, 2, height_ratios=[3, 2, 1, 1, 1])

ax_ecg = fig.add_subplot(grid[0, :])
(line_ecg,) = ax_ecg.plot([], [], lw=1.5, color="#00ff41")
ax_ecg.set_title("Live ECG Stream")
ax_ecg.set_ylim(-0.1, 1.1)
ax_ecg.set_xlim(0, BUF_SZ)

ax_rpp = fig.add_subplot(grid[1, :])
rpp_history = deque(maxlen=100)
(line_rpp,) = ax_rpp.plot([], [], color="#ff3333", lw=2)
ax_rpp.set_title("RPP Trend")
ax_rpp.set_ylim(5000, 25000)
ax_rpp.set_xlim(0, 100)

ax_sbp_txt = fig.add_subplot(grid[2, 0])
ax_sbp_txt.axis("off")
sbp_text = ax_sbp_txt.text(
    0.5,
    0.5,
    "SBP: --",
    fontsize=28,
    weight="bold",
    color="#3399ff",
    ha="center",
    va="center",
)

ax_hr_txt = fig.add_subplot(grid[2, 1])
ax_hr_txt.axis("off")
hr_text = ax_hr_txt.text(
    0.5,
    0.5,
    "HR: --",
    fontsize=28,
    weight="bold",
    color="#ffcc00",
    ha="center",
    va="center",
)

ax_rpp_txt = fig.add_subplot(grid[3, :])
ax_rpp_txt.axis("off")
rpp_text = ax_rpp_txt.text(
    0.5,
    0.70,
    "Current RPP: --",
    fontsize=20,
    weight="bold",
    color="#ff3535",
    ha="center",
    va="center",
)
phase_text = ax_rpp_txt.text(
    0.5,
    0.35,
    "Phase: --",
    fontsize=16,
    weight="bold",
    color="#cccccc",
    ha="center",
    va="center",
)

ax_status_txt = fig.add_subplot(grid[4, :])
ax_status_txt.axis("off")
load_status_text = ax_status_txt.text(
    0.25,
    0.65,
    "Status Beban: --",
    fontsize=18,
    weight="bold",
    color="#ffffff",
    ha="center",
    va="center",
)
recovery_status_text = ax_status_txt.text(
    0.75,
    0.65,
    "Status Recovery: --",
    fontsize=18,
    weight="bold",
    color="#ffffff",
    ha="center",
    va="center",
)
delta_text = ax_status_txt.text(
    0.25, 0.22, "ΔRPP: --", fontsize=14, color="#cccccc", ha="center", va="center"
)
recovery_pct_text = ax_status_txt.text(
    0.75, 0.22, "%Recovery: --", fontsize=14, color="#cccccc", ha="center", va="center"
)

ax_menu = fig.add_axes([0.05, 0.05, 0.90, 0.90])
ax_menu.axis("off")
menu_title = ax_menu.text(
    0.5,
    0.85,
    "MAIN MENU",
    fontsize=28,
    weight="bold",
    color="#ffffff",
    ha="center",
    va="center",
)
menu_subtitle = ax_menu.text(
    0.5,
    0.75,
    "Pilih fase praktisi sebelum monitoring dimulai",
    fontsize=16,
    color="#cccccc",
    ha="center",
    va="center",
)
menu_opt1 = ax_menu.text(
    0.5,
    0.58,
    "[1] REST",
    fontsize=22,
    weight="bold",
    color="#33cc66",
    ha="center",
    va="center",
)
menu_opt2 = ax_menu.text(
    0.5,
    0.48,
    "[2] POST-EXERCISE",
    fontsize=22,
    weight="bold",
    color="#ffcc00",
    ha="center",
    va="center",
)
menu_opt3 = ax_menu.text(
    0.5,
    0.38,
    "[3] RECOVERY",
    fontsize=22,
    weight="bold",
    color="#ff6666",
    ha="center",
    va="center",
)
menu_info = ax_menu.text(
    0.5,
    0.20,
    "Saat monitoring: tekan [M] kembali ke menu, [Q] keluar",
    fontsize=13,
    color="#aaaaaa",
    ha="center",
    va="center",
)
menu_msg = ax_menu.text(
    0.5, 0.10, last_info_message, fontsize=14, color="#66ccff", ha="center", va="center"
)

monitor_axes = [ax_ecg, ax_rpp, ax_sbp_txt, ax_hr_txt, ax_rpp_txt, ax_status_txt]


def show_menu():
    ax_menu.set_visible(True)
    menu_msg.set_text(last_info_message)
    for ax in monitor_axes:
        ax.set_visible(False)


def show_monitor():
    ax_menu.set_visible(False)
    for ax in monitor_axes:
        ax.set_visible(True)


#########################################
##### 9. Keyboard Event #################
##########################################


def on_key(event):
    if app_mode == "MENU":
        if event.key == "1":
            start_phase_session("REST")
        elif event.key == "2":
            start_phase_session("POST-EXERCISE")
        elif event.key == "3":
            start_phase_session("RECOVERY")
        elif event.key in ["q", "Q"]:
            save_current_session()
            save_all_logs()
            plt.close(fig)

    elif app_mode == "MONITOR":
        if event.key in ["m", "M"]:
            back_to_menu()
        elif event.key in ["q", "Q"]:
            save_current_session()
            save_all_logs()
            plt.close(fig)


#########################################
##### 10. Update Animasi ################
##########################################

total_comp_times = []
preprocess_feat_times = []
predict_times = []


def update(frame):
    global baseline_rpp, post_peak_rpp
    global last_hr_val, last_sbp_val, last_rpp_val
    global last_load_label, last_load_color
    global last_recovery_label, last_recovery_color
    global last_delta_rpp, last_recovery_pct

    try:
        if app_mode == "MENU":
            show_menu()
            return (
                line_ecg,
                line_rpp,
                sbp_text,
                hr_text,
                rpp_text,
                phase_text,
                load_status_text,
                recovery_status_text,
                delta_text,
                recovery_pct_text,
            )

        show_monitor()

        current_data = list(buf)
        if len(current_data) > 0:
            raw_np = np.array(current_data)
            norm_view = (raw_np - raw_np.min()) / (raw_np.max() - raw_np.min() + 1e-9)
            line_ecg.set_data(range(len(norm_view)), norm_view)

        if len(buf) >= BUF_SZ and sample_count % 60 == 0:
            start_total = time.perf_counter()

            start_pf = time.perf_counter()
            raw = np.array(buf)
            ecg_clean = notch_filter(bandpass_filter(raw))
            ecg_norm = (ecg_clean - ecg_stats["mean"]) / ecg_stats["std"]

            y_der = np.gradient(ecg_norm, 1 / FS)
            y_sq = y_der**2
            y_mwi = np.convolve(y_sq, mwi_kernel, mode="same")

            feats = np.array(
                [
                    shape_factor(ecg_clean),
                    mobility(ecg_clean),
                    skewness(ecg_clean),
                    complexity(ecg_clean),
                    cm10(ecg_clean),
                ]
            ).reshape(1, -1)
            end_pf = time.perf_counter()

            start_pred = time.perf_counter()
            feats_scaled = feat_scaler.transform(feats)
            sbp_pred = model_sbp.predict(feats_scaled)[0]

            peaks, _ = find_peaks(
                y_mwi, distance=int(0.333 * FS), height=np.mean(y_mwi)
            )
            hr_val = (len(peaks) / BUF_SEC) * 60
            rpp_val = hr_val * sbp_pred
            end_pred = time.perf_counter()
            end_total = time.perf_counter()

            preprocess_feat_times.append(end_pf - start_pf)
            predict_times.append(end_pred - start_pred)
            total_comp_times.append(end_total - start_total)

            phase = selected_phase

            if phase == "REST":
                rest_rpp_values.append(rpp_val)
                baseline_rpp = np.mean(rest_rpp_values)

            elif phase == "POST-EXERCISE":
                post_rpp_values.append(rpp_val)
                if post_peak_rpp is None or rpp_val > post_peak_rpp:
                    post_peak_rpp = rpp_val

            elif phase == "RECOVERY":
                recovery_rpp_values.append(rpp_val)

            load_label, load_color = classify_load(rpp_val, baseline_rpp, phase)

            delta_rpp = None
            if baseline_rpp is not None:
                delta_rpp = rpp_val - baseline_rpp

            if phase == "RECOVERY":
                rec_label, rec_color, rec_drop, rec_pct = classify_recovery(
                    rpp_val, baseline_rpp, post_peak_rpp
                )
            else:
                rec_label, rec_color, rec_drop, rec_pct = (
                    "Belum masuk recovery",
                    "#aaaaaa",
                    None,
                    None,
                )

            last_hr_val = hr_val
            last_sbp_val = sbp_pred
            last_rpp_val = rpp_val
            last_load_label = load_label
            last_load_color = load_color
            last_recovery_label = rec_label
            last_recovery_color = rec_color
            last_delta_rpp = delta_rpp
            last_recovery_pct = rec_pct

            sbp_text.set_text(f"SBP: {sbp_pred:.1f} mmHg")
            hr_text.set_text(f"HR: {hr_val:.1f} BPM")
            rpp_text.set_text(f"Current RPP: {rpp_val:.0f}")
            phase_text.set_text(f"Phase: {phase} | Session: {session_id}")

            load_status_text.set_text(f"Status Beban: {load_label}")
            load_status_text.set_color(load_color)

            recovery_status_text.set_text(f"Status Recovery: {rec_label}")
            recovery_status_text.set_color(rec_color)

            delta_text.set_text(
                f"ΔRPP: {delta_rpp:+.0f}" if delta_rpp is not None else "ΔRPP: --"
            )
            recovery_pct_text.set_text(
                f"%Recovery: {rec_pct:.1f}%" if rec_pct is not None else "%Recovery: --"
            )

            rpp_history.append(rpp_val)
            line_rpp.set_data(range(len(rpp_history)), list(rpp_history))

            current_session_logs.append(
                {
                    "session_id": session_id,
                    "session_start": session_start_time,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "sample_count": sample_count,
                    "phase": phase,
                    "hr_bpm": hr_val,
                    "sbp_mmhg": sbp_pred,
                    "rpp": rpp_val,
                    "baseline_rpp": baseline_rpp,
                    "delta_rpp": delta_rpp,
                    "post_peak_rpp": post_peak_rpp,
                    "recovery_drop": rec_drop,
                    "recovery_pct": rec_pct,
                    "load_status": load_label,
                    "recovery_status": rec_label,
                }
            )

    except KeyboardInterrupt:
        save_current_session()
        save_all_logs()
        plt.close(fig)

    return (
        line_ecg,
        line_rpp,
        sbp_text,
        hr_text,
        rpp_text,
        phase_text,
        load_status_text,
        recovery_status_text,
        delta_text,
        recovery_pct_text,
    )


#########################################
##### 11. Report Saat Close #############
##########################################


def on_close(event):
    save_current_session()
    save_all_logs()

    if not total_comp_times:
        print("\nData benchmark tidak mencukupi.")
        close_shimmer()
        return

    print("\n" + "=" * 40)
    print("          HASIL BENCHMARK")
    print("=" * 40)
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
    print("\n" + "-" * 40)
    print(f"Rata-rata waktu komputasi = {avg_total:.4f} detik")
    print(f"Rasio terhadap window {BUF_SEC}s = {ratio:.4f}%")
    print("=" * 40)

    close_shimmer()


#########################################
##### 12. Jalankan Sistem ###############
##########################################

try:
    init_shimmer()

    show_menu()
    fig.canvas.mpl_connect("close_event", on_close)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.tight_layout()
    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)

    print("Aplikasi dimulai.")
    print("Di MENU: tekan 1=REST, 2=POST-EXERCISE, 3=RECOVERY, Q=keluar")
    print("Saat MONITOR: tekan M=kembali ke menu, Q=keluar")

    plt.show()

except KeyboardInterrupt:
    print("\nInterupsi terdeteksi, menutup sistem...")
    save_current_session()
    save_all_logs()
    close_shimmer()
    plt.close("all")

except SerialException:
    print("\nKoneksi serial gagal / terputus.")
    save_current_session()
    save_all_logs()
    close_shimmer()
    plt.close("all")

except Exception as e:
    print(f"\nTerjadi error: {e}")
    save_current_session()
    save_all_logs()
    close_shimmer()
    plt.close("all")

finally:
    print("Selesai.")
    sys.exit(0)
