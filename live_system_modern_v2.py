import sys
import time
import warnings
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
import joblib
import serial
from serial import SerialException

from PySide6.QtCore import QTimer, QObject
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import pyqtgraph as pg
from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names",
)

# =========================
# 1. KONFIGURASI SISTEM
# =========================
PORT = "/dev/rfcomm0"
FS = 125
BUF_SEC = 20
BUF_SZ = FS * BUF_SEC

V_REF = 2.42
GAIN = 4
SENS_MV = V_REF / (GAIN * (2**23 - 1)) * 1000
CH_ECG = EChannelType.EXG_ADS1292R_1_CH1_24BIT

# Relative load thresholds terhadap baseline REST
LOAD_MILD_RATIO = 1.20
LOAD_MOD_RATIO = 1.50
LOAD_HIGH_RATIO = 1.80

# Recovery thresholds
GOOD_RECOVERY_PCT = 50.0
MOD_RECOVERY_PCT = 25.0
ABNORMAL_HRR_1MIN = 12.0

# Hemodynamic support thresholds
SBP_EXAGGERATED = 180.0  # konservatif untuk early warning monitoring
SBP_VERY_HIGH = 210.0
SBP_RECOVERY_HIGH = 140.0

mwi_win = int(0.150 * FS)
mwi_kernel = np.ones(mwi_win) / mwi_win


# =========================
# 2. UTILITAS SINYAL
# =========================
def to_signed24(x):
    return x - 0x1000000 if (x & 0x800000) else x


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


def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=FS, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def notch_filter(data, f0=50.0, Q=30.0, fs=FS):
    b, a = iirnotch(f0 / (0.5 * fs), Q)
    return filtfilt(b, a, data)


def safe_mean(values):
    return float(np.mean(values)) if values else None


def classify_load(rpp, baseline_rpp, phase):
    if baseline_rpp is None:
        return "Baseline belum tersedia", "#94a3b8", None, None

    delta_rpp = rpp - baseline_rpp
    ratio = rpp / (baseline_rpp + 1e-9)

    if phase == "REST":
        if ratio <= 1.10:
            return "Baseline stabil", "#22c55e", delta_rpp, ratio
        elif ratio <= 1.20:
            return "Baseline sedikit meningkat", "#facc15", delta_rpp, ratio
        return "Baseline perlu ditinjau", "#f97316", delta_rpp, ratio

    if phase == "POST-EXERCISE":
        if ratio < LOAD_MILD_RATIO:
            return "Beban mendekati baseline", "#22c55e", delta_rpp, ratio
        elif ratio < LOAD_MOD_RATIO:
            return "Beban meningkat ringan", "#38bdf8", delta_rpp, ratio
        elif ratio < LOAD_HIGH_RATIO:
            return "Beban meningkat sedang", "#facc15", delta_rpp, ratio
        return "Beban meningkat tinggi", "#ef4444", delta_rpp, ratio

    if phase == "RECOVERY":
        if ratio <= 1.20:
            return "Beban mendekati baseline", "#22c55e", delta_rpp, ratio
        elif ratio <= 1.50:
            return "Beban masih di atas baseline", "#facc15", delta_rpp, ratio
        return "Beban masih tinggi", "#ef4444", delta_rpp, ratio

    return "Status beban tidak diketahui", "#94a3b8", delta_rpp, ratio


def classify_recovery(
    current_rpp,
    baseline_rpp,
    post_peak_rpp,
    current_hr,
    peak_hr,
    elapsed_recovery_sec,
):
    if baseline_rpp is None:
        return "Baseline REST belum tersedia", "#94a3b8", None, None, None
    if post_peak_rpp is None:
        return "Peak POST-EXERCISE belum tersedia", "#94a3b8", None, None, None

    denom = post_peak_rpp - baseline_rpp
    if denom <= 1e-9:
        return "Recovery belum dapat dihitung", "#94a3b8", None, None, None

    recovery_drop = post_peak_rpp - current_rpp
    recovery_pct = (recovery_drop / denom) * 100.0

    hrr_val = None
    if peak_hr is not None:
        hrr_val = peak_hr - current_hr

    # Jika sudah melewati ~1 menit recovery, pakai kombinasi %recovery + HRR
    if (
        elapsed_recovery_sec is not None
        and elapsed_recovery_sec >= 60
        and hrr_val is not None
    ):
        if recovery_pct >= GOOD_RECOVERY_PCT and hrr_val > ABNORMAL_HRR_1MIN:
            return "Recovery cepat", "#22c55e", recovery_drop, recovery_pct, hrr_val
        elif recovery_pct >= MOD_RECOVERY_PCT and hrr_val > ABNORMAL_HRR_1MIN:
            return "Recovery cukup", "#facc15", recovery_drop, recovery_pct, hrr_val
        return "Recovery lambat", "#ef4444", recovery_drop, recovery_pct, hrr_val

    # Sebelum 1 menit, tampilkan progres recovery sementara
    if recovery_pct >= GOOD_RECOVERY_PCT:
        return "Recovery menuju baik", "#22c55e", recovery_drop, recovery_pct, hrr_val
    elif recovery_pct >= MOD_RECOVERY_PCT:
        return (
            "Recovery sedang berlangsung",
            "#facc15",
            recovery_drop,
            recovery_pct,
            hrr_val,
        )
    return "Awal recovery", "#38bdf8", recovery_drop, recovery_pct, hrr_val


def classify_hemodynamic_flag(phase, sbp, hr, baseline_sbp, baseline_hr):
    if phase == "REST":
        if sbp >= 140:
            return "SBP rest meningkat", "#f97316"
        if hr >= 100:
            return "HR rest tinggi", "#f97316"
        return "Hemodinamik stabil", "#22c55e"

    if phase == "POST-EXERCISE":
        if sbp >= SBP_VERY_HIGH:
            return "SBP exercise sangat tinggi", "#ef4444"
        if sbp >= SBP_EXAGGERATED:
            return "SBP exercise tinggi", "#f97316"
        if baseline_hr is not None and hr <= baseline_hr + 10:
            return "Kenaikan HR kurang adekuat", "#facc15"
        return "Respons exercise sesuai", "#22c55e"

    if phase == "RECOVERY":
        if sbp >= SBP_RECOVERY_HIGH:
            return "SBP recovery masih tinggi", "#f97316"
        return "Recovery hemodinamik sesuai", "#22c55e"

    return "Flag tidak tersedia", "#94a3b8"


# =========================
# 2A. QUALITY CHECK ECG
# =========================
MIN_HR_BPM = 40
MAX_HR_BPM = 220
MAX_FLATLINE_PCT = 0.20
MAX_CLIP_PCT = 0.05
MAX_BASELINE_STD = 0.35
MIN_PEAKS = 8
MAX_PEAKS = 80
MAX_RR_CV = 0.25


def moving_average(x, w):
    if w <= 1:
        return x.copy()
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def robust_zscore(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    return 0.6745 * (x - med) / mad


def detect_flatline_ratio(x, eps=1e-6):
    dx = np.abs(np.diff(x))
    return np.mean(dx < eps)


def detect_clip_ratio(x):
    z = robust_zscore(x)
    return np.mean(np.abs(z) > 8.0)


def ecg_quality_check(ecg_raw, ecg_clean, y_mwi, peaks, fs=FS):
    reasons = []

    # 1) Flatline / data nyangkut
    flat_ratio = detect_flatline_ratio(ecg_raw)
    if flat_ratio > MAX_FLATLINE_PCT:
        reasons.append(f"flatline tinggi ({flat_ratio:.2f})")

    # 2) Clip / outlier ekstrem
    clip_ratio = detect_clip_ratio(ecg_raw)
    if clip_ratio > MAX_CLIP_PCT:
        reasons.append(f"outlier ekstrem ({clip_ratio:.2f})")

    # 3) Baseline wander besar
    baseline = moving_average(ecg_clean, int(1.0 * fs))
    baseline_std = np.std(baseline) / (np.std(ecg_clean) + 1e-9)
    if baseline_std > MAX_BASELINE_STD:
        reasons.append(f"baseline drift tinggi ({baseline_std:.2f})")

    # 4) Jumlah peak tidak masuk akal
    n_peaks = len(peaks)
    if n_peaks < MIN_PEAKS or n_peaks > MAX_PEAKS:
        reasons.append(f"jumlah peak tidak wajar ({n_peaks})")

    # 5) HR range check
    hr_est = (n_peaks / BUF_SEC) * 60.0
    if hr_est < MIN_HR_BPM or hr_est > MAX_HR_BPM:
        reasons.append(f"HR tidak masuk akal ({hr_est:.1f} bpm)")

    # 6) RR interval stability check
    if n_peaks >= 3:
        rr = np.diff(peaks) / fs
        rr_cv = np.std(rr) / (np.mean(rr) + 1e-9)
        if rr_cv > MAX_RR_CV:
            reasons.append(f"RR tidak stabil ({rr_cv:.2f})")
    else:
        rr_cv = None

    is_good = len(reasons) == 0
    return {
        "is_good": is_good,
        "hr_est": hr_est,
        "flat_ratio": flat_ratio,
        "clip_ratio": clip_ratio,
        "baseline_std_ratio": baseline_std,
        "rr_cv": rr_cv,
        "reasons": "; ".join(reasons) if reasons else "OK",
    }


class ShimmerReader(QObject):
    def __init__(self):
        super().__init__()
        self.buf = deque(maxlen=BUF_SZ)
        self.sample_count = 0
        self.ser = None
        self.shim = None

    def handler(self, pkt: DataPacket):
        try:
            raw_counts = pkt[CH_ECG]
            signed = to_signed24(raw_counts)
            ecg_mv = signed * SENS_MV
            self.buf.append(ecg_mv)
            self.sample_count += 1
        except Exception:
            pass

    def init_shimmer(self):
        self.ser = serial.Serial(PORT, baudrate=DEFAULT_BAUDRATE, timeout=None)
        time.sleep(2)
        self.shim = ShimmerBluetooth(self.ser)
        self.shim.initialize()
        if CH_ECG not in self.shim.get_data_types():
            raise RuntimeError("ECG channel tidak aktif di Shimmer.")
        self.shim.add_stream_callback(self.handler)
        self.shim.start_streaming()

    def close(self):
        try:
            if self.shim is not None:
                self.shim.stop_streaming()
        except Exception:
            pass
        time.sleep(0.3)
        try:
            if self.ser is not None:
                self.ser.close()
        except Exception:
            pass


class MetricCard(QFrame):
    def __init__(self, title, value="--", accent="#22c55e"):
        super().__init__()
        self.setObjectName("MetricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(6)

        self.title = QLabel(title)
        self.title.setObjectName("CardTitle")
        self.value = QLabel(value)
        self.value.setObjectName("CardValue")
        self.value.setStyleSheet(f"color: {accent};")
        self.sub = QLabel("Realtime")
        self.sub.setObjectName("CardSub")

        layout.addWidget(self.title)
        layout.addWidget(self.value)
        layout.addWidget(self.sub)

    def update_card(self, value, sub=None, color=None):
        self.value.setText(value)
        if sub is not None:
            self.sub.setText(sub)
        if color is not None:
            self.value.setStyleSheet(f"color: {color};")


class RPPMonitorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RPP Monitoring Real-Time | Shimmer3 ECG")
        self.resize(1600, 920)

        self.reader = ShimmerReader()
        self.model_sbp = joblib.load("rf2_sbp.pkl")
        self.feat_scaler = joblib.load("feat_scaler.pkl")
        self.ecg_stats = joblib.load("params.pkl")

        self.app_mode = "READY"
        self.selected_phase = None
        self.session_id = 0
        self.session_start_time = None
        self.rest_done = False
        self.post_done = False

        self.all_window_logs = []
        self.current_session_logs = []
        self.rest_rpp_values = []
        self.post_rpp_values = []
        self.recovery_rpp_values = []
        self.rest_hr_values = []
        self.rest_sbp_values = []

        self.baseline_rpp = None
        self.baseline_hr = None
        self.baseline_sbp = None

        self.post_peak_rpp = None
        self.post_peak_hr = None
        self.post_peak_sbp = None

        self.recovery_start_time = None

        self.rpp_history = deque(maxlen=180)

        self.total_comp_times = []
        self.preprocess_feat_times = []
        self.predict_times = []

        self._build_ui()
        self._apply_theme()
        self._connect_signals()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_dashboard)
        self.timer.start(50)

        self._set_status_message("Siap. Hubungkan Shimmer lalu pilih fase.")

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)
        main.setContentsMargins(18, 18, 18, 18)
        main.setSpacing(18)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(320)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(18, 18, 18, 18)
        side.setSpacing(14)

        badge = QLabel("RPP Monitoring Suite")
        badge.setObjectName("AppBadge")
        title = QLabel("Estimasi Rate Pressure Product")
        title.setObjectName("MainTitle")
        subtitle = QLabel("Shimmer3 ECG • REST → POST-EXERCISE → RECOVERY")
        subtitle.setObjectName("SubTitle")

        self.conn_label = QLabel("Status device: Belum terhubung")
        self.conn_label.setObjectName("InfoPill")

        self.btn_connect = QPushButton("Hubungkan Shimmer")
        self.btn_start = QPushButton("Mulai Monitoring")
        self.btn_menu = QPushButton("Akhiri Fase / Kembali")
        self.btn_save = QPushButton("Simpan Semua Log")

        self.phase_combo = QComboBox()
        self.phase_combo.addItems(["REST", "POST-EXERCISE", "RECOVERY"])

        self.phase_hint = QLabel("Urutan fase wajib: REST → POST-EXERCISE → RECOVERY")
        self.phase_hint.setWordWrap(True)
        self.phase_hint.setObjectName("HintLabel")

        self.status_panel = QFrame()
        self.status_panel.setObjectName("StatusPanel")
        sp = QVBoxLayout(self.status_panel)
        sp.setContentsMargins(14, 14, 14, 14)
        sp.setSpacing(8)
        sp.addWidget(QLabel("Status sistem"))
        self.status_message = QLabel("-")
        self.status_message.setObjectName("StatusMessage")
        self.status_message.setWordWrap(True)
        sp.addWidget(self.status_message)

        side.addWidget(badge)
        side.addWidget(title)
        side.addWidget(subtitle)
        side.addSpacing(8)
        side.addWidget(self.conn_label)
        side.addWidget(self.phase_combo)
        side.addWidget(self.btn_connect)
        side.addWidget(self.btn_start)
        side.addWidget(self.btn_menu)
        side.addWidget(self.btn_save)
        side.addWidget(self.phase_hint)
        side.addWidget(self.status_panel)
        side.addStretch(1)

        body = QVBoxLayout()
        body.setSpacing(18)

        top_cards = QGridLayout()
        top_cards.setHorizontalSpacing(14)
        top_cards.setVerticalSpacing(14)
        self.card_hr = MetricCard("Heart Rate", "-- BPM", "#fbbf24")
        self.card_sbp = MetricCard("Predicted SBP", "-- mmHg", "#60a5fa")
        self.card_rpp = MetricCard("Current RPP", "--", "#f87171")
        self.card_phase = MetricCard("Fase Aktif", "--", "#34d399")
        self.card_load = MetricCard("Status Beban", "--", "#c084fc")
        self.card_recovery = MetricCard("Status Recovery", "--", "#38bdf8")

        cards = [
            self.card_hr,
            self.card_sbp,
            self.card_rpp,
            self.card_phase,
            self.card_load,
            self.card_recovery,
        ]
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        for card, pos in zip(cards, positions):
            top_cards.addWidget(card, *pos)

        plot_row = QGridLayout()
        plot_row.setHorizontalSpacing(18)
        plot_row.setVerticalSpacing(18)

        self.ecg_plot = pg.PlotWidget(title="Live ECG Stream")
        self.ecg_plot.setObjectName("PlotCard")
        self.ecg_plot.showGrid(x=True, y=True, alpha=0.15)
        self.ecg_plot.setLabel("left", "Amplitude (norm)")
        self.ecg_plot.setLabel("bottom", "Samples")
        self.ecg_curve = self.ecg_plot.plot(pen=pg.mkPen("#22c55e", width=2))

        self.rpp_plot = pg.PlotWidget(title="RPP Trend")
        self.rpp_plot.setObjectName("PlotCard")
        self.rpp_plot.showGrid(x=True, y=True, alpha=0.15)
        self.rpp_plot.setLabel("left", "RPP")
        self.rpp_plot.setLabel("bottom", "Window")
        self.rpp_curve = self.rpp_plot.plot(pen=pg.mkPen("#fb7185", width=2.5))

        plot_row.addWidget(self.ecg_plot, 0, 0)
        plot_row.addWidget(self.rpp_plot, 1, 0)

        extra_panel = QFrame()
        extra_panel.setObjectName("MetricCard")
        extra_layout = QVBoxLayout(extra_panel)
        extra_layout.setContentsMargins(18, 18, 18, 18)
        extra_layout.setSpacing(10)

        extra_title = QLabel("Analitik Fase")
        extra_title.setObjectName("SectionTitle")
        self.delta_label = QLabel("ΔRPP: --")
        self.recovery_pct_label = QLabel("%Recovery: --")
        self.baseline_label = QLabel("Baseline RPP: --")
        self.peak_label = QLabel("Peak Post-Exercise: --")
        self.benchmark_label = QLabel("Benchmark: --")
        self.hemo_label = QLabel("Flag hemodinamik: --")
        self.hrr_label = QLabel("HRR: --")
        self.qc_label = QLabel("Kualitas Sinyal: --")

        for w in [
            self.delta_label,
            self.recovery_pct_label,
            self.baseline_label,
            self.peak_label,
            self.benchmark_label,
            self.hemo_label,
            self.hrr_label,
            self.qc_label,
        ]:
            w.setObjectName("DetailText")
            extra_layout.addWidget(w)

        extra_layout.insertWidget(0, extra_title)
        extra_layout.addStretch(1)

        plot_row.addWidget(extra_panel, 0, 1, 2, 1)
        plot_row.setColumnStretch(0, 3)
        plot_row.setColumnStretch(1, 1)

        body.addLayout(top_cards)
        body.addLayout(plot_row)

        main.addWidget(sidebar)
        main.addLayout(body, 1)

    def _apply_theme(self):
        self.setStyleSheet("""
        QWidget {
            background: #0b1220;
            color: #e5eefb;
            font-family: 'Inter', 'Segoe UI', sans-serif;
            font-size: 11pt;
        }
        QMainWindow {
            background: #0b1220;
        }
        QFrame#Sidebar {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #111827, stop:1 #0f172a);
            border: 1px solid #1f2937;
            border-radius: 22px;
        }
        QFrame#MetricCard, QFrame#StatusPanel {
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 20px;
        }
        QLabel#AppBadge {
            color: #67e8f9;
            font-size: 10pt;
            font-weight: 700;
            letter-spacing: 1px;
        }
        QLabel#MainTitle {
            font-size: 20pt;
            font-weight: 800;
            color: #f8fafc;
        }
        QLabel#SubTitle, QLabel#HintLabel, QLabel#CardSub {
            color: #94a3b8;
        }
        QLabel#InfoPill {
            background: #0f1a2d;
            border: 1px solid #1e293b;
            border-radius: 14px;
            padding: 10px 12px;
            color: #cbd5e1;
            font-weight: 600;
        }
        QLabel#CardTitle, QLabel#SectionTitle {
            color: #93c5fd;
            font-size: 10pt;
            font-weight: 700;
        }
        QLabel#CardValue {
            font-size: 24pt;
            font-weight: 800;
            color: #f8fafc;
        }
        QLabel#DetailText, QLabel#StatusMessage {
            color: #dbeafe;
            font-size: 11pt;
        }
        QPushButton {
            background: #172554;
            border: 1px solid #1d4ed8;
            color: #eff6ff;
            border-radius: 14px;
            padding: 12px 14px;
            font-weight: 700;
        }
        QPushButton:hover {
            background: #1d4ed8;
        }
        QPushButton:pressed {
            background: #1e40af;
        }
        QComboBox {
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 12px;
            min-height: 22px;
        }
        QComboBox QAbstractItemView {
            background: #111827;
            selection-background-color: #1d4ed8;
            border: 1px solid #334155;
        }
        """)
        pg.setConfigOptions(antialias=True)
        bg = "#111827"
        fg = "#cbd5e1"
        for pw in [self.ecg_plot, self.rpp_plot]:
            pw.setBackground(bg)
            pw.getAxis("left").setTextPen(fg)
            pw.getAxis("bottom").setTextPen(fg)
            pw.getAxis("left").setPen(pg.mkPen(fg))
            pw.getAxis("bottom").setPen(pg.mkPen(fg))
            pw.getPlotItem().titleLabel.item.setDefaultTextColor(fg)

    def _connect_signals(self):
        self.btn_connect.clicked.connect(self.connect_shimmer)
        self.btn_start.clicked.connect(self.start_phase_session)
        self.btn_menu.clicked.connect(self.back_to_menu)
        self.btn_save.clicked.connect(self.save_all_logs)

        exit_action = QAction("Keluar", self)
        exit_action.triggered.connect(self.close)
        self.addAction(exit_action)

    def _set_status_message(self, text):
        self.status_message.setText(text)

    def connect_shimmer(self):
        try:
            self.reader.init_shimmer()
            self.conn_label.setText(f"Status device: Terhubung ke {PORT}")
            self._set_status_message(
                "Shimmer berhasil terhubung. Silakan mulai fase REST."
            )
        except Exception as e:
            self.conn_label.setText("Status device: Gagal terhubung")
            QMessageBox.critical(self, "Koneksi gagal", str(e))

    def reset_session_display_only(self):
        self.current_session_logs = []
        self.rpp_history.clear()
        self.card_hr.update_card("-- BPM", "Realtime")
        self.card_sbp.update_card("-- mmHg", "Prediksi")
        self.card_rpp.update_card("--", "RPP aktif")
        self.card_phase.update_card("--", "Pilih fase")
        self.card_load.update_card("--", "Belum ada")
        self.card_recovery.update_card("--", "Belum ada")
        self.delta_label.setText("ΔRPP: --")
        self.recovery_pct_label.setText("%Recovery: --")
        self.hemo_label.setText("Flag hemodinamik: --")
        self.hrr_label.setText("HRR: --")
        self.qc_label.setText("Kualitas Sinyal: --")
        self.qc_label.setStyleSheet("")

    def start_phase_session(self):
        phase_name = self.phase_combo.currentText()
        if phase_name == "POST-EXERCISE" and not self.rest_done:
            self._set_status_message("Jalankan fase REST terlebih dahulu.")
            return
        if phase_name == "RECOVERY" and not self.post_done:
            self._set_status_message("Jalankan fase POST-EXERCISE terlebih dahulu.")
            return

        self.session_id += 1
        self.selected_phase = phase_name
        self.session_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.reset_session_display_only()
        self.app_mode = "MONITOR"

        if phase_name == "RECOVERY":
            self.recovery_start_time = time.time()

        self.card_phase.update_card(
            phase_name, f"Session #{self.session_id}", "#34d399"
        )
        self._set_status_message(f"Monitoring fase {phase_name} dimulai.")

    def back_to_menu(self):
        if self.selected_phase == "REST":
            self.rest_done = True
        elif self.selected_phase == "POST-EXERCISE":
            self.post_done = True

        self.save_current_session()
        self.selected_phase = None
        self.app_mode = "READY"
        self._set_status_message("Fase diakhiri. Pilih fase berikutnya.")

    def save_current_session(self):
        if not self.current_session_logs:
            return
        df_session = pd.DataFrame(self.current_session_logs)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"rpp_session_{self.session_id}_{timestamp}.csv"
        df_session.to_csv(fname, index=False)
        self.all_window_logs.extend(self.current_session_logs)
        self.current_session_logs = []
        self._set_status_message(f"Sesi tersimpan ke {fname}")

    def save_all_logs(self):
        self.save_current_session()
        if not self.all_window_logs:
            self._set_status_message("Belum ada log untuk disimpan.")
            return
        df_all = pd.DataFrame(self.all_window_logs)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"rpp_monitoring_all_{timestamp}.csv"
        df_all.to_csv(fname, index=False)
        self._set_status_message(f"Seluruh log disimpan ke {fname}")

    def update_dashboard(self):
        current_data = list(self.reader.buf)
        if current_data:
            raw_np = np.array(current_data)
            norm_view = (raw_np - raw_np.min()) / (raw_np.max() - raw_np.min() + 1e-9)
            self.ecg_curve.setData(norm_view)

        if self.app_mode != "MONITOR":
            return

        if len(self.reader.buf) >= BUF_SZ and self.reader.sample_count % 60 == 0:
            try:
                start_total = time.perf_counter()
                start_pf = time.perf_counter()

                raw = np.array(self.reader.buf)
                ecg_clean = notch_filter(bandpass_filter(raw))
                ecg_norm = (ecg_clean - self.ecg_stats["mean"]) / self.ecg_stats["std"]
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
                feats_scaled = self.feat_scaler.transform(feats)
                sbp_pred = self.model_sbp.predict(feats_scaled)[0]

                peaks, _ = find_peaks(
                    y_mwi, distance=int(0.333 * FS), height=np.mean(y_mwi)
                )
                hr_val = (len(peaks) / BUF_SEC) * 60
                rpp_val = hr_val * sbp_pred

                end_pred = time.perf_counter()
                end_total = time.perf_counter()

                self.preprocess_feat_times.append(end_pf - start_pf)
                self.predict_times.append(end_pred - start_pred)
                self.total_comp_times.append(end_total - start_total)

                qc_result = ecg_quality_check(raw, ecg_clean, y_mwi, peaks)
                qc_color = "#22c55e" if qc_result["is_good"] else "#ef4444"
                self.qc_label.setText(
                    f"Kualitas Sinyal: {'BAIK' if qc_result['is_good'] else 'BERMASALAH'}"
                )
                self.qc_label.setStyleSheet(f"color: {qc_color};")

                phase = self.selected_phase

                if phase == "REST":
                    self.rest_rpp_values.append(rpp_val)
                    self.rest_hr_values.append(hr_val)
                    self.rest_sbp_values.append(sbp_pred)

                    self.baseline_rpp = safe_mean(self.rest_rpp_values)
                    self.baseline_hr = safe_mean(self.rest_hr_values)
                    self.baseline_sbp = safe_mean(self.rest_sbp_values)

                elif phase == "POST-EXERCISE":
                    self.post_rpp_values.append(rpp_val)
                    if self.post_peak_rpp is None or rpp_val > self.post_peak_rpp:
                        self.post_peak_rpp = rpp_val
                    if self.post_peak_hr is None or hr_val > self.post_peak_hr:
                        self.post_peak_hr = hr_val
                    if self.post_peak_sbp is None or sbp_pred > self.post_peak_sbp:
                        self.post_peak_sbp = sbp_pred

                elif phase == "RECOVERY":
                    self.recovery_rpp_values.append(rpp_val)

                load_label, load_color, delta_rpp, rpp_ratio = classify_load(
                    rpp_val, self.baseline_rpp, phase
                )

                rec_label, rec_color, rec_drop, rec_pct, hrr_val = (
                    "Belum masuk recovery",
                    "#94a3b8",
                    None,
                    None,
                    None,
                )

                elapsed_rec_sec = None
                if phase == "RECOVERY" and self.recovery_start_time is not None:
                    elapsed_rec_sec = time.time() - self.recovery_start_time
                    rec_label, rec_color, rec_drop, rec_pct, hrr_val = (
                        classify_recovery(
                            current_rpp=rpp_val,
                            baseline_rpp=self.baseline_rpp,
                            post_peak_rpp=self.post_peak_rpp,
                            current_hr=hr_val,
                            peak_hr=self.post_peak_hr,
                            elapsed_recovery_sec=elapsed_rec_sec,
                        )
                    )

                hemo_label, hemo_color = classify_hemodynamic_flag(
                    phase=phase,
                    sbp=sbp_pred,
                    hr=hr_val,
                    baseline_sbp=self.baseline_sbp,
                    baseline_hr=self.baseline_hr,
                )

                self.card_hr.update_card(
                    f"{hr_val:.1f} BPM", "Deteksi R-peak", "#fbbf24"
                )
                self.card_sbp.update_card(
                    f"{sbp_pred:.1f} mmHg", "Estimasi model RF", "#60a5fa"
                )
                self.card_rpp.update_card(
                    f"{rpp_val:.0f}", "Rate Pressure Product", "#f87171"
                )
                self.card_load.update_card(
                    load_label,
                    f"Rasio vs baseline: {rpp_ratio:.2f}x"
                    if rpp_ratio is not None
                    else "Status beban",
                    load_color,
                )
                self.card_recovery.update_card(
                    rec_label, "Respons pemulihan", rec_color
                )

                self.delta_label.setText(
                    f"ΔRPP: {delta_rpp:+.0f}" if delta_rpp is not None else "ΔRPP: --"
                )
                self.recovery_pct_label.setText(
                    f"%Recovery: {rec_pct:.1f}%"
                    if rec_pct is not None
                    else "%Recovery: --"
                )
                self.baseline_label.setText(
                    f"Baseline RPP: {self.baseline_rpp:.0f}"
                    if self.baseline_rpp is not None
                    else "Baseline RPP: --"
                )
                self.peak_label.setText(
                    f"Peak Post-Exercise: {self.post_peak_rpp:.0f}"
                    if self.post_peak_rpp is not None
                    else "Peak Post-Exercise: --"
                )
                self.hemo_label.setText(f"Flag hemodinamik: {hemo_label}")
                self.hrr_label.setText(
                    f"HRR: {hrr_val:.1f} bpm" if hrr_val is not None else "HRR: --"
                )

                if self.total_comp_times:
                    avg_total = np.mean(self.total_comp_times)
                    self.benchmark_label.setText(f"Benchmark: {avg_total:.4f} s/window")

                self.rpp_history.append(rpp_val)
                self.rpp_curve.setData(list(self.rpp_history))

                self.current_session_logs.append(
                    {
                        "session_id": self.session_id,
                        "session_start": self.session_start_time,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "sample_count": self.reader.sample_count,
                        "phase": phase,
                        "hr_bpm": hr_val,
                        "sbp_mmhg": sbp_pred,
                        "rpp": rpp_val,
                        "baseline_hr": self.baseline_hr,
                        "baseline_sbp": self.baseline_sbp,
                        "baseline_rpp": self.baseline_rpp,
                        "rpp_ratio_to_baseline": rpp_ratio,
                        "delta_rpp": delta_rpp,
                        "post_peak_hr": self.post_peak_hr,
                        "post_peak_sbp": self.post_peak_sbp,
                        "post_peak_rpp": self.post_peak_rpp,
                        "recovery_drop": rec_drop,
                        "recovery_pct": rec_pct,
                        "hrr_bpm": hrr_val,
                        "load_status": load_label,
                        "recovery_status": rec_label,
                        "hemodynamic_flag": hemo_label,
                        "qc_status": "BAIK" if qc_result["is_good"] else "BERMASALAH",
                        "qc_reasons": qc_result["reasons"],
                        "qc_flat_ratio": qc_result["flat_ratio"],
                        "qc_clip_ratio": qc_result["clip_ratio"],
                        "qc_baseline_std_ratio": qc_result["baseline_std_ratio"],
                        "qc_rr_cv": qc_result["rr_cv"],
                    }
                )

            except Exception as e:
                self._set_status_message(f"Error update: {e}")

    def closeEvent(self, event):
        self.save_current_session()
        self.save_all_logs()
        self.reader.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("RPP Monitoring Real-Time")
    font = QFont("Inter", 10)
    app.setFont(font)
    win = RPPMonitorWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except SerialException as e:
        print(f"Koneksi serial gagal: {e}")
        sys.exit(1)
