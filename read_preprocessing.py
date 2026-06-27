import os
import sys
import warnings
from collections import deque

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch

from PySide6.QtCore import QTimer, QObject
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import pyqtgraph as pg

warnings.filterwarnings("ignore")

FS = 125
BUF_SEC = 20
BUF_SZ = FS * BUF_SEC

CHUNK_SPEED = int(FS * 0.05)


def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=FS, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def notch_filter(data, f0=50.0, Q=30.0, fs=FS):
    b, a = iirnotch(f0 / (0.5 * fs), Q)
    return filtfilt(b, a, data)


class FileReader(QObject):
    def __init__(self, filepath=None):
        super().__init__()
        self.buf = deque(maxlen=BUF_SZ)
        self.sample_count = 0
        self.filepath = filepath
        self.data = None
        self.total_samples = 0
        self.pos = 0
        self.finished = False
        self.loaded = False

    def load_file(self, filepath):
        df = pd.read_csv(filepath)
        self.data = df["raw_ecg_mv"].values
        self.total_samples = len(self.data)
        self.filepath = filepath
        self.pos = 0
        self.sample_count = 0
        self.finished = False
        self.loaded = True
        self.buf.clear()

    def feed_chunk(self):
        if self.finished or not self.loaded:
            return
        end = min(self.pos + CHUNK_SPEED, self.total_samples)
        for i in range(self.pos, end):
            self.buf.append(float(self.data[i]))
            self.sample_count += 1
        self.pos = end
        if self.pos >= self.total_samples:
            self.finished = True

    @property
    def progress_pct(self):
        if self.total_samples == 0:
            return 0.0
        return min(100.0, self.pos / self.total_samples * 100.0)


class PreprocessMonitorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG Preprocessing Stages | CSV File")
        self.resize(1400, 900)

        self.reader = FileReader()
        self.file_loaded = False

        self._build_ui()
        self._apply_theme()
        self._connect_signals()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(50)

        self._set_status("Pilih file CSV ECG untuk memulai.")

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)
        main.setContentsMargins(18, 18, 18, 18)
        main.setSpacing(18)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(280)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(18, 18, 18, 18)
        side.setSpacing(14)

        badge = QLabel("Preprocessing Viewer")
        badge.setObjectName("AppBadge")
        title = QLabel("ECG Signal\nPreprocessing Stages")
        title.setObjectName("MainTitle")
        subtitle = QLabel("Bandpass & Notch Filter Pipeline")
        subtitle.setObjectName("SubTitle")

        self.conn_label = QLabel("Status file: Belum dipilih")
        self.conn_label.setObjectName("InfoPill")

        self.btn_open = QPushButton("Pilih File CSV ECG")
        self.btn_start = QPushButton("Mulai / Restart")

        self.status_panel = QFrame()
        self.status_panel.setObjectName("StatusPanel")
        sp = QVBoxLayout(self.status_panel)
        sp.setContentsMargins(14, 14, 14, 14)
        sp.setSpacing(8)
        sp.addWidget(QLabel("Status"))
        self.status_message = QLabel("-")
        self.status_message.setObjectName("StatusMessage")
        self.status_message.setWordWrap(True)
        sp.addWidget(self.status_message)

        side.addWidget(badge)
        side.addWidget(title)
        side.addWidget(subtitle)
        side.addSpacing(8)
        side.addWidget(self.conn_label)
        side.addWidget(self.btn_open)
        side.addWidget(self.btn_start)
        side.addWidget(self.status_panel)
        side.addStretch(1)

        body = QVBoxLayout()
        body.setSpacing(14)

        self.raw_plot = pg.PlotWidget(title="Raw ECG Signal")
        self.raw_plot.setObjectName("PlotCard")
        self.raw_plot.showGrid(x=True, y=True, alpha=0.15)
        self.raw_plot.setLabel("left", "Amplitude (mV)")
        self.raw_plot.setLabel("bottom", "Samples")
        self.raw_curve = self.raw_plot.plot(pen=pg.mkPen("#22c55e", width=1.5))

        self.bp_plot = pg.PlotWidget(title="Bandpass Filter (0.5 - 40 Hz)")
        self.bp_plot.setObjectName("PlotCard")
        self.bp_plot.showGrid(x=True, y=True, alpha=0.15)
        self.bp_plot.setLabel("left", "Amplitude (mV)")
        self.bp_plot.setLabel("bottom", "Samples")
        self.bp_curve = self.bp_plot.plot(pen=pg.mkPen("#60a5fa", width=1.5))

        self.notch_plot = pg.PlotWidget(title="Notch Filter (50 Hz)")
        self.notch_plot.setObjectName("PlotCard")
        self.notch_plot.showGrid(x=True, y=True, alpha=0.15)
        self.notch_plot.setLabel("left", "Amplitude (mV)")
        self.notch_plot.setLabel("bottom", "Samples")
        self.notch_curve = self.notch_plot.plot(pen=pg.mkPen("#facc15", width=1.5))

        self.clean_plot = pg.PlotWidget(title="Bandpass + Notch Filter")
        self.clean_plot.setObjectName("PlotCard")
        self.clean_plot.showGrid(x=True, y=True, alpha=0.15)
        self.clean_plot.setLabel("left", "Amplitude (mV)")
        self.clean_plot.setLabel("bottom", "Samples")
        self.clean_curve = self.clean_plot.plot(pen=pg.mkPen("#f87171", width=1.5))

        body.addWidget(self.raw_plot)
        body.addWidget(self.bp_plot)
        body.addWidget(self.notch_plot)
        body.addWidget(self.clean_plot)

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
        QFrame#StatusPanel {
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
            font-size: 18pt;
            font-weight: 800;
            color: #f8fafc;
        }
        QLabel#SubTitle {
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
        QLabel#StatusMessage {
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
        """)
        pg.setConfigOptions(antialias=True)
        bg = "#111827"
        fg = "#cbd5e1"
        for pw in [self.raw_plot, self.bp_plot, self.notch_plot, self.clean_plot]:
            pw.setBackground(bg)
            pw.getAxis("left").setTextPen(fg)
            pw.getAxis("bottom").setTextPen(fg)
            pw.getAxis("left").setPen(pg.mkPen(fg))
            pw.getAxis("bottom").setPen(pg.mkPen(fg))
            pw.getPlotItem().titleLabel.item.setDefaultTextColor(fg)

    def _connect_signals(self):
        self.btn_open.clicked.connect(self.open_csv_file)
        self.btn_start.clicked.connect(self.start_playback)

        exit_action = QAction("Keluar", self)
        exit_action.triggered.connect(self.close)
        self.addAction(exit_action)

    def _set_status(self, text):
        self.status_message.setText(text)

    def open_csv_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Pilih File CSV ECG",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not filepath:
            return
        try:
            self.reader.load_file(filepath)
            self.file_loaded = True
            fname = os.path.basename(filepath)
            total_sec = self.reader.total_samples / FS
            self.conn_label.setText(
                f"File: {fname} | {self.reader.total_samples} sampel ({total_sec:.1f} s)"
            )
            self._set_status(f"File berhasil dimuat: {fname}. Klik Mulai untuk playback.")
        except Exception as e:
            self.conn_label.setText("Status file: Gagal dimuat")
            self._set_status(f"Gagal memuat file: {e}")
            self.file_loaded = False

    def start_playback(self):
        if not self.file_loaded:
            self._set_status("Pilih file CSV ECG terlebih dahulu.")
            return
        self.reader.pos = 0
        self.reader.sample_count = 0
        self.reader.finished = False
        self.reader.buf.clear()
        self._set_status("Playback dimulai...")

    def update_plots(self):
        self.reader.feed_chunk()

        current_data = list(self.reader.buf)
        if not current_data:
            self.reader.feed_chunk()
            current_data = list(self.reader.buf)
            if not current_data:
                return

        if self.reader.finished:
            self._set_status(
                f"File selesai ({self.reader.total_samples} sampel). "
                "Klik Mulai untuk mengulang."
            )

        raw = np.array(current_data, dtype=float)

        if len(raw) < 10:
            return

        self.raw_curve.setData(raw)

        if len(raw) >= 20:
            bp_signal = bandpass_filter(raw)
            notch_only = notch_filter(raw)
            clean = notch_filter(bandpass_filter(raw))

            self.bp_curve.setData(bp_signal)
            self.notch_curve.setData(notch_only)
            self.clean_curve.setData(clean)

    def closeEvent(self, event):
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ECG Preprocessing Viewer")
    font = QFont("Inter", 10)
    app.setFont(font)
    win = PreprocessMonitorWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
