#!/usr/bin/env python3
"""Quick-look GUI for PicoScope 3416E MSO waveform acquisition.

    picodaq-gui [--config config_acquisition.yaml] [--sim]
    (or: python -m picodaq.gui ...)

Tune mode : threshold / batch size / batch timeout can be changed while
            acquiring (applied between batches); saving is optional and goes
            to <base>_tune_NNNNN.h5.
Run mode  : settings are locked (GUI and worker); always saved.

Acquisition + HDF5 writing run in a separate process (picodaq.acquisition.worker_main).
"""
import argparse
import copy
import re
import multiprocessing as mp
import queue
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .acquisition import data_dir, next_file_id, output_path, worker_main
from .config import load_config

BASE_RE = re.compile(r"^[A-Za-z0-9._-]+$")  # allowed in the file name base
N_WAVE_DRAW = 20  # waveforms kept on screen per channel
UPDATE_MS = 200
MAX_RATE_POINTS = 20000


class QuickLookHists:
    def __init__(self, ql):
        amp_max = float(ql.get("amp_max_v", 2.0))
        self.amp_edges = np.linspace(0, amp_max, int(ql.get("amp_bins", 512)) + 1)
        self.amp2d_edges = np.linspace(0, amp_max, int(ql.get("amp2d_bins", 256)) + 1)
        r0, r1 = ql.get("ratio_range", [0.0, 2.0])
        self.ratio_edges = np.linspace(r0, r1, int(ql.get("ratio_bins", 200)) + 1)
        self.reset()

    def reset(self):
        self.anode = np.zeros(len(self.amp_edges) - 1)
        self.cathode = np.zeros(len(self.amp_edges) - 1)
        self.h2d = np.zeros((len(self.amp2d_edges) - 1, len(self.ratio_edges) - 1))
        self.events = 0

    def fill(self, a, c):
        self.anode += np.histogram(a, self.amp_edges)[0]
        self.cathode += np.histogram(c, self.amp_edges)[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = c / a
        ok = np.isfinite(r) & (a > 0)
        self.h2d += np.histogram2d(a[ok], r[ok], (self.amp2d_edges, self.ratio_edges))[0]
        self.events += len(a)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config_path, simulate=False):
        super().__init__()
        self.setWindowTitle("PicoScope QL")
        self.ctx = mp.get_context("spawn")
        self.proc = None
        self.queue = None
        self.cmd_queue = None
        self.stop_event = None
        self.running = False
        self.run_mode = "tune"
        self.config = None
        self.marks = []
        self._build_ui(config_path, simulate)
        self._reset_run_state()
        try:
            self.load_config(init=True)
        except Exception as e:
            self.log(f"config not loaded: {e}")
        self._update_controls()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(UPDATE_MS)

    # ------------------------------------------------------------ UI
    def _build_ui(self, config_path, simulate):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)

        # row 1: run control
        row1 = QtWidgets.QHBoxLayout()
        self.cfg_edit = QtWidgets.QLineEdit(str(config_path))
        browse = QtWidgets.QPushButton("…")
        browse.setFixedWidth(30)
        browse.clicked.connect(self.browse_config)
        self.browse_btn = browse
        self.mode_box = QtWidgets.QComboBox()
        self.mode_box.addItems(["Tune", "Run"])
        self.mode_box.setToolTip("Tune: settings adjustable while acquiring.\nRun: settings locked, always saved.")
        self.mode_box.currentIndexChanged.connect(self._update_controls)
        self.save_box = QtWidgets.QCheckBox("Save")
        self.save_box.toggled.connect(self._update_file_label)
        self.dur_spin = QtWidgets.QDoubleSpinBox()
        self.dur_spin.setRange(0, 1e5)
        self.dur_spin.setDecimals(1)
        self.dur_spin.setSuffix(" min")
        self.dur_spin.setSpecialValueText("∞")
        self.sim_box = QtWidgets.QCheckBox("Simulate")
        self.sim_box.setChecked(simulate)
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        for w in (QtWidgets.QLabel("Config"), self.cfg_edit, browse, QtWidgets.QLabel("Mode"),
                  self.mode_box, self.save_box, QtWidgets.QLabel("Duration"), self.dur_spin,
                  self.sim_box, self.start_btn, self.stop_btn):
            row1.addWidget(w)
        vbox.addLayout(row1)

        # row 2: trigger / batch settings
        row2 = QtWidgets.QHBoxLayout()
        self.thr_spin = QtWidgets.QDoubleSpinBox()
        self.thr_spin.setRange(1, 2000)
        self.thr_spin.setDecimals(0)
        self.thr_spin.setSuffix(" mV")
        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(1, 10000)
        self.timeout_spin = QtWidgets.QDoubleSpinBox()
        self.timeout_spin.setRange(0, 600)
        self.timeout_spin.setDecimals(1)
        self.timeout_spin.setSuffix(" s")
        self.timeout_spin.setSpecialValueText("off")
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.setToolTip("Send the settings to the running acquisition (Tune mode only)")
        self.apply_btn.clicked.connect(self.apply_settings)
        self.reset_btn = QtWidgets.QPushButton("Reset QL")
        self.reset_btn.clicked.connect(self.reset_ql)
        self.log_box = QtWidgets.QCheckBox("log y")
        self.log_box.toggled.connect(self.set_logy)
        self.lock_label = QtWidgets.QLabel("")
        for w in (QtWidgets.QLabel("Threshold"), self.thr_spin, QtWidgets.QLabel("Batch"),
                  self.batch_spin, QtWidgets.QLabel("Batch timeout"), self.timeout_spin,
                  self.apply_btn, self.lock_label):
            row2.addWidget(w)
        row2.addStretch(1)
        row2.addWidget(self.reset_btn)
        row2.addWidget(self.log_box)
        vbox.addLayout(row2)

        # row 3: output file (editable while idle)
        row3 = QtWidgets.QHBoxLayout()
        self.dir_edit = QtWidgets.QLineEdit()
        self.dir_edit.setToolTip("Output folder. strftime codes are expanded at start, e.g. data/%Y%m%d")
        self.dir_btn = QtWidgets.QPushButton("…")
        self.dir_btn.setFixedWidth(30)
        self.dir_btn.clicked.connect(self.browse_dir)
        self.base_edit = QtWidgets.QLineEdit()
        self.base_edit.setMaximumWidth(220)
        self.base_edit.setToolTip("File name base: <base>_NNNNN.h5 (Run), <base>_tune_NNNNN.h5 (Tune).\n"
                                  "Letters, digits, . _ - only. NNNNN counts up; existing files are never overwritten.")
        self.dir_edit.textChanged.connect(self._update_file_label)
        self.base_edit.textChanged.connect(self._update_file_label)
        for w in (QtWidgets.QLabel("Save to"), self.dir_edit, self.dir_btn, QtWidgets.QLabel("Name"),
                  self.base_edit):
            row3.addWidget(w)
        vbox.addLayout(row3)
        self.file_label = QtWidgets.QLabel("")
        self.file_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        f = self.file_label.font()
        f.setBold(True)
        f.setPointSizeF(f.pointSizeF() * 1.15)
        self.file_label.setFont(f)
        vbox.addWidget(self.file_label)

        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.status = QtWidgets.QLabel("idle")
        self.status.setFont(mono)
        self.rate_label = QtWidgets.QLabel("")
        self.rate_label.setFont(mono)
        vbox.addWidget(self.status)
        vbox.addWidget(self.rate_label)

        pg.setConfigOptions(antialias=False)
        self.glw = pg.GraphicsLayoutWidget()
        vbox.addWidget(self.glw, 1)
        self.p_wave = {"anode": self.glw.addPlot(row=0, col=0),
                       "cathode": self.glw.addPlot(row=1, col=0)}
        self.p_spec = {"anode": self.glw.addPlot(row=0, col=1),
                       "cathode": self.glw.addPlot(row=1, col=1)}
        self.p_2d = self.glw.addPlot(row=2, col=0, title="C/A vs anode PH")
        self.p_rate = self.glw.addPlot(row=2, col=1, title="Trigger rate")

        colors = {"anode": (255, 190, 0), "cathode": (0, 200, 255)}
        self.wave_items, self.spec_items = {}, {}
        for key in ("anode", "cathode"):
            p = self.p_wave[key]
            p.setLabel("bottom", "time", "s")
            p.setLabel("left", "voltage", "V")
            p.showGrid(x=True, y=True, alpha=0.3)
            self.wave_items[key] = [p.plot(pen=pg.mkPen((*colors[key], 70)))
                                    for _ in range(N_WAVE_DRAW)]
            self.wave_items[key][-1].setPen(pg.mkPen(colors[key], width=2))  # newest
            s = self.p_spec[key]
            s.setLabel("bottom", "pulse height", "V")
            s.setLabel("left", "counts")
            s.showGrid(x=True, y=True, alpha=0.3)
            self.spec_items[key] = s.plot(stepMode="center", pen=pg.mkPen(colors[key]))
        self.p_wave["cathode"].setXLink(self.p_wave["anode"])
        self.p_spec["cathode"].setXLink(self.p_spec["anode"])

        self.img = pg.ImageItem()
        self.img.setColorMap(pg.colormap.get("viridis"))
        self.p_2d.addItem(self.img)
        self.p_2d.setLabel("bottom", "anode PH", "V")
        self.p_2d.setLabel("left", "cathode / anode")

        self.p_rate.setLabel("bottom", "elapsed", "s")
        self.p_rate.setLabel("left", "rate", "Hz")
        self.p_rate.showGrid(x=True, y=True, alpha=0.3)
        self.p_rate.addLegend(offset=(-10, 10))
        self.rate_in_item = self.p_rate.plot(pen=pg.mkPen((230, 230, 230)), name="input (captured+missed)")
        self.rate_acc_item = self.p_rate.plot(pen=pg.mkPen((90, 220, 120)), name="accepted")

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(90)
        vbox.addWidget(self.log_view)

    def log(self, text):
        self.log_view.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {text}")

    def browse_config(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Config", "", "YAML (*.yaml *.yml)")
        if path:
            self.cfg_edit.setText(path)
            try:
                self.load_config(init=True)
            except Exception as e:
                self.log(f"config error: {e}")

    def browse_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Output folder", self.dir_edit.text())
        if path:
            self.dir_edit.setText(path)

    def _output_settings(self):
        """(data_dir, base) from the GUI fields; raises ValueError if the name is not usable."""
        folder = self.dir_edit.text().strip()
        base = self.base_edit.text().strip()
        if not folder:
            raise ValueError("output folder is empty")
        if not BASE_RE.match(base):
            raise ValueError(f"file name {base!r}: use letters, digits, '.', '_' or '-' only")
        return folder, base

    def _gui_mode(self):
        return "run" if self.mode_box.currentIndex() == 1 else "tune"

    def _update_file_label(self, *_):
        if not hasattr(self, "file_label"):
            return
        if self.running:
            if not self.saving:
                self.file_label.setText("Not saving")
            elif self.save_path:
                self.file_label.setText(f"Saving: {self.save_path}  ({self.file_count} wf)")
            else:
                self.file_label.setText("Saving: waiting for the first batch…")
            return
        mode = self._gui_mode()
        if mode == "tune" and not self.save_box.isChecked():
            self.file_label.setText("Not saving (Tune, Save off)")
            return
        if self.config is None:
            self.file_label.setText("")
            return
        try:
            folder, base = self._output_settings()
        except ValueError as e:
            self.file_label.setText(f"⚠ {e}")
            return
        cfg = {"output": dict(self.config["output"], data_dir=folder)}
        b = base if mode == "run" else f"{base}_tune"
        path = output_path(cfg, next_file_id(cfg, int(self.config["output"].get("start_id", 0)), b), b)
        self.file_label.setText(f"Next file: {path.resolve()}")

    def load_config(self, init=False):
        self.config = load_config(self.cfg_edit.text())
        proc = self.config["processing"]
        self.anode_ch = proc.get("anode", "A")
        self.cathode_ch = proc.get("cathode", "B")
        self.hists = QuickLookHists(self.config["ql"])
        self.p_wave["anode"].setTitle(f"Anode ({self.anode_ch})")
        self.p_wave["cathode"].setTitle(f"Cathode ({self.cathode_ch})")
        method = f"trapezoid {proc['shaper']['rise_us']:g}/{proc['shaper']['flat_us']:g} us, {proc['amplitude']}"
        self.p_spec["anode"].setTitle(f"Anode PH ({self.anode_ch}, {method})")
        self.p_spec["cathode"].setTitle(f"Cathode PH ({self.cathode_ch}, polarity-corrected)")
        if init:
            self.sim_box.setChecked(self.sim_box.isChecked() or bool(self.config.get("simulate")))
            self.thr_spin.setValue(float(self.config["trigger"]["threshold_mv"]))
            self.batch_spin.setValue(int(self.config["output"]["rapid_block_captures"]))
            self.timeout_spin.setValue(float(self.config["capture"].get("batch_timeout_s") or 0))
            self.dir_edit.setText(str(self.config["output"]["data_dir"]))
            self.base_edit.setText(str(self.config["output"]["base"]))
            self.setWindowTitle(f"PicoScope QL — {self.cfg_edit.text()}")
        self.redraw()

    # ------------------------------------------------------ mode / locks
    def _update_controls(self):
        mode = "run" if self.mode_box.currentIndex() == 1 else "tune"
        idle = not self.running
        locked = self.running and self.run_mode == "run"
        self.mode_box.setEnabled(idle)
        self.cfg_edit.setEnabled(idle)
        self.browse_btn.setEnabled(idle)
        self.sim_box.setEnabled(idle)
        self.dur_spin.setEnabled(idle)
        for w in (self.dir_edit, self.dir_btn, self.base_edit):
            w.setEnabled(idle)
        if mode == "run":
            self.save_box.setChecked(True)
        self.save_box.setEnabled(idle and mode == "tune")
        for w in (self.thr_spin, self.batch_spin, self.timeout_spin):
            w.setEnabled(not locked)
        self.apply_btn.setEnabled(self.running and self.run_mode == "tune")
        self.start_btn.setEnabled(idle)
        self.stop_btn.setEnabled(self.running and not self.stop_event.is_set()
                                 if self.stop_event is not None else False)
        self.lock_label.setText("🔒 locked (Run)" if locked else "")
        self._update_file_label()

    # ------------------------------------------------------- run control
    def _reset_run_state(self):
        self.times = None
        self.waves = {"anode": deque(maxlen=N_WAVE_DRAW), "cathode": deque(maxlen=N_WAVE_DRAW)}
        self.rate_t = deque(maxlen=MAX_RATE_POINTS)
        self.rate_in = deque(maxlen=MAX_RATE_POINTS)
        self.rate_acc = deque(maxlen=MAX_RATE_POINTS)
        self.total = 0
        self.file = "-"
        self.file_count = 0
        self.save_path = None
        self.saving = False
        self.last = None  # last rate stats
        self.threshold_now = None
        self.trigger_info = None
        self.run_t0 = time.time()
        for m in self.marks:
            self.p_rate.removeItem(m)
        self.marks = []

    def start(self):
        if self.running:
            return
        try:
            self.load_config()
        except Exception as e:
            self.log(f"config error: {e}")
            return
        mode = self._gui_mode()
        save = True if mode == "run" else self.save_box.isChecked()
        try:
            folder, base = self._output_settings()
        except ValueError as e:
            self.log(f"cannot start: {e}")
            return
        config = copy.deepcopy(self.config)
        config["output"]["data_dir"] = folder
        config["output"]["base"] = base
        config["simulate"] = self.sim_box.isChecked()
        config["trigger"]["threshold_mv"] = float(self.thr_spin.value())
        config["output"]["rapid_block_captures"] = int(self.batch_spin.value())
        config["capture"]["batch_timeout_s"] = float(self.timeout_spin.value()) or None
        self._reset_run_state()
        self.saving = save
        self.hists.reset()
        self.queue = self.ctx.Queue(maxsize=64)
        self.cmd_queue = self.ctx.Queue()
        self.stop_event = self.ctx.Event()
        duration = self.dur_spin.value() or None
        self.proc = self.ctx.Process(
            target=worker_main,
            args=(config, duration, self.stop_event, self.queue, self.cmd_queue, mode, save),
        )
        self.proc.start()
        self.running = True
        self.run_mode = mode
        self._update_controls()
        self.log(f"{mode.upper()} started ({'SIM' if config['simulate'] else 'scope'}), "
                 f"thr {config['trigger']['threshold_mv']:g} mV, "
                 f"{'saving to ' + str(data_dir(config)) if save else 'not saving'}")

    def apply_settings(self):
        if not (self.running and self.run_mode == "tune"):
            return
        cmd = {"threshold_mv": float(self.thr_spin.value()),
               "rapid_block_captures": int(self.batch_spin.value()),
               "batch_timeout_s": float(self.timeout_spin.value())}
        self.cmd_queue.put(cmd)

    def stop(self):
        if self.running and self.stop_event is not None:
            self.stop_event.set()
            self._update_controls()
            self.log("stopping…")

    def reset_ql(self):
        self.hists.reset()
        for d in self.waves.values():
            d.clear()
        self.redraw()

    def _finish(self):
        self._drain_and_join(timeout=10)
        self.running = False
        self._update_controls()
        self.log(f"stopped: {self.total} waveforms")
        self.update_status()

    def _drain_and_join(self, timeout):
        # a process that has put data on a Queue does not exit until it is drained
        if self.proc is None:
            return
        deadline = time.time() + timeout
        while self.proc.is_alive() and time.time() < deadline:
            self._drain()
            self.proc.join(0.1)
        if self.proc.is_alive():
            self.log("worker did not exit, terminating")
            self.proc.terminate()
            self.proc.join(2)
        self._drain()
        self.proc = None

    def _drain(self):
        changed = False
        if self.queue is None:
            return changed
        for _ in range(500):
            try:
                msg = self.queue.get_nowait()
            except (queue.Empty, OSError, ValueError):
                break
            changed |= self.handle(msg)
        return changed

    # ---------------------------------------------------------- polling
    def poll(self):
        if self.queue is None:
            return
        if self._drain():
            self.redraw()
        if self.running and self.proc is not None and not self.proc.is_alive():
            self._finish()
        if self.running:
            self.update_status()

    def handle(self, msg):
        kind = msg.get("type")
        if kind == "info":
            self.times = msg["times_s"]
            self.trigger_info = msg.get("trigger_info")
            self.log(f"scope serial {msg['serial']}, trigger info "
                     f"{'available' if self.trigger_info else 'NOT available'}")
            return False
        if kind == "batch":
            thr = msg.get("threshold_mv")
            t_rel = msg["t1"] - self.run_t0
            if self.threshold_now is not None and thr != self.threshold_now:
                line = pg.InfiniteLine(pos=t_rel, angle=90, pen=pg.mkPen((255, 120, 120), style=QtCore.Qt.DashLine),
                                       label=f"{thr:g} mV", labelOpts={"position": 0.9, "color": (255, 120, 120)})
                self.p_rate.addItem(line)
                self.marks.append(line)
            self.threshold_now = thr
            self.total = msg["total"]
            if msg["n"] == 0:
                self.rate_t.append(t_rel)
                self.rate_in.append(0.0)
                self.rate_acc.append(0.0)
                return True
            a = msg["amp"].get(self.anode_ch)
            c = msg["amp"].get(self.cathode_ch)
            if a is not None:
                self.hists.fill(a, c if c is not None else np.zeros_like(a))
            for key, ch in (("anode", self.anode_ch), ("cathode", self.cathode_ch)):
                for w in msg["wave"].get(ch, []):
                    self.waves[key].append(w)
            stats = msg.get("rates")
            if stats is None:  # no trigger info: fall back to wall-clock throughput
                stats = {"input_rate": np.nan,
                         "accepted_rate": msg["n"] / max(msg["t1"] - msg["t0"], 1e-9), "live": np.nan}
            self.last = stats
            self.rate_t.append(t_rel)
            self.rate_in.append(stats["input_rate"])
            self.rate_acc.append(stats["accepted_rate"])
            self.file = msg["file"]
            self.file_count = msg["file_count"]
            self.save_path = msg.get("path")
            self._update_file_label()
            return True
        if kind == "log":
            self.log(msg["text"])
        elif kind == "error":
            self.log("ERROR in acquisition:\n" + msg["text"])
        elif kind == "stopped" and self.running:
            self._finish()
        return False

    # ---------------------------------------------------------- drawing
    def set_logy(self, on):
        for p in self.p_spec.values():
            p.setLogMode(y=on)
        self.redraw()

    def redraw(self):
        if self.times is not None:
            for key, items in self.wave_items.items():
                ws = list(self.waves[key])
                offset = len(items) - len(ws)  # newest goes to the last (highlighted) item
                for i, item in enumerate(items):
                    j = i - offset
                    if 0 <= j < len(ws) and len(ws[j]) == len(self.times):
                        item.setData(self.times, ws[j])
                    else:
                        item.clear()
        logy = self.log_box.isChecked()
        for key, h in (("anode", self.hists.anode), ("cathode", self.hists.cathode)):
            y = np.where(h > 0, h, 0.5) if logy else h
            self.spec_items[key].setData(self.hists.amp_edges, y)
        x0, x1 = self.hists.amp2d_edges[[0, -1]]
        y0, y1 = self.hists.ratio_edges[[0, -1]]
        self.img.setImage(np.log1p(self.hists.h2d), autoLevels=True)
        self.img.setRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
        t = np.asarray(self.rate_t)
        self.rate_in_item.setData(t, np.asarray(self.rate_in), connect="finite")
        self.rate_acc_item.setData(t, np.asarray(self.rate_acc), connect="finite")

    def update_status(self):
        state = f"{self.run_mode.upper()}" if self.running else "idle"
        el = int(time.time() - self.run_t0) if self.running else 0
        self.status.setText(
            f"{state} | {self.file} ({self.file_count} wf) | total {self.total} | "
            f"QL {self.hists.events} | elapsed {el // 3600:02d}:{el // 60 % 60:02d}:{el % 60:02d}"
        )
        s = self.last
        if s is None:
            self.rate_label.setText("")
            return
        thr = f"{self.threshold_now:g} mV" if self.threshold_now is not None else "-"
        live = f"{100 * s['live']:.1f} %" if np.isfinite(s["live"]) else "n/a"
        # average over the last ~10 batches
        n = min(10, len(self.rate_in))
        def mean(x):
            x = np.asarray(x, float)
            return float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else np.nan

        def hz(v):
            return f"{v:.1f} Hz" if np.isfinite(v) else "n/a"

        avg_in = mean(list(self.rate_in)[-n:])
        avg_acc = mean(list(self.rate_acc)[-n:])
        self.rate_label.setText(
            f"threshold {thr} | input {hz(s['input_rate'])} | accepted {hz(s['accepted_rate'])} | "
            f"live {live} | (10-batch avg: in {hz(avg_in)}, acc {hz(avg_acc)})"
        )

    def closeEvent(self, event):
        if self.running:
            self.stop_event.set()
            self._drain_and_join(timeout=15)
        event.accept()


def main(argv=None):
    parser = argparse.ArgumentParser(description="PicoScope quick-look GUI")
    parser.add_argument("--config", default="config_acquisition.yaml")
    parser.add_argument("--sim", action="store_true", help="simulated pulses, no scope")
    args = parser.parse_args(argv)
    app = pg.mkQApp("PicoScope QL")
    win = MainWindow(Path(args.config), simulate=args.sim)
    win.resize(1400, 1000)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    mp.freeze_support()
    main()
