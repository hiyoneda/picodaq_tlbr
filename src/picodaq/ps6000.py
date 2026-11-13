"""PicoScope 6000 series (6402C/D, 6403C/D, 6404C/D, 6407; legacy "ps6000" API).

pypicosdk does not cover this driver, so it is called directly with ctypes.
Prototypes and constants follow Pico's picosdk-python-wrappers (picosdk/ps6000.py)
and the PicoScope 6000 Series Programmer's Guide.

Select it with  scope: {driver: ps6000}  in the acquisition config.

Differences from the 3000E (psospa) backend
- 8-bit ADC (values are returned scaled to +-32512).
- ps6000GetTriggerInfoBulk gives per-segment timestamps, but there is no
  missedTriggers count: input rate and live fraction cannot be measured
  (trig["missed"] is None; the GUI shows the accepted rate only).
- Optional per-channel analogue offset and bandwidth limiter.
"""
import ctypes
import platform
import time
import warnings
from ctypes import POINTER, Structure, byref, c_float, c_int16, c_int32, c_uint16, c_uint32, c_uint64, c_void_p
from pathlib import Path

import numpy as np

from .config import enabled_channels
from .timing import TS_MASK

MAX_ADC = 32512
CHANNELS = {"A": 0, "B": 1, "C": 2, "D": 3}
RANGE_MV = {"mV10": 10, "mV20": 20, "mV50": 50, "mV100": 100, "mV200": 200, "mV500": 500,
            "V1": 1000, "V2": 2000, "V5": 5000, "V10": 10000, "V20": 20000, "V50": 50000}
RANGE_INDEX = {name: i for i, name in enumerate(RANGE_MV)}  # PS6000_RANGE
COUPLING = {"AC": 0, "DC": 1, "DC_1M": 1, "DC_50R": 2}  # PS6000_COUPLING
BANDWIDTH = {"full": 0, "20MHz": 1, "25MHz": 2}  # PS6000_BANDWIDTH_LIMITER
DIRECTION = {"above": 0, "below": 1, "rising": 2, "falling": 3}  # PS6000_THRESHOLD_DIRECTION
RATIO_NONE = 0
PICO_VARIANT_INFO, PICO_BATCH_AND_SERIAL = 3, 4

PICO_OK = 0x0
PICO_USB3_0_DEVICE_NON_USB3_0_PORT = 0x11E  # works on USB 2, only slower
STATUS_NAMES = {
    0x03: "PICO_NOT_FOUND (no PicoScope 6000 found: USB cable, power supply, other program?)",
    0x04: "PICO_FW_FAIL", 0x05: "PICO_OPEN_OPERATION_IN_PROGRESS", 0x07: "PICO_NOT_RESPONDING",
    0x0C: "PICO_INVALID_HANDLE", 0x0D: "PICO_INVALID_PARAMETER", 0x0E: "PICO_INVALID_TIMEBASE",
    0x0F: "PICO_INVALID_VOLTAGE_RANGE", 0x10: "PICO_INVALID_CHANNEL",
    0x11: "PICO_INVALID_TRIGGER_CHANNEL", 0x18: "PICO_DATA_NOT_AVAILABLE",
    0x1D: "PICO_TOO_MANY_SAMPLES", 0x1E: "PICO_TOO_MANY_SEGMENTS",
    0x25: "PICO_NO_SAMPLES_AVAILABLE", 0x26: "PICO_SEGMENT_OUT_OF_RANGE", 0x27: "PICO_BUSY",
    0x2D: "PICO_MEMORY", 0x37: "PICO_INVALID_BUFFER", 0x41: "PICO_INVALID_STATE",
    0x42: "PICO_NOT_ENOUGH_SEGMENTS", 0x43: "PICO_DRIVER_FUNCTION", 0x45: "PICO_INVALID_COUPLING",
    0x50: "PICO_INVALID_ANALOGUE_OFFSET", 0x10E: "PICO_INVALID_BANDWIDTH",
}


class PS6000_TRIGGER_INFO(Structure):
    _pack_ = 1
    _fields_ = [("status", c_uint32), ("segmentIndex", c_uint32), ("triggerIndex", c_uint32),
                ("triggerTime", c_uint64), ("timeUnits", c_uint16), ("reserved0", c_uint16),
                ("timeStampCounter", c_uint64)]


# name: (restype, argtypes); pointers are c_void_p (byref objects / addresses)
PROTOTYPES = {
    "ps6000OpenUnit": (c_uint32, [c_void_p, c_void_p]),
    "ps6000CloseUnit": (c_uint32, [c_int16]),
    "ps6000GetUnitInfo": (c_uint32, [c_int16, c_void_p, c_int16, c_void_p, c_uint32]),
    "ps6000SetChannel": (c_uint32, [c_int16, c_int32, c_int16, c_int32, c_int32, c_float, c_int32]),
    "ps6000GetAnalogueOffset": (c_uint32, [c_int16, c_int32, c_int32, c_void_p, c_void_p]),
    "ps6000GetTimebase2": (c_uint32, [c_int16, c_uint32, c_uint32, c_void_p, c_int16, c_void_p, c_uint32]),
    "ps6000SetSimpleTrigger": (c_uint32, [c_int16, c_int16, c_int32, c_int16, c_int32, c_uint32, c_int16]),
    "ps6000MemorySegments": (c_uint32, [c_int16, c_uint32, c_void_p]),
    "ps6000SetNoOfCaptures": (c_uint32, [c_int16, c_uint32]),
    "ps6000SetDataBufferBulk": (c_uint32, [c_int16, c_int32, c_void_p, c_uint32, c_uint32, c_int32]),
    "ps6000RunBlock": (c_uint32, [c_int16, c_uint32, c_uint32, c_uint32, c_int16, c_void_p, c_uint32,
                                  c_void_p, c_void_p]),
    "ps6000IsReady": (c_uint32, [c_int16, c_void_p]),
    "ps6000Stop": (c_uint32, [c_int16]),
    "ps6000GetNoOfCaptures": (c_uint32, [c_int16, c_void_p]),
    "ps6000GetValuesBulk": (c_uint32, [c_int16, c_void_p, c_uint32, c_uint32, c_uint32, c_int32, c_void_p]),
    # the official wrapper declares c_uint16 here; PICO_STATUS is 32 bit
    "ps6000GetTriggerInfoBulk": (c_uint32, [c_int16, c_void_p, c_uint32, c_uint32]),
}


class Ps6000Error(RuntimeError):
    pass


def _check(status, what):
    status = int(status)
    if status == PICO_OK:
        return
    name = STATUS_NAMES.get(status, "see PicoStatus.h")
    raise Ps6000Error(f"{what}: status 0x{status:X} {name}")


def library_candidates(sdk_path=None, name="ps6000"):
    system = platform.system()
    fname = {"Windows": f"{name}.dll", "Darwin": f"lib{name}.dylib"}.get(system, f"lib{name}.so")
    dirs = []
    if sdk_path:
        p = Path(sdk_path)
        dirs += [p, p / f"lib{name}", p / "lib", p / "Libraries" / f"lib{name}"]
    if system == "Darwin":
        fw = Path("/Library/Frameworks/PicoSDK.framework/Libraries")
        dirs += [fw / f"lib{name}", fw]
    elif system == "Windows":
        dirs += [Path("C:/Program Files/Pico Technology/SDK/lib")]
    else:
        dirs += [Path("/opt/picoscope/lib")]
    return [d / fname for d in dirs]


def load_library(sdk_path=None):
    """Load libps6000 from sdk_path or the standard PicoSDK locations."""
    tried = []
    for path in library_candidates(sdk_path):
        tried.append(str(path))
        if path.exists():
            return ctypes.CDLL(str(path))
    from ctypes.util import find_library

    found = find_library("ps6000")
    if found:
        return ctypes.CDLL(found)
    raise FileNotFoundError("libps6000 not found (is PicoSDK with the PicoScope 6000 driver installed?). "
                            "Tried: " + ", ".join(tried) + ". Set scope.sdk_path in the config.")


def timebase_for_rate(rate_hz):
    """ps6000 timebase: n < 5 -> 2^n / 5 GHz, n >= 5 -> (n - 4) / 156.25 MHz."""
    if rate_hz > 156.25e6:
        return int(np.clip(round(np.log2(5e9 / rate_hz)), 0, 4))
    return max(5, int(round(156.25e6 / rate_hz)) + 4)


def _prepare(lib):
    for name, (res, args) in PROTOTYPES.items():
        f = getattr(lib, name, None)
        if f is not None and hasattr(f, "argtypes"):  # a real CDLL function (fakes in the tests are plain callables)
            f.restype, f.argtypes = res, args


class Ps6000Backend:
    """Same interface as backends.PicoBackend (capture / set_threshold / stop / close)."""

    driver = "ps6000"

    def __init__(self, config, lib=None):
        scope_cfg = config.get("scope", {}) or {}
        self.lib = lib if lib is not None else load_library(scope_cfg.get("sdk_path"))
        _prepare(self.lib)
        cap = config["capture"]
        if cap.get("loop", "poll") != "poll":
            warnings.warn("capture.loop is ignored for the ps6000 driver (always poll)")
        self.samples = int(cap["samples"])
        self.pre = int(round(self.samples * cap["pre_trigger_fraction"]))
        self.post = self.samples - self.pre

        h = c_int16()
        st = self.lib.ps6000OpenUnit(byref(h), None)
        if int(st) == PICO_USB3_0_DEVICE_NON_USB3_0_PORT:
            warnings.warn("PicoScope 6000 is on a USB 2 port: works, but readout is slower")
        else:
            _check(st, "ps6000OpenUnit")
        self.handle = h.value
        try:
            self.serial = self._info(PICO_BATCH_AND_SERIAL)
            self.model = self._info(PICO_VARIANT_INFO) or "6000"
            self.channels = enabled_channels(config)
            self.range_v, self.offset_v = {}, {}
            for name, idx in CHANNELS.items():
                ch = config["channels"].get(name, {}) or {}
                on = bool(ch.get("enabled")) and name in self.channels
                rng = ch.get("range", "V2")
                if rng not in RANGE_INDEX:
                    raise ValueError(f"channel {name}: range {rng!r}; use one of {list(RANGE_INDEX)}")
                cpl = ch.get("coupling", "DC_1M")
                bw = ch.get("bandwidth", "full")
                off = float(ch.get("analog_offset_v", 0.0)) if on else 0.0
                if off:
                    self._check_offset(name, off, COUPLING[cpl], RANGE_INDEX[rng])
                _check(self.lib.ps6000SetChannel(self.handle, idx, int(on), COUPLING[cpl], RANGE_INDEX[rng],
                                                 c_float(off), BANDWIDTH[bw]), f"ps6000SetChannel {name}")
                if on:
                    self.range_v[name] = RANGE_MV[rng] / 1000.0
                    self.offset_v[name] = off
            self.trigger_cfg = dict(config["trigger"])
            if self.trigger_cfg["enabled"]:
                self.set_threshold(self.trigger_cfg["threshold_mv"])
            self.timebase = timebase_for_rate(cap["sample_rate_hz"])
            dt_ns, max_samples = c_float(), c_uint32()
            _check(self.lib.ps6000GetTimebase2(self.handle, self.timebase, self.samples, byref(dt_ns), 1,
                                               byref(max_samples), 0), "ps6000GetTimebase2")
            self.dt = dt_ns.value * 1e-9
            self.times = (np.arange(self.samples) - self.pre) * self.dt
            self.trigger_info_ok = True
        except Exception:
            self.close()
            raise

    # ------------------------------------------------------------ helpers
    def _info(self, what):
        buf, req = ctypes.create_string_buffer(64), c_int16()
        if int(self.lib.ps6000GetUnitInfo(self.handle, byref(buf), 64, byref(req), what)) != PICO_OK:
            return ""
        return buf.value.decode(errors="replace")

    def _check_offset(self, name, off, coupling, rng):
        vmax, vmin = c_float(), c_float()
        if int(self.lib.ps6000GetAnalogueOffset(self.handle, rng, coupling, byref(vmax), byref(vmin))) != PICO_OK:
            return  # let SetChannel decide
        if not vmin.value <= off <= vmax.value:
            raise ValueError(f"channel {name}: analog_offset_v {off} outside "
                             f"[{vmin.value:g}, {vmax.value:g}] V for this range")

    def threshold_counts(self, threshold_mv, channel=None):
        ch = channel or self.trigger_cfg["source"]
        counts = (threshold_mv * 1e-3 + self.offset_v[ch]) / self.range_v[ch] * MAX_ADC
        if abs(counts) > MAX_ADC:
            warnings.warn(f"threshold {threshold_mv} mV is outside the range of channel {ch}; clipped")
        return int(np.clip(round(counts), -MAX_ADC, MAX_ADC))

    def set_threshold(self, threshold_mv):
        t = self.trigger_cfg
        src = t["source"]
        if src not in self.range_v:
            raise ValueError(f"trigger source {src} is not an enabled channel")
        auto_ms = int(min(max(t.get("auto_trigger_ms", 0), 0), 32767))
        _check(self.lib.ps6000SetSimpleTrigger(self.handle, 1, CHANNELS[src], self.threshold_counts(threshold_mv),
                                               DIRECTION[t.get("direction", "rising")], 0, auto_ms),
               "ps6000SetSimpleTrigger")
        t["threshold_mv"] = threshold_mv

    def _volts(self, raw, ch):
        return (raw.astype(np.float32) * np.float32(self.range_v[ch] / MAX_ADC)
                - np.float32(self.offset_v[ch]))

    def _trigger_info(self, n):
        if not self.trigger_info_ok:
            return None
        info = (PS6000_TRIGGER_INFO * n)()
        st = int(self.lib.ps6000GetTriggerInfoBulk(self.handle, byref(info), 0, n - 1))
        if st != PICO_OK:
            self.trigger_info_ok = False  # older driver / not supported: stop asking
            warnings.warn(f"ps6000GetTriggerInfoBulk returned 0x{st:X}: no timestamps")
            return None
        return {
            "timestamp": np.array([info[i].timeStampCounter & TS_MASK for i in range(n)], dtype=np.uint64),
            "missed": None,  # not provided by the ps6000 API
            "status": np.array([info[i].status for i in range(n)], dtype=np.int64),
        }

    # ------------------------------------------------------------ capture
    def capture(self, n, stop_event=None, timeout_s=None):
        lib, h = self.lib, self.handle
        max_samples = c_uint32()
        _check(lib.ps6000MemorySegments(h, n, byref(max_samples)), "ps6000MemorySegments")
        if self.samples > max_samples.value:
            raise Ps6000Error(f"{self.samples} samples per segment > {max_samples.value} with {n} segments; "
                              "use fewer rapid_block_captures")
        _check(lib.ps6000SetNoOfCaptures(h, n), "ps6000SetNoOfCaptures")
        bufs = {ch: np.zeros((n, self.samples), np.int16) for ch in self.channels}
        for ch, arr in bufs.items():
            for i in range(n):
                _check(lib.ps6000SetDataBufferBulk(h, CHANNELS[ch], arr[i].ctypes.data, self.samples, i, RATIO_NONE),
                       "ps6000SetDataBufferBulk")
        _check(lib.ps6000RunBlock(h, self.pre, self.post, self.timebase, 1, None, 0, None, None), "ps6000RunBlock")
        t0 = time.monotonic()
        ready = c_int16(0)
        while True:
            _check(lib.ps6000IsReady(h, byref(ready)), "ps6000IsReady")
            if ready.value:
                n_done = n
                break
            stop = stop_event is not None and stop_event.is_set()
            late = bool(timeout_s) and time.monotonic() - t0 > timeout_s
            if stop or late:
                _check(lib.ps6000Stop(h), "ps6000Stop")
                ncap = c_uint32()
                _check(lib.ps6000GetNoOfCaptures(h, byref(ncap)), "ps6000GetNoOfCaptures")
                n_done = int(ncap.value)
                break
            time.sleep(0.002)
        if n_done == 0:
            return None
        nsamp = c_uint32(self.samples)
        overflow = np.zeros(n_done, np.int16)
        _check(lib.ps6000GetValuesBulk(h, byref(nsamp), 0, n_done - 1, 1, RATIO_NONE, overflow.ctypes.data),
               "ps6000GetValuesBulk")
        m = int(nsamp.value)
        data = {ch: self._volts(arr[:n_done, :m], ch) for ch, arr in bufs.items()}
        return data, self.times[:m].copy(), self._trigger_info(n_done)

    def stop(self):
        self.lib.ps6000Stop(self.handle)

    def close(self):
        try:
            self.lib.ps6000CloseUnit(self.handle)
        except Exception:
            pass
