"""Scope backends.

capture(n, stop_event=None, timeout_s=None) returns (data, times, trig) or None:
    data  {channel: (m, samples) float32 [V]}, m <= n
    times (samples,) [s], 0 = trigger
    trig  {"timestamp", "missed", "status"} arrays (m,), or None if unavailable
After each capture, backend.last_overflow is a (m,) uint8 bit mask of the channels
that went over range somewhere in the record (bit i = channel i, A = bit 0),
or None if the driver did not report it.
"""
import ctypes
import time
import warnings

import numpy as np

from .calibration import LINES
from .config import CHANNEL_BITS, enabled_channels
from .timing import TS_MASK, TS_RESET


def overflow_bits(overflow, n):
    """pypicosdk get_values_bulk overflow (per capture: list of channel names, or an
    int bit mask) -> (n,) uint8 bit mask; None if it does not match n captures."""
    if overflow is None or len(overflow) < n:
        return None
    out = np.zeros(n, np.uint8)
    for i, o in enumerate(list(overflow)[:n]):
        if isinstance(o, (int, np.integer)):
            out[i] = int(o) & 0xFF
        else:
            out[i] = sum(1 << CHANNEL_BITS[str(c).upper()] for c in o if str(c).upper() in CHANNEL_BITS)
    return out


def _trig_from_info(info):
    return {
        "timestamp": np.array([int(i["timeStampCounter"]) & TS_MASK for i in info], dtype=np.uint64),
        "missed": np.array([int(i["missedTriggers"]) for i in info], dtype=np.uint64),
        "status": np.array([int(i["status"]) for i in info], dtype=np.int64),
    }


class PicoBackend:
    driver = "psospa"
    """PicoScope 3000E (psospa driver) via pypicosdk.

    capture.loop = "poll"   : RunBlock + own IsReady polling. Stop / timeout read
                              out the segments captured so far (default).
    capture.loop = "simple" : pypicosdk run_simple_rapid_block_capture (blocks until
                              the batch is full; behaviour of run_acqusition.py).
    The poll loop uses pypicosdk internals (_call_attr_function, _adc_to_), checked
    against pypicosdk 1.7.5.
    """

    def __init__(self, config):
        import pypicosdk as psdk

        self.psdk = psdk
        cap = config["capture"]
        self.samples = cap["samples"]
        self.pretrig = round(cap["pre_trigger_fraction"] * 100)
        self.loop = cap.get("loop", "poll")
        self.last_overflow = None
        scope = psdk.psospa()
        resolution = getattr(psdk.RESOLUTION, f"_{config['scope']['resolution']}")
        scope.open_unit(resolution=resolution)
        self.scope = scope
        try:
            self.channels = enabled_channels(config)
            for name in self.channels:
                scope.set_channel(
                    channel=getattr(psdk.CHANNEL, name),
                    range=getattr(psdk.RANGE, config["channels"][name]["range"]),
                )
            self.trigger_cfg = dict(config["trigger"])
            if self.trigger_cfg["enabled"]:
                self.set_threshold(self.trigger_cfg["threshold_mv"])
            rate = cap["sample_rate_hz"] / 1e6
            self.timebase = scope.sample_rate_to_timebase(rate, unit=psdk.SAMPLE_RATE.MSPS)
            self.serial = scope.get_unit_serial()
            self.model = config["scope"].get("model", "3416E MSO")
        except Exception:
            self.close()
            raise

    def set_threshold(self, threshold_mv):
        t = self.trigger_cfg
        self.scope.set_simple_trigger(
            channel=getattr(self.psdk.CHANNEL, t["source"]),
            threshold=int(round(threshold_mv)),
            auto_trigger=t["auto_trigger_ms"] * 1000,
        )
        t["threshold_mv"] = threshold_mv

    def _volts(self, buffers):
        return {
            name: np.asarray(buffers[getattr(self.psdk.CHANNEL, name)], dtype=np.float32)
            for name in self.channels
        }

    def _trigger_info(self, n):
        """Trigger info of segments 0 .. n-1.

        psospaGetTriggerInfo(handle, info*, firstSegmentIndex, segmentCount) takes the
        NUMBER of segments. pypicosdk 1.7.5 get_trigger_info passes the last index
        (first + n - 1) instead, so the last segment of every batch came back empty
        (Timestamp = 0, MissedTriggers = 0). Fixed upstream after 1.7.5 (pyPicoSDK #189).
        The driver is called here the same way pypicosdk calls it, with the count n,
        so this works with either pypicosdk version."""
        scope = self.scope
        try:
            info = (self.psdk.PICO_TRIGGER_INFO * n)()
            scope._call_attr_function("GetTriggerInfo", scope.handle, ctypes.byref(info), 0, n)
        except Exception:
            return None
        rows = [{"status": i.status_, "segmentIndex": i.segmentIndex_,
                 "missedTriggers": i.missedTriggers_, "timeStampCounter": i.timeStampCounter_}
                for i in info]
        seg = [int(r["segmentIndex"]) for r in rows]
        if seg != list(range(n)) and not getattr(self, "_seg_warned", False):
            # an entry the driver did not fill stays all zero (segmentIndex 0)
            self._seg_warned = True
            warnings.warn(f"trigger info: segment indices {seg} are not 0..{n - 1}")
        return _trig_from_info(rows)

    def capture(self, n, stop_event=None, timeout_s=None):
        scope = self.scope
        if self.loop == "simple":
            buffers, times = scope.run_simple_rapid_block_capture(
                timebase=self.timebase, samples=self.samples, captures=n,
                output_unit="v", time_unit="s", pre_trig_percent=self.pretrig,
            )
            self.last_overflow = None  # not reported by run_simple_rapid_block_capture
            return self._volts(buffers), np.asarray(times), self._trigger_info(n)

        # same setup as run_simple_rapid_block_capture, but IsReady is polled here
        scope.memory_segments(n)
        scope.set_no_of_captures(n)
        buffers = scope.set_data_buffer_for_enabled_channels(self.samples, captures=n)
        scope.run_block_capture(self.timebase, self.samples, self.pretrig)
        t_start = time.monotonic()
        ready = ctypes.c_int16(0)
        while True:
            scope._call_attr_function("IsReady", scope.handle, ctypes.byref(ready))
            if ready.value:
                n_done = n
                break
            stop = stop_event is not None and stop_event.is_set()
            late = bool(timeout_s) and time.monotonic() - t_start > timeout_s
            if stop or late:
                scope.stop()
                n_done = int(scope.get_no_of_captures())
                break
            time.sleep(0.002)
        if n_done == 0:
            self.last_overflow = None
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # pypicosdk warns once per over-range capture
            actual, over = scope.get_values_bulk(self.samples, 0, n_done - 1, wait_for_ready=False)
        self.last_overflow = overflow_bits(over, n_done)
        buffers = {ch: arr[:n_done, :actual] for ch, arr in buffers.items()}
        buffers = scope._adc_to_(buffers, unit="v")
        times = scope.get_time_axis(self.timebase, actual, pre_trig_percent=self.pretrig, unit="s")
        return self._volts(buffers), np.asarray(times), self._trigger_info(n_done)

    def stop(self):
        self.scope.stop()

    def close(self):
        try:
            self.scope.close_unit()
        except Exception:
            pass


def csa_response(t, amp, t_col, tau):
    """Charge-sensitive preamp output for a charge amp collected uniformly over t_col,
    decay constant tau. A pole-zero-corrected trapezoid with flat top > t_col
    returns exactly amp."""
    tc = np.clip(t, 0, None)
    inside = (tau / t_col) * (1 - np.exp(-np.minimum(tc, t_col) / tau))
    after = np.exp(-np.clip(tc - t_col, 0, None) / tau)
    return amp * inside * np.where(tc > t_col, after, 1.0)


class SimBackend:
    """Planar TlBr-like pulses from a Poisson source (Cs-137-like spectrum) with
    threshold, dead time (post-trigger + re-arm + pre-trigger fill + readout),
    timestamps and missedTriggers. For testing without hardware."""

    def __init__(self, config):
        self.channels = enabled_channels(config)
        cap = config["capture"]
        self.samples = cap["samples"]
        self.fs = cap["sample_rate_hz"]
        pretrig = int(round(self.samples * cap["pre_trigger_fraction"]))
        self.times = (np.arange(self.samples) - pretrig) / self.fs
        self.pre = pretrig / self.fs
        self.post = (self.samples - pretrig) / self.fs
        self.rearm = 2e-6
        self.raw_rate = float(config.get("sim_rate_hz", 600.0))
        self.threshold = config["trigger"]["threshold_mv"] * 1e-3
        chans = config["processing"]["channels"]
        self.tau = [chans.get(ch, {}).get("tau_us", 100.0) * 1e-6 for ch in self.channels]
        self.gain = 1 / float(config.get("sim_gain_kev_per_v", 3367.0))  # V / keV
        lines = LINES[config.get("sim_source", "Cs137")]
        self.line_e = np.array([l[0] for l in lines])
        self.line_p = np.array([l[1] for l in lines]) / sum(l[1] for l in lines)
        self.rng = np.random.default_rng(config.get("sim_seed"))
        self.serial = "SIMULATED"
        self.model = "simulator"
        self.driver = "sim"
        self.wall0 = time.monotonic()
        self.ts0 = int(self.rng.integers(0, 1 << 40))
        self.t_arr = 0.0
        self.pending = None
        self.dead_until = 0.0
        self.reset_next = True
        self.realtime = bool(config.get("sim_realtime", True))
        self.t_virtual = 0.0
        from .ps6000 import RANGE_MV

        self.range_v = {ch: RANGE_MV.get(config["channels"][ch].get("range", "V2"), 2000) / 1000
                        for ch in self.channels}
        self.last_overflow = None

    def set_threshold(self, threshold_mv):
        self.threshold = threshold_mv * 1e-3
        self.reset_next = True  # real scopes may reset the timestamp on a settings change

    def _draw(self):
        """Energy [keV] and depth (0 = anode side). 35 % full-energy line (chosen by
        emission probability, sigma = sqrt(4^2 + (0.005 E)^2) keV), 40 % Compton
        continuum of a line, 25 % low-energy background."""
        rng = self.rng
        u = rng.random()
        e_line = rng.choice(self.line_e, p=self.line_p)
        if u < 0.35:
            e = rng.normal(e_line, np.hypot(4.0, 0.005 * e_line))
        elif u < 0.75:
            k = 2 * e_line / 511.0
            e = rng.uniform(0, e_line * k / (1 + k))
        else:
            e = rng.exponential(40.0)
        return e, rng.random()

    def _next(self):
        if self.pending is not None:
            a, self.pending = self.pending, None
            return a
        self.t_arr += self.rng.exponential(1 / self.raw_rate)
        e, z = self._draw()
        return self.t_arr, e, z, e * self.gain * (0.6 + 0.4 * z) > self.threshold

    def _now(self):
        return time.monotonic() - self.wall0 if self.realtime else self.t_virtual

    def capture(self, n, stop_event=None, timeout_s=None):
        armed_at = self._now() + self.pre  # pre-trigger buffer must fill first
        limit = armed_at + timeout_s if timeout_s else None
        if not self.realtime and limit is None:
            limit = armed_at + 10.0  # virtual clock: give up after 10 s without triggers
        missed, events, miss_list = 0, [], []

        def consume(upto):
            """Process arrivals up to time `upto` (and `limit`)."""
            nonlocal missed
            while len(events) < n:
                arr = self._next()
                t, trig = arr[0], arr[3]
                if t > upto or (limit is not None and t > limit):
                    self.pending = arr
                    return
                if not trig:
                    continue
                if t < max(armed_at, self.dead_until):
                    missed += 1  # readout gap, post-trigger, re-arm or pre-trigger fill
                    continue
                events.append(arr[:3])
                miss_list.append(missed)
                missed = 0
                self.dead_until = t + self.post + self.rearm + self.pre

        if self.realtime:
            while True:
                now = self._now()
                consume(now)
                if len(events) == n and now >= events[-1][0] + self.post:
                    break
                if (limit is not None and now >= limit) or (
                        stop_event is not None and stop_event.is_set()):
                    keep = [i for i, ev in enumerate(events) if ev[0] + self.post <= now]
                    events = [events[i] for i in keep]
                    miss_list = [miss_list[i] for i in keep]
                    break
                time.sleep(0.005)
        else:
            consume(np.inf)
            end = events[-1][0] + self.post if len(events) == n else limit
            self.t_virtual = max(self.t_virtual, end)
        if not events:
            self.last_overflow = None
            return None
        k = len(events)
        t_ev = np.array([ev[0] for ev in events])
        e = np.array([ev[1] for ev in events])
        z = np.array([ev[2] for ev in events])
        t = self.times[None, :]
        t_col = (2.0 + 8.0 * z)[:, None] * 1e-6  # electron drift time grows with depth
        amp_a = (e * self.gain * (0.6 + 0.4 * z))[:, None]
        amp_c = (e * self.gain * 1.3 * z)[:, None]
        sig = [csa_response(t, amp_a, t_col, self.tau[0]),
               -csa_response(t, amp_c, t_col, self.tau[min(1, len(self.tau) - 1)])]
        data = {}
        over = np.zeros(k, np.uint8)
        for i, ch in enumerate(self.channels):
            s = sig[i] if i < 2 else 0.0
            v = (s + self.rng.normal(0, 4e-3, (k, self.samples))).astype(np.float32)
            r = np.float32(self.range_v[ch])
            over |= (np.abs(v).max(axis=1) >= r).astype(np.uint8) << CHANNEL_BITS[ch]
            data[ch] = np.clip(v, -r, r)  # the ADC saturates at the range
        self.last_overflow = over
        status = np.zeros(k, np.int64)
        if self.reset_next:
            status[0] = TS_RESET
            self.reset_next = False
        trig = {
            "timestamp": ((np.round(t_ev * self.fs).astype(np.int64) + self.ts0) & TS_MASK).astype(np.uint64),
            "missed": np.array(miss_list, np.uint64),
            "status": status,
        }
        return data, self.times.copy(), trig

    def stop(self):
        pass

    def close(self):
        pass


def make_backend(config):
    """simulate: true -> SimBackend; scope.driver: ps6000 -> Ps6000Backend (6402C etc.);
    otherwise PicoBackend (pypicosdk psospa, 3000E)."""
    if config.get("simulate"):
        return SimBackend(config)
    driver = (config.get("scope") or {}).get("driver", "psospa")
    if driver == "ps6000":
        from .ps6000 import Ps6000Backend

        return Ps6000Backend(config)
    if driver != "psospa":
        raise ValueError(f"scope.driver {driver!r}: use psospa (3000E) or ps6000 (6402C etc.)")
    return PicoBackend(config)
