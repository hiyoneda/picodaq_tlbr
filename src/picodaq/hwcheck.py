"""Hardware test run: checks the acquisition loop and the trigger info on the real scope.

    picodaq-check --config config_acquisition.yaml [--pulser-hz 1000] [--save]
    picodaq-check --sim            # dry run without hardware

Best with a pulser (known rate) or a source on channel A. Nothing is written
unless --save is given (then hwcheck_YYYYmmdd_HHMMSS.h5 in output.data_dir).
"""
import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np

from .acquisition import data_dir
from .backends import make_backend
from .config import load_config
from .io import WaveformWriter
from .timing import RateMeter, is_reset, livetime_summary, ts_diff

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"


class Report:
    def __init__(self):
        self.rows = []

    def add(self, name, status, detail):
        self.rows.append({"check": name, "status": status, "detail": detail})
        print(f"[{status:4s}] {name}: {detail}", flush=True)


def _concat(trigs):
    trigs = [t for t in trigs if t is not None]
    if not trigs:
        return None
    return {k: None if any(t[k] is None for t in trigs) else np.concatenate([t[k] for t in trigs])
            for k in trigs[0]}


def _auto_high_threshold(cfg):
    """95 % of the trigger channel's range (the 6402C may run at +-500 mV)."""
    from .ps6000 import RANGE_MV

    rng = cfg["channels"][cfg["trigger"]["source"]].get("range", "V2")
    return 0.95 * RANGE_MV.get(rng, 2000)


def run_check(config, pulser_hz=None, batch=32, n_batches=10, high_threshold_mv=None,
              save=False):
    rep = Report()
    cfg = load_config(config)
    samples = cfg["capture"]["samples"]
    record_s = samples / cfg["capture"]["sample_rate_hz"]
    thr0 = float(cfg["trigger"]["threshold_mv"])
    if high_threshold_mv is None:
        high_threshold_mv = _auto_high_threshold(cfg)

    # 1. open
    t = time.monotonic()
    be = make_backend(cfg)
    rep.add("open", PASS, f"model {getattr(be, 'model', '?')}, serial {be.serial}, "
            f"driver {getattr(be, 'driver', '?')}, channels {be.channels}, "
            f"loop={getattr(be, 'loop', '-')}, {time.monotonic() - t:.1f} s")
    writer = None
    try:
        # 2. one full batch
        t0 = time.time()
        r = be.capture(batch, timeout_s=10.0)
        t1 = time.time()
        if r is None:
            rep.add("first batch", FAIL, "no trigger within 10 s - check signal / threshold")
            return rep
        data, times, trig = r
        n = next(iter(data.values())).shape[0]
        dt = float(times[1] - times[0])
        rep.add("first batch", PASS if n == batch else WARN,
                f"{n}/{batch} waveforms in {t1 - t0:.2f} s, dt = {dt * 1e9:.2f} ns, "
                f"samples = {times.size}")
        over = getattr(be, "last_overflow", None)
        if over is None:
            rep.add("over range", INFO, "対象外: the driver did not report over-range (simple loop?)")
        else:
            from .config import CHANNEL_BITS

            k = {ch: int(((over >> CHANNEL_BITS[ch]) & 1).sum()) for ch in data}
            rep.add("over range", WARN if any(k.values()) else PASS,
                    ", ".join(f"{ch} {v}/{n}" for ch, v in k.items())
                    + (" - pulses exceed the range: widen it or lower the gain" if any(k.values()) else ""))
        for ch, wf in data.items():
            bad = not np.all(np.isfinite(wf))
            rep.add(f"data {ch}", FAIL if bad else PASS,
                    f"shape {wf.shape}, min {wf.min() * 1e3:.1f} mV, max {wf.max() * 1e3:.1f} mV"
                    + (" (NaN/inf!)" if bad else ""))
        if save:
            path = data_dir(cfg) / time.strftime("hwcheck_%Y%m%d_%H%M%S.h5")
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = WaveformWriter(path, be.channels, samples, times, cfg, be.serial, mode="hwcheck",
                                    model=getattr(be, "model", "3416E MSO"))
            writer.append(data, trig, t0, t1, thr0)
        has_missed = trig is not None and trig.get("missed") is not None
        if trig is None:
            rep.add("trigger info", FAIL, "trigger info failed - no timestamps / missedTriggers")
        else:
            st = trig["status"]
            miss = (f"missed in batch {int(trig['missed'][1:].sum())}" if has_missed
                    else "missedTriggers: not provided by this driver (対象外)")
            rep.add("trigger info", PASS, f"status flags {sorted(set(int(s) for s in st))}, {miss}")
            if not has_missed:
                rep.add("missedTriggers", INFO, "対象外 (ps6000): input rate / live fraction cannot be "
                        "measured; accepted rate from timestamps only")
            if n >= 2:
                d = ts_diff(trig["timestamp"][1:], trig["timestamp"][:-1]) * dt
                rep.add("event spacing", INFO,
                        f"min {d.min() * 1e6:.1f} us, median {np.median(d) * 1e6:.1f} us "
                        f"(record {record_s * 1e6:.1f} us; min >= record means the pre-trigger "
                        f"part must refill before the next trigger)")

        # 3. continuity between two RunBlock calls
        last_ts, last_end = (int(trig["timestamp"][-1]), t1) if trig is not None else (None, t1)
        t2 = time.time()
        r2 = be.capture(batch, timeout_s=10.0)
        t3 = time.time()
        if r2 is not None and r2[2] is not None and last_ts is not None:
            tr2 = r2[2]
            if save:
                writer.append(r2[0], tr2, t2, t3, thr0)
            reset = bool(is_reset(tr2["status"][:1])[0])
            gap = ts_diff(int(tr2["timestamp"][0]), last_ts) * dt
            ts_span = ts_diff(int(tr2["timestamp"][-1]), int(trig["timestamp"][0])) * dt
            wall_span = t3 - t0
            if reset:
                rep.add("timestamp across batches", WARN,
                        "timestamp reset flag on the first segment of every batch -> "
                        "rates/live time only within batches")
            else:
                ok = 0.5 < ts_span / wall_span <= 1.02
                rep.add("timestamp across batches", PASS if ok else WARN,
                        f"timestamp span {ts_span:.3f} s vs wall clock {wall_span:.3f} s, "
                        f"gap first->second batch {gap * 1e3:.2f} ms (wall readout gap "
                        f"{(t2 - last_end) * 1e3:.1f} ms)")
                if has_missed:
                    m0 = int(tr2["missed"][0])
                    exp = f", expected ~{pulser_hz * gap:.1f} from pulser" if pulser_hz else ""
                    rep.add("missed during readout", INFO,
                            f"missedTriggers of first segment = {m0}{exp}")
                else:
                    rep.add("missed during readout", INFO, "対象外 (no missedTriggers)")

        # 4. rates over several batches
        meter = RateMeter(dt)
        trigs = []
        for _ in range(n_batches):
            a = time.time()
            rr = be.capture(batch, timeout_s=5.0)
            b = time.time()
            if rr is None:
                continue
            trigs.append(rr[2])
            meter.update(rr[2])
            if save:
                writer.append(rr[0], rr[2], a, b, thr0)
        tr = _concat(trigs)
        if tr is not None:
            s = livetime_summary(tr["timestamp"], tr["missed"], tr["status"], dt)
            if has_missed:
                rep.add("rates", INFO,
                        f"real {s['real_time_s']:.2f} s, input {s['input_rate']:.1f} Hz, "
                        f"accepted {s['accepted_rate']:.1f} Hz, live {100 * s['live_fraction']:.1f} %")
            else:
                rep.add("rates", INFO, f"real {s['real_time_s']:.2f} s, "
                        f"accepted {s['accepted_rate']:.1f} Hz; input / live 対象外")
            if pulser_hz:
                # without missedTriggers: accepted rate inside a batch, fine if the pulser
                # period is longer than one record
                key, name = ("input_rate", "input") if has_missed else ("accepted_rate", "accepted")
                dev = s[key] / pulser_hz - 1
                rep.add(f"{name} rate vs pulser", PASS if abs(dev) < 0.02 else WARN,
                        f"{s[key]:.2f} Hz vs {pulser_hz:g} Hz ({100 * dev:+.2f} %)")

        # 5. threshold change -> timestamp reset?
        be.set_threshold(thr0 + 5)
        rr = be.capture(4, timeout_s=5.0)
        if rr is not None and rr[2] is not None:
            rep.add("threshold change", INFO,
                    f"reset flag after set_simple_trigger: {bool(is_reset(rr[2]['status'][:1])[0])}")
        be.set_threshold(thr0)

        # 6. batch timeout with (almost) no triggers
        be.set_threshold(high_threshold_mv)
        a = time.monotonic()
        rr = be.capture(1000, timeout_s=1.0)
        el = time.monotonic() - a
        got = 0 if rr is None else next(iter(rr[0].values())).shape[0]
        rep.add("batch timeout 1 s", PASS if 0.9 < el < 1.6 else FAIL,
                f"returned after {el:.2f} s with {got} waveforms (threshold {high_threshold_mv:g} mV)")

        # 7. Stop during a batch
        ev = threading.Event()
        threading.Timer(0.5, ev.set).start()
        a = time.monotonic()
        be.capture(1000, stop_event=ev, timeout_s=None)
        el = time.monotonic() - a
        rep.add("stop during batch", PASS if el < 0.8 else FAIL, f"stop at 0.5 s -> returned at {el:.2f} s")
        be.set_threshold(thr0)

        # 8. old loop for comparison
        if getattr(be, "loop", None) != "poll":
            rep.add("simple loop", INFO, "対象外 for this driver")
        else:
            be.loop = "simple"
            try:
                a = time.monotonic()
                rr = be.capture(8)
                rep.add("simple loop", PASS,
                        f"8 waveforms in {time.monotonic() - a:.2f} s, shapes "
                        f"{[v.shape for v in rr[0].values()]}, trigger info {rr[2] is not None}")
            except Exception as e:
                rep.add("simple loop", WARN, repr(e))
            finally:
                be.loop = "poll"
    except Exception as e:
        rep.add("exception", FAIL, repr(e))
        raise
    finally:
        if writer is not None:
            writer.close()
            rep.add("saved", INFO, str(writer.path))
        be.close()
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(description="PicoScope hardware test run")
    p.add_argument("--config", default="config_acquisition.yaml")
    p.add_argument("--sim", action="store_true", help="dry run with the simulator")
    p.add_argument("--pulser-hz", type=float, default=None, help="known input rate for comparison")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--batches", type=int, default=10)
    p.add_argument("--high-threshold-mv", type=float, default=None,
                   help="threshold that should give (almost) no triggers (default 95%% of the range)")
    p.add_argument("--save", action="store_true", help="save the test waveforms")
    p.add_argument("--json", default=None, help="write the report to this file")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    if a.sim:
        cfg["simulate"] = True
    rep = run_check(cfg, a.pulser_hz, a.batch, a.batches, a.high_threshold_mv, a.save)
    n_fail = sum(r["status"] == FAIL for r in rep.rows)
    n_warn = sum(r["status"] == WARN for r in rep.rows)
    print(f"\n{len(rep.rows)} checks: {n_fail} FAIL, {n_warn} WARN")
    if a.json:
        Path(a.json).write_text(json.dumps(rep.rows, indent=2, ensure_ascii=False))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
