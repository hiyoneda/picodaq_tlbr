"""Acquisition loop (used by the GUI subprocess and the CLI).

    picodaq-acquire [duration_min] [--config X] [--sim] [--mode run|tune] [--no-save]
"""
import argparse
import multiprocessing as mp
import queue
import time
import traceback
from pathlib import Path

from .backends import make_backend
from .config import load_config
from .io import WaveformWriter
from .shaping import processors_from_config
from .timing import RateMeter

import numpy as np


def output_path(config, file_id, base=None):
    output = config["output"]
    return data_dir(config) / f"{base or output['base']}_{file_id:05d}.h5"


def data_dir(config):
    """output.data_dir with strftime codes expanded, e.g. data/%Y%m%d -> data/20261111."""
    return Path(time.strftime(str(config["output"]["data_dir"])))


def next_file_id(config, file_id, base=None):
    while output_path(config, file_id, base).exists():
        file_id += 1
    return file_id


def captures_per_file(config, channels):
    capture, output = config["capture"], config["output"]
    bytes_per_waveform = capture["samples"] * len(channels) * 4
    return max(1, output["max_file_size_mb"] * 1024**2 // bytes_per_waveform)


class QuickLook:
    """Pulse heights for the GUI, with the same ChannelProcessor as the analysis."""

    def __init__(self, config, channels, times):
        dt = float(times[1] - times[0])
        trig_idx = int(np.argmin(np.abs(times)))
        self.proc = processors_from_config(config["processing"], channels, dt, trig_idx)
        self.n_wave = int(config["ql"].get("waveforms_per_batch", 4))

    def message(self, data):
        return {
            "amp": {ch: self.proc[ch].amplitude(wf).astype(np.float32) for ch, wf in data.items()},
            "wave": {ch: wf[: self.n_wave].copy() for ch, wf in data.items()},
        }


def _send(q, msg, block):
    if q is None:
        return
    try:
        if block:
            q.put(msg, timeout=5)
        else:
            q.put_nowait(msg)
    except queue.Full:
        pass  # GUI is behind: drop QL data, never block acquisition


def run_acquisition(config, duration_min=None, stop_event=None, ql_queue=None,
                    cmd_queue=None, mode="run", save=True, backend=None):
    """mode 'run' : settings locked, normally save=True.
    mode 'tune': cmd_queue may change threshold_mv / rapid_block_captures /
                 batch_timeout_s between batches; files are <base>_tune_NNNNN.h5.
    Returns the number of waveforms acquired."""
    config = load_config(config)
    backend = backend or make_backend(config)
    channels = backend.channels
    capture, output = config["capture"], config["output"]
    file_capacity = captures_per_file(config, channels)
    batch_size = min(output["rapid_block_captures"], file_capacity)
    timeout_s = capture.get("batch_timeout_s")
    threshold = float(config["trigger"]["threshold_mv"])
    target = capture["total_waveforms"]
    base = output["base"] if mode == "run" else f"{output['base']}_tune"
    file_id = next_file_id(config, output["start_id"], base)
    deadline = None if duration_min is None else time.monotonic() + max(1.0, duration_min) * 60.0
    parent = mp.parent_process()
    writer = path = ql = rates = None
    acquired = 0

    def log(text):
        if ql_queue is None:
            print(text, flush=True)
        else:
            _send(ql_queue, {"type": "log", "text": text}, block=True)

    try:
        while (target is None or acquired < target) and (
            deadline is None or time.monotonic() < deadline
        ):
            if stop_event is not None and stop_event.is_set():
                break
            if parent is not None and not parent.is_alive():
                break  # GUI died: stop instead of running unattended

            # settings changes, applied between batches (tune mode only)
            while cmd_queue is not None:
                try:
                    cmd = cmd_queue.get_nowait()
                except queue.Empty:
                    break
                if mode != "tune":
                    log(f"ignored {cmd}: run mode is locked")
                    continue
                if "threshold_mv" in cmd and float(cmd["threshold_mv"]) != threshold:
                    threshold = float(cmd["threshold_mv"])
                    backend.set_threshold(threshold)
                    log(f"threshold -> {threshold:g} mV")
                if "rapid_block_captures" in cmd and int(cmd["rapid_block_captures"]) != batch_size:
                    batch_size = max(1, min(int(cmd["rapid_block_captures"]), file_capacity))
                    log(f"batch size -> {batch_size}")
                if "batch_timeout_s" in cmd and (float(cmd["batch_timeout_s"]) or None) != timeout_s:
                    timeout_s = float(cmd["batch_timeout_s"]) or None
                    log(f"batch timeout -> {timeout_s} s")

            if save and writer is None:
                path = output_path(config, file_id, base)
                path.parent.mkdir(parents=True, exist_ok=True)
            remaining = file_capacity - writer.count if (save and writer is not None) else file_capacity
            count = min(batch_size, remaining, target - acquired if target is not None else batch_size)

            t0 = time.time()
            result = backend.capture(count, stop_event, timeout_s)
            t1 = time.time()
            if result is None:  # nothing triggered within the timeout (or stopped)
                _send(ql_queue, {"type": "batch", "n": 0, "t0": t0, "t1": t1, "total": acquired,
                                 "threshold_mv": threshold, "rates": None}, block=False)
                continue
            data, times, trig = result
            n = next(iter(data.values())).shape[0]

            if ql is None:
                ql = QuickLook(config, channels, times)
                rates = RateMeter(float(times[1] - times[0]))
                _send(ql_queue, {"type": "info", "times_s": times, "channels": channels,
                                 "serial": backend.serial, "trigger_info": trig is not None},
                      block=True)
                if trig is None:
                    log("WARNING: trigger info not available (no timestamps / missed triggers)")
                elif trig.get("missed") is None:
                    log("note: this scope gives no missedTriggers -> accepted rate only "
                        "(input rate / live fraction n/a)")
            stats = rates.update(trig)

            if save:
                if writer is None:
                    run_cfg = dict(config)
                    run_cfg["trigger"] = dict(config["trigger"], threshold_mv=threshold)
                    writer = WaveformWriter(path, channels, capture["samples"], times, run_cfg,
                                            backend.serial, mode,
                                            model=getattr(backend, "model", "3416E MSO"))
                    log(f"opened {path.name}")
                writer.append(data, trig, t0, t1, threshold)
            acquired += n

            if ql_queue is not None:
                msg = {"type": "batch", "n": n, "t0": t0, "t1": t1, "total": acquired,
                       "file": path.name if save else "(not saved)",
                       "path": str(path.resolve()) if save else None,
                       "file_count": writer.count if writer is not None else 0,
                       "threshold_mv": threshold, "rates": stats}
                msg.update(ql.message(data))
                _send(ql_queue, msg, block=False)
            else:
                if not stats:
                    r = ""
                elif np.isfinite(stats["input_rate"]):
                    r = f", input {stats['input_rate']:.1f} Hz, live {100 * stats['live']:.1f} %"
                else:
                    r = f", accepted {stats['accepted_rate']:.1f} Hz"
                where = f"{path.name}: {writer.count}" if save else f"(not saved): {acquired}"
                print(f"{where} waveforms{r}", flush=True)

            if save and writer.count >= file_capacity:
                writer.close()
                writer = None
                log(f"closed {path.name}")
                file_id = next_file_id(config, file_id + 1, base)
    except KeyboardInterrupt:
        backend.stop()
        print("stopped")
    finally:
        if writer is not None:
            writer.close()
            log(f"closed {path.name}")
        backend.close()
    return acquired


def worker_main(config, duration_min, stop_event, ql_queue, cmd_queue=None, mode="run", save=True):
    """Entry point of the GUI subprocess."""
    try:
        n = run_acquisition(config, duration_min, stop_event, ql_queue, cmd_queue, mode, save)
        _send(ql_queue, {"type": "log", "text": f"run finished: {n} waveforms"}, block=True)
    except Exception:
        _send(ql_queue, {"type": "error", "text": traceback.format_exc()}, block=True)
    finally:
        _send(ql_queue, {"type": "stopped"}, block=True)


def main(argv=None):
    p = argparse.ArgumentParser(description="PicoScope rapid-block acquisition (no GUI)")
    p.add_argument("duration", nargs="?", type=float, help="acquisition time in minutes (min. 1)")
    p.add_argument("--config", default="config_acquisition.yaml")
    p.add_argument("--sim", action="store_true", help="simulated pulses, no scope")
    p.add_argument("--mode", choices=["run", "tune"], default="run")
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args(argv)
    config = load_config(a.config)
    if a.sim:
        config["simulate"] = True
    run_acquisition(config, a.duration, mode=a.mode, save=not a.no_save)


if __name__ == "__main__":
    main()
