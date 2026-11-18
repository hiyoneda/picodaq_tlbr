"""End-to-end acquisition with the simulator (no hardware, simulated clock)."""
import multiprocessing as mp
import queue

import numpy as np
import pytest

from picodaq.acquisition import captures_per_file, run_acquisition
from picodaq.io import WaveformReader, run_files
from picodaq.timing import TS_RESET


def test_run_writes_files_with_rollover(config):
    config["capture"]["total_waveforms"] = 300
    config["output"]["max_file_size_mb"] = 4  # -> 104 waveforms per file
    cap = captures_per_file(config, ["A", "B"])
    n = run_acquisition(config, mode="run")
    assert n == 300
    files = run_files(config["output"]["data_dir"])
    assert len(files) == -(-300 // cap)
    total, thr = 0, set()
    for f in files:
        with WaveformReader(f) as r:
            total += r.n_events
            assert r.run_mode == "run"
            assert r.has_trigger_info
            thr |= set(r.batch_info()["threshold_mv"])
    assert total == 300 and thr == {30.0}
    with WaveformReader(files[0]) as r:
        s = r.rate_summary()
        assert 0 < s["live_fraction"] <= 1 and s["input_rate"] > s["accepted_rate"]
        assert r.trigger_info()["status"][0] & TS_RESET


def test_tune_mode_applies_commands_and_run_mode_refuses(config):
    config["capture"]["total_waveforms"] = 200
    for mode, expect in (("tune", 150.0), ("run", 30.0)):
        cmd, ql = queue.Queue(), queue.Queue(maxsize=1000)
        cmd.put({"threshold_mv": 150, "rapid_block_captures": 16, "batch_timeout_s": 1.0})
        run_acquisition(dict(config), ql_queue=ql, cmd_queue=cmd, mode=mode, save=True)
        msgs = []
        while not ql.empty():
            msgs.append(ql.get())
        batches = [m for m in msgs if m["type"] == "batch" and m["n"]]
        logs = [m["text"] for m in msgs if m["type"] == "log"]
        assert batches[-1]["threshold_mv"] == expect
        if mode == "tune":
            assert {m["n"] for m in batches[1:-1]} == {16}  # last one is cut at total_waveforms
            assert any("threshold -> 150" in t for t in logs)
            assert all(f.name.startswith("tlbr_tune_") for f in
                       sorted(__import__("pathlib").Path(config["output"]["data_dir"]).glob("*tune*")))
        else:
            assert any("run mode is locked" in t for t in logs)
        b = batches[-1]
        assert set(b["amp"]) == {"A", "B"} and b["wave"]["A"].shape[0] == 4
        assert b["rates"] is None or 0 < b["rates"]["live"] <= 1


def test_tune_without_saving_writes_nothing(config, tmp_path):
    config["capture"]["total_waveforms"] = 64
    assert run_acquisition(config, mode="tune", save=False) == 64
    assert not (tmp_path / "data").exists()


def test_ql_pulse_heights_follow_cs137_like_spectrum(config):
    """Quick-look amplitudes (shared ChannelProcessor) put the 662 keV line where the
    simulator puts it: anode = E * gain * (0.6 + 0.4 z)."""
    config["capture"]["total_waveforms"] = 600
    ql = queue.Queue(maxsize=1000)
    run_acquisition(config, ql_queue=ql, save=False, mode="tune")
    amps = []
    while not ql.empty():
        m = ql.get()
        if m["type"] == "batch" and m["n"]:
            amps.append(m["amp"]["A"])
    a = np.concatenate(amps)
    assert a.size == 600
    # photopeak band reaches up to 662 keV / 3367 keV/V = 0.197 V
    assert np.percentile(a, 99) == pytest.approx(0.197, rel=0.1)
    assert np.all(a > 0.0)


def test_worker_main_in_subprocess(config):
    from picodaq.acquisition import worker_main

    ctx = mp.get_context("spawn")
    q, stop = ctx.Queue(), ctx.Event()
    config["capture"]["total_waveforms"] = 64
    p = ctx.Process(target=worker_main, args=(config, None, stop, q, None, "run", True))
    p.start()
    seen = []
    while True:
        m = q.get(timeout=30)
        seen.append(m["type"])
        if m["type"] == "stopped":
            break
    p.join(10)
    assert "error" not in seen and "batch" in seen and "info" in seen


@pytest.mark.parametrize("realtime", [False, True])
def test_simulator_without_triggers_respects_timeout_and_stop(config, realtime):
    import threading
    import time

    from picodaq.backends import SimBackend

    config["sim_realtime"] = realtime
    be = SimBackend(config)
    be.set_threshold(1900)  # nothing passes
    t = time.monotonic()
    assert be.capture(100, timeout_s=0.3) is None
    if realtime:
        assert 0.25 < time.monotonic() - t < 0.8
        ev = threading.Event()
        threading.Timer(0.2, ev.set).start()
        t = time.monotonic()
        assert be.capture(100, stop_event=ev) is None
        assert time.monotonic() - t < 0.5


def test_over_range_is_recorded_and_vetoed(config):
    """Simulator with a +-100 mV anode range: large pulses saturate. The file
    flags them, the QL message counts them, and the event file drops them from GOOD."""
    pytest.importorskip("astropy")
    from picodaq.events import EventFile

    config["channels"]["A"]["range"] = "mV100"
    config["capture"]["total_waveforms"] = 400
    ql = queue.Queue(maxsize=1000)
    run_acquisition(config, ql_queue=ql, mode="run")
    msgs = []
    while not ql.empty():
        msgs.append(ql.get())
    n_ql = sum(m["overflow"]["A"] for m in msgs if m.get("type") == "batch" and m.get("overflow"))
    f = run_files(config["output"]["data_dir"])[0]
    with WaveformReader(f) as r:
        ov = r.overflow()
        nA = r.overflow_count()["A"]
        assert 20 < nA < r.n_events and r.overflow_count()["B"] == 0
        assert n_ql == nA
        vmax = np.abs(r.waveforms("A")).max(axis=1)
        assert np.all(vmax[(ov & 1) == 1] >= 0.0999) and np.all(vmax[(ov & 1) == 0] < 0.1)
    ef = EventFile.from_waveforms(f, verbose=False)
    good = np.asarray(ef.events["GOOD"])
    assert not good[np.asarray(ef.events["OVERFLOW"]) != 0].any()
    assert ef.files["N_OVERFLOW"][0] == nA
    ef.recompute_good(reject_overflow=False)
    assert np.asarray(ef.events["GOOD"]).sum() > good.sum()
