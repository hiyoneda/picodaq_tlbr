"""Ps6000Backend (PicoScope 6402C, legacy ps6000 driver) against tests/fake_ps6000.py."""
import threading
import time

import numpy as np
import pytest

from picodaq import ps6000
from picodaq.ps6000 import MAX_ADC, Ps6000Backend, timebase_for_rate


@pytest.fixture
def cfg6402(tmp_path):
    from pathlib import Path

    from picodaq.config import load_config

    cfg = load_config(Path(__file__).parent.parent / "config_6402c.yaml")
    cfg["output"]["data_dir"] = str(tmp_path / "data")
    return cfg


@pytest.fixture
def fake():
    import importlib
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    try:
        return importlib.import_module("fake_ps6000").FakeLib()
    finally:
        sys.path.pop(0)


def test_timebase():
    assert timebase_for_rate(20e6) == 12  # (12 - 4) / 156.25 MHz = 51.2 ns
    assert timebase_for_rate(156.25e6) == 5
    assert timebase_for_rate(5e9) == 0 and timebase_for_rate(1.25e9) == 2


def test_open_and_settings(cfg6402, fake):
    cfg6402["channels"]["A"]["analog_offset_v"] = -0.2
    b = Ps6000Backend(cfg6402, lib=fake)
    assert b.model == "6402C" and b.serial == "FK123/0001" and b.driver == "ps6000"
    assert b.channels == ["A", "B"]
    a = fake.channels[0]
    assert a == dict(enabled=1, coupling=1, range=5, offset=pytest.approx(-0.2), bw=1)  # DC_1M, mV500, 20MHz
    assert fake.channels[2]["enabled"] == 0 and fake.channels[2]["offset"] == 0.0
    # 30 mV on A: (0.030 - 0.2) / 0.5 * 32512, rising
    assert fake.trigger["thr"] == round((0.030 - 0.2) / 0.5 * MAX_ADC) and fake.trigger["direction"] == 2
    assert b.dt == pytest.approx(51.2e-9) and b.times.size == 5000
    assert np.argmin(np.abs(b.times)) == 1000
    b.set_threshold(80.0)
    assert fake.trigger["thr"] == round((0.080 - 0.2) / 0.5 * MAX_ADC)
    with pytest.warns(UserWarning):
        b.set_threshold(900.0)  # outside +-500 mV
    assert fake.trigger["thr"] == MAX_ADC
    b.close()
    assert fake.closed


def test_bad_range_closes(cfg6402, fake):
    cfg6402["channels"]["B"]["range"] = "V3"
    with pytest.raises(ValueError):
        Ps6000Backend(cfg6402, lib=fake)
    assert fake.closed
    cfg6402["channels"]["B"]["range"] = "mV500"
    cfg6402["channels"]["A"]["analog_offset_v"] = -0.8
    with pytest.raises(ValueError, match="analog_offset_v"):
        Ps6000Backend(cfg6402, lib=fake)


def test_full_batch_values_and_timestamps(cfg6402, fake):
    cfg6402["channels"]["B"]["analog_offset_v"] = 0.1
    b = Ps6000Backend(cfg6402, lib=fake)
    data, times, trig = b.capture(16)
    assert data["A"].shape == (16, 5000) and data["A"].dtype == np.float32
    # fake: 1000*(ch+1) + seg counts after the trigger
    assert data["A"][3, 2000] == pytest.approx((1000 + 3) / MAX_ADC * 0.5, rel=1e-5)
    assert data["B"][0, 0] == pytest.approx(-0.1)  # counts 0 -> -offset
    assert trig["missed"] is None and trig["timestamp"].dtype == np.uint64
    d = np.diff(trig["timestamp"].astype(np.int64)) * b.dt
    assert np.allclose(d, 1 / fake.rate, rtol=1e-3)
    assert ("RunBlock", 1000, 4000, 12, 1) in fake.calls
    np.testing.assert_array_equal(b.last_overflow, [0b10 if i % 5 == 2 else 0 for i in range(16)])
    b.close()


def test_partial_stop_none(cfg6402, fake):
    b = Ps6000Backend(cfg6402, lib=fake)
    t = time.monotonic()
    data, _, trig = b.capture(1000, timeout_s=0.2)
    assert 0.19 < time.monotonic() - t < 0.4
    n = data["A"].shape[0]
    assert 0 < n < 1000 and len(trig["timestamp"]) == n
    ev = threading.Event()
    threading.Timer(0.1, ev.set).start()
    t = time.monotonic()
    data, _, _ = b.capture(1000, stop_event=ev)
    assert time.monotonic() - t < 0.3 and data["A"].shape[0] < 1000
    fake.rate = 1e-3
    assert b.capture(10, timeout_s=0.05) is None
    b.close()


def test_trigger_info_failure(cfg6402, fake):
    fake.trigger_info_status = 0x43
    b = Ps6000Backend(cfg6402, lib=fake)
    with pytest.warns(UserWarning):
        assert b.capture(4)[2] is None
    assert b.capture(4)[2] is None  # not asked again
    b.close()


def test_status_error_message(cfg6402, fake):
    fake.ps6000OpenUnit = lambda h, s: 0x3
    with pytest.raises(ps6000.Ps6000Error, match="NOT_FOUND"):
        Ps6000Backend(cfg6402, lib=fake)


def test_library_not_found(tmp_path):
    import ctypes.util

    if ctypes.util.find_library("ps6000"):
        pytest.skip("a real libps6000 is installed")
    with pytest.raises(FileNotFoundError, match="sdk_path"):
        ps6000.load_library(tmp_path)


def test_acquisition_and_hwcheck(cfg6402, fake, monkeypatch, tmp_path):
    """Whole chain with the fake: file has model 6402C and no missedTriggers; hwcheck has no FAIL."""
    from picodaq import ps6000 as mod
    from picodaq.acquisition import run_acquisition
    from picodaq.hwcheck import FAIL, run_check
    from picodaq.io import WaveformReader, run_files

    monkeypatch.setattr(mod, "load_library", lambda path=None: fake)
    cfg6402["capture"]["total_waveforms"] = 64
    run_acquisition(cfg6402, mode="run")
    files = run_files(cfg6402["output"]["data_dir"], base="tlbr6402c")
    assert files
    with WaveformReader(files[0]) as r:
        assert r.model == "6402C" and not r.missed_available
        assert r.trigger_info()["missed"] is None
        s = r.rate_summary()
        assert np.isnan(s["input_rate"]) and np.isnan(s["live_fraction"]) and s["accepted_rate"] > 0
        t, _ = r.event_times()
        assert t.size == r.n_events
    rep = run_check(cfg6402, pulser_hz=None, batch=16, n_batches=3)
    rows = {x["check"]: x for x in rep.rows}
    assert not [x for x in rep.rows if x["status"] == FAIL], rep.rows
    assert rows["missedTriggers"]["status"] == "INFO" and "対象外" in rows["simple loop"]["detail"]
    # auto high threshold = 95 % of +-500 mV
    assert "475 mV" in rows["batch timeout 1 s"]["detail"]


def test_ctypes_against_compiled_shim(cfg6402, tmp_path):
    """Load a real shared library (tests/shim/fake_libps6000.c) through load_library(sdk_path):
    checks the ctypes prototypes, pointer passing and the TRIGGER_INFO layout."""
    import ctypes
    import platform
    import shutil
    import subprocess
    from pathlib import Path

    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler")
    src = Path(__file__).parent / "shim" / "fake_libps6000.c"
    ext = {"Darwin": "dylib", "Windows": "dll"}.get(platform.system(), "so")
    lib_dir = tmp_path / "libps6000"
    lib_dir.mkdir()
    out = lib_dir / f"libps6000.{ext}"
    flags = ["-dynamiclib"] if ext == "dylib" else ["-shared", "-fPIC"]
    subprocess.run([cc, *flags, "-O1", "-o", str(out), str(src)], check=True)
    cfg6402["scope"]["sdk_path"] = str(tmp_path)
    cfg6402["capture"]["samples"] = 1000
    b = Ps6000Backend(cfg6402)
    assert isinstance(b.lib, ctypes.CDLL) and b.model == "6402C" and b.serial == "SHIM/0001"
    assert ctypes.c_int16.in_dll(b.lib, "last_threshold").value == round(0.030 / 0.5 * MAX_ADC)
    assert ctypes.sizeof(ps6000.PS6000_TRIGGER_INFO) == 32  # packed
    data, times, trig = b.capture(8)
    assert data["B"][5, 10] == pytest.approx((100 + 5) / MAX_ADC * 0.5, rel=1e-5)
    assert trig["timestamp"][3] == (0x00FFFFFFFFFFFF00 + 3000) & ((1 << 56) - 1)
    b.close()
