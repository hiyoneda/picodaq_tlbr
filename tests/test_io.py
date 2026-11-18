import h5py
import numpy as np
import pytest
import yaml

from picodaq.io import WaveformReader, WaveformWriter, run_files
from picodaq.shaping import processors_from_config
from picodaq.timing import TS_RESET


def _times(n=500, pre=100, dt=50e-9):
    return (np.arange(n) - pre) * dt


def test_roundtrip(tmp_path, config):
    t = _times()
    path = tmp_path / "tlbr_00000.h5"
    rng = np.random.default_rng(0)
    with WaveformWriter(path, ["A", "B"], t.size, t, config, "SER", "run") as w:
        for k in range(3):
            data = {ch: rng.normal(size=(4, t.size)).astype(np.float32) for ch in "AB"}
            trig = {"timestamp": np.arange(4, dtype=np.uint64) * 1000 + 4000 * k,
                    "missed": np.ones(4, np.uint64),
                    "status": np.array([TS_RESET if k == 0 else 0, 0, 0, 0])}
            w.append(data, trig, 1.0 + k, 1.5 + k, 30.0 + k)
        assert w.count == 12
    with WaveformReader(path) as r:
        assert r.n_events == len(r) == 12
        assert r.channels == ["A", "B"]
        assert r.trigger_index == 100
        assert r.dt == pytest.approx(50e-9)
        assert r.run_mode == "run" and r.serial == "SER"
        assert r.config["processing"]["channels"]["B"]["polarity"] == -1
        np.testing.assert_allclose(r.waveforms("A", slice(8, 12)), data["A"], rtol=1e-6)
        chunks = list(r.iter_chunks(chunk=5))
        assert [(a, b) for a, b, _ in chunks] == [(0, 5), (5, 10), (10, 12)]
        assert r.has_trigger_info
        t_ev, epoch = r.event_times()
        assert t_ev[-1] == pytest.approx(11000 * 50e-9)
        assert set(epoch) == {0}
        s = r.rate_summary()
        assert s["accepted_rate"] == pytest.approx(11 / (11000 * 50e-9))
        b = r.batch_info()
        np.testing.assert_array_equal(b["threshold_mv"], [30, 31, 32])
        # processors from the config stored in the file
        proc = processors_from_config(r.config["processing"], r.channels, r.dt, r.trigger_index)
        assert proc["B"].polarity == -1


def test_missing_trigger_info_is_flagged(tmp_path, config):
    t = _times()
    path = tmp_path / "x.h5"
    with WaveformWriter(path, ["A"], t.size, t, config, "SER") as w:
        w.append({"A": np.zeros((2, t.size), np.float32)}, None, 0, 1, 30)
    with WaveformReader(path) as r:
        assert not r.has_trigger_info
        assert r.event_times() is None and r.rate_summary() is None


def test_reads_original_run_acqusition_files(tmp_path):
    """Files from the original script: only Time + Waveforms + attrs."""
    t = _times()
    path = tmp_path / "tlbr_00000.h5"
    with h5py.File(path, "w") as f:
        f.attrs["PicoScopeModel"] = "3416E MSO"
        f.attrs["PicoScopeSerial"] = "13076/0138"
        f.attrs["Config"] = yaml.safe_dump({"capture": {"samples": t.size}})
        f.attrs["Waveforms"] = 3
        f.create_dataset("Time", data=t)
        g = f.create_group("Waveforms")
        g.create_dataset("A", data=np.zeros((3, t.size), np.float32))
        g.create_dataset("B", data=np.zeros((3, t.size), np.float32))
    with WaveformReader(path) as r:
        assert r.n_events == 3 and r.channels == ["A", "B"]
        assert r.trigger_info() is None and r.batch_info() is None
        assert r.rate_summary() is None


def test_run_files_excludes_tune(tmp_path):
    for name in ("tlbr_00000.h5", "tlbr_00001.h5", "tlbr_tune_00000.h5", "other.h5"):
        (tmp_path / name).touch()
    assert [p.name for p in run_files(tmp_path)] == ["tlbr_00000.h5", "tlbr_00001.h5"]


def test_overflow_roundtrip(tmp_path, config):
    t = _times()
    path = tmp_path / "o.h5"
    trig = {"timestamp": np.arange(4, dtype=np.uint64), "missed": np.zeros(4, np.uint64),
            "status": np.zeros(4, np.int64)}
    data = {ch: np.zeros((4, t.size), np.float32) for ch in "AB"}
    with WaveformWriter(path, ["A", "B"], t.size, t, config, "SER") as w:
        w.append(data, trig, 0, 1, 30, overflow=np.array([0, 1, 2, 3], np.uint8))
    with WaveformReader(path) as r:
        np.testing.assert_array_equal(r.overflow(), [0, 1, 2, 3])
        assert r.overflow_count() == {"A": 2, "B": 2}
    # a batch without over-range information makes the whole file "not available"
    with WaveformWriter(tmp_path / "p.h5", ["A", "B"], t.size, t, config, "SER") as w:
        w.append(data, trig, 0, 1, 30, overflow=np.zeros(4, np.uint8))
        w.append(data, trig, 1, 2, 30)
    with WaveformReader(tmp_path / "p.h5") as r:
        assert r.overflow() is None and r.overflow_count() is None
