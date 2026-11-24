"""GOOD uses only the anode and cathode, also when more channels are recorded."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("astropy")

from picodaq.config import load_config
from picodaq.events import EventFile
from picodaq.io import WaveformWriter


def test_extra_channel_does_not_reject_events(tmp_path):
    cfg = load_config(Path(__file__).parent.parent / "config_acquisition.yaml")
    n, samples, dt = 20, 2500, 1e-7
    times = (np.arange(samples) - samples // 5) * dt
    rng = np.random.default_rng(0)
    quiet = lambda: rng.normal(0, 1e-3, (n, samples)).astype(np.float32)
    ttl = np.where(times >= 0, 4.0, 0.0).astype(np.float32) + rng.normal(0, 0.02, (n, samples)).astype(np.float32)
    overflow = np.full(n, 1 << 2, np.uint8)  # C over range in every event
    overflow[3] |= 1 << 0  # A over range in event 3
    trig = {"timestamp": np.arange(1, n + 1, dtype=np.uint64) * 10000, "missed": np.ones(n, np.uint64),
            "status": np.zeros(n, np.int64)}
    path = tmp_path / "ext_00000.h5"
    w = WaveformWriter(path, ["A", "B", "C"], samples, times, cfg, "TEST")
    w.append({"A": quiet(), "B": quiet(), "C": ttl.astype(np.float32)}, trig, 0.0, 1.0, 1000.0, overflow=overflow)
    w.close()

    ef = EventFile.from_waveforms([path], verbose=False)
    good = np.asarray(ef.events["GOOD"])
    assert not good[3]  # anode over range: still vetoed
    assert good[np.arange(n) != 3].all()  # C (noisy, over range) does not reject
    np.testing.assert_array_equal(ef.recompute_good(), good)  # same rule as recompute_good
