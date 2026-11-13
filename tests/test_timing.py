import numpy as np
import pytest

from picodaq.timing import TS_MASK, TS_RESET, RateMeter, livetime_summary, ts_diff, unwrap_timestamps


def test_ts_diff_wraps():
    assert ts_diff(5, TS_MASK - 4) == 10
    a = np.array([5, 100], np.uint64)
    b = np.array([TS_MASK - 4, 50], np.uint64)
    np.testing.assert_array_equal(ts_diff(a, b), [10, 50])


def test_unwrap_with_wrap_and_reset():
    ts = np.array([TS_MASK - 10, TS_MASK - 5, 4, 1000, 1010, 1030], np.uint64)
    st = np.array([0, 0, 0, TS_RESET, 0, 0])
    ticks, epoch = unwrap_timestamps(ts, st)
    np.testing.assert_array_equal(epoch, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(ticks, [0, 5, 15, 0, 10, 30])


def _simulate(rate, dead, T, dt=50e-9, seed=0):
    """Poisson triggers, non-paralysable dead time -> (timestamp, missed)."""
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.exponential(1 / rate, int(rate * T * 1.2)))
    t = t[t < T]
    ts, missed, m, free = [], [], 0, -1.0
    for x in t:
        if x < free:
            m += 1
            continue
        ts.append(x)
        missed.append(m)
        m = 0
        free = x + dead
    return np.round(np.array(ts) / dt).astype(np.uint64), np.array(missed, np.uint64)


def test_livetime_summary_recovers_rate_and_live():
    rate, dead = 1000.0, 250e-6
    ts, missed = _simulate(rate, dead, 20.0)
    s = livetime_summary(ts, missed, np.zeros(ts.size, int), 50e-9)
    assert s["input_rate"] == pytest.approx(rate, rel=0.03)
    assert s["live_fraction"] == pytest.approx(1 / (1 + rate * dead), rel=0.03)
    assert s["live_time_s"] == pytest.approx(s["real_time_s"] * s["live_fraction"])


def test_ratemeter_batches():
    ts, missed = _simulate(500.0, 250e-6, 5.0)
    st = np.zeros(ts.size, np.int64)
    st[0] = TS_RESET
    m = RateMeter(50e-9)
    out = []
    for i in range(0, ts.size, 32):
        r = m.update({"timestamp": ts[i:i + 32], "missed": missed[i:i + 32], "status": st[i:i + 32]})
        if r:
            out.append(r)
    assert np.mean([o["input_rate"] for o in out]) == pytest.approx(500, rel=0.05)
    assert m.update(None) is None
