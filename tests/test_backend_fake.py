"""PicoBackend control flow against a stand-in for pypicosdk (no hardware)."""
import threading
import time

import numpy as np
import pytest


@pytest.fixture
def backend(fake_pypicosdk, config):
    from picodaq.backends import PicoBackend

    config["simulate"] = False
    b = PicoBackend(config)
    yield b
    b.close()


def test_full_batch(backend):
    data, times, trig = backend.capture(16)
    assert set(data) == {"A", "B"}
    assert data["A"].shape == (16, 5000) and data["A"].dtype == np.float32
    assert times.size == 5000 and np.argmin(np.abs(times)) == 1000
    assert len(trig["timestamp"]) == 16 and trig["timestamp"].dtype == np.uint64
    ov = backend.last_overflow
    assert ov.dtype == np.uint8 and len(ov) == 16
    assert ov[3] == 0b01 and ov[5] == 0b10 and ov[10] == 0b01 and ov[0] == 0  # A = bit 0, B = bit 1


def test_timeout_returns_partial_batch(backend):
    t = time.monotonic()
    data, _, trig = backend.capture(1000, timeout_s=0.3)
    el = time.monotonic() - t
    assert 0.29 < el < 0.5
    assert 0 < data["A"].shape[0] < 1000 and data["A"].shape[0] == len(trig["missed"])


def test_stop_event(backend):
    ev = threading.Event()
    threading.Timer(0.2, ev.set).start()
    t = time.monotonic()
    data, _, _ = backend.capture(1000, stop_event=ev)
    assert time.monotonic() - t < 0.4
    assert data["A"].shape[0] < 1000


def test_no_trigger_returns_none(backend, fake_pypicosdk, monkeypatch):
    monkeypatch.setattr(fake_pypicosdk, "RATE", 0.0)
    assert backend.capture(10, timeout_s=0.1) is None


def test_simple_loop_and_threshold(backend):
    backend.loop = "simple"
    data, _, trig = backend.capture(4)
    assert data["B"].shape == (4, 5000) and trig is not None
    assert backend.last_overflow is None  # not reported by the simple loop
    backend.set_threshold(80.4)
    assert backend.scope.thr == 80


def test_overflow_bits():
    from picodaq.backends import overflow_bits

    np.testing.assert_array_equal(overflow_bits([[], ["A"], ["B", "A"], 4], 4), [0, 1, 3, 4])
    assert overflow_bits([[]], 2) is None and overflow_bits(None, 1) is None


def test_trigger_info_covers_every_segment(fake_pypicosdk, config):
    """Every segment of a batch gets its trigger info, the last one included
    (pypicosdk 1.7.5 get_trigger_info left the last one empty)."""
    from picodaq.backends import PicoBackend

    config["simulate"] = False
    b = PicoBackend(config)
    for kw in ({}, {"timeout_s": 0.05}):  # full batch, then a partial one (200 Hz, 0.05 s)
        data, _, trig = b.capture(16, **kw)
        n = data["A"].shape[0]
        assert len(trig["timestamp"]) == n
        assert np.all(trig["timestamp"] > 0)  # nothing left at 0
        np.testing.assert_array_equal(trig["timestamp"], 1000 + 100000 * np.arange(n))
        assert b.scope.trigger_info_calls[-1] == (0, n)  # count, not last index
    # the old call through pypicosdk 1.7.5 leaves the last entry empty: what the fix avoids
    old = b.scope.get_trigger_info(0, 16)
    assert old[-1]["timeStampCounter"] == 0 and old[-2]["timeStampCounter"] > 0
    b.close()
