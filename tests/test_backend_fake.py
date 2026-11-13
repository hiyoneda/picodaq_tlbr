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
    backend.set_threshold(80.4)
    assert backend.scope.thr == 80
