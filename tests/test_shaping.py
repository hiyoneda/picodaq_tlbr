import numpy as np
import pytest

from picodaq.backends import csa_response
from picodaq.config import load_config
from picodaq.shaping import (
    ChannelProcessor,
    TrapezoidShaper,
    fit_tail_tau,
    quality_mask,
    rise_time,
    trapezoid,
    tune_tau,
)

DT = 50e-9
N = 5000
TRIG = 1000


def csa_pulse(amp=0.2, tau=100e-6, rise=4e-6, n=N, trig=TRIG, dt=DT):
    """CSA output: charge amp collected linearly over `rise`, decay tau."""
    return csa_response((np.arange(n) - trig) * dt, amp, rise, tau)


def notebook_trapezoid(v, rise, flat, M):
    """Verbatim copy of the notebook implementation (reference)."""
    k, l = int(rise), int(rise) + int(flat)
    n = v.shape[-1]
    d = np.array(v, dtype=np.float64, copy=True)
    d[..., k:] -= v[..., :n - k]
    d[..., l:] -= v[..., :n - l]
    d[..., k + l:] += v[..., :n - k - l]
    p = np.cumsum(d, axis=-1)
    s = np.cumsum(p + M * d, axis=-1)
    return s / (k * M)


def test_trapezoid_matches_notebook():
    v = np.random.default_rng(0).normal(size=(3, N))
    np.testing.assert_allclose(trapezoid(v, 300, 240, 2960.0), notebook_trapezoid(v, 300, 240, 2960.0))


def test_flat_top_equals_step_amplitude():
    tau = 100e-6
    sh = TrapezoidShaper.from_time(DT, 15e-6, 12e-6, tau)
    s = sh(csa_pulse(0.2, tau))
    amp = sh.amplitude(s, TRIG)
    assert amp == pytest.approx(0.2, rel=0.01)
    lo, hi = sh.flattop(TRIG)
    g = int(0.15 * sh.flat)
    flat = s[lo + 80 + g:hi - g]  # after the 4 us (80 sample) collection
    assert np.ptp(flat) < 1e-4  # no droop with the right tau


def test_wrong_tau_gives_droop():
    sh = TrapezoidShaper.from_time(DT, 15e-6, 12e-6, 30e-6)
    s = sh(csa_pulse(0.2, 100e-6))
    lo, hi = sh.flattop(TRIG)
    assert abs(s[hi - 10] - s[lo + 40]) > 0.005


def test_max_method_close_to_flattop():
    sh = TrapezoidShaper.from_time(DT, 15e-6, 12e-6, 100e-6)
    s = sh(csa_pulse(0.2))
    assert sh.amplitude(s, TRIG, "max") == pytest.approx(0.2, rel=0.02)
    with pytest.raises(ValueError):
        sh.amplitude(s, TRIG, "nope")


def test_tune_tau_recovers_tau():
    rng = np.random.default_rng(1)
    v = np.array([csa_pulse(a, 80e-6) for a in rng.uniform(0.1, 0.3, 100)])
    v += rng.normal(0, 2e-3, v.shape)
    sh = TrapezoidShaper.from_time(DT, 15e-6, 12e-6, 1.0)
    taus = np.geomspace(5e-6, 400e-6, 40) / DT
    best, droops = tune_tau(v, sh, taus, TRIG)
    assert best * DT == pytest.approx(80e-6, rel=0.05)
    assert droops.shape == taus.shape


def test_tune_tau_notebook_window_is_biased_by_slow_rise():
    """Notebook setting (droop from flat-top start + 10 %): with an 8 us collection
    time inside a 12 us flat top the start point is still on the rising edge."""
    v = np.array([csa_pulse(0.2, 80e-6, rise=8e-6)])
    sh = TrapezoidShaper.from_time(DT, 15e-6, 12e-6, 1.0)
    taus = np.geomspace(5e-6, 1000e-6, 80) / DT
    good, _ = tune_tau(v, sh, taus, TRIG, n_events=1)
    old, _ = tune_tau(v, sh, taus, TRIG, n_events=1, skip=max(5, int(0.1 * sh.flat)))
    assert good * DT == pytest.approx(80e-6, rel=0.03)
    assert abs(old * DT / 80e-6 - 1) > 0.2


def test_tune_tau_warns_when_flat_top_too_short():
    v = np.array([csa_pulse(0.2, 80e-6, rise=14e-6)])
    sh = TrapezoidShaper.from_time(DT, 15e-6, 12e-6, 1.0)
    with pytest.warns(UserWarning, match="flat top"):
        tune_tau(v, sh, np.geomspace(5e-6, 1000e-6, 40) / DT, TRIG, n_events=1)


def test_fit_tail_and_rise_time():
    time = (np.arange(N) - TRIG) * DT
    w = csa_pulse(0.2, 100e-6, rise=5e-6)
    assert fit_tail_tau(w, time, TRIG) == pytest.approx(100e-6, rel=0.02)
    assert rise_time(w, DT, TRIG) == pytest.approx(0.8 * 5e-6, rel=0.1)  # 10-90 % of a ramp


def test_channel_processor_polarity_baseline_and_features():
    proc = load_config(None)["processing"]
    cp = ChannelProcessor.from_config(proc, "B", DT, TRIG)  # cathode: polarity -1
    assert cp.polarity == -1
    assert cp.baseline_samples == 800
    rng = np.random.default_rng(2)
    raw = -np.array([csa_pulse(a, 27e-6) for a in (0.1, 0.2)]) + 0.05  # offset baseline
    raw += rng.normal(0, 1e-3, raw.shape)
    f = cp.process(raw)
    np.testing.assert_allclose(f["amp"], [0.1, 0.2], rtol=0.02)
    np.testing.assert_allclose(f["baseline"], -0.05, atol=1e-3)
    assert set(f) == {"amp", "baseline", "bl_rms", "bl_slope", "vmax", "late"}
    np.testing.assert_allclose(cp.amplitude(raw), f["amp"])
    assert quality_mask(f, N).all()


def test_late_pulse_flagged():
    cp = ChannelProcessor.from_config(load_config(None)["processing"], "A", DT, TRIG)
    clean = csa_pulse(0.2, 148e-6)
    piled = clean + csa_pulse(0.2, 148e-6, trig=TRIG + 2500)
    f = cp.process(np.array([clean, piled]))
    assert f["late"][0] < 0.01 < f["late"][1]
    assert list(quality_mask(f, N)) == [True, False]


def test_late_flattop_window_removes_ballistic_deficit():
    """Collection 10 us in a 12 us flat top: the notebook window (15-85 %) averages part
    of the rise; a 20 us flat top with a late window gives the full amplitude."""
    short = TrapezoidShaper.from_time(DT, 15e-6, 12e-6, 100e-6)
    long_ = TrapezoidShaper.from_time(DT, 15e-6, 20e-6, 100e-6)
    w = csa_pulse(0.2, 100e-6, rise=10e-6)
    assert short.amplitude(short(w), TRIG) < 0.19
    assert long_.amplitude(long_(w), TRIG, window=(0.6, 0.9)) == pytest.approx(0.2, rel=0.002)
