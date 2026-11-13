"""Pulse processing shared by the quick look and the offline analysis.

Charge-sensitive preamplifier output (step + exponential decay) ->
polarity / baseline correction -> trapezoidal shaping (Jordanov & Knoll,
NIM A 345 (1994) 337) -> pulse height and quality variables.

All lengths inside the classes are in samples; use the *from_time /
from_config constructors to work in seconds or microseconds.
"""
import warnings
from dataclasses import dataclass, replace

import numpy as np


# ------------------------------------------------------------------ filter
def trapezoid(v, rise, flat, M):
    """Trapezoidal shaper with pole-zero correction.

    v    : (..., n) baseline-corrected, positive-polarity waveform(s)
    rise : rise (peaking) length k [samples]
    flat : flat-top length [samples]
    M    : pole-zero parameter = tau / dt
    Output is normalised to the input step amplitude.
    """
    k, l = int(rise), int(rise) + int(flat)
    n = v.shape[-1]
    d = np.array(v, dtype=np.float64, copy=True)
    d[..., k:] -= v[..., :n - k]
    d[..., l:] -= v[..., :n - l]
    d[..., k + l:] += v[..., :n - k - l]
    p = np.cumsum(d, axis=-1)  # first integrator
    s = np.cumsum(p + M * d, axis=-1)  # pole-zero + second integrator
    return s / (k * M)


@dataclass(frozen=True)
class TrapezoidShaper:
    rise: int  # samples
    flat: int  # samples
    tau: float  # samples (= M)

    @classmethod
    def from_time(cls, dt, rise_s, flat_s, tau_s):
        return cls(max(1, int(rise_s / dt)), max(1, int(flat_s / dt)), tau_s / dt)

    def with_tau(self, tau):
        return replace(self, tau=float(tau))

    @property
    def length(self):
        return 2 * self.rise + self.flat

    def __call__(self, v):
        return trapezoid(v, self.rise, self.flat, self.tau)

    def flattop(self, trigger_index):
        """[start, stop) of the flat top for a pulse starting at trigger_index."""
        return trigger_index + self.rise, trigger_index + self.rise + self.flat

    def amplitude(self, shaped, trigger_index, method="flattop", guard_frac=0.15, search=None,
                  window=None):
        """Pulse height from shaped waveform(s).

        flattop : mean over the flat top (fixed window from the trigger, unbiased).
                  window = (f0, f1) as fractions of the flat top; default
                  (guard_frac, 1 - guard_frac) as in the notebook. The flat top is
                  only flat after the charge collection: with collection times
                  close to the flat top, use a longer flat top and a late window,
                  e.g. flat 20 us with window (0.6, 0.9).
        max     : maximum of a moving average around the flat top
                  (robust against a spread in collection time); search = extra
                  samples on each side (default: rise/2)
        """
        lo, hi = self.flattop(trigger_index)
        guard = max(3, int(guard_frac * self.flat))
        if method == "flattop":
            if window is None:
                a, b = lo + guard, hi - guard
            else:
                a, b = lo + int(window[0] * self.flat), lo + max(int(window[1] * self.flat), int(window[0] * self.flat) + 1)
            return shaped[..., a:b].mean(axis=-1)
        if method == "max":
            w = max(3, int(0.3 * self.flat))
            pad = self.rise // 2 if search is None else int(search)
            seg = shaped[..., max(0, lo - pad):hi + pad]
            c = np.cumsum(seg, axis=-1)
            smooth = (c[..., w:] - c[..., :-w]) / w
            return smooth.max(axis=-1)
        raise ValueError(f"unknown amplitude method {method!r}")


# ------------------------------------------------------- parameter helpers
def tune_tau(v, shaper, taus, trigger_index, n_events=60, guard_frac=0.1, skip=None):
    """Pole-zero tau [samples] for which the flat top has zero droop.

    v     : (n, samples) corrected waveforms (largest events are used)
    taus  : candidate taus [samples]
    skip  : samples at the start of the flat top that are not used, because the
            charge collection is still going on there. Default: the collection
            time of the mean waveform (position of its maximum) + guard.
            skip=max(5, guard_frac*flat) reproduces the original notebook, which
            is biased when the collection time is not much shorter than the flat top.
    Returns (best_tau, droops); droop = s[end] / s[start] - 1.
    """
    rise, flat = shaper.rise, shaper.flat
    win = slice(trigger_index, trigger_index + rise)
    sel = np.argsort(v[:, win].mean(1))[-n_events:]
    m = v[sel].mean(0)
    lo, hi = shaper.flattop(trigger_index)
    guard = max(5, int(guard_frac * flat))
    if skip is None:
        t_col = int(np.argmax(m[trigger_index:trigger_index + rise + flat]))
        skip = t_col + guard
        if skip > flat - guard - 5:
            warnings.warn(f"collection time ({t_col} samples) is not shorter than the flat top "
                          f"({flat} samples): tau tuning is biased, use a longer flat top")
            skip = max(guard, flat - guard - 5)
    start = lo + int(skip)
    taus = np.asarray(taus, dtype=float)
    droops = np.empty(taus.size)
    for i, tau in enumerate(taus):
        s = trapezoid(m, rise, flat, tau)
        droops[i] = s[hi - guard] / s[start] - 1.0
    o = np.argsort(droops)
    return float(np.interp(0.0, droops[o], taus[o])), droops


def fit_tail_tau(wave, time, trigger_index, start_s=40e-6, stop_s=None):
    """Single-exponential fit of the tail [s] (cross-check for tune_tau)."""
    dt = float(time[1] - time[0])
    i0 = trigger_index + int(start_s / dt)
    i1 = wave.size - 50 if stop_s is None else trigger_index + int(stop_s / dt)
    y, t = wave[i0:i1], time[i0:i1]
    ok = y > 0
    slope, _ = np.polyfit(t[ok], np.log(y[ok]), 1)
    return -1.0 / slope


def rise_time(wave, dt, trigger_index, lo=0.1, hi=0.9, pre_s=20e-6, window_s=60e-6):
    """lo-hi rise time [s] of a corrected (positive) waveform."""
    i0 = max(0, trigger_index - int(pre_s / dt))
    seg = wave[i0:i0 + int(window_s / dt)]
    pk = seg.max()
    return (int(np.argmax(seg > hi * pk)) - int(np.argmax(seg > lo * pk))) * dt


# --------------------------------------------------------- channel level
@dataclass
class ChannelProcessor:
    """Everything needed to turn raw waveforms of one channel into pulse heights."""

    shaper: TrapezoidShaper
    trigger_index: int
    baseline_samples: int
    polarity: float = 1.0
    method: str = "flattop"
    guard_frac: float = 0.15
    window: tuple = None  # flat-top averaging window as fractions (None: guard_frac on both ends)

    @classmethod
    def from_config(cls, processing, channel, dt, trigger_index):
        """processing: the 'processing' section of the config."""
        ch = processing["channels"].get(channel, {})
        sh = processing["shaper"]
        shaper = TrapezoidShaper.from_time(
            dt, sh["rise_us"] * 1e-6, sh["flat_us"] * 1e-6, ch.get("tau_us", 100.0) * 1e-6
        )
        return cls(
            shaper=shaper,
            trigger_index=int(trigger_index),
            baseline_samples=max(1, int(processing.get("baseline_fraction", 0.8) * trigger_index)),
            polarity=float(ch.get("polarity", 1)),
            method=processing.get("amplitude", "flattop"),
            guard_frac=float(processing.get("guard_frac", 0.15)),
            window=tuple(processing["flattop_window"]) if processing.get("flattop_window") else None,
        )

    def correct(self, raw):
        """Polarity to +, subtract baseline. Returns (v, baseline)."""
        v = np.asarray(raw, dtype=np.float64)
        if v.ndim == 1:
            v = v[None, :]
        v = v * self.polarity
        base = v[:, : self.baseline_samples].mean(axis=1)
        return v - base[:, None], base

    def shape(self, raw):
        v, _ = self.correct(raw)
        return self.shaper(v)

    def amplitude(self, raw):
        v, _ = self.correct(raw)
        return self.shaper.amplitude(self.shaper(v), self.trigger_index, self.method, self.guard_frac,
                                     window=self.window)

    def process(self, raw):
        """Pulse height plus quality-cut variables (as in the analysis notebook).

        amp      : pulse height [V]
        baseline : baseline mean [V] (polarity-corrected)
        bl_rms   : baseline RMS [V]
        bl_slope : baseline slope [V / sample]
        vmax     : maximum of the corrected waveform [V]
        late     : rise after the shaping window [V] (pile-up indicator)
        """
        v = np.asarray(raw, dtype=np.float64)
        if v.ndim == 1:
            v = v[None, :]
        v = v * self.polarity
        nbl = self.baseline_samples
        bl = v[:, :nbl]
        out = {"baseline": bl.mean(1), "bl_rms": bl.std(1)}
        out["bl_slope"] = np.polyfit(np.arange(nbl), bl.T, 1)[0] if nbl > 1 else np.zeros(len(v))
        v = v - out["baseline"][:, None]
        out["vmax"] = v.max(1)
        s = self.shaper(v)
        out["amp"] = self.shaper.amplitude(s, self.trigger_index, self.method, self.guard_frac,
                                           window=self.window)
        hi = self.shaper.flattop(self.trigger_index)[1]
        tail = v[:, hi + self.shaper.rise:]
        out["late"] = (tail - tail[:, :1]).max(1) if tail.shape[1] > 1 else np.zeros(len(v))
        return out

    def process_reader(self, reader, channel, chunk=256, events=None):
        """Run process() over a WaveformReader channel chunk by chunk."""
        parts = []
        for _, _, wf in reader.iter_chunks(chunk=chunk, channels=[channel], events=events):
            parts.append(self.process(wf[channel]))
        if not parts:
            return {}
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def processors_from_config(processing, channels, dt, trigger_index):
    return {ch: ChannelProcessor.from_config(processing, ch, dt, trigger_index) for ch in channels}


def quality_mask(features, samples, bl_rms_max=0.010, drift_max=0.050, late_max=0.050):
    """Notebook quality cut: baseline RMS, baseline drift over the record, late pulse."""
    return (
        (features["bl_rms"] < bl_rms_max)
        & (np.abs(features["bl_slope"]) * samples < drift_max)
        & (features["late"] < late_max)
    )
