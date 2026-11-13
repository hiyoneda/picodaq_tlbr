"""PicoScope trigger timestamps, trigger rates and live time.

timeStampCounter : sample ticks of the scope clock, 56-bit, wraps.
missedTriggers   : trigger events the trigger circuit saw since the previous
                   captured segment while the scope could not capture
                   (post-trigger of the previous record, pre-trigger fill,
                   re-arm, readout between rapid-block batches).
status           : per-segment PICO_STATUS; TS_RESET means this segment's
                   timestamp is not continuous with the previous one.

missed may be None (PicoScope 6000 / ps6000 driver has no missedTriggers):
then only the accepted rate is known; input rate and live fraction are NaN.
"""
import numpy as np

TS_MASK = (1 << 56) - 1
TS_RESET = 0x01000000  # PICO_DEVICE_TIME_STAMP_RESET


def ts_diff(a, b):
    """(a - b) for 56-bit wrapping counters; works on ints and uint64 arrays."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a = np.asarray(a, dtype=np.uint64)
        b = np.asarray(b, dtype=np.uint64)
        return ((a - b) & np.uint64(TS_MASK)).astype(np.int64)
    return (int(a) - int(b)) & TS_MASK


def is_reset(status):
    return (np.asarray(status, dtype=np.int64) & TS_RESET) != 0


def unwrap_timestamps(timestamp, status=None):
    """Return (ticks, epoch).

    ticks : int64, cumulative ticks since the first event of its epoch
    epoch : int, increments at every timestamp reset (time differences across
            epochs are unknown)
    """
    ts = np.asarray(timestamp, dtype=np.uint64)
    n = ts.size
    ticks = np.zeros(n, np.int64)
    epoch = np.zeros(n, np.int64)
    if n == 0:
        return ticks, epoch
    d = ts_diff(ts[1:], ts[:-1])
    brk = np.zeros(n, bool)
    if status is not None:
        brk = is_reset(status)
        brk[0] = False
    epoch = np.cumsum(brk)
    d = np.where(brk[1:], 0, d)
    ticks[1:] = np.cumsum(d)
    # restart the count at each epoch
    starts = np.flatnonzero(np.r_[True, brk[1:]])
    offset = np.repeat(ticks[starts], np.diff(np.r_[starts, n]))
    return ticks - offset, epoch


def livetime_summary(timestamp, missed, status, dt):
    """Real time, input/accepted rate and live fraction, summed over epochs.

    The first segment of every epoch only starts the clock: its missed count
    refers to an unknown interval and is not used.
    """
    ticks, epoch = unwrap_timestamps(timestamp, status)
    have_missed = missed is not None
    missed = np.asarray(missed, dtype=np.int64) if have_missed else np.zeros(len(ticks), np.int64)
    real = n_acc = n_in = 0.0
    epochs = []
    for e in np.unique(epoch):
        sel = np.flatnonzero(epoch == e)
        if sel.size < 2:
            continue
        t = (ticks[sel[-1]] - ticks[sel[0]]) * dt
        acc = sel.size - 1
        inp = acc + int(missed[sel[1:]].sum())
        epochs.append({"real_time_s": t, "accepted": acc, "input": inp})
        real += t
        n_acc += acc
        n_in += inp
    if real <= 0:
        return {"real_time_s": 0.0, "input_rate": np.nan, "accepted_rate": np.nan,
                "live_fraction": np.nan, "live_time_s": np.nan, "epochs": epochs}
    live = n_acc / n_in if n_in and have_missed else np.nan
    return {
        "real_time_s": real,
        "input_rate": n_in / real if have_missed else np.nan,
        "accepted_rate": n_acc / real,
        "live_fraction": live,
        "live_time_s": real * live,
        "epochs": epochs,
    }


class RateMeter:
    """Online input rate / live fraction, one update per rapid-block batch."""

    def __init__(self, dt):
        self.dt = dt
        self.prev_ts = None

    def update(self, trig):
        if trig is None or len(trig["timestamp"]) == 0:
            return None
        ts = [int(x) for x in trig["timestamp"]]
        have_missed = trig.get("missed") is not None
        missed = np.asarray(trig["missed"], dtype=np.int64) if have_missed else np.zeros(len(ts), np.int64)
        reset0 = bool(int(trig["status"][0]) & TS_RESET)
        prev, self.prev_ts = self.prev_ts, ts[-1]
        n = len(ts)
        if prev is not None and not reset0:
            span = ts_diff(ts[-1], prev) * self.dt  # full cycle including readout
            n_acc, n_in = n, n + int(missed.sum())
        elif n >= 2:
            span = ts_diff(ts[-1], ts[0]) * self.dt  # inside this batch only
            n_acc, n_in = n - 1, n - 1 + int(missed[1:].sum())
        else:
            return None
        if span <= 0 or n_in <= 0:
            return None
        if not have_missed:
            return {"input_rate": np.nan, "accepted_rate": n_acc / span, "live": np.nan, "span_s": span}
        return {"input_rate": n_in / span, "accepted_rate": n_acc / span,
                "live": n_acc / n_in, "span_s": span}
