"""HDF5 waveform files.

Layout (one file = up to output.max_file_size_mb of waveforms):
    attrs   PicoScopeModel, PicoScopeSerial, Config (YAML), RunMode,
            Waveforms (count), ActualSampleRateHz, TimestampUnitS
    Time                     (samples,)          time axis [s], 0 = trigger
    Waveforms/<ch>           (n, samples) f4      [V], as recorded (raw polarity)
    Trigger/Timestamp        (n,) u8             scope ticks, 56-bit
    Trigger/MissedTriggers   (n,) u8
    Trigger/Status           (n,) i8             -1 = not available
    Trigger/Overflow         (n,) u1             over-range bit mask, bit i = channel i (A = 0);
                                                 attrs OverflowAvailable = 0 if not reported
    BatchInfo                (n_batches, 4) f8   t_start_unix, t_end_unix, n, threshold_mv
Files written by the original run_acqusition.py have only Time and Waveforms;
WaveformReader handles both.
"""
from pathlib import Path

import h5py
import numpy as np
import yaml

from .timing import livetime_summary, unwrap_timestamps

BATCH_COLUMNS = ("t_start_unix_s", "t_end_unix_s", "n_waveforms", "threshold_mv")


class WaveformWriter:
    def __init__(self, path, channels, samples, times, config, serial, mode="run",
                 model="3416E MSO"):
        output = config["output"]
        compression = output.get("compression")
        self.path = Path(path)
        self.file = h5py.File(path, "w")
        a = self.file.attrs
        a["PicoScopeModel"] = model
        a["PicoScopeSerial"] = serial
        a["Config"] = yaml.safe_dump(config, sort_keys=False)
        a["RunMode"] = mode
        a["TimestampUnitS"] = float(times[1] - times[0])
        a["ActualSampleRateHz"] = 1.0 / float(times[1] - times[0])
        a["Waveforms"] = 0
        a["MissedTriggersAvailable"] = 1  # 0 for the ps6000 driver (no missedTriggers)
        a["OverflowAvailable"] = 1  # 0 once a batch came without over-range information
        self.file.create_dataset("Time", data=times)
        group = self.file.create_group("Waveforms")
        chunk_events = min(output["rapid_block_captures"], 16)
        self.data = {
            ch: group.create_dataset(
                ch, shape=(0, samples), maxshape=(None, samples), dtype=np.float32,
                chunks=(chunk_events, samples), compression=compression,
            )
            for ch in channels
        }
        trig = self.file.create_group("Trigger")
        self.trig = {
            name: trig.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dt, chunks=(4096,))
            for name, dt in (("Timestamp", np.uint64), ("MissedTriggers", np.uint64),
                             ("Status", np.int64), ("Overflow", np.uint8))
        }
        self.trig["Overflow"].attrs["Bits"] = "bit i = channel i (A=0, B=1, C=2, D=3): over range in the record"
        self.trig["Timestamp"].attrs["Unit"] = "ticks of TimestampUnitS, 56-bit wrapping counter"
        self.batches = self.file.create_dataset(
            "BatchInfo", shape=(0, 4), maxshape=(None, 4), dtype=np.float64, chunks=(256, 4)
        )
        self.batches.attrs["Columns"] = ", ".join(BATCH_COLUMNS)

    @property
    def count(self):
        return next(iter(self.data.values())).shape[0]

    def append(self, buffers, trig, t_start, t_end, threshold_mv, overflow=None):
        """overflow: (count,) uint8 over-range bit mask, or None if not reported."""
        count = next(iter(buffers.values())).shape[0]
        start = self.count
        for ch, d in self.data.items():
            d.resize((start + count, d.shape[1]))
            d[-count:] = buffers[ch]
        if trig is None:
            trig = {"timestamp": np.zeros(count, np.uint64), "missed": np.zeros(count, np.uint64),
                    "status": np.full(count, -1, np.int64)}
        elif trig.get("missed") is None:
            trig = dict(trig, missed=np.zeros(count, np.uint64))
            self.file.attrs["MissedTriggersAvailable"] = 0
        if overflow is None or len(overflow) != count:
            overflow = np.zeros(count, np.uint8)
            self.file.attrs["OverflowAvailable"] = 0
        trig = dict(trig, overflow=overflow)
        for name, key in (("Timestamp", "timestamp"), ("MissedTriggers", "missed"), ("Status", "status"),
                          ("Overflow", "overflow")):
            d = self.trig[name]
            d.resize((start + count,))
            d[-count:] = trig[key]
        nb = self.batches.shape[0]
        self.batches.resize((nb + 1, 4))
        self.batches[nb] = (t_start, t_end, count, threshold_mv)
        self.file.attrs["Waveforms"] = self.count
        self.file.flush()

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class WaveformReader:
    """Read one HDF5 file. Waveforms are returned as recorded (raw polarity, V);
    use shaping.ChannelProcessor for polarity / baseline / pulse height."""

    def __init__(self, path):
        self.path = Path(path)
        self.file = h5py.File(path, "r")
        f = self.file
        self.time = f["Time"][:]
        self.dt = float(np.mean(np.diff(self.time)))
        self.fs = 1.0 / self.dt
        self.samples = self.time.size
        self.trigger_index = int(np.argmin(np.abs(self.time)))
        self.channels = list(f["Waveforms"])
        self.n_events = f["Waveforms"][self.channels[0]].shape[0]
        self.config = yaml.safe_load(f.attrs["Config"]) if "Config" in f.attrs else {}
        self.model = f.attrs.get("PicoScopeModel", "")
        self.serial = f.attrs.get("PicoScopeSerial", "")
        self.missed_available = bool(f.attrs.get("MissedTriggersAvailable", 1))
        self.run_mode = f.attrs.get("RunMode", "")
        self.timestamp_unit = float(f.attrs.get("TimestampUnitS", self.dt))

    def __len__(self):
        return self.n_events

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.file.close()

    def waveforms(self, channel, events=slice(None)):
        """(n, samples) float64 [V], raw polarity."""
        v = np.asarray(self.file[f"Waveforms/{channel}"][events], dtype=np.float64)
        return v[None, :] if v.ndim == 1 else v

    def iter_chunks(self, chunk=256, channels=None, events=None):
        """Yield (start, stop, {channel: (m, samples) float64}). events: optional slice."""
        channels = channels or self.channels
        sl = events or slice(None)
        start, stop, _ = sl.indices(self.n_events)
        for i in range(start, stop, chunk):
            j = min(i + chunk, stop)
            yield i, j, {ch: self.waveforms(ch, slice(i, j)) for ch in channels}

    # ------------------------------------------------------------ trigger
    @property
    def has_trigger_info(self):
        if "Trigger" not in self.file:
            return False
        st = self.file["Trigger/Status"]
        return st.shape[0] > 0 and bool(np.any(st[:] >= 0))

    def trigger_info(self):
        """dict(timestamp, missed, status) or None for old files.
        missed is None when the scope does not provide it (PicoScope 6000)."""
        if "Trigger" not in self.file:
            return None
        g = self.file["Trigger"]
        missed = g["MissedTriggers"][:] if self.missed_available else None
        return {"timestamp": g["Timestamp"][:], "missed": missed,
                "status": g["Status"][:]}

    def overflow(self):
        """(n,) uint8 over-range bit mask (bit i = channel i, A = 0), or None
        for files without it (older files, simple loop)."""
        f = self.file
        if "Trigger" not in f or "Overflow" not in f["Trigger"] or not f.attrs.get("OverflowAvailable", 0):
            return None
        return f["Trigger"]["Overflow"][:]

    def overflow_count(self):
        """{channel: number of over-range events}, or None."""
        ov = self.overflow()
        if ov is None:
            return None
        from .config import CHANNEL_BITS

        return {ch: int(((ov >> CHANNEL_BITS[ch]) & 1).sum()) for ch in self.channels}

    def event_times(self):
        """(t [s] since the first event of the epoch, epoch) per event, or None."""
        tr = self.trigger_info() if self.has_trigger_info else None
        if tr is None:
            return None
        ticks, epoch = unwrap_timestamps(tr["timestamp"], tr["status"])
        return ticks * self.timestamp_unit, epoch

    def rate_summary(self):
        """Real time, input / accepted rate, live fraction (see timing.livetime_summary)."""
        tr = self.trigger_info() if self.has_trigger_info else None
        if tr is None:
            return None
        return livetime_summary(tr["timestamp"], tr["missed"], tr["status"], self.timestamp_unit)

    def batch_info(self):
        """dict of column arrays, or None for old files."""
        if "BatchInfo" not in self.file:
            return None
        b = self.file["BatchInfo"][:]
        return {name: b[:, i] for i, name in enumerate(BATCH_COLUMNS[: b.shape[1]])}

    def __repr__(self):
        return (f"WaveformReader({self.path.name!r}, events={self.n_events}, channels={self.channels}, "
                f"dt={self.dt * 1e9:.1f} ns, trigger_info={self.has_trigger_info})")


def run_files(data_dir, base="tlbr"):
    """Sorted list of <data_dir>/<base>_NNNNN.h5 (tune files excluded)."""
    return sorted(p for p in Path(data_dir).glob(f"{base}_[0-9]*.h5"))
