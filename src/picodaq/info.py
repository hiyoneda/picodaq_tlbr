"""Summary of recorded HDF5 files.

    picodaq-info data/20261111/tlbr_00000.h5 [more files ...] [--process 2000]
"""
import argparse

import numpy as np

from .config import load_config
from .io import WaveformReader
from .shaping import processors_from_config, quality_mask


def summarize(path, process=0, config=None):
    lines = []
    with WaveformReader(path) as r:
        lines.append(repr(r))
        lines.append(f"  model {r.model}, serial {r.serial}, mode {r.run_mode or '-'}, "
                     f"record {r.samples * r.dt * 1e6:.1f} us, pre-trigger {r.trigger_index * r.dt * 1e6:.1f} us")
        b = r.batch_info()
        if b is not None and len(b["n_waveforms"]):
            wall = b["t_end_unix_s"][-1] - b["t_start_unix_s"][0]
            thr = np.unique(b["threshold_mv"])
            lines.append(f"  {len(b['n_waveforms'])} batches, wall clock {wall:.1f} s, "
                         f"threshold(s) {', '.join(f'{t:g}' for t in thr)} mV")
        s = r.rate_summary()
        if s is not None:
            def fmt(v, f):
                return f.format(v) if np.isfinite(v) else "n/a"

            lines.append(f"  trigger info: real time {s['real_time_s']:.2f} s ({len(s['epochs'])} epoch(s)), "
                         f"input {fmt(s['input_rate'], '{:.1f} Hz')}, "
                         f"accepted {fmt(s['accepted_rate'], '{:.1f} Hz')}, "
                         f"live {fmt(100 * s['live_fraction'], '{:.1f} %')}"
                         + ("" if r.missed_available else "  (no missedTriggers in this file)"))
        else:
            lines.append("  trigger info: not in this file")
        oc = r.overflow_count()
        if oc is None:
            lines.append("  over range: not recorded in this file")
        else:
            lines.append("  over range: " + ", ".join(f"{ch} {k} ({100 * k / max(r.n_events, 1):.2f} %)"
                                                      for ch, k in oc.items()))
        if process:
            cfg = load_config(config if config is not None else r.config)
            proc = processors_from_config(cfg["processing"], r.channels, r.dt, r.trigger_index)
            m = min(process, r.n_events)
            feats = {ch: proc[ch].process_reader(r, ch, events=slice(0, m)) for ch in r.channels}
            good = np.ones(m, bool)
            for ch, f in feats.items():
                good &= quality_mask(f, r.samples)
            lines.append(f"  first {m} events: quality cut keeps {100 * good.mean():.1f} %")
            for ch, f in feats.items():
                a = f["amp"][good]
                lines.append(f"    {ch}: amp median {np.median(a) * 1e3:.1f} mV, "
                             f"99th pct {np.percentile(a, 99) * 1e3:.1f} mV, "
                             f"baseline RMS {np.median(f['bl_rms']) * 1e3:.2f} mV")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description="Summarise picodaq HDF5 files")
    p.add_argument("files", nargs="+")
    p.add_argument("--process", type=int, default=0, metavar="N",
                   help="also shape the first N events (processing settings from the file)")
    p.add_argument("--config", default=None, help="override processing settings with this config")
    a = p.parse_args(argv)
    for f in a.files:
        print(summarize(f, a.process, a.config))


if __name__ == "__main__":
    main()
