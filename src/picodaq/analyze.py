"""Offline analysis CLI: waveforms -> events -> depth correction -> energy.

    picodaq-analyze tau     run_Ba/data/tlbr_00000.h5            # pole-zero tau, rise times
    picodaq-analyze events  run_Ba/data -o ba_events.fits        # shape all waveforms
    picodaq-analyze depth   ba_events.fits -o depth.fits --window 0.10 0.135 --plot depth.png
    picodaq-analyze calib   ba_events.fits:Ba133 eu_events.fits:Eu152 --depth depth.fits \\
                            -o calib.fits --plot calib.png
    picodaq-analyze energy  ba_events.fits eu_events.fits --depth depth.fits --calib calib.fits
    picodaq-analyze run     analysis.yaml                        # all of the above
                            (calibration.method: car2d adds the optional 2-D calibration)

Settings come from analysis.yaml (--analysis) with command-line options on top.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

from .config import deep_merge, load_config

ANALYSIS_DEFAULTS = {
    "data_root": None,  # runs are relative to this (itself relative to the yaml file)
    "output_dir": "analysis",
    "runs": {},
    "processing": None,  # None: the settings stored in the first HDF5 file; dict: overrides
    "max_events": None,
    "cuts": {"bl_rms_max": 0.010, "drift_max": 0.050, "late_max": 0.050},
    "depth": {"source": None, "amp_window": None, "car_range": [0.25, 1.40], "car_bins": 12,
              "car_ref": 1.0, "degree": 3},
    "calibration": {"model": "linear", "approx_kev_per_v": None, "min_amp": 0.015,
                    "window_frac": 0.06, "min_window_kev": 10.0, "lines": {},
                    "method": "two_step",  # two_step | car2d (optional 2-D calibration)
                    "car2d": {"car_bins": 10, "deg_off": 1, "deg_gain": 3, "quadratic": False}},
}


def _resolve(p, base):
    p = Path(os.path.expandvars(str(p))).expanduser()
    return Path(os.path.normpath(p if p.is_absolute() else base / p))


def load_analysis(path=None):
    """Read analysis.yaml. Relative paths are taken relative to the yaml file:
    data_root (environment variable PICODAQ_DATA overrides it), the runs (relative
    to data_root) and output_dir. So the repository can be cloned anywhere."""
    user = {}
    base = Path.cwd()
    if path:
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        base = Path(path).resolve().parent
    acfg = deep_merge(ANALYSIS_DEFAULTS, user)
    root = os.environ.get("PICODAQ_DATA") or acfg.get("data_root")
    root = _resolve(root, base) if root else base
    acfg["data_root"] = str(root)
    acfg["runs"] = {src: [str(_resolve(p, root)) for p in (ps if isinstance(ps, (list, tuple)) else [ps])]
                    for src, ps in acfg["runs"].items()}
    acfg["output_dir"] = str(_resolve(acfg["output_dir"], base))
    return acfg


def _processing(acfg, first_file):
    """Processing settings: those of the first file, with analysis.yaml overrides."""
    from .io import WaveformReader

    with WaveformReader(first_file) as r:
        proc = load_config(r.config)["processing"]
    return deep_merge(proc, acfg["processing"] or {})


def _savefig(fig, path):
    if path:
        fig.savefig(path, dpi=120)
        print(f"wrote {path}")


# ------------------------------------------------------------------ steps
def step_tau(files, acfg, n_events=2000):
    from .io import WaveformReader
    from .shaping import ChannelProcessor, TrapezoidShaper, fit_tail_tau, rise_time, tune_tau

    for f in files:
        with WaveformReader(f) as r:
            proc = deep_merge(load_config(r.config)["processing"], acfg["processing"] or {})
            sh = proc["shaper"]
            print(f"{Path(f).name}: {min(n_events, r.n_events)} events, rise {sh['rise_us']} us, "
                  f"flat {sh['flat_us']} us")
            for ch in r.channels:
                cp = ChannelProcessor.from_config(proc, ch, r.dt, r.trigger_index)
                v, _ = cp.correct(r.waveforms(ch, slice(0, n_events)))
                shaper = TrapezoidShaper.from_time(r.dt, sh["rise_us"] * 1e-6, sh["flat_us"] * 1e-6, 1.0)
                taus = np.geomspace(5e-6, 1000e-6, 200) / r.dt
                best, _ = tune_tau(v, shaper, taus, r.trigger_index)
                old, _ = tune_tau(v, shaper, taus, r.trigger_index, skip=max(5, int(0.1 * shaper.flat)))
                sel = np.argsort(v[:, r.trigger_index:r.trigger_index + shaper.rise].mean(1))[-60:]
                m = v[sel].mean(0)
                print(f"  {ch}: tau {best * r.dt * 1e6:.1f} us (notebook window {old * r.dt * 1e6:.1f} us, "
                      f"tail fit {fit_tail_tau(m, r.time, r.trigger_index) * 1e6:.1f} us), "
                      f"10-90 % rise {rise_time(m, r.dt, r.trigger_index) * 1e6:.1f} us, "
                      f"config tau {proc['channels'].get(ch, {}).get('tau_us')} us")


def step_events(paths, out, acfg):
    from .events import EventFile, _expand

    files = _expand(paths)
    proc = _processing(acfg, files[0])
    ef = EventFile.from_waveforms(files, processing=proc, cuts=acfg["cuts"], max_events=acfg["max_events"])
    ef.write(out)
    print(f"wrote {out}: {len(ef)} events, good {100 * np.mean(ef.events['GOOD']):.1f} %")
    return ef


def step_depth(ef, out, acfg, plot=None):
    from .depth import DepthCorrection

    d = acfg["depth"]
    if not d["amp_window"]:
        raise SystemExit("depth: give amp_window (V) around one strong line, e.g. --window 0.10 0.135")
    dc = DepthCorrection.fit(ef.amp("anode"), ef.column("CAR"), tuple(d["amp_window"]),
                             tuple(d["car_range"]), d["car_bins"], d["car_ref"], d["degree"])
    dc.write(out)
    print(f"wrote {out}: CAR {dc.car_lim[0]:.2f}-{dc.car_lim[1]:.2f}, "
          f"{int(np.isfinite(dc.peak).sum())} slices, peak at ref {dc.amp_ref * 1e3:.2f} mV")
    if plot:
        from .plots import depth_figure

        _savefig(depth_figure(ef, dc), plot)
    return dc


def step_calib(efs, dc, out, acfg, plot=None):
    from .calibration import calibrate

    c = acfg["calibration"]
    spectra = {}
    for src, ef in efs.items():
        ef.apply_depth(dc)
        spectra[src] = ef.selected("AMP_CORR")
    kw = {"min_amp": c["min_amp"], "window_frac": c["window_frac"], "min_window_kev": c["min_window_kev"]}
    cal = calibrate(spectra, c["model"], c["approx_kev_per_v"], **kw)
    if c["lines"]:  # restrict lines per source, then refit around the result
        from .calibration import EnergyCalibration, measure_lines

        pts = []
        for src, a in spectra.items():
            pts += measure_lines(a, src, cal, energies=c["lines"].get(src), **kw)
        cal = EnergyCalibration.fit(pts, c["model"])
    cal.write(out)
    print(f"wrote {out}\n{cal.summary()}" + (f"\n  ({cal.note})" if cal.note else ""))
    if plot:
        from .plots import calibration_figure

        _savefig(calibration_figure(cal), plot)
    return cal


def step_energy(efs, dc, cal, plot=None):
    for src, ef in efs.items():
        ef.apply_depth(dc)
        ef.apply_calibration(cal)
    if plot:
        from .plots import spectra_figure

        _savefig(spectra_figure(efs), plot)


def step_car2d(efs, dc, cal, out, acfg):
    """Optional 2-D calibration; ENERGY is replaced, and the two methods are compared
    line by line (centroid and FWHM in the calibrated spectra)."""
    from .carcal import CarCalibration, line_table
    from .plots import carcal_figure

    c = acfg["calibration"]
    two = {src: ef.selected("ENERGY") for src, ef in efs.items()}  # two-step energies
    data = {src: (ef.amp("anode"), ef.column("CAR")) for src, ef in efs.items()}
    cc = CarCalibration.fit(data, dc, cal, min_amp=c["min_amp"], window_frac=c["window_frac"],
                            min_window_kev=c["min_window_kev"], **c["car2d"])
    cc.write(out / "carcal.fits")
    _savefig(carcal_figure(cc), out / "carcal.png")
    for ef in efs.values():
        ef.apply_carcal(cc)
    new = {src: ef.selected("ENERGY") for src, ef in efs.items()}
    a = {(s, e): (m, f) for s, e, m, f in line_table(two)}
    b = {(s, e): (m, f) for s, e, m, f in line_table(new)}
    rows = [cc.summary(), "", "line              two-step: centroid  FWHM      2-D: centroid  FWHM   [keV]"]
    for k in sorted(set(a) | set(b), key=lambda k: k[1]):
        fa = "{:9.2f} {:6.2f}".format(*a[k]) if k in a else " " * 16
        fb = "{:9.2f} {:6.2f}".format(*b[k]) if k in b else ""
        rows.append(f"{k[0]:6s} {k[1]:8.2f}        {fa}        {fb}")
    text = "\n".join(rows)
    print(text)
    from .plots import spectra_figure

    _savefig(spectra_figure(efs), out / "spectra_car2d.png")
    return text


def run_all(acfg, force=False):
    out = Path(acfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    from .events import EventFile

    if not acfg["runs"]:
        raise SystemExit("analysis.yaml: no runs given")
    efs = {}
    for src, paths in acfg["runs"].items():
        fn = out / f"{src}_events.fits"
        if fn.exists() and not force:
            efs[src] = EventFile.read(fn)
            print(f"using {fn} ({len(efs[src])} events; --force to rebuild)")
        else:
            efs[src] = step_events(paths, fn, acfg)
    dsrc = acfg["depth"]["source"] or next(iter(efs))
    dc = step_depth(efs[dsrc], out / "depth.fits", acfg, out / "depth.png")
    cal = step_calib(efs, dc, out / "calib.fits", acfg, out / "calib.png")
    step_energy(efs, dc, cal, out / "spectra.png")
    summary = cal.summary() + "\n"
    if acfg["calibration"]["method"] == "car2d":
        summary += "\n" + step_car2d(efs, dc, cal, out, acfg) + "\n"
    for src, ef in efs.items():
        ef.write(out / f"{src}_events.fits")
        print(f"wrote {out / f'{src}_events.fits'} (with AMP_CORR, ENERGY)")
    (out / "summary.txt").write_text(summary)
    with open(out / "analysis_used.yaml", "w") as f:
        yaml.safe_dump(acfg, f, sort_keys=False)


# -------------------------------------------------------------------- CLI
def main(argv=None):
    import matplotlib

    matplotlib.use("Agg")
    p = argparse.ArgumentParser(prog="picodaq-analyze", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis", default=None, help="analysis.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("tau", help="pole-zero tau and rise times from waveforms")
    s.add_argument("files", nargs="+")
    s.add_argument("--events", type=int, default=2000)

    s = sub.add_parser("events", help="shape waveforms into an event FITS file")
    s.add_argument("paths", nargs="+", help="HDF5 files or directories")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--max-events", type=int, default=None, help="per file")

    s = sub.add_parser("depth", help="fit the CAR depth correction")
    s.add_argument("events")
    s.add_argument("-o", "--output", default="depth.fits")
    s.add_argument("--window", nargs=2, type=float, metavar=("LO", "HI"), help="V, around one strong line")
    s.add_argument("--plot", default=None)

    s = sub.add_parser("calib", help="energy calibration from known sources")
    s.add_argument("inputs", nargs="+", metavar="EVENTS:SOURCE", help="e.g. ba.fits:Ba133")
    s.add_argument("--depth", required=True)
    s.add_argument("-o", "--output", default="calib.fits")
    s.add_argument("--model", choices=["linear0", "linear", "quadratic"])
    s.add_argument("--approx", type=float, help="starting gain keV/V")
    s.add_argument("--plot", default=None)

    s = sub.add_parser("energy", help="add AMP_CORR and ENERGY columns to event files")
    s.add_argument("events", nargs="+")
    s.add_argument("--depth", required=True)
    s.add_argument("--calib", required=True)
    s.add_argument("--plot", default=None)

    s = sub.add_parser("run", help="everything from analysis.yaml")
    s.add_argument("config", help="analysis.yaml")
    s.add_argument("--force", action="store_true", help="rebuild event files")

    a = p.parse_args(argv)
    acfg = load_analysis(a.config if a.cmd == "run" else a.analysis)

    from .calibration import EnergyCalibration
    from .depth import DepthCorrection
    from .events import EventFile

    if a.cmd == "tau":
        step_tau(a.files, acfg, a.events)
    elif a.cmd == "events":
        if a.max_events:
            acfg["max_events"] = a.max_events
        step_events(a.paths, a.output, acfg)
    elif a.cmd == "depth":
        if a.window:
            acfg["depth"]["amp_window"] = a.window
        step_depth(EventFile.read(a.events), a.output, acfg, a.plot)
    elif a.cmd == "calib":
        if a.model:
            acfg["calibration"]["model"] = a.model
        if a.approx:
            acfg["calibration"]["approx_kev_per_v"] = a.approx
        efs = {}
        for item in a.inputs:
            fn, _, src = item.rpartition(":")
            if not fn:
                raise SystemExit(f"{item}: use EVENTS:SOURCE, e.g. ba.fits:Ba133")
            efs[src] = EventFile.read(fn)
        step_calib(efs, DepthCorrection.read(a.depth), a.output, acfg, a.plot)
    elif a.cmd == "energy":
        dc, cal = DepthCorrection.read(a.depth), EnergyCalibration.read(a.calib)
        efs = {Path(fn).stem: EventFile.read(fn) for fn in a.events}
        step_energy(efs, dc, cal, a.plot)
        for fn, ef in zip(a.events, efs.values()):
            ef.write(fn)
            print(f"updated {fn}")
    elif a.cmd == "run":
        run_all(acfg, a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
