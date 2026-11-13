"""Event file: one row per recorded waveform, stored as FITS.

    ef = EventFile.from_waveforms(["data/20260909/run_Ba/data/tlbr_00000.h5"])
    ef.apply_depth(DepthCorrection.read("depth.fits"))
    ef.apply_calibration(EnergyCalibration.read("calib.fits"))
    ef.write("ba_events.fits")

FITS layout
    EVENTS  FILE_ID EVENT_ID TIME EPOCH, per channel <X>: AMP_X BASE_X BLRMS_X
            BLSLOPE_X VMAX_X LATE_X, then GOOD CAR (+ AMP_CORR DEPTH_OK ENERGY)
            header: processing and cut parameters (+ the processing YAML as HISTORY)
    FILES   one row per HDF5 file: name, events, real/live time, rates
    DEPTH   depth correction used for AMP_CORR (if applied)
    CALIB   energy calibration used for ENERGY (if applied)
    CARCAL  optional 2-D calibration used for ENERGY instead (carcal.py)
Needs astropy (pip install picodaq[analysis]).
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from .calibration import EnergyCalibration
from .carcal import CarCalibration
from .config import load_config
from .depth import DepthCorrection
from .io import WaveformReader
from .shaping import processors_from_config, quality_mask

FEATURE_COLUMNS = {"amp": ("AMP", "V"), "baseline": ("BASE", "V"), "bl_rms": ("BLRMS", "V"),
                   "bl_slope": ("BLSLOPE", None), "vmax": ("VMAX", "V"), "late": ("LATE", "V")}
DEFAULT_CUTS = {"bl_rms_max": 0.010, "drift_max": 0.050, "late_max": 0.050}


def _expand(paths):
    out = []
    for p in map(Path, paths if isinstance(paths, (list, tuple)) else [paths]):
        if p.is_dir():
            out += sorted(q for q in p.glob("*.h5") if not q.name.startswith("._"))
        else:
            out.append(p)
    return out


@dataclass
class EventFile:
    events: object  # astropy Table
    files: object  # astropy Table
    meta: dict = field(default_factory=dict)
    depth: DepthCorrection = None
    calib: EnergyCalibration = None
    carcal: CarCalibration = None

    # ------------------------------------------------------------ build
    @classmethod
    def from_waveforms(cls, paths, processing=None, cuts=None, chunk=256, max_events=None,
                       verbose=True):
        """Shape every waveform of the given HDF5 files / directories.

        processing : 'processing' dict (config section) applied to ALL files;
                     None = the one stored in the first file (defaults for old files)
        cuts       : quality cuts for GOOD (shaping.quality_mask arguments)
        max_events : per file (for quick looks)
        """
        from astropy.table import Table, vstack

        files = _expand(paths)
        if not files:
            raise FileNotFoundError(f"no HDF5 files in {paths}")
        cuts = {**DEFAULT_CUTS, **(cuts or {})}
        parts, rows, samples = [], [], None
        for fid, path in enumerate(files):
            with WaveformReader(path) as r:
                samples = samples or r.samples
                if processing is None:
                    processing = load_config(r.config)["processing"]
                anode, cathode = processing["anode"], processing["cathode"]
                proc = processors_from_config(processing, r.channels, r.dt, r.trigger_index)
                ev = slice(0, min(max_events, r.n_events)) if max_events else None
                feats = {ch: proc[ch].process_reader(r, ch, chunk=chunk, events=ev) for ch in r.channels}
                n = len(feats[r.channels[0]]["amp"])
                t = Table()
                t["FILE_ID"] = np.full(n, fid, np.int16)
                t["EVENT_ID"] = np.arange(n, dtype=np.int32)
                times = r.event_times()
                t["TIME"] = times[0][:n] if times is not None else np.full(n, np.nan)
                t["TIME"].unit = "s"
                t["EPOCH"] = (times[1][:n] if times is not None else np.zeros(n)).astype(np.int16)
                good = np.ones(n, bool)
                for ch in r.channels:
                    for key, (name, unit) in FEATURE_COLUMNS.items():
                        t[f"{name}_{ch}"] = feats[ch][key].astype(np.float32)
                        if unit:
                            t[f"{name}_{ch}"].unit = unit
                    good &= quality_mask(feats[ch], r.samples, **cuts)
                t["GOOD"] = good
                a, c = feats[anode]["amp"], feats[cathode]["amp"]
                with np.errstate(divide="ignore", invalid="ignore"):
                    t["CAR"] = np.where(a > 0, c / a, np.nan).astype(np.float32)
                parts.append(t)
                s = r.rate_summary() or {}
                rows.append({"FILE_ID": fid, "FILENAME": str(path), "N_EVENTS": n,
                             "N_IN_FILE": r.n_events, "RUN_MODE": str(r.run_mode),
                             "SERIAL": str(r.serial), "DT": r.dt,
                             "REAL_TIME": s.get("real_time_s", np.nan),
                             "LIVE_TIME": s.get("live_time_s", np.nan),
                             "INPUT_RATE": s.get("input_rate", np.nan),
                             "LIVE_FRAC": s.get("live_fraction", np.nan)})
                if verbose:
                    print(f"{path.name}: {n} events, good {100 * good.mean():.1f} %", flush=True)
        events = vstack(parts, metadata_conflicts="silent")
        ftab = Table(rows=rows)
        for c, u in (("DT", "s"), ("REAL_TIME", "s"), ("LIVE_TIME", "s"), ("INPUT_RATE", "Hz")):
            ftab[c].unit = u
        meta = {"anode": processing["anode"], "cathode": processing["cathode"],
                "processing": processing, "cuts": cuts, "samples": samples,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        return cls(events, ftab, meta)

    # ------------------------------------------------------------ access
    @property
    def anode(self):
        return self.meta["anode"]

    @property
    def cathode(self):
        return self.meta["cathode"]

    def __len__(self):
        return len(self.events)

    def column(self, name, good=True):
        x = np.asarray(self.events[name])
        return x[np.asarray(self.events["GOOD"])] if good else x

    def amp(self, which="anode", good=True):
        ch = {"anode": self.anode, "cathode": self.cathode}.get(which, which)
        return self.column(f"AMP_{ch}", good)

    def recompute_good(self, **cuts):
        """Re-apply the quality cuts with new limits (no reshaping needed)."""
        cuts = {**self.meta["cuts"], **cuts}
        good = np.ones(len(self.events), bool)
        for ch in (self.anode, self.cathode):
            f = {k: np.asarray(self.events[f"{n}_{ch}"]) for k, (n, _) in FEATURE_COLUMNS.items()}
            good &= quality_mask(f, int(self.meta["samples"]), **cuts)
        self.events["GOOD"] = good
        self.meta["cuts"] = cuts
        return good

    # -------------------------------------------------- correction steps
    def apply_depth(self, dc):
        self.depth = dc
        a = np.asarray(self.events[f"AMP_{self.anode}"], float)
        car = np.asarray(self.events["CAR"], float)
        self.events["AMP_CORR"] = dc.apply(a, car).astype(np.float32)
        self.events["AMP_CORR"].unit = "V"
        self.events["DEPTH_OK"] = dc.in_range(car)
        return self.events["AMP_CORR"]

    def apply_calibration(self, cal):
        if "AMP_CORR" not in self.events.colnames:
            raise RuntimeError("apply_depth() first: the calibration is for the depth-corrected anode")
        self.calib = cal
        self.events["ENERGY"] = cal.to_keV(np.asarray(self.events["AMP_CORR"], float)).astype(np.float32)
        self.events["ENERGY"].unit = "keV"
        return self.events["ENERGY"]

    def apply_carcal(self, cc):
        """Optional 2-D calibration: ENERGY from the raw anode amplitude and CAR
        (replaces ENERGY of apply_calibration; DEPTH_OK = inside its CAR range)."""
        self.carcal = cc
        a = np.asarray(self.events[f"AMP_{self.anode}"], float)
        car = np.asarray(self.events["CAR"], float)
        self.events["ENERGY"] = cc.to_keV(a, car).astype(np.float32)
        self.events["ENERGY"].unit = "keV"
        self.events["DEPTH_OK"] = cc.in_range(car)
        return self.events["ENERGY"]

    def selected(self, column="ENERGY"):
        """Values of `column` for GOOD events inside the depth-correction range."""
        m = np.asarray(self.events["GOOD"])
        if "DEPTH_OK" in self.events.colnames:
            m &= np.asarray(self.events["DEPTH_OK"])
        x = np.asarray(self.events[column], float)[m]
        return x[np.isfinite(x)]

    # --------------------------------------------------------------- FITS
    def _header_meta(self):
        p, c = self.meta["processing"], self.meta["cuts"]
        m = {"ANODE": self.anode, "CATHODE": self.cathode,
             "RISE_US": p["shaper"]["rise_us"], "FLAT_US": p["shaper"]["flat_us"],
             "AMPMETH": p.get("amplitude", "flattop"), "BLFRAC": p.get("baseline_fraction", 0.8),
             "SAMPLES": int(self.meta["samples"]), "BLSLOPEU": "V/sample", "BLRMSMAX": c["bl_rms_max"], "DRIFTMAX": c["drift_max"], "LATEMAX": c["late_max"],
             "CREATED": self.meta.get("created", "")}
        for ch, v in p["channels"].items():
            m[f"TAU_{ch}"] = v.get("tau_us", np.nan)
            m[f"POL_{ch}"] = v.get("polarity", 1)
        m["HISTORY"] = ["processing:"] + ["  " + l for l in yaml.safe_dump(p, sort_keys=False).splitlines()]
        return m

    def write(self, path, overwrite=True):
        from astropy.io import fits

        ev = self.events.copy(copy_data=False)
        ev.meta = self._header_meta()
        hdus = [fits.PrimaryHDU(), fits.table_to_hdu(ev), fits.table_to_hdu(self.files)]
        hdus[1].name, hdus[2].name = "EVENTS", "FILES"
        hdus[0].header["CREATOR"] = "picodaq"
        if self.depth is not None:
            hdus.append(self.depth.to_hdu())
        if self.calib is not None:
            hdus.append(self.calib.to_hdu())
        if self.carcal is not None:
            hdus.append(self.carcal.to_hdu())
        fits.HDUList(hdus).writeto(path, overwrite=overwrite)

    @classmethod
    def read(cls, path):
        from astropy.io import fits
        from astropy.table import Table

        with fits.open(path) as f:
            events = Table(f["EVENTS"].data)
            files = Table(f["FILES"].data)
            h = f["EVENTS"].header
            for hdu_name, tab in (("EVENTS", events), ("FILES", files)):
                hh = f[hdu_name].header
                for i, name in enumerate(tab.colnames, start=1):
                    if f"TUNIT{i}" in hh:
                        tab[name].unit = hh[f"TUNIT{i}"]
            hist = [str(x) for x in h.get("HISTORY", [])]
            processing = yaml.safe_load("\n".join(l[2:] for l in hist[1:])) if hist[:1] == ["processing:"] else {}
            meta = {"anode": h["ANODE"], "cathode": h["CATHODE"], "processing": processing,
                    "cuts": {"bl_rms_max": h["BLRMSMAX"], "drift_max": h["DRIFTMAX"], "late_max": h["LATEMAX"]},
                    "samples": h["SAMPLES"], "created": h.get("CREATED", "")}
            depth = DepthCorrection.from_hdu(f["DEPTH"]) if "DEPTH" in f else None
            calib = EnergyCalibration.from_hdu(f["CALIB"]) if "CALIB" in f else None
            carcal = CarCalibration.from_hdu(f["CARCAL"]) if "CARCAL" in f else None
        return cls(events, files, meta, depth, calib, carcal)
