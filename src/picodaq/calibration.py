"""Energy calibration from known gamma-ray lines.

    points = measure_lines(amp_corr, "Ba133", approx_gain_kev_per_v=3000)
    points += measure_lines(amp_eu, "Eu152", approx_gain_kev_per_v=3000)
    cal = EnergyCalibration.fit(points, model="linear")
    energy = cal.to_keV(amp_corr)
    cal.write("calib.fits")

Line energies [keV] and emission probabilities [%] from the NNDC / LNHB
evaluations (rounded); only lines that a ~3 % FWHM detector separates are listed.
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np

from .spectrum import FWHM_PER_SIGMA, find_peaks, fit_peak, histogram

LINES = {
    "Ba133": [(81.00, 32.9), (276.40, 7.16), (302.85, 18.34), (356.01, 62.05), (383.85, 8.94)],
    "Eu152": [(121.78, 28.53), (244.70, 7.55), (344.28, 26.59), (778.90, 12.93), (964.08, 14.51),
              (1085.84, 10.11), (1112.08, 13.67), (1408.01, 20.87)],
    "Cs137": [(661.66, 85.1)],
}

MODELS = {"linear0": 1, "linear": 1, "quadratic": 2}  # polynomial degree in amplitude


@dataclass
class CalPoint:
    source: str
    energy: float  # keV
    amp: float  # V (depth corrected)
    amp_err: float
    sigma: float  # V
    sigma_err: float
    counts: float
    chi2_ndof: float = np.nan


def source_lines(source, energies=None):
    """[(E, intensity)] of a source, optionally restricted to the given energies."""
    lines = LINES[source]
    if energies is None:
        return list(lines)
    keep = [min(lines, key=lambda l: abs(l[0] - e)) for e in energies]
    return [l for l in lines if l in keep]


def guess_gain(amp, source, min_amp=0.0, bins=500):
    """keV/V from the most prominent peak = the most intense line of the source."""
    x, y, _ = histogram(amp, bins=bins)
    peaks, _ = find_peaks(x, y, min_x=min_amp)
    if not len(peaks):
        raise RuntimeError("no peak found to guess the gain")
    e_main = max(LINES[source], key=lambda l: l[1])[0]
    return e_main / peaks[0]


def measure_lines(amp, source, approx=None, energies=None, window_frac=0.06, min_window_kev=10.0,
                  bins=60, min_amp=0.0, min_counts=50, min_separation_kev=None):
    """Fit every listed line of `source` in the amplitude spectrum `amp` [V].

    approx : keV/V (float), anything with a to_amp(E) method (an EnergyCalibration,
             e.g. a first-pass result), or None (gain guessed from the strongest peak)
    Each line is searched in E +- max(window_frac * E, min_window_kev), narrowed to
    half the distance to the neighbouring line of the source. Of two lines closer
    than min_separation_kev (default: 2 x the window) only the more intense is used.
    Lines that fail or have fewer than min_counts counts are skipped.
    """
    amp = np.asarray(amp, float)
    amp = amp[np.isfinite(amp)]
    if approx is None:
        approx = guess_gain(amp, source, min_amp=min_amp)
    to_amp = approx.to_amp if hasattr(approx, "to_amp") else (lambda e: e / approx)
    lines = source_lines(source, energies)
    all_e = np.array([l[0] for l in LINES[source]])
    points = []
    for e, inten in lines:
        half_kev = max(window_frac * e, min_window_kev)
        sep = min_separation_kev if min_separation_kev is not None else half_kev
        others = [(oe, oi) for oe, oi in LINES[source] if oe != e and abs(oe - e) < sep]
        if any(oi > inten for _, oi in others):
            continue  # a stronger line is too close
        near = np.abs(all_e[all_e != e] - e)
        if near.size:
            half_kev = min(half_kev, max(near.min() / 2, 0.5 * min_window_kev))
        lo, hi = float(to_amp(e - half_kev)), float(to_amp(e + half_kev))
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            continue
        x, y, _ = histogram(amp, bins=bins, range=(lo, hi))
        try:
            f = fit_peak(x, y, lo, hi, sigma0=0.2 * (hi - lo) / 2)
        except RuntimeError:
            continue
        n = f.counts(x[1] - x[0])
        if n < min_counts:
            continue
        points.append(CalPoint(source, e, f.mu, f.mu_err, f.sigma, f.errors.get("sigma", np.nan),
                               n, f.chi2 / max(f.ndof, 1)))
    return points


def calibrate(spectra, model="linear", approx=None, passes=2, max_resid_fwhm=0.5, **kw):
    """Multi-source calibration in two passes.

    spectra : {source: depth-corrected amplitudes [V]}, e.g. {"Ba133": a_ba, "Cs137": a_cs}
    approx  : keV/V for the first pass (None = guessed from the strongest peak of the
              Cs137 or Ba133 spectrum and used for all sources)
    Pass 1 measures the lines around approx and fits; later passes re-measure around
    the fitted calibration. Points whose residual exceeds max_resid_fwhm x the median
    FWHM of all lines are dropped before the final fit. kw goes to measure_lines.
    """
    if approx is None:
        # only sources with one dominant line give a safe guess (Eu-152 has two
        # lines of similar intensity, 121.8 and 344.3 keV)
        src = next((s for s in ("Cs137", "Ba133") if s in spectra), None)
        if src is None:
            raise ValueError("give approx (keV/V): no Cs137 or Ba133 spectrum to guess it from")
        approx = guess_gain(spectra[src], src, min_amp=kw.get("min_amp", 0.0))
    cal = None
    for i in range(max(1, passes)):
        points = []
        for src, a in spectra.items():
            points += measure_lines(a, src, cal if cal is not None else approx, **kw)
        # the first pass is always linear: robust against a rough starting gain
        cal = EnergyCalibration.fit(points, model if (i > 0 or passes == 1) else
                                    ("linear0" if len(points) < 2 else "linear"))
    # a line fitted on the wrong peak has a large residual (and often a wide FWHM):
    # compare with the median FWHM of all lines
    keep = np.abs(cal.residuals()) <= max_resid_fwhm * np.median(cal.fwhm_keV())
    if not keep.all() and keep.sum() >= {"linear0": 1, "linear": 2, "quadratic": 3}[model]:
        cal = EnergyCalibration.fit([p for p, k in zip(cal.points, keep) if k], model)
        cal.note = f"dropped {int((~keep).sum())} outlier point(s)"
    return cal


@dataclass
class EnergyCalibration:
    """E [keV] = polynomial in amplitude [V].

    model: linear0 (E = c1 a), linear (c0 + c1 a), quadratic (c0 + c1 a + c2 a^2)
    coef : [c0, c1, c2] (c2 = 0 unless quadratic)
    """

    model: str
    coef: list
    points: list = field(default_factory=list)
    created: str = ""
    note: str = ""

    @classmethod
    def fit(cls, points, model="linear", note=""):
        if model not in MODELS:
            raise ValueError(f"model must be one of {list(MODELS)}")
        a = np.array([p.amp for p in points])
        e = np.array([p.energy for p in points])
        err = np.array([p.amp_err for p in points])
        need = {"linear0": 1, "linear": 2, "quadratic": 3}[model]
        if a.size < need:
            raise ValueError(f"{model} needs >= {need} points, got {a.size}")
        # weight by the amplitude error propagated with a first slope estimate
        slope = np.sum(a * e) / np.sum(a * a)
        w = 1.0 / np.maximum(np.nan_to_num(err, nan=0) * slope, 0.1)
        if model == "linear0":
            c1 = np.sum(w**2 * a * e) / np.sum(w**2 * a * a)
            coef = [0.0, float(c1), 0.0]
        else:
            deg = MODELS[model]
            c = np.polyfit(a, e, deg, w=w)[::-1]
            coef = [float(c[0]), float(c[1]), float(c[2]) if deg == 2 else 0.0]
        return cls(model, coef, list(points), datetime.now(timezone.utc).isoformat(timespec="seconds"), note)

    # ------------------------------------------------------------ apply
    def to_keV(self, amp):
        c0, c1, c2 = self.coef
        a = np.asarray(amp, float)
        return c0 + c1 * a + c2 * a * a

    def to_amp(self, energy):
        c0, c1, c2 = self.coef
        e = np.asarray(energy, float)
        if c2 == 0:
            return (e - c0) / c1
        disc = c1 * c1 - 4 * c2 * (c0 - e)
        with np.errstate(invalid="ignore"):
            return np.where(disc >= 0, (-c1 + np.sqrt(np.abs(disc))) / (2 * c2), np.nan)

    def gain(self, amp):
        """dE/da [keV / V] at amplitude amp."""
        return self.coef[1] + 2 * self.coef[2] * np.asarray(amp, float)

    def residuals(self):
        return np.array([self.to_keV(p.amp) - p.energy for p in self.points])

    def fwhm_keV(self):
        return np.array([FWHM_PER_SIGMA * p.sigma * self.gain(p.amp) for p in self.points])

    def summary(self):
        lines = [f"model {self.model}: E = {self.coef[0]:.3f} + {self.coef[1]:.2f} a"
                 + (f" + {self.coef[2]:.2f} a^2" if self.model == "quadratic" else "") + "  [keV, a in V]"]
        for p, r, fw in zip(self.points, self.residuals(), self.fwhm_keV()):
            lines.append(f"  {p.source:6s} {p.energy:8.2f} keV: a = {p.amp * 1e3:8.3f} mV, "
                         f"resid {r:+6.2f} keV, FWHM {fw:6.2f} keV ({100 * fw / p.energy:.2f} %)")
        return "\n".join(lines)

    # --------------------------------------------------------------- FITS
    def to_hdu(self):
        from astropy.io import fits
        from astropy.table import Table

        t = Table(rows=[asdict(p) for p in self.points] if self.points else None,
                  names=list(CalPoint.__dataclass_fields__))
        t.rename_columns(list(t.colnames), [c.upper() for c in t.colnames])
        t["RESID_KEV"] = self.residuals() if self.points else []
        t["FWHM_KEV"] = self.fwhm_keV() if self.points else []
        for c, u in (("ENERGY", "keV"), ("AMP", "V"), ("AMP_ERR", "V"), ("SIGMA", "V"),
                     ("SIGMA_ERR", "V"), ("RESID_KEV", "keV"), ("FWHM_KEV", "keV")):
            t[c].unit = u
        hdu = fits.table_to_hdu(t)
        hdu.name = "CALIB"
        h = hdu.header
        h["MODEL"] = (self.model, "E = C0 + C1 a + C2 a^2, a in V, E in keV")
        for i, c in enumerate(self.coef):
            h[f"C{i}"] = c
        h["CREATED"] = self.created
        if self.note:
            h["NOTE"] = self.note[:68]
        return hdu

    def write(self, path, overwrite=True):
        from astropy.io import fits

        fits.HDUList([fits.PrimaryHDU(), self.to_hdu()]).writeto(path, overwrite=overwrite)

    @classmethod
    def from_hdu(cls, hdu):
        from astropy.table import Table

        h = hdu.header
        t = Table(hdu.data)
        pts = [CalPoint(str(r["SOURCE"]).strip(), float(r["ENERGY"]), float(r["AMP"]),
                        float(r["AMP_ERR"]), float(r["SIGMA"]), float(r["SIGMA_ERR"]),
                        float(r["COUNTS"]), float(r["CHI2_NDOF"])) for r in t]
        return cls(h["MODEL"], [h["C0"], h["C1"], h["C2"]], pts, h.get("CREATED", ""), h.get("NOTE", ""))

    @classmethod
    def read(cls, path):
        from astropy.io import fits

        with fits.open(path) as f:
            return cls.from_hdu(f["CALIB"])
