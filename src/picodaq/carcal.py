"""Optional 2-D calibration: energy from the anode amplitude and the CAR together.

The default analysis is two steps: depth correction with one line (depth.py), then
an energy calibration of the corrected amplitude (calibration.py). This module
instead fits one model to the peak positions of EVERY line of EVERY source in
every CAR slice:

    a = off(c) + E * gain(c)  [+ q * E**2]

a : raw anode amplitude [V], c : CAR, E : energy [keV]
off(c), gain(c) : polynomials in c (degrees deg_off, deg_gain), q : a constant

Each line only has to cover the CAR range where its events are; the polynomials
join them smoothly. A depth-dependent offset or gain is absorbed. Per event the
model is inverted for E. The two-step result is only used to find the peaks.

    cc = CarCalibration.fit({"Ba133": (a_ba, car_ba), "Cs137": (a_cs, car_cs)}, dc, cal)
    energy = cc.to_keV(a, car)          # NaN outside the fitted CAR range
    cc.write("carcal.fits")
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np

from .calibration import measure_lines
from .spectrum import FWHM_PER_SIGMA


@dataclass
class CarPoint:
    source: str
    energy: float  # keV
    car: float  # mean CAR of the slice
    amp: float  # raw anode amplitude [V]
    amp_err: float
    sigma: float  # V
    counts: float
    used: bool = True


class _SliceApprox:
    """Where the two-step result expects a line in one CAR slice (raw amplitude)."""

    def __init__(self, dc, cal, car):
        self.dc, self.cal, self.car = dc, cal, car

    def to_amp(self, energy):
        return self.cal.to_amp(energy) / self.dc.factor(self.car)


def _poly(coef, c):
    return sum(k * np.asarray(c, float) ** i for i, k in enumerate(coef))


@dataclass
class CarCalibration:
    coef_off: list  # increasing powers of CAR
    coef_gain: list
    q: float
    car_lim: tuple
    points: list = field(default_factory=list)
    created: str = ""
    note: str = ""

    # --------------------------------------------------------------- fit
    @classmethod
    def fit(cls, data, dc, cal, car_range=None, car_bins=10, deg_off=1, deg_gain=3,
            quadratic=False, min_counts=30, max_resid_fwhm=0.5, **kw):
        """data : {source: (raw anode amplitude [V], CAR)} (GOOD events)
        dc, cal : two-step DepthCorrection and EnergyCalibration (starting point)
        car_range : default dc.car_lim. kw goes to calibration.measure_lines."""
        lo, hi = car_range or dc.car_lim
        edges = np.linspace(lo, hi, car_bins + 1)
        points = []
        for src, (a, car) in data.items():
            a, car = np.asarray(a, float), np.asarray(car, float)
            for c0, c1 in zip(edges[:-1], edges[1:]):
                m = np.isfinite(a) & (car >= c0) & (car < c1)
                if m.sum() < min_counts:
                    continue
                cm = float(car[m].mean())
                for p in measure_lines(a[m], src, _SliceApprox(dc, cal, cm), min_counts=min_counts, **kw):
                    points.append(CarPoint(src, p.energy, cm, p.amp, p.amp_err, p.sigma, p.counts))
        cc = cls._solve(points, deg_off, deg_gain, quadratic)
        # drop slice/line points that caught another peak, then refit
        r, fw = cc.residuals(), cc.fwhm_keV()
        bad = np.abs(r) > max_resid_fwhm * np.median(fw)
        if bad.any():
            for p, b in zip(points, bad):
                p.used = not b
            cc = cls._solve(points, deg_off, deg_gain, quadratic)
            cc.note = f"dropped {int(bad.sum())} of {len(points)} points"
        cc.car_lim = (float(min(p.car for p in points if p.used)), float(max(p.car for p in points if p.used)))
        return cc

    @classmethod
    def _solve(cls, points, deg_off, deg_gain, quadratic):
        use = [p for p in points if p.used]
        n_par = deg_off + 1 + deg_gain + 1 + int(quadratic)
        if len(use) < n_par + 1:
            raise RuntimeError(f"{len(use)} points for {n_par} parameters")
        c = np.array([p.car for p in use])
        e = np.array([p.energy for p in use])
        a = np.array([p.amp for p in use])
        err = np.array([p.amp_err for p in use])
        err = np.where(np.isfinite(err) & (err > 0), err, np.nanmedian(err))
        cols = [c**i for i in range(deg_off + 1)] + [e * c**i for i in range(deg_gain + 1)]
        if quadratic:
            cols.append(e**2)
        X = np.column_stack(cols) / err[:, None]
        sol, *_ = np.linalg.lstsq(X, a / err, rcond=None)
        off = list(map(float, sol[:deg_off + 1]))
        gain = list(map(float, sol[deg_off + 1:deg_off + deg_gain + 2]))
        q = float(sol[-1]) if quadratic else 0.0
        lim = (float(c.min()), float(c.max()))
        return cls(off, gain, q, lim, list(points), datetime.now(timezone.utc).isoformat(timespec="seconds"))

    # ------------------------------------------------------------- apply
    def off(self, car):
        return _poly(self.coef_off, car)

    def gain(self, car):
        """dA/dE at E = 0 [V / keV]."""
        return _poly(self.coef_gain, car)

    def in_range(self, car):
        car = np.asarray(car, float)
        return (car >= self.car_lim[0]) & (car <= self.car_lim[1])

    def to_amp(self, energy, car):
        e = np.asarray(energy, float)
        return self.off(car) + e * self.gain(car) + self.q * e * e

    def to_keV(self, amp, car):
        a, c = np.asarray(amp, float), np.asarray(car, float)
        g, d = self.gain(c), a - self.off(c)
        if self.q == 0:
            e = d / g
        else:
            disc = g * g + 4 * self.q * d
            with np.errstate(invalid="ignore"):
                e = np.where(disc >= 0, (-g + np.sqrt(np.abs(disc))) / (2 * self.q), np.nan)
        return np.where(self.in_range(c), e, np.nan)

    # ------------------------------------------------------------ report
    def residuals(self):
        """keV, for every point (used or not)."""
        return np.array([float(self.to_keV_unclipped(p.amp, p.car)) - p.energy for p in self.points])

    def to_keV_unclipped(self, amp, car):
        g, d = self.gain(car), amp - self.off(car)
        return d / g if self.q == 0 else (-g + np.sqrt(g * g + 4 * self.q * d)) / (2 * self.q)

    def fwhm_keV(self):
        return np.array([FWHM_PER_SIGMA * p.sigma / (self.gain(p.car) + 2 * self.q * p.energy)
                         for p in self.points])

    def summary(self):
        r = self.residuals()
        lines = [f"2-D calibration: a = off(CAR) + E gain(CAR)" + (" + q E^2" if self.q else "")
                 + f", CAR {self.car_lim[0]:.2f}-{self.car_lim[1]:.2f}, "
                 f"{sum(p.used for p in self.points)} of {len(self.points)} points used"]
        keys = sorted({(p.source, p.energy) for p in self.points}, key=lambda k: k[1])
        for src, e in keys:
            idx = [i for i, p in enumerate(self.points) if (p.source, p.energy) == (src, e) and p.used]
            if idx:
                lines.append(f"  {src:6s} {e:8.2f} keV: {len(idx):2d} CAR slices, "
                             f"rms resid {np.sqrt(np.mean(r[idx] ** 2)):5.2f} keV")
        return "\n".join(lines)

    # -------------------------------------------------------------- FITS
    def to_hdu(self):
        from astropy.io import fits
        from astropy.table import Table

        t = Table(rows=[asdict(p) for p in self.points], names=list(CarPoint.__dataclass_fields__))
        t.rename_columns(list(t.colnames), [c.upper() for c in t.colnames])
        t["RESID_KEV"] = self.residuals()
        for col, u in (("ENERGY", "keV"), ("AMP", "V"), ("AMP_ERR", "V"), ("SIGMA", "V"), ("RESID_KEV", "keV")):
            t[col].unit = u
        hdu = fits.table_to_hdu(t)
        hdu.name = "CARCAL"
        h = hdu.header
        h["MODEL"] = ("a = sum OFFi c^i + E sum GAINi c^i + Q E^2", "a V, E keV, c CAR")
        h["DEG_OFF"], h["DEG_GAIN"] = len(self.coef_off) - 1, len(self.coef_gain) - 1
        for i, k in enumerate(self.coef_off):
            h[f"OFF{i}"] = k
        for i, k in enumerate(self.coef_gain):
            h[f"GAIN{i}"] = k
        h["Q"] = self.q
        h["CARLO"], h["CARHI"] = self.car_lim
        h["CREATED"] = self.created
        if self.note:
            h["NOTE"] = self.note[:68]
        return hdu

    def write(self, path, overwrite=True):
        from astropy.io import fits

        fits.HDUList([fits.PrimaryHDU(), self.to_hdu()]).writeto(path, overwrite=overwrite)

    @classmethod
    def from_hdu(cls, hdu):
        h, d = hdu.header, hdu.data
        pts = [CarPoint(str(r["SOURCE"]).strip(), float(r["ENERGY"]), float(r["CAR"]), float(r["AMP"]),
                        float(r["AMP_ERR"]), float(r["SIGMA"]), float(r["COUNTS"]), bool(r["USED"]))
               for r in d]
        return cls([h[f"OFF{i}"] for i in range(h["DEG_OFF"] + 1)],
                   [h[f"GAIN{i}"] for i in range(h["DEG_GAIN"] + 1)], h["Q"],
                   (h["CARLO"], h["CARHI"]), pts, h.get("CREATED", ""), h.get("NOTE", ""))

    @classmethod
    def read(cls, path):
        from astropy.io import fits

        with fits.open(path) as f:
            return cls.from_hdu(f["CARCAL"])


def line_table(energies):
    """Measure every known line in calibrated spectra {source: energies [keV]}:
    -> list of (source, E, centroid keV, FWHM keV). For comparing methods."""
    rows = []
    for src, e in energies.items():
        for p in measure_lines(e, src, approx=1.0):
            rows.append((src, p.energy, p.amp, FWHM_PER_SIGMA * p.sigma))
    return rows
