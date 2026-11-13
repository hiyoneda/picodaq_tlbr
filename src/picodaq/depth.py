"""Depth correction of the anode pulse height with the cathode/anode ratio (CAR).

In planar TlBr the holes are hardly collected, so the anode signal (mostly the
electron drift) depends on the interaction depth; CAR is the depth indicator.
The photopeak position of one strong line is measured in CAR slices, a
polynomial is fitted, and the anode amplitude is scaled to its value at CAR_REF:

    dc = DepthCorrection.fit(amp_a, car, amp_window=(0.10, 0.13))   # e.g. Ba-133 356 keV
    amp_corr = dc.apply(amp_a, car)          # NaN outside the fitted CAR range
    dc.write("depth.fits")
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .spectrum import fit_peak


@dataclass
class DepthCorrection:
    coef: list  # np.polyval order (highest power first)
    car_lim: tuple  # CAR range where the correction is valid
    car_ref: float = 1.0
    car: np.ndarray = field(default_factory=lambda: np.array([]))  # slice centres
    peak: np.ndarray = field(default_factory=lambda: np.array([]))  # fitted peak [V]
    peak_err: np.ndarray = field(default_factory=lambda: np.array([]))
    amp_window: tuple = (np.nan, np.nan)
    created: str = ""

    @property
    def amp_ref(self):
        return float(np.polyval(self.coef, self.car_ref))

    @classmethod
    def fit(cls, amp, car, amp_window, car_range=(0.25, 1.40), car_bins=12, car_ref=1.0,
            degree=3, bins=30, min_events=25, fit_frac=(0.08, 0.06), clip_frac=0.03):
        """Fit the photopeak position vs CAR.

        amp_window : (lo, hi) [V] where the line lies in every CAR slice; it may be
                     wide: in each slice the strongest peak inside it is taken and
                     fitted in [peak x (1 - fit_frac[0]), peak x (1 + fit_frac[1])],
                     so neighbouring weaker lines (Ba-133 303 / 384 keV) stay out.
        clip_frac  : slices more than this fraction off the curve fitted without them
                     are rejected (one at a time)
        """
        amp, car = np.asarray(amp, float), np.asarray(car, float)
        edges = np.linspace(*car_range, car_bins + 1)
        centres = 0.5 * (edges[1:] + edges[:-1])
        pos = np.full(car_bins, np.nan)
        err = np.full(car_bins, np.nan)
        lo, hi = amp_window
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            x = amp[(car >= a) & (car < b) & (amp >= lo) & (amp <= hi)]
            if x.size < min_events:
                continue
            counts, e = np.histogram(x, bins=2 * bins, range=amp_window)
            c = 0.5 * (e[1:] + e[:-1])
            m0 = c[int(np.argmax(np.convolve(counts, np.ones(3) / 3, "same")))]
            flo, fhi = m0 * (1 - fit_frac[0]), m0 * (1 + fit_frac[1])
            counts, e = np.histogram(x, bins=bins, range=(flo, fhi))
            c = 0.5 * (e[1:] + e[:-1])
            try:
                f = fit_peak(c, counts, flo, fhi, mu0=m0, sigma0=0.2 * (fhi - flo) / 2)
            except RuntimeError:
                continue
            pos[i], err[i] = f.mu, f.mu_err
        ok = np.isfinite(pos)
        # robust polynomial: a slice whose fit picked another peak sits far from the
        # curve fitted WITHOUT it (leave-one-out); drop the worst one while it is
        # off by more than clip_frac, then refit
        def _fit(m):
            w = 1 / np.where(np.isfinite(err[m]) & (err[m] > 0), err[m], np.nanmedian(err[m]))
            return np.polyfit(centres[m], pos[m], degree, w=w)

        while True:
            if ok.sum() < degree + 1:
                raise RuntimeError(f"only {ok.sum()} CAR slices fitted; need {degree + 1}")
            if ok.sum() < degree + 3:
                break
            loo = np.full(car_bins, 0.0)
            for i in np.flatnonzero(ok):
                m = ok.copy()
                m[i] = False
                loo[i] = abs(pos[i] - np.polyval(_fit(m), centres[i])) / pos[i]
            worst = int(np.argmax(loo))
            if loo[worst] <= clip_frac:
                break
            ok[worst] = False
            pos[worst] = np.nan  # rejected slices stay in the table as NaN
        coef = _fit(ok)
        return cls(list(map(float, coef)), (float(centres[ok].min()), float(centres[ok].max())),
                   float(car_ref), centres, pos, err, tuple(amp_window),
                   datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def in_range(self, car):
        car = np.asarray(car, float)
        return (car >= self.car_lim[0]) & (car <= self.car_lim[1])

    def factor(self, car):
        c = np.clip(np.asarray(car, float), *self.car_lim)
        return self.amp_ref / np.polyval(self.coef, c)

    def apply(self, amp, car, outside="nan"):
        """Corrected amplitude. outside: 'nan' (default) or 'clip' (notebook behaviour)."""
        out = np.asarray(amp, float) * self.factor(car)
        if outside == "nan":
            out = np.where(self.in_range(car), out, np.nan)
        return out

    # --------------------------------------------------------------- FITS
    def to_hdu(self):
        from astropy.io import fits
        from astropy.table import Table

        t = Table({"CAR": self.car, "PEAK": self.peak, "PEAK_ERR": self.peak_err})
        t["PEAK"].unit = t["PEAK_ERR"].unit = "V"
        hdu = fits.table_to_hdu(t)
        hdu.name = "DEPTH"
        h = hdu.header
        h["DEGREE"] = (len(self.coef) - 1, "peak(CAR) = sum COEFi CAR^(DEGREE-i)")
        for i, c in enumerate(self.coef):
            h[f"COEF{i}"] = c
        h["CARLO"], h["CARHI"] = self.car_lim
        h["CARREF"] = self.car_ref
        h["AMPREF"] = (self.amp_ref, "V, peak position at CARREF")
        h["AMPLO"], h["AMPHI"] = self.amp_window
        h["CREATED"] = self.created
        return hdu

    def write(self, path, overwrite=True):
        from astropy.io import fits

        fits.HDUList([fits.PrimaryHDU(), self.to_hdu()]).writeto(path, overwrite=overwrite)

    @classmethod
    def from_hdu(cls, hdu):
        h, d = hdu.header, hdu.data
        coef = [h[f"COEF{i}"] for i in range(h["DEGREE"] + 1)]
        return cls(coef, (h["CARLO"], h["CARHI"]), h["CARREF"], np.array(d["CAR"]),
                   np.array(d["PEAK"]), np.array(d["PEAK_ERR"]), (h["AMPLO"], h["AMPHI"]),
                   h.get("CREATED", ""))

    @classmethod
    def read(cls, path):
        from astropy.io import fits

        with fits.open(path) as f:
            return cls.from_hdu(f["DEPTH"])
