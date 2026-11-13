"""Histograms and peak fitting (Gaussian + linear background), as in the notebook.

Needs scipy (pip install picodaq[analysis]).
"""
from dataclasses import dataclass, field

import numpy as np

SQRT2PI = np.sqrt(2 * np.pi)
FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))  # 2.3548


def histogram(x, bins=256, range=None):
    """-> (centers, counts, edges). range None: 0 .. 1.1 x 99.9th percentile."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if range is None:
        range = (0.0, float(np.percentile(x, 99.9)) * 1.1 if x.size else 1.0)
    counts, edges = np.histogram(x, bins=bins, range=range)
    return 0.5 * (edges[:-1] + edges[1:]), counts, edges


def gauss_lin(x, area, mu, sigma, a, b):
    """Gaussian (area = counts x bin width) on a linear background a + b x."""
    return area / (SQRT2PI * sigma) * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + a + b * x


@dataclass
class PeakFit:
    area: float
    mu: float
    sigma: float
    a: float
    b: float
    errors: dict = field(default_factory=dict)  # 1-sigma errors of the parameters
    chi2: float = np.nan
    ndof: int = 0
    window: tuple = (np.nan, np.nan)

    @property
    def fwhm(self):
        return FWHM_PER_SIGMA * self.sigma

    @property
    def mu_err(self):
        return self.errors.get("mu", np.nan)

    def counts(self, bin_width):
        return self.area / bin_width

    def model(self, x):
        return gauss_lin(x, self.area, self.mu, self.sigma, self.a, self.b)


def fit_peak(x, y, lo, hi, mu0=None, sigma0=None):
    """Fit gauss_lin to histogram (x = bin centres, y = counts) inside [lo, hi].

    Poisson weights sqrt(max(y, 1)). Raises RuntimeError when the fit fails or the
    peak ends up outside the window.
    """
    from scipy.optimize import curve_fit

    m = (x >= lo) & (x <= hi)
    xs, ys = np.asarray(x)[m], np.asarray(y, dtype=float)[m]
    if xs.size < 6 or ys.sum() < 10:
        raise RuntimeError("too few bins / counts in the fit window")
    dx = xs[1] - xs[0]
    mu0 = float(xs[np.argmax(ys)]) if mu0 is None else float(mu0)
    sigma0 = 0.1 * (hi - lo) if sigma0 is None else float(sigma0)
    p0 = [max(ys.sum() - ys.min() * xs.size, 1.0) * dx, mu0, sigma0, float(ys.min()), 0.0]
    lower = [0.0, lo, dx / 4, -np.inf, -np.inf]
    upper = [np.inf, hi, hi - lo, np.inf, np.inf]
    try:
        popt, pcov = curve_fit(gauss_lin, xs, ys, p0=p0, sigma=np.sqrt(np.maximum(ys, 1)),
                               absolute_sigma=True, bounds=(lower, upper), maxfev=40000)
    except (RuntimeError, ValueError) as e:
        raise RuntimeError(f"peak fit failed: {e}") from e
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
    resid = (ys - gauss_lin(xs, *popt)) / np.sqrt(np.maximum(ys, 1))
    names = ("area", "mu", "sigma", "a", "b")
    fit = PeakFit(*popt, errors=dict(zip(names, perr)), chi2=float(resid @ resid),
                  ndof=int(xs.size - 5), window=(lo, hi))
    if not lo < fit.mu < hi:
        raise RuntimeError("fitted peak outside the window")
    return fit


def find_peaks(centers, counts, min_x=None, smooth=3, prominence_frac=0.05):
    """Peak positions (bin centres), most prominent first. Thin wrapper of
    scipy.signal.find_peaks on a moving average."""
    from scipy.signal import find_peaks as _fp

    c = np.convolve(counts, np.ones(smooth) / smooth, "same") if smooth > 1 else np.asarray(counts, float)
    if min_x is not None:
        c = np.where(centers >= min_x, c, 0)
    idx, prop = _fp(c, prominence=prominence_frac * c.max() if c.max() > 0 else 0)
    order = np.argsort(prop["prominences"])[::-1]
    return centers[idx[order]], prop["prominences"][order]
