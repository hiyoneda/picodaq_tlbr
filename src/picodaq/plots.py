"""Quality-assurance figures for the analysis steps (matplotlib)."""
import numpy as np

from .calibration import LINES
from .spectrum import histogram


def _plt():
    import matplotlib.pyplot as plt  # the CLI selects the Agg backend itself

    return plt


def depth_figure(ef, dc, amp_max=None):
    """CAR vs anode amplitude with the fitted photopeak positions and the correction curve,
    and the anode spectrum before / after the correction."""
    plt = _plt()
    a, car = ef.amp("anode"), ef.column("CAR")
    amp_max = amp_max or np.nanpercentile(a, 99.5) * 1.1
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].hist2d(car[np.isfinite(car)], a[np.isfinite(car)] * 1e3, bins=(80, 120),
                 range=[[0, 2], [0, amp_max * 1e3]], cmap="viridis", cmin=1)
    ok = np.isfinite(dc.peak)
    ax[0].errorbar(dc.car[ok], dc.peak[ok] * 1e3, dc.peak_err[ok] * 1e3, fmt="o", color="w", ms=4,
                   label="fitted peak")
    xf = np.linspace(*dc.car_lim, 200)
    ax[0].plot(xf, np.polyval(dc.coef, xf) * 1e3, "r-", label="correction curve")
    ax[0].axvline(dc.car_ref, color="w", ls=":", lw=0.8)
    ax[0].axhspan(dc.amp_window[0] * 1e3, dc.amp_window[1] * 1e3, color="w", alpha=0.08)
    ax[0].set_xlabel("cathode / anode (CAR)")
    ax[0].set_ylabel("anode pulse height [mV]")
    ax[0].legend(fontsize=8)
    corr = dc.apply(ef.amp("anode"), car)
    for label, x in (("raw (CAR in range)", a[dc.in_range(car)]), ("depth corrected", corr)):
        c, y, _ = histogram(x, bins=300, range=(0, amp_max))
        ax[1].step(c * 1e3, y, where="mid", lw=0.9, label=label)
    ax[1].set_xlabel("anode pulse height [mV]")
    ax[1].set_ylabel("counts / bin")
    ax[1].legend(fontsize=8)
    fig.suptitle(f"depth correction: CAR {dc.car_lim[0]:.2f}-{dc.car_lim[1]:.2f}, ref {dc.car_ref}")
    fig.tight_layout()
    return fig


def calibration_figure(cal):
    """Amplitude vs energy with the model, residuals and FWHM vs energy."""
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    srcs = sorted({p.source for p in cal.points})
    e = np.array([p.energy for p in cal.points])
    a = np.array([p.amp for p in cal.points])
    ae = np.array([p.amp_err for p in cal.points])
    r, fw = cal.residuals(), cal.fwhm_keV()
    for i, s in enumerate(srcs):
        m = np.array([p.source == s for p in cal.points])
        ax[0].errorbar(a[m] * 1e3, e[m], xerr=ae[m] * 1e3, fmt="o", ms=4, color=f"C{i}", label=s)
        ax[1].errorbar(e[m], r[m], yerr=cal.gain(a[m]) * ae[m], fmt="o", ms=4, color=f"C{i}")
        ax[2].plot(e[m], fw[m], "o", ms=4, color=f"C{i}")
    xa = np.linspace(0, a.max() * 1.1, 200)
    ax[0].plot(xa * 1e3, cal.to_keV(xa), "k-", lw=0.8)
    ax[0].set_xlabel("depth-corrected anode [mV]")
    ax[0].set_ylabel("energy [keV]")
    ax[0].legend(fontsize=8)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("energy [keV]")
    ax[1].set_ylabel("residual [keV]")
    ax[2].set_xlabel("energy [keV]")
    ax[2].set_ylabel("FWHM [keV]")
    fig.suptitle(f"energy calibration ({cal.model}): " + ", ".join(f"C{i} = {c:.4g}" for i, c in enumerate(cal.coef)))
    fig.tight_layout()
    return fig


def spectra_figure(efs, e_max=None, bin_kev=2.0, logy=True):
    """Calibrated spectra of several event files {source: EventFile} with the known lines."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    e_max = e_max or max(np.nanpercentile(ef.selected("ENERGY"), 99.5) for ef in efs.values()) * 1.1
    for i, (src, ef) in enumerate(efs.items()):
        c, y, _ = histogram(ef.selected("ENERGY"), bins=int(e_max / bin_kev), range=(0, e_max))
        ax.step(c, y, where="mid", lw=0.9, color=f"C{i}", label=src)
        for e, _ in LINES.get(src, []):
            if e < e_max:
                ax.axvline(e, color=f"C{i}", ls=":", lw=0.7)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("energy [keV]")
    ax.set_ylabel(f"counts / {bin_kev:g} keV")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def carcal_figure(cc):
    """2-D calibration: gain(CAR) and off(CAR) with the per-point estimates, and
    the residuals of every line vs CAR."""
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    pts = [p for p in cc.points if p.used]
    c = np.array([p.car for p in pts])
    e = np.array([p.energy for p in pts])
    a = np.array([p.amp for p in pts])
    r = cc.residuals()[[i for i, p in enumerate(cc.points) if p.used]]
    xc = np.linspace(*cc.car_lim, 200)
    g_pt = (a - cc.off(c) - cc.q * e * e) / e
    sc = ax[0].scatter(c, g_pt * 1e6, c=e, s=12, cmap="viridis")
    ax[0].plot(xc, cc.gain(xc) * 1e6, "r-")
    ax[0].set_xlabel("CAR")
    ax[0].set_ylabel("gain [mV / MeV]")
    fig.colorbar(sc, ax=ax[0], label="line energy [keV]")
    ax[1].plot(xc, cc.off(xc) * 1e3, "r-")
    ax[1].set_xlabel("CAR")
    ax[1].set_ylabel("offset [mV]")
    sc = ax[2].scatter(c, r, c=e, s=12, cmap="viridis")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xlabel("CAR")
    ax[2].set_ylabel("residual [keV]")
    fig.suptitle("2-D calibration (CAR x energy)" + (f" - {cc.note}" if cc.note else ""))
    fig.tight_layout()
    return fig
