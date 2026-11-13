"""Offline analysis chain: spectrum, depth correction, calibration, event FITS, CLI.

The simulator has a known response: anode = E / sim_gain * (0.6 + 0.4 z),
CAR = 1.3 z / (0.6 + 0.4 z), so after the depth correction every line must
land on one straight line through the origin.
"""
import numpy as np
from pathlib import Path
import pytest

pytest.importorskip("scipy")
pytest.importorskip("astropy")

from picodaq.acquisition import run_acquisition
from picodaq.calibration import (LINES, CalPoint, EnergyCalibration, calibrate, guess_gain,
                                 measure_lines)
from picodaq.depth import DepthCorrection
from picodaq.events import EventFile
from picodaq.spectrum import fit_peak, gauss_lin, histogram

GAIN = 3367.0  # keV/V in the simulator


def test_fit_peak_recovers_gaussian():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0.2, 0.004, 5000), rng.uniform(0.15, 0.25, 2000)]
    c, y, _ = histogram(x, bins=100, range=(0.15, 0.25))
    f = fit_peak(c, y, 0.17, 0.23)
    assert f.mu == pytest.approx(0.2, abs=3e-4)
    assert f.sigma == pytest.approx(0.004, rel=0.05)
    assert f.counts(c[1] - c[0]) == pytest.approx(5000, rel=0.05)
    assert f.fwhm == pytest.approx(2.3548 * f.sigma, rel=1e-4)
    with pytest.raises(RuntimeError):
        fit_peak(c, np.zeros_like(y), 0.17, 0.23)


def test_calibration_models_and_inverse(tmp_path):
    pts = [CalPoint("X", e, (e - 2.0) / 3000 + 1e-6 * (e / 1000) ** 2, 1e-5, 0.003, 1e-4, 1000)
           for e in (100.0, 300.0, 600.0, 1200.0)]
    for model in ("linear0", "linear", "quadratic"):
        cal = EnergyCalibration.fit(pts, model)
        a = np.array([p.amp for p in pts])
        np.testing.assert_allclose(cal.to_amp(cal.to_keV(a)), a, rtol=1e-9)
        if model != "linear0":
            assert np.abs(cal.residuals()).max() < 1.0
    with pytest.raises(ValueError):
        EnergyCalibration.fit(pts[:1], "linear")
    cal.write(tmp_path / "c.fits")
    back = EnergyCalibration.read(tmp_path / "c.fits")
    assert back.model == "quadratic" and np.allclose(back.coef, cal.coef)
    assert [p.energy for p in back.points] == [p.energy for p in pts]


def test_line_tables():
    assert max(LINES["Ba133"], key=lambda l: l[1])[0] == pytest.approx(356.01)
    assert LINES["Cs137"][0][0] == pytest.approx(661.66)


@pytest.fixture(scope="module")
def sim_runs(tmp_path_factory):
    """Simulated Ba-133 / Eu-152 / Cs-137 runs -> EventFile per source."""
    from picodaq.config import load_config
    from pathlib import Path

    root = tmp_path_factory.mktemp("sim")
    cfg0 = load_config(Path(__file__).parent.parent / "config_acquisition.yaml")
    out = {}
    for src, n in (("Ba133", 12000), ("Eu152", 12000), ("Cs137", 4000)):
        cfg = load_config(cfg0)
        cfg.update(simulate=True, sim_realtime=False, sim_seed=7, sim_source=src)
        cfg["output"]["data_dir"] = str(root / src)
        cfg["output"]["rapid_block_captures"] = 512
        cfg["capture"]["total_waveforms"] = n
        run_acquisition(cfg, mode="run")
        # collection takes up to 10 us in the simulator: long flat top, late window
        proc = dict(cfg0["processing"], shaper={"rise_us": 15.0, "flat_us": 20.0},
                    flattop_window=[0.6, 0.9])
        out[src] = EventFile.from_waveforms(root / src, processing=proc, verbose=False)
    return root, out


def test_depth_correction_and_full_chain(sim_runs, tmp_path):
    _, efs = sim_runs
    ba = efs["Ba133"]
    # 356 keV: 356 / GAIN * (0.6 .. 1.0) = 0.063 .. 0.106 V
    dc = DepthCorrection.fit(ba.amp("anode"), ba.column("CAR"), (0.055, 0.112), car_range=(0.1, 1.25))
    assert np.isfinite(dc.peak).sum() >= 10
    # z at CAR = 1: 1.3 z = 0.6 + 0.4 z -> z = 2/3 -> 356 / GAIN * (0.6 + 0.4 * 2/3)
    assert dc.amp_ref == pytest.approx(356.01 / GAIN * (0.6 + 0.4 * 2 / 3), rel=0.01)
    spectra = {}
    for src, ef in efs.items():
        ef.apply_depth(dc)
        spectra[src] = ef.selected("AMP_CORR")
    cal = calibrate(spectra, "linear", min_amp=0.02)
    assert len(cal.points) >= 8 and {p.source for p in cal.points} == set(efs)
    assert np.abs(cal.residuals()).max() < 2.0  # keV
    assert abs(cal.coef[0]) < 2.0  # the simulator has no offset
    for ef in efs.values():
        ef.apply_calibration(cal)
    e = efs["Cs137"].selected("ENERGY")
    c, y, _ = histogram(e, bins=120, range=(600, 720))
    assert fit_peak(c, y, 620, 700).mu == pytest.approx(661.66, abs=1.5)
    # guess_gain picks the dominant line of Ba-133
    assert guess_gain(spectra["Ba133"], "Ba133", min_amp=0.02) == pytest.approx(cal.coef[1], rel=0.03)
    # Eu-152 alone cannot be guessed safely
    with pytest.raises(ValueError):
        calibrate({"Eu152": spectra["Eu152"]}, "linear")

    # FITS round trip of the event file with DEPTH and CALIB
    path = tmp_path / "ba.fits"
    ba.write(path)
    back = EventFile.read(path)
    assert len(back) == len(ba) and back.anode == "A" and back.cathode == "B"
    assert back.events["ENERGY"].unit == "keV" and back.events["AMP_A"].unit == "V"
    assert back.meta["processing"]["channels"]["B"]["polarity"] == -1
    assert back.depth.coef == pytest.approx(dc.coef) and back.calib.coef == pytest.approx(cal.coef)
    np.testing.assert_allclose(back.events["ENERGY"], ba.events["ENERGY"])
    good_before = np.asarray(back.events["GOOD"]).copy()
    back.recompute_good(bl_rms_max=1e-9)
    assert not np.asarray(back.events["GOOD"]).any() and good_before.any()


def test_measure_lines_skips_weaker_close_line():
    rng = np.random.default_rng(3)
    # Eu-152 1085.8 / 1112.1 keV are 26 keV apart: only the stronger 1112 is used
    a = np.r_[rng.normal(1085.84, 6, 3000), rng.normal(1112.08, 6, 4000), rng.uniform(1000, 1200, 2000)] / 3000
    pts = measure_lines(a, "Eu152", approx=3000.0, energies=[1085.84, 1112.08])
    assert [p.energy for p in pts] == [1112.08]
    assert pts[0].amp * 3000 == pytest.approx(1112.08, abs=1.5)


def test_cli_run(sim_runs, tmp_path):
    import yaml

    from picodaq.analyze import main

    root, _ = sim_runs
    cfg = {"output_dir": str(tmp_path / "out"),
           "processing": {"shaper": {"rise_us": 15.0, "flat_us": 20.0}, "flattop_window": [0.6, 0.9]},
           "runs": {"Ba133": [str(root / "Ba133")], "Cs137": [str(root / "Cs137")]},
           "depth": {"source": "Ba133", "amp_window": [0.055, 0.112], "car_range": [0.1, 1.25]}}
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(cfg))
    assert main(["run", str(tmp_path / "a.yaml")]) == 0
    out = tmp_path / "out"
    for name in ("Ba133_events.fits", "Cs137_events.fits", "depth.fits", "calib.fits",
                 "depth.png", "calib.png", "spectra.png", "summary.txt"):
        assert (out / name).exists(), name
    cal = EnergyCalibration.read(out / "calib.fits")
    assert np.abs(cal.residuals()).max() < 2.0
    # step-by-step commands on the same files
    assert main(["depth", str(out / "Ba133_events.fits"), "-o", str(tmp_path / "d.fits"),
                 "--window", "0.055", "0.112"]) == 0
    assert main(["calib", f"{out / 'Ba133_events.fits'}:Ba133", "--depth", str(tmp_path / "d.fits"),
                 "-o", str(tmp_path / "c.fits"), "--model", "linear0"]) == 0
    assert EnergyCalibration.read(tmp_path / "c.fits").model == "linear0"
    assert main(["tau", str(next((root / "Ba133").glob("*.h5"))), "--events", "300"]) == 0


def test_car2d_calibration(sim_runs, tmp_path):
    """Optional 2-D calibration: same lines, ENERGY from (amplitude, CAR) directly."""
    from picodaq.carcal import CarCalibration, line_table

    _, efs = sim_runs
    ba = efs["Ba133"]
    dc = DepthCorrection.fit(ba.amp("anode"), ba.column("CAR"), (0.055, 0.112), car_range=(0.1, 1.25))
    for ef in efs.values():
        ef.apply_depth(dc)
    cal = calibrate({s: ef.selected("AMP_CORR") for s, ef in efs.items()}, "linear", min_amp=0.02)
    data = {s: (ef.amp("anode"), ef.column("CAR")) for s, ef in efs.items()}
    cc = CarCalibration.fit(data, dc, cal, min_amp=0.02)
    assert sum(p.used for p in cc.points) >= 30
    used = [i for i, p in enumerate(cc.points) if p.used]
    assert np.sqrt(np.mean(cc.residuals()[used] ** 2)) < 3.0  # per slice: few counts
    # the simulator has no offset: off(CAR) ~ 0 compared with 356 keV x gain
    assert np.all(np.abs(cc.off(np.linspace(*cc.car_lim, 5))) < 0.01 * 356 * cc.gain(1.0))
    for ef in efs.values():
        ef.apply_carcal(cc)
    rows = {(s, e): (m, f) for s, e, m, f in line_table({s: ef.selected("ENERGY") for s, ef in efs.items()})}
    assert rows[("Cs137", 661.66)][0] == pytest.approx(661.66, abs=1.5)
    assert rows[("Ba133", 356.01)][0] == pytest.approx(356.01, abs=1.5)
    # FITS: own file and inside the event file
    cc.write(tmp_path / "cc.fits")
    back = CarCalibration.read(tmp_path / "cc.fits")
    np.testing.assert_allclose(back.to_keV(0.09, 1.0), cc.to_keV(0.09, 1.0))
    efs["Cs137"].write(tmp_path / "cs.fits")
    assert EventFile.read(tmp_path / "cs.fits").carcal.coef_gain == pytest.approx(cc.coef_gain)
    assert np.isnan(cc.to_keV(0.09, cc.car_lim[1] + 0.1))


def test_cli_run_car2d(sim_runs, tmp_path):
    import yaml

    from picodaq.analyze import main

    root, _ = sim_runs
    cfg = {"output_dir": str(tmp_path / "out"),
           "processing": {"shaper": {"rise_us": 15.0, "flat_us": 20.0}, "flattop_window": [0.6, 0.9]},
           "runs": {"Ba133": [str(root / "Ba133")], "Cs137": [str(root / "Cs137")]},
           "depth": {"source": "Ba133", "amp_window": [0.055, 0.112], "car_range": [0.1, 1.25]},
           "calibration": {"method": "car2d"}}
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(cfg))
    assert main(["run", str(tmp_path / "a.yaml")]) == 0
    out = tmp_path / "out"
    for name in ("carcal.fits", "carcal.png", "spectra_car2d.png"):
        assert (out / name).exists(), name
    assert "2-D calibration" in (out / "summary.txt").read_text()


def test_analysis_yaml_relative_paths(tmp_path, monkeypatch):
    """Relative paths in analysis.yaml are taken from the yaml file, not the current directory."""
    import yaml

    from picodaq.analyze import load_analysis

    repo = tmp_path / "TlBr" / "picodaq"
    repo.mkdir(parents=True)
    (repo / "a.yaml").write_text(yaml.safe_dump(
        {"data_root": "../data", "output_dir": "analysis/x", "runs": {"Ba133": ["20260909/run_Ba/data"]}}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PICODAQ_DATA", raising=False)
    acfg = load_analysis(repo / "a.yaml")
    assert acfg["runs"]["Ba133"] == [str(tmp_path / "TlBr" / "data" / "20260909" / "run_Ba" / "data")]
    assert acfg["output_dir"] == str(repo / "analysis" / "x")
    monkeypatch.setenv("PICODAQ_DATA", str(tmp_path / "elsewhere"))
    assert load_analysis(repo / "a.yaml")["runs"]["Ba133"][0].startswith(str(tmp_path / "elsewhere"))
    # the shipped example has no absolute paths
    shipped = load_analysis(Path(__file__).parent.parent / "analysis.yaml")
    assert all(p.startswith(str(tmp_path / "elsewhere")) for ps in shipped["runs"].values() for p in ps)
