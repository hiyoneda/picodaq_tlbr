import sys
from pathlib import Path

import pytest

from picodaq.config import load_config

HERE = Path(__file__).parent


@pytest.fixture
def config(tmp_path):
    cfg = load_config(HERE.parent / "config_acquisition.yaml")
    cfg["output"]["data_dir"] = str(tmp_path / "data")
    cfg["simulate"] = True
    cfg["sim_realtime"] = False  # simulated clock: fast, deterministic
    cfg["sim_seed"] = 1
    return cfg


@pytest.fixture
def fake_pypicosdk(monkeypatch):
    """Install tests/fake_pypicosdk.py as 'pypicosdk'."""
    import importlib

    monkeypatch.syspath_prepend(str(HERE))
    mod = importlib.import_module("fake_pypicosdk")
    monkeypatch.setitem(sys.modules, "pypicosdk", mod)
    return mod
