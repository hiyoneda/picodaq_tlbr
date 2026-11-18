"""Configuration: YAML file merged onto defaults."""
import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "scope": {"resolution": "10BIT"},
    "channels": {},
    "trigger": {"enabled": True, "source": "A", "threshold_mv": 30, "auto_trigger_ms": 0},
    "capture": {
        "samples": 5000,
        "sample_rate_hz": 20_000_000,
        "pre_trigger_fraction": 0.2,
        "total_waveforms": None,
        "loop": "poll",
        "batch_timeout_s": 2.0,
    },
    "output": {
        "data_dir": "data",
        "base": "tlbr",
        "start_id": 0,
        "max_file_size_mb": 1024,
        "rapid_block_captures": 32,
        "compression": "lzf",
    },
    # pulse processing: shared by the quick look and the offline analysis
    "processing": {
        "anode": "A",
        "cathode": "B",
        "baseline_fraction": 0.8,  # baseline = first 80 % of the pre-trigger samples
        "shaper": {"rise_us": 15.0, "flat_us": 12.0},
        "amplitude": "flattop",  # flattop | max
        "guard_frac": 0.15,
        "flattop_window": None,  # e.g. [0.6, 0.9]: average only the late part of the flat top
        "channels": {
            "A": {"polarity": 1, "tau_us": 148.0},
            "B": {"polarity": -1, "tau_us": 27.0},
        },
    },
    # quick-look display only
    "ql": {
        "waveforms_per_batch": 4,
        "amp_max_v": 2.0,
        "amp_bins": 512,
        "amp2d_bins": 256,
        "ratio_range": [0.0, 2.0],
        "ratio_bins": 200,
    },
    "simulate": False,
    "sim_rate_hz": 600.0,
}


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(source=None):
    """Path, YAML string, dict or None -> config dict with defaults filled in."""
    if source is None:
        user = {}
    elif isinstance(source, dict):
        user = source
    elif isinstance(source, Path) or (isinstance(source, str) and "\n" not in source):
        with open(source) as f:
            user = yaml.safe_load(f) or {}
    else:
        user = yaml.safe_load(source) or {}
    return deep_merge(DEFAULTS, user)


# bit of each channel in the over-range masks (Trigger/Overflow, backend.last_overflow)
CHANNEL_BITS = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7}


def enabled_channels(config):
    return [name for name, value in config["channels"].items() if value.get("enabled")]
