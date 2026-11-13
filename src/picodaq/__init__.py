"""picodaq: PicoScope 3416E waveform acquisition, quick look and shared analysis tools.

Shared with offline analysis (no GUI / scope dependencies):
    picodaq.config   load_config, enabled_channels
    picodaq.io       WaveformWriter, WaveformReader
    picodaq.shaping  TrapezoidShaper, ChannelProcessor, tune_tau, ...
    picodaq.timing   timestamps, rates, live time
    picodaq.spectrum histograms, Gaussian + linear peak fit        (scipy)
    picodaq.depth    DepthCorrection (CAR)                          (scipy, astropy for FITS)
    picodaq.calibration  LINES, EnergyCalibration, calibrate        (scipy, astropy for FITS)
    picodaq.events   EventFile: event table as FITS                 (astropy)
    picodaq.carcal   CarCalibration: optional 2-D (CAR x energy) calibration
"""
from .config import enabled_channels, load_config
from .io import WaveformReader, WaveformWriter
from .shaping import (
    ChannelProcessor,
    TrapezoidShaper,
    fit_tail_tau,
    processors_from_config,
    quality_mask,
    rise_time,
    trapezoid,
    tune_tau,
)
from .timing import RateMeter, livetime_summary, unwrap_timestamps
from .calibration import LINES, CalPoint, EnergyCalibration, calibrate, measure_lines
from .carcal import CarCalibration
from .depth import DepthCorrection
from .events import EventFile
from .spectrum import PeakFit, fit_peak, histogram

__version__ = "0.1.0"
