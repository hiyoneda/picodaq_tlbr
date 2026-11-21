"""Minimal stand-in for pypicosdk to exercise PicoBackend's control flow."""
import ctypes, time, types
import numpy as np
RESOLUTION = types.SimpleNamespace(_10BIT=10)
CHANNEL = types.SimpleNamespace(A=0, B=1, C=2, D=3)
RANGE = types.SimpleNamespace(V2=9)
SAMPLE_RATE = types.SimpleNamespace(MSPS=1)
RATE = 200.0  # triggers / s


class PICO_TRIGGER_INFO(ctypes.Structure):  # same fields as pypicosdk.constants
    _fields_ = [("status_", ctypes.c_int32), ("segmentIndex_", ctypes.c_uint64),
                ("triggerIndex_", ctypes.c_uint64), ("triggerTime_", ctypes.c_double),
                ("timeUnits_", ctypes.c_int32), ("missedTriggers_", ctypes.c_uint64),
                ("timeStampCounter_", ctypes.c_uint64)]
class psospa:
    handle = 1
    def __init__(self): self.trigger_info_calls = []
    def open_unit(self, resolution): self.ch = []
    def set_channel(self, channel, range): self.ch.append(channel)
    def set_simple_trigger(self, channel, threshold, auto_trigger): self.thr = threshold
    def sample_rate_to_timebase(self, rate, unit): return 3
    def get_unit_serial(self): return "FAKE/0001"
    def memory_segments(self, n): self.nseg = n
    def set_no_of_captures(self, n): self.ncap = n
    def set_data_buffer_for_enabled_channels(self, samples, captures):
        self.bufs = {c: np.zeros((captures, samples), np.int16) for c in self.ch}; return self.bufs
    def run_block_capture(self, tb, samples, pretrig):
        self.t_run = time.monotonic(); self.stopped_at = None; self.samples = samples
    def _done(self):
        t = (self.stopped_at or time.monotonic()) - self.t_run
        return min(self.ncap, int(t * RATE))
    def _call_attr_function(self, name, handle, *args):
        if name == "IsReady":
            (ready_ref,) = args
            ctypes.cast(ready_ref, ctypes.POINTER(ctypes.c_int16)).contents.value = int(self._done() >= self.ncap)
        elif name == "GetTriggerInfo":
            # like the psospa driver: the last argument is the NUMBER of segments
            info_ref, first, count = args
            self.trigger_info_calls.append((first, count))
            info = info_ref._obj
            for k in range(count):
                i = first + k
                info[k].status_, info[k].segmentIndex_ = 0, i
                info[k].missedTriggers_, info[k].timeStampCounter_ = i % 3, 1000 + 100000 * i
        else:
            raise AssertionError(name)
    def stop(self): self.stopped_at = time.monotonic()
    def get_no_of_captures(self): return self._done()
    def get_values_bulk(self, samples, a, b, wait_for_ready=True):
        for c, arr in self.bufs.items(): arr[a:b + 1] = (c + 1) * 100
        # like pypicosdk: per capture the list of channel names that went over range
        over = [(["A"] if i % 7 == 3 else []) + (["B"] if i % 11 == 5 else []) for i in range(b + 1)]
        return samples, over
    def _adc_to_(self, buffers, unit): return {c: v.astype(np.float64) / 32512 * 2 for c, v in buffers.items()}
    def get_time_axis(self, tb, n, pre_trig_percent, unit): return (np.arange(n) - n * pre_trig_percent // 100) * 50e-9
    def get_trigger_info(self, first, n):
        # pypicosdk 1.7.5: passes first + n - 1 as the count -> the last entry stays empty
        info = (PICO_TRIGGER_INFO * n)()
        self._call_attr_function("GetTriggerInfo", self.handle, ctypes.byref(info), first, first + n - 1)
        return [dict(status=i.status_, segmentIndex=i.segmentIndex_, triggerIndex=0, triggerTime=0.0,
                     timeUnits=0, missedTriggers=i.missedTriggers_, timeStampCounter=i.timeStampCounter_)
                for i in info]
    def close_unit(self): pass
    def run_simple_rapid_block_capture(self, timebase, samples, captures, output_unit, time_unit, pre_trig_percent):
        time.sleep(captures / RATE)
        return ({c: np.full((captures, samples), 0.1) for c in self.ch},
                self.get_time_axis(timebase, samples, pre_trig_percent, time_unit))
