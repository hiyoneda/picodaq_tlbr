"""Stand-in for libps6000 (PicoScope 6000 legacy driver) for tests without hardware.

Called exactly like the ctypes CDLL: out-parameters arrive as ctypes.byref()
objects (._obj is the target), buffers as integer addresses. Triggers arrive
at RATE Hz on a clock that starts at RunBlock.
"""
import ctypes
import time

import numpy as np

OK = 0


class FakeLib:
    RATE = 2000.0  # Hz

    def __init__(self, rate=None, trigger_info_status=OK):
        self.rate = self.RATE if rate is None else rate
        self.trigger_info_status = trigger_info_status
        self.calls = []
        self.channels = {}
        self.trigger = None
        self.segments = 1
        self.n_captures = 1
        self.buffers = {}
        self.t_run = None
        self.stopped_at = None
        self.t_open = time.monotonic()  # the timestamp clock runs from open
        self.closed = False

    # helpers
    def _log(self, name, *args):
        self.calls.append((name,) + args)

    def _done(self, now=None):
        if self.t_run is None:
            return 0
        if self.trigger and self.trigger["thr"] > 0.9 * 32512:
            return 0  # "high threshold": no pulses that large
        t = (self.stopped_at or time.monotonic()) if now is None else now
        return int(min(self.n_captures, (t - self.t_run) * self.rate))

    # API
    def ps6000OpenUnit(self, handle_ref, serial):
        handle_ref._obj.value = 7
        return OK

    def ps6000CloseUnit(self, h):
        self.closed = True
        return OK

    def ps6000GetUnitInfo(self, h, buf_ref, n, req_ref, info):
        text = {3: b"6402C", 4: b"FK123/0001"}.get(info, b"")
        buf_ref._obj.value = text
        req_ref._obj.value = len(text)
        return OK

    def ps6000GetAnalogueOffset(self, h, rng, coupling, max_ref, min_ref):
        lim = 0.5 if rng <= 5 else 2.5  # roughly the 6402C 1 MOhm limits
        max_ref._obj.value, min_ref._obj.value = lim, -lim
        return OK

    def ps6000SetChannel(self, h, ch, enabled, coupling, rng, offset, bw):
        off = offset.value if isinstance(offset, ctypes.c_float) else offset
        self.channels[ch] = dict(enabled=enabled, coupling=coupling, range=rng, offset=off, bw=bw)
        return OK

    def ps6000GetTimebase2(self, h, tb, n, dt_ref, oversample, max_ref, seg):
        self._log("GetTimebase2", tb, n, oversample)
        dt_ref._obj.value = (2 ** tb / 5.0) if tb < 5 else (tb - 4) / 0.15625  # ns
        max_ref._obj.value = 1 << 30
        return OK

    def ps6000SetSimpleTrigger(self, h, enable, ch, thr, direction, delay, auto_ms):
        self.trigger = dict(enable=enable, ch=ch, thr=thr, direction=direction, auto_ms=auto_ms)
        return OK

    def ps6000MemorySegments(self, h, n, max_ref):
        self.segments = n
        max_ref._obj.value = (1 << 30) // n
        return OK

    def ps6000SetNoOfCaptures(self, h, n):
        if n > self.segments:
            return 0x42  # PICO_NOT_ENOUGH_SEGMENTS
        self.n_captures = n
        self.buffers = {}
        return OK

    def ps6000SetDataBufferBulk(self, h, ch, addr, n, seg, mode):
        self.buffers[(ch, seg)] = (addr, n)
        return OK

    def ps6000RunBlock(self, h, pre, post, tb, oversample, t_ref, seg, cb, param):
        self._log("RunBlock", pre, post, tb, oversample)
        self.pre, self.post = pre, post
        self.t_run, self.stopped_at = time.monotonic(), None
        return OK

    def ps6000IsReady(self, h, ready_ref):
        ready_ref._obj.value = int(self._done() >= self.n_captures)
        return OK

    def ps6000Stop(self, h):
        if self.t_run is not None and self.stopped_at is None:
            self.stopped_at = time.monotonic()
        return OK

    def ps6000GetNoOfCaptures(self, h, n_ref):
        n_ref._obj.value = self._done()
        return OK

    def ps6000GetValuesBulk(self, h, n_ref, first, last, ratio, mode, overflow_addr):
        n = n_ref._obj.value
        for (ch, seg), (addr, length) in self.buffers.items():
            if first <= seg <= last:
                arr = np.ctypeslib.as_array((ctypes.c_int16 * length).from_address(addr))
                # step of ch*1000 + seg counts at the trigger point
                arr[:] = 0
                arr[self.pre:] = 1000 * (ch + 1) + seg
        return OK

    def ps6000GetTriggerInfoBulk(self, h, info_ref, first, last):
        if self.trigger_info_status != OK:
            return self.trigger_info_status
        info = info_ref._obj
        period = 1.0 / self.rate
        for i in range(first, last + 1):
            info[i].status = OK
            info[i].segmentIndex = i
            t = self.t_run - self.t_open + (i + 1) * period
            info[i].timeStampCounter = int(round(t / 51.2e-9))
        return OK
