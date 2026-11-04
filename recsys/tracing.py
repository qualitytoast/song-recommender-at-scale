"""Where a request's time goes, for the timing checks (scripts/serving_check.py --trace, scripts/tail_report.py).

Python's garbage collector stops every thread of its process while it runs. GC counts its milliseconds (via
gc.callbacks), so a request or a shard call can report how much of its time went to collection pauses. `timed`
wraps a shard's handler so its reply says when the handler started and how long it ran (shard.proto's Work).
"""
import functools
import gc
import time


class GcClock:
    """Milliseconds this process has spent in the garbage collector since it started."""

    def __init__(self):
        self.ms, self.started = 0.0, None
        gc.callbacks.append(self.callback)

    def callback(self, phase, info):  # called holding the interpreter lock, one collection at a time
        if phase == "start":
            self.started = time.perf_counter()
        elif self.started is not None:
            self.ms += (time.perf_counter() - self.started) * 1000
            self.started = None


GC = GcClock()  # one per process


def timed(handler):
    """A shard handler whose reply's `work` gives the handler's start (Unix time), its milliseconds, and the garbage
    collector's milliseconds within them."""
    @functools.wraps(handler)
    def run(self, request, context):
        start_s, start, gc_ms = time.time(), time.perf_counter(), GC.ms
        reply = handler(self, request, context)
        reply.work.start_s, reply.work.ms, reply.work.gc_ms = start_s, (time.perf_counter() - start) * 1000, GC.ms - gc_ms
        return reply
    return run
