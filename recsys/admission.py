"""Overload protection: a front desk turns requests away fast instead of letting them queue for seconds.

Two limits per front desk:
  max_pending  requests admitted at once (waiting or being answered); one more gets 503 right away
  max_wait     a request that has waited this long without being started is dropped with 503: the person
               has likely given up, and answering it would only delay the requests behind it
A 503 carries Retry-After: 1, and nginx tries one other front-desk copy first, so a person only sees
"busy" when every copy is full. Without this, a front desk past capacity answers everyone, slowly
(at 1,000 users back to back: 2.3 s at p50, 4.7 s at p99).
"""
import threading
import time


class Overloaded(RuntimeError):
    """The front desk is full, or the request waited too long: HTTP 503, try again shortly."""


class Admission:
    def __init__(self, max_pending, max_wait_ms=None):
        self.max_pending = max_pending
        self.max_wait = None if max_wait_ms is None else max_wait_ms / 1000
        self.pending = 0
        self.lock = threading.Lock()
        self.stats = {"admitted": 0, "rejected_full": 0, "rejected_waited": 0}

    def enter(self):
        """Admit a request (True), or refuse it because max_pending are already in (False)."""
        with self.lock:
            if self.pending >= self.max_pending:
                self.stats["rejected_full"] += 1
                return False
            self.pending += 1
            self.stats["admitted"] += 1
            return True

    def leave(self):
        with self.lock:
            self.pending -= 1

    def deadline(self, arrived):
        """When a request that arrived at `arrived` (time.monotonic()) must have started, or None."""
        return None if self.max_wait is None else arrived + self.max_wait

    def check_started(self, arrived):
        """Raise Overloaded if a request that arrived at `arrived` waited longer than max_wait to start."""
        deadline = self.deadline(arrived)
        if deadline is not None and time.monotonic() > deadline:
            with self.lock:
                self.stats["rejected_waited"] += 1
            raise Overloaded(f"busy: waited over {self.max_wait * 1000:.0f} ms to start; try again")

    def report(self):
        return {**self.stats, "pending": self.pending, "max_pending": self.max_pending,
                "max_wait_ms": None if self.max_wait is None else self.max_wait * 1000}
