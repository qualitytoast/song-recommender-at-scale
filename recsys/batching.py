"""Dynamic batching (stage 6): requests that arrive within `wait_ms` of each other (up to `max_batch`) are
answered by one call that runs them through the models together, each shard asked once per round for all
of them (ShardedRecommender.recommend_batch). Waiting a moment for company costs each request up to
wait_ms; doing the work together raises how many requests per second one front desk can answer.
"""
import queue
import threading
import time
from concurrent.futures import Future

from recsys.admission import Overloaded


class Batcher:
    def __init__(self, run_batch, max_batch=16, wait_ms=2.0):
        """run_batch(list of requests) -> one result per request (an Exception for a failed one)."""
        self.run_batch, self.max_batch, self.wait = run_batch, max_batch, wait_ms / 1000
        self.queue = queue.Queue()
        self.sizes = []  # every batch's size, for reports
        self.dropped = 0  # requests that waited past their deadline: never computed
        self.closed, self.lock = False, threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, request, deadline=None):
        """The request's result (waits for its batch); raises the request's own error, if it had one, or
        Overloaded if it couldn't start by `deadline` (time.monotonic()) or the batcher is closed."""
        future = Future()
        with self.lock:
            if self.closed:
                raise Overloaded("busy: this version was just replaced; try again")
            self.queue.put((request, future, deadline))
        return future.result()

    def close(self):
        """End the thread once what's queued is answered (else it, and the recommender it runs, live forever: a
        front desk that switches versions, recsys/switch.py, closes the old version's batcher)."""
        with self.lock:
            self.closed = True
            self.queue.put(None)  # after everything queued so far: those are still answered
        self.thread.join()
        # run_batch is the recommender's own method, and the recommender holds this batcher: a cycle that only the
        # garbage collector frees, which on the server kept every old version (~150 MB more a switch, 2026-11-04)
        self.run_batch = None

    def _loop(self):
        closing = False
        while not closing:
            first = self.queue.get()
            if first is None:
                break
            batch = [first]
            deadline = time.monotonic() + self.wait
            while len(batch) < self.max_batch and (left := deadline - time.monotonic()) > 0:
                try:
                    item = self.queue.get(timeout=left)
                except queue.Empty:
                    break
                if item is None:
                    closing = True
                    break
                batch.append(item)
            now = time.monotonic()
            stale = [(r, f) for r, f, d in batch if d is not None and now > d]  # waited too long: dropped
            batch = [(r, f) for r, f, d in batch if d is None or now <= d]
            for _, future in stale:
                self.dropped += 1
                future.set_exception(Overloaded("busy: waited too long to start; try again"))
            if not batch:
                continue
            self.sizes.append(len(batch))
            try:
                results = self.run_batch([request for request, _ in batch])
            except Exception as e:  # the whole batch failed: each request gets the error
                results = [e] * len(batch)
            for (_, future), result in zip(batch, results):
                future.set_exception(result) if isinstance(result, Exception) else future.set_result(result)

    def report(self):
        return {"batches": len(self.sizes), "mean_size": round(sum(self.sizes) / len(self.sizes), 2) if self.sizes else None,
                "largest": max(self.sizes, default=None), "dropped_waited_too_long": self.dropped}
