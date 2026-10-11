"""Dynamic batching (stage 6): requests that arrive within `wait_ms` of each other (up to `max_batch`) are
answered by one call that runs them through the models together, each shard asked once per round for all
of them (ShardedRecommender.recommend_batch). Waiting a moment for company costs each request up to
wait_ms; doing the work together raises how many requests per second one front desk can answer.
"""
import queue
import threading
import time
from concurrent.futures import Future


class Batcher:
    def __init__(self, run_batch, max_batch=16, wait_ms=2.0):
        """run_batch(list of requests) -> one result per request (an Exception for a failed one)."""
        self.run_batch, self.max_batch, self.wait = run_batch, max_batch, wait_ms / 1000
        self.queue = queue.Queue()
        self.sizes = []  # every batch's size, for reports
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, request):
        """The request's result (waits for its batch); raises the request's own error, if it had one."""
        future = Future()
        self.queue.put((request, future))
        return future.result()

    def _loop(self):
        while True:
            batch = [self.queue.get()]
            deadline = time.monotonic() + self.wait
            while len(batch) < self.max_batch and (left := deadline - time.monotonic()) > 0:
                try:
                    batch.append(self.queue.get(timeout=left))
                except queue.Empty:
                    break
            self.sizes.append(len(batch))
            try:
                results = self.run_batch([request for request, _ in batch])
            except Exception as e:  # the whole batch failed: each request gets the error
                results = [e] * len(batch)
            for (_, future), result in zip(batch, results):
                future.set_exception(result) if isinstance(result, Exception) else future.set_result(result)

    def report(self):
        return {"batches": len(self.sizes), "mean_size": round(sum(self.sizes) / len(self.sizes), 2) if self.sizes else None,
                "largest": max(self.sizes, default=None)}
