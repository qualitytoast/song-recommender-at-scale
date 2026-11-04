"""Health checks for containers, without starting Python for each one.

Docker runs a container's health check as a new process every 2 s. Starting Python and importing gRPC for it cost
~205 ms of CPU each time on the server's 2 cores, and every one of the slowest 1% of requests overlapped such a check
(RESULTS.md, "Serving: the slowest 1% of requests"). So each server process checks itself (SelfCheck): a thread asks
the process's own server, every 2 s, whether it's serving (a shard: the gRPC health service over a connection kept
open; the front desk: GET /live), and if it is, writes the time to a file. Docker's check only reads that file, a
shell one-liner in docker-compose.yml: healthy if it was written in the last 10 s. A process too stuck to answer
itself, or to run the thread at all, stops writing, so Docker sees the file go stale and autoheal restarts it.

    python -m recsys.shard ... --health-file /dev/shm/healthy    # each server writes its file
    python -m recsys.serve ... --health-file /dev/shm/healthy

/dev/shm is memory, not disk: a busy disk can't delay the writes, and a restarted container starts without the
file (a stale one can't pass for healthy while it loads). A one-off check by hand (exit 0 if serving, 1 otherwise):

    python -m recsys.healthcheck grpc localhost:50051
    python -m recsys.healthcheck http http://localhost:8000/live
"""
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

TIMEOUT = 4.0  # seconds a check's answer may take
EVERY = 2.0    # seconds between a process's self-checks


def grpc_serving(target, timeout=TIMEOUT):
    """A check: whether the gRPC server at target says SERVING (its health service), over one connection kept
    open for every call."""
    import grpc
    from grpc_health.v1 import health_pb2, health_pb2_grpc
    stub = health_pb2_grpc.HealthStub(grpc.insecure_channel(target))
    return lambda: stub.Check(health_pb2.HealthCheckRequest(), timeout=timeout).status == \
        health_pb2.HealthCheckResponse.SERVING


def http_ok(url, timeout=TIMEOUT):
    """A check: whether GET url answers 200."""
    def check():
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    return check


class SelfCheck:
    """Every `every` seconds, run check() and, if it says serving, write the time (whole seconds, Unix time) to path:
    to a new file first, renamed over the old one, so a reader never sees half of it. A daemon thread: it ends with
    the process. A check that fails or raises, or a write that fails, writes nothing, and the file goes stale."""

    def __init__(self, check, path, every=EVERY):
        self.check, self.path, self.every = check, Path(path), every
        self.thread = threading.Thread(target=self.run, name="self-check", daemon=True)
        self.thread.start()

    def run(self):
        while True:
            try:
                if self.check():
                    fresh = self.path.with_name(self.path.name + ".new")
                    fresh.write_text(f"{int(time.time())}\n")
                    os.replace(fresh, self.path)
            except Exception:
                pass  # not serving, or couldn't say so: no write
            time.sleep(self.every)


def main(kind, target):
    try:
        return 0 if (grpc_serving(target) if kind == "grpc" else http_ok(target))() else 1
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
