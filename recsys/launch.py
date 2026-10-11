"""Start the sharded system on this machine with one command: N shard servers and the front desk.

    python -m recsys.launch --shards 4      # the newest bundle; the front desk on http://127.0.0.1:8000

Cuts the bundle into N slices first if they don't exist yet (recsys.shard_slices), starts each shard
(recsys.shard) on its own port, then the front desk (recsys.serve --shards ...), and waits until the
front desk's /health answers (it only starts once every shard has checked out). Ctrl-C (or SIGTERM)
stops the front desk, then the shards; they get SIGTERM, since a background job ignores SIGINT, and so do
its children. In containers, docker compose up does the same (docker/).
"""
import argparse
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from recsys.serve import latest_bundle

ROOT = Path(__file__).resolve().parents[1]


def start(args, log):
    return subprocess.Popen([sys.executable, "-m", *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)


def wait_healthy(url, processes, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if any(p.poll() is not None for p in processes):
            raise RuntimeError("a process exited during startup; see its log")
        try:
            return urllib.request.urlopen(f"{url}/health", timeout=2).read().decode()
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"{url} not healthy after {timeout}s")


def main():
    ap = argparse.ArgumentParser(description="Start N shard servers and the front desk.")
    ap.add_argument("--bundle", help="bundle folder (default: the newest in bundles/)")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--port", type=int, default=8000, help="the front desk's port")
    ap.add_argument("--shard-port", type=int, default=50051, help="shard i listens on this + i")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--deadline", type=float, default=2.0)
    ap.add_argument("--logs", default="runs/serving/launch", help="one log file per process")
    args = ap.parse_args()
    bundle = Path(args.bundle) if args.bundle else latest_bundle()
    slices = bundle / f"shards-{args.shards}"
    if not slices.exists():
        subprocess.run([sys.executable, "-m", "recsys.shard_slices", "--bundle", str(bundle), "--shards",
                        str(args.shards)], cwd=ROOT, check=True)
    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # stop cleanly, even during startup
    try:
        addresses = []
        for i in range(args.shards):
            port = args.shard_port + i
            processes.append(start(["recsys.shard", "--slice", str(slices / f"shard-{i}"), "--port", str(port),
                                    "--host", "127.0.0.1"], open(logs / f"shard-{i}.log", "w")))
            addresses.append(f"127.0.0.1:{port}")
        processes.append(start(["recsys.serve", "--bundle", str(bundle), "--device", args.device, "--port",
                                str(args.port), "--shards", ",".join(addresses), "--deadline", str(args.deadline)],
                               open(logs / "frontdesk.log", "w")))
        start_time = time.time()
        health = wait_healthy(f"http://127.0.0.1:{args.port}", processes, timeout=300)
        print(f"ready after {time.time() - start_time:.0f}s: {health} (logs in {logs}/); Ctrl-C to stop", flush=True)
        while all(p.poll() is None for p in processes):
            time.sleep(1)
        print("a process stopped; shutting everything down", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        for p in reversed(processes):  # the front desk first, then the shards; SIGTERM, as background jobs
            p.terminate()              # ignore Ctrl-C's SIGINT (and pass that on to their children)
        for p in reversed(processes):
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
