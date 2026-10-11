"""Load tests: throughput and response times with 1 or more front-desk copies (docker compose system).

    python scripts/load_test.py --copies 1 3 --users 1 16 64 --seconds 30 --out runs/serving/load.json

For each number of front-desk copies: docker compose up --scale frontdesk=N (nginx restarted so it sees
every copy), then for each number of simulated users a headless locust run (scripts/locustfile.py) of
--seconds. Reports requests per second and p50/p99 per endpoint, and failures.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "PATH": f"{Path.home()}/.docker/bin:{os.environ['PATH']}"}


SETTINGS = {}  # environment for docker compose (--set KEY=VALUE): e.g. BATCH, REDIS_URL


def compose(*args, extra=()):
    subprocess.run(["docker", "compose", *args, *extra], cwd=ROOT, env={**ENV, **SETTINGS}, check=True,
                   capture_output=True)


def wait_healthy(url="http://127.0.0.1:8000/health", timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        try:
            if json.loads(urllib.request.urlopen(url, timeout=2).read())["status"] == "ok":
                return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError("front desk not healthy")


def run_locust(users, seconds, prefix):
    subprocess.run([sys.executable, "-m", "locust", "-f", "scripts/locustfile.py", "--headless", "-u", str(users),
                    "-r", str(users), "--run-time", f"{seconds}s", "--host", "http://127.0.0.1:8000", "--csv", prefix,
                    "--only-summary"], cwd=ROOT, env=ENV, capture_output=True)
    rows = {r["Name"]: r for r in csv.DictReader(open(f"{prefix}_stats.csv"))}
    pick = lambda r: {"requests": int(r["Request Count"]), "failures": int(r["Failure Count"]),
                      "per_s": float(r["Requests/s"]), "p50_ms": float(r["50%"]), "p99_ms": float(r["99%"])}
    return {name: pick(rows[name]) for name in ("/recommend", "/songs", "Aggregated") if name in rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--copies", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--users", type=int, nargs="+", default=[1, 16, 64])
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--shard-copies", type=int, default=1, help="copies of each shard")
    ap.add_argument("--set", nargs="*", default=[], help="KEY=VALUE settings for docker compose (BATCH, REDIS_URL, ...)")
    ap.add_argument("--label", default="", help="added to each result")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    SETTINGS.update(kv.split("=", 1) for kv in args.set)
    results = []
    for copies in args.copies:
        scale = ["--scale", f"frontdesk={copies}"] + [x for i in range(4) for x in ("--scale", f"shard-{i}={args.shard_copies}")]
        compose("up", "-d", extra=scale)
        compose("restart", "nginx")  # nginx looks up the copies' addresses when it starts
        wait_healthy()
        time.sleep(3)
        for users in args.users:
            prefix = str(ROOT / f"runs/serving/locust_{args.label or 'run'}_{copies}x_{users}u")
            r = run_locust(users, args.seconds, prefix)
            results.append({"label": args.label, "settings": SETTINGS, "frontdesk_copies": copies,
                            "shard_copies": args.shard_copies, "users": users, **r,
                            "health": json.loads(urllib.request.urlopen("http://127.0.0.1:8000/health").read())})
            rec, songs = r.get("/recommend", {}), r.get("/songs", {})
            print(f"{args.label} {copies} front desk(s), {users:>3} users: /recommend {rec.get('per_s', 0):7.1f}/s p50 {rec.get('p50_ms', 0):5.0f} "
                  f"p99 {rec.get('p99_ms', 0):5.0f} ms | /songs {songs.get('per_s', 0):7.1f}/s p50 {songs.get('p50_ms', 0):5.0f} "
                  f"p99 {songs.get('p99_ms', 0):5.0f} ms | failures {r['Aggregated']['failures']}", flush=True)
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
