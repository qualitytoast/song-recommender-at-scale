"""Break shards while requests flow, and record what users see (docker compose system, stage 3).

    python scripts/chaos.py --cases runs/serving/cases --out runs/serving/chaos.json

Sends requests one after another (the check cases, cycled) to the front desk the whole time:
  normal       baseline
  crash        shard-1's program is killed inside its container; Docker's restart policy starts the
               container again, and the front desk uses the shard again once its health check passes
  hang         shard-2 is frozen (docker compose pause): alive but silent, so calls run into the deadline
  recovered    shard-2 unfrozen
Each request is recorded as ok, partial (an answer from the other shards, flagged) or error, with its
time; the summary gives counts and p50/p99 per phase and how long until answers were complete again.
"""
import argparse
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

KILL_SHARD = ("import os; [os.kill(int(p), 9) for p in os.listdir('/proc') if p.isdigit() "
              "and b'recsys.shard' in open(f'/proc/{p}/cmdline', 'rb').read()]")


def compose(*args):
    env = {**os.environ, "PATH": f"{Path.home()}/.docker/bin:{os.environ['PATH']}"}
    return subprocess.run(["docker", "compose", *args], capture_output=True, text=True, env=env)


def request(url, case):
    body = json.dumps({"songs": case["songs"], "name": case["name"]}).encode()
    start = time.perf_counter()
    try:
        r = json.loads(urllib.request.urlopen(urllib.request.Request(
            f"{url}/recommend", data=body, headers={"content-type": "application/json"}), timeout=10).read())
        status = "partial" if r.get("partial") else "ok"
        missing = r.get("missing_shards", [])
    except (urllib.error.URLError, TimeoutError) as e:
        status, missing = "error", [str(getattr(e, "code", e))]
    return status, (time.perf_counter() - start) * 1000, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=20, help="length of each phase")
    args = ap.parse_args()
    cases = json.loads((Path(args.cases) / "cases.json").read_text())["cases"]
    log, phase, stop = [], ["normal"], threading.Event()

    def load():
        i, start = 0, time.perf_counter()
        while not stop.is_set():
            status, ms, missing = request(args.url, cases[i % len(cases)])
            log.append({"t": time.perf_counter() - start, "phase": phase[0], "status": status, "ms": ms,
                        "missing": missing})
            i += 1
    worker = threading.Thread(target=load)
    worker.start()
    marks = {}
    time.sleep(args.seconds)
    phase[0], marks["crash"] = "crash", log[-1]["t"]
    compose("exec", "-T", "shard-1", "python", "-c", KILL_SHARD)
    time.sleep(args.seconds * 2)
    phase[0], marks["hang"] = "hang", log[-1]["t"]
    compose("pause", "shard-2")
    time.sleep(args.seconds)
    phase[0], marks["recovered"] = "recovered", log[-1]["t"]
    compose("unpause", "shard-2")
    time.sleep(args.seconds)
    stop.set()
    worker.join()

    summary = {}
    for name in ("normal", "crash", "hang", "recovered"):
        rows = [r for r in log if r["phase"] == name]
        ms = np.array([r["ms"] for r in rows])
        summary[name] = {"requests": len(rows), **{s: sum(r["status"] == s for r in rows) for s in ("ok", "partial", "error")},
                         "p50_ms": float(np.percentile(ms, 50)), "p99_ms": float(np.percentile(ms, 99)),
                         "missing": sorted({str(m) for r in rows for m in r["missing"]})}
    for name in ("crash", "recovered"):  # seconds from the event until answers were complete for good
        rows = [r for r in log if r["t"] >= marks[name]]
        bad = [r["t"] for r in rows if r["status"] != "ok" and (name == "recovered" or r["phase"] == "crash")]
        summary[name]["complete_again_after_s"] = (max(bad) - marks[name]) if bad else 0.0
    Path(args.out).write_text(json.dumps({"summary": summary, "marks": marks, "log": log}))
    for name, s in summary.items():
        print(f"{name:<10} {s['requests']:>6,} requests: {s['ok']:>6,} ok, {s['partial']:>5,} partial, {s['error']:>4,} error"
              f" | p50 {s['p50_ms']:6.1f} ms, p99 {s['p99_ms']:7.1f} ms | missing {s['missing']}"
              + (f" | complete again after {s['complete_again_after_s']:.1f} s" if "complete_again_after_s" in s else ""))


if __name__ == "__main__":
    main()
