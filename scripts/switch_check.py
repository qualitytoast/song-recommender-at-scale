"""Switch the running system to an update and back under steady traffic (recsys/switch.py), and check that no
request failed and no answer mixed two versions.

    python scripts/switch_check.py --bundle bundles/<bundle> --update updates/<folder> \\
        --cases runs/serving/lb_n3_16_cases --out runs/serving/switch_mac.json

The system (docker compose, its front desk following <bundle>/switch.json) must be serving the bundle itself.
Threads send the cases' songs to --url the whole time, with the bundle's own artist limit and mix, as the
site does. Meanwhile: prepare (the update as standby), wait until the front desk has it ready, flip, hold,
flip back (the undo), hold, finish. From the first step on, every answer must be HTTP 200 and not partial, from
one of the two versions, and the same top 10 as every other answer to the same case from the same version: an
answer that mixed versions would differ (run without the answer cache, REDIS_URL=, so every answer is computed). Reports
answers per version, errors, partial answers, latency overall and within 2 s of each flip, how long after each
flip the old version still answered, and each step's time.
"""
import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys import switch  # noqa: E402


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def wait(condition, timeout, what):
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        try:
            if condition():
                return time.perf_counter() - start
        except (urllib.error.URLError, OSError, KeyError, TypeError):
            pass
        time.sleep(0.2)
    raise TimeoutError(f"waited {timeout:.0f} s for {what}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--update", required=True, help="the update's folder, relative to the bundle")
    ap.add_argument("--cases", required=True, help="a folder with cases.json (scripts/serving_check.py)")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--hold", type=float, default=20.0, help="seconds on each version after a flip")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    bundle, url = Path(args.bundle), args.url.rstrip("/")
    path = bundle / "switch.json"
    base_v = json.loads((bundle / "manifest.json").read_text())["version"]
    update_v = json.loads((bundle / args.update / "manifest.json").read_text())["version"]
    if switch.read(path) != {"live": switch.BASE, "standby": None} or get(f"{url}/health")["version"] != base_v:
        raise SystemExit(f"the system must serve the bundle itself ({base_v}) with no standby to start")
    cases = json.loads((Path(args.cases) / "cases.json").read_text())["cases"]
    log, stop, t0 = [], threading.Event(), time.perf_counter()

    def traffic(k):
        i = k
        while not stop.is_set():
            case = i % len(cases)
            body = json.dumps({"songs": cases[case]["songs"], "name": cases[case]["name"]}).encode()
            start = time.perf_counter()
            try:
                r = get(urllib.request.Request(f"{url}/recommend", data=body,
                                               headers={"content-type": "application/json"}))
                status, version, top = ("partial" if r["partial"] else "ok"), r["version"], \
                    tuple(x["uri"] for x in r["recommendations"])
                missing = r.get("missing_shards")
            except (urllib.error.URLError, OSError) as e:
                status, version, top, missing = f"error {getattr(e, 'code', type(e).__name__)}", None, None, None
            log.append({"t": start - t0, "ms": (time.perf_counter() - start) * 1000, "status": status,
                        "version": version, "case": case, "top": top, "missing": missing})
            i += args.threads

    workers = [threading.Thread(target=traffic, args=(k,)) for k in range(args.threads)]
    for w in workers:
        w.start()
    marks, steps = {}, {}
    try:
        time.sleep(args.hold)
        marks["prepare"] = time.perf_counter() - t0
        switch.prepare(path, args.update)
        steps["standby ready (s)"] = wait(lambda: (get(f"{url}/health")["switch"]["standby"] or {}).get("version")
                                          == update_v, 600, "the standby")
        time.sleep(args.hold)  # the standby loaded; the live version still answers
        marks["flip"] = time.perf_counter() - t0
        switch.flip(path)
        steps["flip seen by /health (s)"] = wait(lambda: get(f"{url}/health")["version"] == update_v, 60, "the flip")
        time.sleep(args.hold)
        marks["undo"] = time.perf_counter() - t0
        switch.flip(path)
        steps["undo seen by /health (s)"] = wait(lambda: get(f"{url}/health")["version"] == base_v, 60, "the undo")
        time.sleep(args.hold)
        marks["finish"] = time.perf_counter() - t0
        switch.finish(path)
        steps["standby let go (s)"] = wait(lambda: get(f"{url}/health")["switch"]["standby"] is None, 60, "finish")
        time.sleep(args.hold)
    finally:
        stop.set()
        for w in workers:
            w.join()
    health = get(f"{url}/health")

    answers = {}  # (version, case) -> the set of top 10s answered
    for r in log:
        if r["version"]:
            answers.setdefault((r["version"], r["case"]), set()).add(r["top"])
    ms = np.array([r["ms"] for r in log])
    near = lambda mark: [r["ms"] for r in log if abs(r["t"] - marks[mark]) < 2]
    after = lambda mark, version, until: [r["t"] - marks[mark] for r in log
                                          if r["version"] == version and marks[mark] <= r["t"] < until]
    pct = lambda a, q: round(float(np.percentile(a, q)), 1) if len(a) else None
    summary = {
        "bundle": base_v, "update": update_v, "threads": args.threads, "seconds": round(time.perf_counter() - t0, 1),
        "requests": len(log), "answers": {v: sum(r["version"] == v for r in log) for v in (base_v, update_v)},
        "errors": sum(r["status"].startswith("error") for r in log), "partial": sum(r["status"] == "partial" for r in log),
        "statuses": sorted({r["status"] for r in log}),
        "versions_seen": sorted({r["version"] for r in log if r["version"]}),
        "cases_with_two_answers_from_one_version": sum(len(tops) > 1 for tops in answers.values()),
        "answers_differing_between_versions": sum(answers.get((update_v, c), set()) != answers.get((base_v, c), set())
                                                  for c in range(len(cases)) if (update_v, c) in answers
                                                  and (base_v, c) in answers),
        "old_version_answers_after_flip_s": max(after("flip", base_v, marks["undo"]), default=None),
        "old_version_answers_after_undo_s": max(after("undo", update_v, marks["finish"]), default=None),
        "latency_ms": {"all": {"p50": pct(ms, 50), "p99": pct(ms, 99)},
                       **{f"within 2 s of {m}": {"p50": pct(near(m), 50), "p99": pct(near(m), 99)} for m in marks}},
        # a front desk's first calls after an idle spell can miss a shard's deadline whether or not anything
        # switches (its circuit breaker then skips the shard for a second): counted apart, before the first step
        "problems_before_prepare": sum(r["status"] != "ok" for r in log if r["t"] < marks["prepare"]),
        "problems_from_prepare_on": sum(r["status"] != "ok" for r in log if r["t"] >= marks["prepare"]),
        "problems": [{"t": round(r["t"], 2), "status": r["status"], "missing_shards": r["missing"]}
                     for r in log if r["status"] != "ok"][:50],
        "steps": {k: round(v, 1) for k, v in steps.items()}, "marks_s": {k: round(v, 1) for k, v in marks.items()},
        "health_after": {k: health[k] for k in ("status", "version", "switch", "answers")}}
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    switched = [r for r in log if r["t"] >= marks["prepare"]]
    tops = {}
    for r in switched:
        if r["version"] and r["status"] == "ok":
            tops.setdefault((r["version"], r["case"]), set()).add(r["top"])
    ok = (not summary["problems_from_prepare_on"] and not any(len(v) > 1 for v in tops.values())
          and summary["versions_seen"] == sorted([base_v, update_v]))
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
