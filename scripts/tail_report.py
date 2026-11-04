"""Where the slowest requests' time went: a timing check's per-request traces (scripts/serving_check.py serve --trace)
set against what the server's CPUs were doing meanwhile (scripts/machine_sampler.py). Runs anywhere (numpy).

    python scripts/tail_report.py --requests runs/serving/tail/requests_<label>.jsonl \\
        --machine runs/serving/tail/machine_<label>.jsonl --out runs/serving/tail/report_<label>.json

The slowest 1% of requests by the front desk's total time are set against typical ones (the 40th-60th percentile):
  stages       each stage's milliseconds (timings_ms), and "other": the total minus the stages (the batcher's wait,
               the top 10 picked, the answer built)
  shard rounds each round's slowest call, its round trip split into the wait before the shard started on it, the
               shard's work, and the wait after (the reply's way back); and how many of the round's shards were slow
               (round trip over twice the median of that kind of call): one alone points at that shard's process,
               all of them at the machine
  cases        whether the same cases are slow on every pass (serving_check --repeat): slow by their size, or by luck
  machine      during each request: each container's CPU and its threads' wait for a CPU, the machine's busy CPU and
               steal, threads ready to run, and the health checks running (each a new Python process); and the same
               for the whole machine before the first request (the sampler's quiet minute) and during the run
  garbage      the front desk's and the shards' garbage-collection pauses
With --baseline (another run's requests file), the totals' p50 and p99 against it, with bootstrap 95% ranges of the
changes and whether the keep bar for a speed-up is met. CPU is given in cores (CPU milliseconds per millisecond of the
request; the server has 2). The machine is sampled
every 50 ms, so a request's share of it is interpolated, smeared over about that much on either side.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROUNDS = ("shards: input songs", "shards: search", "shards: candidates")
TICK_MS = 10.0  # /proc/stat counts in clock ticks of 1/100 s


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def calls_of(record, stage):
    """A request's shard calls in one round, with their round trip, wait before, work and wait after (ms)."""
    out = []
    for c in record["trace"]["shard_calls"]:
        if c["stage"] != stage or c.get("failed") or "arrived_s" not in c:
            continue
        trip = (c["arrived_s"] - c["sent_s"]) * 1000
        before = (c["start_s"] - c["sent_s"]) * 1000
        out.append({**c, "trip": trip, "before": before, "after": trip - before - c["work_ms"]})
    return out


def machine_counters(path):
    """Sample times and {counter name: values} from a machine_sampler file, every counter in ms."""
    header, *rows = load_jsonl(path)
    t = np.array([r["t"] for r in rows])
    cpu = lambda *kinds: np.array([sum(r["cpu"][k] for k in kinds) for r in rows], dtype=float) * TICK_MS
    out = {"busy": cpu("user", "nice", "system", "irq", "softirq", "steal"), "steal": cpu("steal"),
           "waited": np.array([r["waited_us"] for r in rows], dtype=float) / 1000}
    for name in header["groups"]:
        for i, kind in enumerate(("cpu", "waited")):
            # a group missing from a sample (a container restarting) keeps its last value
            values, last = [], 0.0
            for r in rows:
                last = r["groups"][name][i] / 1000 if name in r["groups"] else last
                values.append(last)
            out[f"{kind}: {name}"] = np.array(values)
    running = np.array([r["running"] for r in rows], dtype=float)
    return t, out, running


def health_checks(path):
    """[(start, end, container)] of every exec (health check) in an events file that both started and ended."""
    starts, out = {}, []
    for e in load_jsonl(path):
        if e["action"] == "exec_start":
            starts[e["exec"]] = e
        elif e["exec"] in starts:
            s = starts.pop(e["exec"])
            out.append((s["t"], e["t"], s["container"]))
    return out


def group_name(name):
    return "shards" if name.startswith("shard-") else name


def summarize(values):
    values = np.asarray(values, dtype=float)
    return round(float(values.mean()), 2) if len(values) else None


def report(requests, machine=None, events=None):
    records = [r for r in requests if r.get("trace") and not r["cached"]]
    total = np.array([r["timings_ms"]["total"] for r in records])
    p40, p60, p99 = np.percentile(total, [40, 60, 99])
    groups = {"slow": total >= p99, "typical": (total >= p40) & (total <= p60)}
    out = {"requests": len(records), "partial": sum(r["partial"] for r in records),
           "total_ms": {f"p{q}": round(float(np.percentile(total, q)), 2) for q in (50, 90, 99, 99.9)},
           "max_ms": round(float(total.max()), 2),
           "groups": {name: int(mask.sum()) for name, mask in groups.items()}}

    # stages: the slow ones' extra time, stage by stage
    stage_names = [s for s in records[0]["timings_ms"] if s not in ("total", "batch size")]
    stage = {s: np.array([r["timings_ms"].get(s, 0.0) for r in records]) for s in stage_names}
    stage["other"] = total - sum(stage.values())
    out["stages_ms"] = {s: {g: summarize(v[m]) for g, m in groups.items()} for s, v in stage.items()}

    # shard rounds: work or waiting; one shard slow or all of them
    by_round = {s: [calls_of(r, s) for r in records] for s in ROUNDS}
    medians = {}
    for s in ROUNDS:
        for c in (c for calls in by_round[s] for c in calls):
            medians.setdefault((s, c["call"]), []).append(c["trip"])
    medians = {k: float(np.median(v)) for k, v in medians.items()}
    out["shard_rounds"] = {}
    for s in ROUNDS:
        slowest = [max(calls, key=lambda c: c["trip"]) if calls else None for calls in by_round[s]]
        slow_shards = []  # per request: (shards with a slow call, shards in the round)
        for calls in by_round[s]:
            slow = {c["shard"] for c in calls if c["trip"] > 2 * medians[(s, c["call"])]}
            slow_shards.append((len(slow), len({c["shard"] for c in calls})))
        entry = {}
        for g, m in groups.items():
            picked = [c for c, keep in zip(slowest, m) if keep and c is not None]
            counts = [n for (n, _), keep in zip(slow_shards, m) if keep and n > 0]
            entry[g] = {"slowest call: trip": summarize([c["trip"] for c in picked]),
                        "before": summarize([c["before"] for c in picked]),
                        "work": summarize([c["work_ms"] for c in picked]),
                        "after": summarize([c["after"] for c in picked]),
                        "rounds with a slow call": len(counts),
                        "slow shards in those rounds": {n: counts.count(n) for n in sorted(set(counts))}}
        out["shard_rounds"][s] = entry

    # cases: the same cases slow on every pass, or different ones
    slow_cases = [r["case"] for r, keep in zip(records, groups["slow"]) if keep]
    passes = max(r["pass"] for r in records) + 1
    times_slow = {c: slow_cases.count(c) for c in set(slow_cases)}
    lengths = np.array([r["length"] for r in records])
    out["cases"] = {"passes": passes, "slow requests": len(slow_cases), "distinct cases": len(times_slow),
                    "slow requests whose case was slow in another pass too": sum(n for n in times_slow.values() if n > 1),
                    "if slowness were luck, about": round(len(slow_cases) * (passes - 1) * 0.01, 1),
                    "mean length": {"slow": summarize(lengths[groups["slow"]]), "all": summarize(lengths)}}

    # garbage collection
    desk_gc = np.array([r["trace"]["gc_ms"] for r in records])
    shard_gc = np.array([sum(c.get("gc_ms", 0.0) for c in r["trace"]["shard_calls"]) for r in records])
    out["garbage_ms"] = {"front desk": {g: summarize(desk_gc[m]) for g, m in groups.items()},
                         "shards (sum of calls)": {g: summarize(shard_gc[m]) for g, m in groups.items()},
                         "requests with a front-desk pause over 1 ms": int((desk_gc > 1).sum())}

    # the machine during each request
    start = np.array([r["trace"]["start_s"] for r in records])
    end = start + total / 1000
    if machine:
        t, counters, running = machine
        inside = (start >= t[0]) & (end <= t[-1])
        width = (end - start) * 1000
        rate = {name: (np.interp(end, t, v) - np.interp(start, t, v)) / width for name, v in counters.items()}
        cores = {}
        for name, v in rate.items():
            kind, _, who = name.partition(": ")
            key = f"{kind}: {group_name(who)}" if who else name
            cores[key] = cores.get(key, 0.0) + v
        cpu_names = [k for k in cores if k.startswith("cpu: ")]
        cores["cpu: anything else"] = cores["busy"] - sum(cores[k] for k in cpu_names)
        ready = np.array([running[(t >= a - 0.05) & (t <= b + 0.05)].mean() if ((t >= a - 0.05) & (t <= b + 0.05)).any()
                          else np.nan for a, b in zip(start, end)])
        out["machine_cores"] = {k: {g: summarize(v[m & inside]) for g, m in groups.items()} for k, v in sorted(cores.items())}
        out["machine_cores"]["threads ready to run"] = {g: summarize(ready[m & inside & ~np.isnan(ready)])
                                                        for g, m in groups.items()}
        # the machine as a whole: before the first request (the sampler starts a quiet minute early) and during the run
        out["machine_cores_before_and_during"] = {}
        for when, (a, b) in {"before": (t[0], start.min()), "during": (start.min(), end.max())}.items():
            if b - a < 10:  # under 10 s of samples: not enough to say
                continue
            per = {}
            for name, v in counters.items():
                kind, _, who = name.partition(": ")
                key = f"{kind}: {group_name(who)}" if who else name
                per[key] = per.get(key, 0.0) + (np.interp(b, t, v) - np.interp(a, t, v)) / ((b - a) * 1000)
            per["cpu: anything else"] = per["busy"] - sum(v for k, v in per.items() if k.startswith("cpu: "))
            out["machine_cores_before_and_during"][f"{when} ({b - a:.0f} s)"] = {k: round(v, 3) for k, v in sorted(per.items())}
    if events is not None:
        checks = [c for c in events if c[0] <= end.max() and c[1] >= start.min()]
        overlap = np.array([sum(1 for a, b, _ in checks if a < e and b > s) for s, e in zip(start, end)])
        durations = np.array([(b - a) * 1000 for a, b, _ in checks])
        span = end.max() - start.min()
        out["health_checks"] = {
            "checks during the run": len(checks), "per second": round(len(checks) / span, 2),
            "duration_ms": {"p50": summarize([np.median(durations)]), "p99": summarize([np.percentile(durations, 99)])}
            if len(durations) else None,
            "requests overlapping one": {g: round(float((overlap[m] > 0).mean()), 3) for g, m in groups.items()},
            "checks overlapping, mean": {g: summarize(overlap[m]) for g, m in groups.items()},
            # every request by how many checks ran during it: what the total looks like with none
            "total_ms by checks running": {
                label: {"requests": int(m.sum()), "p50": round(float(np.percentile(total[m], 50)), 2),
                        "p99": round(float(np.percentile(total[m], 99)), 2)}
                for label, m in {"0": overlap == 0, "1": overlap == 1, "2": overlap == 2, "3+": overlap >= 3}.items()
                if m.sum() >= 20}}
    return out


def totals(requests):
    return np.array([r["timings_ms"]["total"] for r in requests if r.get("trace") and not r["cached"]])


def compare(base, new, rounds=2000, seed=0):
    """Two runs' total p50 and p99, and for each the 95% range of new minus base over bootstrap rounds (both runs
    resampled with replacement): a range below 0 says new is faster beyond the runs' own noise. Also whether new
    passes the keep bar for a speed-up (p99 at least 5% lower, its range below 0, p50 at most 5% higher)."""
    rng = np.random.default_rng(seed)
    out = {}
    for q in (50, 99):
        b, n = float(np.percentile(base, q)), float(np.percentile(new, q))
        diffs = [np.percentile(rng.choice(new, len(new)), q) - np.percentile(rng.choice(base, len(base)), q)
                 for _ in range(rounds)]
        out[f"p{q}"] = {"baseline": round(b, 2), "new": round(n, 2), "change_ms": round(n - b, 2),
                        "change_pct": round(100 * (n - b) / b, 1),
                        "95% range_ms": [round(float(x), 2) for x in np.percentile(diffs, [2.5, 97.5])]}
    out["passes the keep bar"] = bool(out["p99"]["change_pct"] <= -5 and out["p99"]["95% range_ms"][1] < 0
                                      and out["p50"]["change_pct"] <= 5)
    return out


def show(out):
    print(f"{out['requests']:,} requests ({out['partial']} partial): total " +
          ", ".join(f"{k} {v} ms" for k, v in out["total_ms"].items()) + f", max {out['max_ms']} ms; "
          f"slowest 1%: {out['groups']['slow']}, typical (40th-60th): {out['groups']['typical']}")
    print("\nstages (ms)                      slow   typical")
    for s, v in out["stages_ms"].items():
        print(f"  {s:<28} {v['slow']:7.2f}  {v['typical']:7.2f}")
    print("\nshard rounds: the slowest call (ms)   trip  before   work  after | rounds with a slow call: shards slow")
    for s, v in out["shard_rounds"].items():
        for g in ("slow", "typical"):
            e = v[g]
            print(f"  {s:<22} {g:<8} {e['slowest call: trip']:6.2f} {e['before']:6.2f} {e['work']:6.2f} "
                  f"{e['after']:6.2f} | {e['rounds with a slow call']}: {e['slow shards in those rounds']}")
    print("\ncases:", json.dumps(out["cases"]))
    print("garbage (ms):", json.dumps(out["garbage_ms"]))
    if "machine_cores" in out:
        print("\nmachine during the request (cores)  slow   typical")
        for k, v in out["machine_cores"].items():
            print(f"  {k:<32} {v['slow'] if v['slow'] is not None else float('nan'):6.3f}  "
                  f"{v['typical'] if v['typical'] is not None else float('nan'):6.3f}")
        spans = out["machine_cores_before_and_during"]
        if spans:
            print("\nthe machine (cores)                " + "  ".join(f"{k:>14}" for k in spans))
            for name in next(iter(spans.values())):
                print(f"  {name:<32} " + "  ".join(f"{v.get(name, float('nan')):14.3f}" for v in spans.values()))
    if "against_baseline" in out:
        c = out["against_baseline"]
        print(f"\nagainst the baseline ({c['baseline']}):")
        for q in ("p50", "p99"):
            v = c[q]
            print(f"  total {q}: {v['baseline']} -> {v['new']} ms ({v['change_pct']:+.1f}%, 95% range of the change "
                  f"{v['95% range_ms'][0]:+.2f} to {v['95% range_ms'][1]:+.2f} ms)")
        print(f"  passes the keep bar: {c['passes the keep bar']}")
    if "health_checks" in out:
        h = {k: v for k, v in out["health_checks"].items() if k != "total_ms by checks running"}
        print("\nhealth checks:", json.dumps(h))
        print("total by health checks running during the request:")
        for k, v in out["health_checks"]["total_ms by checks running"].items():
            print(f"  {k:>2}: {v['requests']:6,} requests, p50 {v['p50']:7.2f} ms, p99 {v['p99']:7.2f} ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--requests", required=True, help="requests_<label>.jsonl from serving_check.py serve --trace")
    ap.add_argument("--machine", help="machine_<label>.jsonl from machine_sampler.py (its .events.jsonl beside it)")
    ap.add_argument("--out", help="where to save the report (JSON)")
    ap.add_argument("--baseline", help="another run's requests_<label>.jsonl to compare the totals with")
    args = ap.parse_args()
    machine = events = None
    if args.machine:
        machine = machine_counters(args.machine)
        events_path = Path(args.machine).with_suffix(".events.jsonl")
        events = health_checks(events_path) if events_path.exists() else None
    requests = load_jsonl(args.requests)
    out = report(requests, machine, events)
    if args.baseline:
        out["against_baseline"] = {"baseline": args.baseline, **compare(totals(load_jsonl(args.baseline)), totals(requests))}
    show(out)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
