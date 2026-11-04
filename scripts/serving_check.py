"""Check a running server against the offline evaluation, and time it.

    # 1. test cases with their offline top 10 (no server running: this loads the training data)
    python scripts/serving_check.py cases --ranker configs/ranker_1m_any_length.toml --out runs/serving/cases
    # 2. send them to a server (python -m recsys.serve ...) one at a time over HTTP
    python scripts/serving_check.py serve --cases runs/serving/cases --url http://127.0.0.1:8000 --label cpu
    # timing the slowest requests: straight to a front desk (nginx removes the check headers), every answer computed,
    # each request's trace kept (requests_<label>.jsonl, for scripts/tail_report.py), the cases sent 5 times
    python scripts/serving_check.py serve --cases ... --url http://<front desk>:8000 --label tail --fresh --trace --repeat 5
    # a 16-bit bundle (export --vector-bits 16): first take the expected answers from it, in one process
    python scripts/serving_check.py reference --cases runs/serving/cases --bundle bundles/<bundle> --out <folder>

cases: per_length held-out cases of each length 1..input_length from the ranker's any-length held-out
set (a fixed random sample), each with its songs as Spotify URIs, its playlist name as text (the
known name words, which encode back to the same IDs), and the top 10 the offline evaluation's code
gives it (cached shortlists, the trained ranker).
reference: the same cases with the top 10 the one-process recommender gives on a bundle (no artist limit,
no mixing), for a bundle whose song vectors are rounded (a 16-bit one differs from the full-precision
offline evaluation in ~2% of the orders), so the served answers can still be checked for exact equality.
serve: sends each case to the server once (after a few warm-up requests), compares its top 10 with
the offline one (same list, same order), the NDCG@10 of both, and reports the time per stage (the
server's own timings) and end to end (as the client sees it), p50 and p99, and the memory of the
server and its search process. Saves everything to <cases>/check_<label>.json. --repeat sends the cases that many
times, --concurrency that many at once; --fresh asks for answers computed despite the cache (X-Recsys-Fresh), --trace
for each answer's trace (X-Recsys-Trace: per shard call, recsys/frontdesk.py), saved with the client's own send and
receive times, one line per request, to <cases>/requests_<label>.jsonl (no songs in it, only the case's number).
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_cases(args):
    import torch
    from recsys.ranker import (CandidateRanker, cached_shortlists, check_vocab, feature_fn, load_ranker_config,
                               load_retriever, make_featurizer, rerank)
    from recsys.search import song_vectors
    from recsys.train import pick_device, run_dir_for

    rc = load_ranker_config(args.ranker)
    if rc.windows != "any_length":
        raise ValueError("the check uses any-length cases; this ranker config has windows = full")
    device = pick_device()
    cfg, ds, retriever = load_retriever(rc, args.seed, device)
    L = cfg.data.input_length
    rng = np.random.RandomState(0)
    rows = np.sort(np.concatenate([rng.choice(np.flatnonzero(ds.LI_test == k), args.per_length, replace=False)
                                   for k in range(1, L + 1)]))
    sets = [(ds.XI_test, ds.N_test, ds.Y_test, ds.LI_test)]  # the whole sample: its shortlists are cached
    ((ids, scores, _),), from_cache = cached_shortlists(rc, run_dir_for(rc.retriever, args.seed) / "best.pt",
                                                       retriever, song_vectors(retriever), device, sets)
    ids, scores = ids[rows], scores[rows]
    X, N, Y, lengths = ds.X_test[rows], ds.N_test[rows], ds.Y_test[rows], ds.LI_test[rows]
    featurizer = make_featurizer(rc, ds, retriever, device)
    ranker = CandidateRanker(retriever, len(featurizer.names), rc.correction_hidden, rc.predict_steps).to(device)
    path = run_dir_for(args.ranker, args.seed) / "best.pt"
    checkpoint = torch.load(path, weights_only=True)
    check_vocab(checkpoint, ds.vocab, path)
    ranker.load_state_dict(checkpoint["model"])
    songs, begin, end = ds.H_test
    histories = (songs, begin[rows], end[rows])
    features = feature_fn(featurizer, ids, scores, histories, np.arange(len(rows)), device, None, lengths)
    reranked = rerank(ranker, X, N, ids, scores, device, features=features, lengths=lengths)
    order = np.argsort(-reranked, axis=1, kind="stable")[:, :10]
    top10 = np.take_along_axis(ids, order, axis=1)
    cases = [{"songs": [ds.vocab[j] for j in X[r, :lengths[r]]],
              "name": " ".join(ds.name_words[w - 1] for w in N[r] if w),
              "target": ds.vocab[Y[r]], "length": int(lengths[r]),
              "offline_top10": [ds.vocab[j] for j in top10[r]]} for r in range(len(rows))]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.json").write_text(json.dumps({"ranker": args.ranker, "shortlists_from_cache": bool(from_cache),
                                                "cases": cases}))
    print(f"{len(cases):,} cases ({args.per_length} of each length 1-{L}) -> {out / 'cases.json'}")


def make_reference(args):
    """The cases' expected top 10s from the one-process recommender on a bundle (no artist limit, no mixing):
    the reference for a bundle whose answers differ slightly from the full-precision offline evaluation
    (export --vector-bits 16), so the served answers can still be checked for being exactly the same."""
    from recsys.recommender import Recommender

    data = json.loads((Path(args.cases) / "cases.json").read_text())
    rec = Recommender(args.bundle, "cpu")
    try:
        for case in data["cases"]:
            case["offline_top10"] = [r["uri"] for r in rec.recommend(case["songs"], case["name"], max_per_artist=0,
                                                                       mix=0)["recommendations"]]
    finally:
        rec.close()
    data["reference"] = f"one process, bundle {rec.version}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.json").write_text(json.dumps(data))
    print(f"{len(data['cases']):,} cases, expected answers from {data['reference']} -> {out / 'cases.json'}")


def post(url, body, headers=None):
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json", **(headers or {})})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def ndcg10(top10, target):
    return 1 / np.log2(top10.index(target) + 2) if target in top10 else 0.0


def memory_mb(pattern):
    """Resident memory (MB) of the processes whose command line matches pattern."""
    pids = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
    rss = [int(x) for p in pids for x in subprocess.run(["ps", "-o", "rss=", "-p", p], capture_output=True,
                                                         text=True).stdout.split()]
    return round(sum(rss) / 1024)


def check_server(args):
    data = json.loads((Path(args.cases) / "cases.json").read_text())
    cases = data["cases"]
    url = args.url.rstrip("/")
    headers = {**({"X-Recsys-Fresh": "1"} if args.fresh else {}), **({"X-Recsys-Trace": "1"} if args.trace else {})}
    health = json.loads(urllib.request.urlopen(f"{url}/health").read())

    def send(case):  # as offline: no artist limit, no mixing
        sent, start = time.time(), time.perf_counter()
        reply = post(f"{url}/recommend", {"songs": case["songs"], "name": case["name"], "max_per_artist": 0, "mix": 0},
                     headers)
        return reply, sent, time.time(), (time.perf_counter() - start) * 1000

    for case in cases[:args.warmup]:  # first requests pay one-off costs (e.g. GPU kernels); not timed
        send(case)
    jobs = [(r, i) for r in range(args.repeat) for i in range(len(cases))]
    started = time.perf_counter()
    with ThreadPoolExecutor(args.concurrency) as pool:  # one thread: one request at a time
        answers = list(pool.map(lambda job: send(cases[job[1]]), jobs))
    seconds = time.perf_counter() - started
    same_list, same_set, stages, end_to_end = 0, 0, {}, []
    gains = {"offline": [], "served": []}
    by_length = {}
    records = []
    for (round_, i), (reply, sent, received, total_ms) in zip(jobs, answers):
        case = cases[i]
        end_to_end.append(total_ms)
        served = [r["uri"] for r in reply["recommendations"]]
        same_list += served == case["offline_top10"]
        same_set += set(served) == set(case["offline_top10"])
        for stage, ms in reply["timings_ms"].items():
            stages.setdefault(stage, []).append(ms)
        g_off, g_srv = ndcg10(case["offline_top10"], case["target"]), ndcg10(served, case["target"])
        gains["offline"].append(g_off)
        gains["served"].append(g_srv)
        by_length.setdefault(case["length"], []).append(served == case["offline_top10"])
        if args.trace:
            records.append({"case": i, "pass": round_, "length": case["length"], "sent_s": sent,
                            "received_s": received, "end_to_end_ms": round(total_ms, 3), "same_top10": served == case["offline_top10"],
                            "partial": reply.get("partial", False), "cached": reply.get("cached", False),
                            "timings_ms": reply["timings_ms"], "trace": reply.get("trace")})
    pct = lambda a, q: float(np.percentile(a, q))
    result = {"label": args.label, "health": health, "cases": len(cases), "requests": len(jobs),
              "repeat": args.repeat, "concurrency": args.concurrency, "fresh": args.fresh, "seconds": round(seconds, 1),
              "same_top10_in_order": same_list, "same_top10_as_set": same_set,
              "same_top10_by_length": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_length.items())},
              "ndcg@10": {k: float(np.mean(v)) for k, v in gains.items()},
              "latency_ms": {**{s: {"p50": pct(v, 50), "p99": pct(v, 99)} for s, v in stages.items()},
                             "end to end (client)": {"p50": pct(end_to_end, 50), "p99": pct(end_to_end, 99)}},
              "memory_mb": {"server": memory_mb("recsys.serve"), "search process": memory_mb("recsys.search_worker")}}
    (Path(args.cases) / f"check_{args.label}.json").write_text(json.dumps(result, indent=2))
    if args.trace:
        with open(Path(args.cases) / f"requests_{args.label}.jsonl", "w") as f:
            f.writelines(json.dumps(record) + "\n" for record in records)
    print(f"{args.label}: {len(jobs):,} requests in {seconds:.0f} s ({args.concurrency} at a time) | same top 10 as "
          f"offline: {same_list:,} in order, {same_set:,} as a set "
          f"| NDCG@10 served {result['ndcg@10']['served']:.4f} vs offline {result['ndcg@10']['offline']:.4f}")
    print("  by length: " + ", ".join(f"{k}: {v}" for k, v in result["same_top10_by_length"].items()))
    for stage, t in result["latency_ms"].items():
        print(f"  {stage:<22} p50 {t['p50']:7.2f} ms   p99 {t['p99']:7.2f} ms")
    print(f"  memory: server {result['memory_mb']['server']:,} MB, search process "
          f"{result['memory_mb']['search process']:,} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("cases")
    c.add_argument("--ranker", required=True)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--per-length", type=int, default=200)
    c.add_argument("--out", required=True)
    r = sub.add_parser("reference", help="expected answers from the one-process recommender on a bundle")
    r.add_argument("--cases", required=True)
    r.add_argument("--bundle", required=True)
    r.add_argument("--out", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--cases", required=True)
    s.add_argument("--url", default="http://127.0.0.1:8000")
    s.add_argument("--label", required=True)
    s.add_argument("--warmup", type=int, default=50)
    s.add_argument("--repeat", type=int, default=1, help="send the cases this many times")
    s.add_argument("--concurrency", type=int, default=1, help="requests at once")
    s.add_argument("--fresh", action="store_true", help="answers computed despite the cache (X-Recsys-Fresh)")
    s.add_argument("--trace", action="store_true", help="keep each request's trace (X-Recsys-Trace)")
    args = ap.parse_args()
    {"cases": make_cases, "reference": make_reference, "serve": check_server}[args.mode](args)
