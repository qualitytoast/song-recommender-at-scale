"""Check a running server against the offline evaluation, and time it.

    # 1. test cases with their offline top 10 (no server running: this loads the training data)
    python scripts/serving_check.py cases --ranker configs/ranker_1m_any_length.toml --out runs/serving/cases
    # 2. send them to a server (python -m recsys.serve ...) one at a time over HTTP
    python scripts/serving_check.py serve --cases runs/serving/cases --url http://127.0.0.1:8000 --label cpu

cases: per_length held-out cases of each length 1..input_length from the ranker's any-length held-out
set (a fixed random sample), each with its songs as Spotify URIs, its playlist name as text (the
known name words, which encode back to the same IDs), and the top 10 the offline evaluation's code
gives it (cached shortlists, the trained ranker).
serve: sends each case to the server once (after a few warm-up requests), compares its top 10 with
the offline one (same list, same order), the NDCG@10 of both, and reports the time per stage (the
server's own timings) and end to end (as the client sees it), p50 and p99, and the memory of the
server and its search process. Saves everything to <cases>/check_<label>.json.
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
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
    ranker = CandidateRanker(retriever, len(featurizer.names), rc.correction_hidden).to(device)
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


def post(url, body):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
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
    health = json.loads(urllib.request.urlopen(f"{url}/health").read())
    for case in cases[:args.warmup]:  # first requests pay one-off costs (e.g. GPU kernels); not timed
        post(f"{url}/recommend", {"songs": case["songs"], "name": case["name"], "max_per_artist": 0, "mix": 0})
    same_list, same_set, stages, end_to_end = 0, 0, {}, []
    gains = {"offline": [], "served": []}
    by_length = {}
    for case in cases:
        start = time.perf_counter()
        reply = post(f"{url}/recommend", {"songs": case["songs"], "name": case["name"],
                                          "max_per_artist": 0, "mix": 0})  # as offline: no limit, no mixing
        end_to_end.append((time.perf_counter() - start) * 1000)
        served = [r["uri"] for r in reply["recommendations"]]
        same_list += served == case["offline_top10"]
        same_set += set(served) == set(case["offline_top10"])
        for stage, ms in reply["timings_ms"].items():
            stages.setdefault(stage, []).append(ms)
        g_off, g_srv = ndcg10(case["offline_top10"], case["target"]), ndcg10(served, case["target"])
        gains["offline"].append(g_off)
        gains["served"].append(g_srv)
        by_length.setdefault(case["length"], []).append(served == case["offline_top10"])
    pct = lambda a, q: float(np.percentile(a, q))
    result = {"label": args.label, "health": health, "cases": len(cases),
              "same_top10_in_order": same_list, "same_top10_as_set": same_set,
              "same_top10_by_length": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_length.items())},
              "ndcg@10": {k: float(np.mean(v)) for k, v in gains.items()},
              "latency_ms": {**{s: {"p50": pct(v, 50), "p99": pct(v, 99)} for s, v in stages.items()},
                             "end to end (client)": {"p50": pct(end_to_end, 50), "p99": pct(end_to_end, 99)}},
              "memory_mb": {"server": memory_mb("recsys.serve"), "search process": memory_mb("recsys.search_worker")}}
    (Path(args.cases) / f"check_{args.label}.json").write_text(json.dumps(result, indent=2))
    print(f"{args.label}: {len(cases):,} requests | same top 10 as offline: {same_list:,} in order, {same_set:,} as a set "
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
    s = sub.add_parser("serve")
    s.add_argument("--cases", required=True)
    s.add_argument("--url", default="http://127.0.0.1:8000")
    s.add_argument("--label", required=True)
    s.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()
    make_cases(args) if args.mode == "cases" else check_server(args)
