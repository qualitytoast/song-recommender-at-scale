"""Search benchmark: exact search vs FAISS indexes, with and without sharding.

    # speed and memory at a given catalog size, with stand-in vectors
    python scripts/search_benchmark.py synthetic --songs 1053328
    # speed and search recall on a trained retriever's real vectors and held-out queries
    python scripts/search_benchmark.py real --config configs/retriever_200k.toml --seed 1

For each method: build time, memory, batch speed (queries per second for a big
batch, as when building shortlists), single-request latency (one query at a
time, median and 99th percentile, measured inside the search process and end to
end through the pipe), and search recall@k (share of exact search's top k it
returns). In "real" mode also: how often the true next song is in the top k.

Stand-in vectors are random, with no cluster structure, so approximate methods
find fewer true neighbours there than on real vectors: use "synthetic" for speed
and memory, "real" for recall. Run it when nothing else is using the machine.
Results are printed and saved to runs/search_benchmark/<mode>_<songs>.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys.search import ExactSearch, SearchWorker, query_vectors, search_recall, song_vectors  # noqa: E402
from recsys.train import pick_device  # noqa: E402


def methods(n_songs):
    """(name, build parameters) to compare. IVF uses ~4*sqrt(songs) clusters (a common
    starting point); sharded IVF splits clusters and probes across 4 shards, so it
    visits the same share of the catalog as the unsharded version."""
    nlist = 2 ** int(round(np.log2(4 * np.sqrt(n_songs))))
    out = [("flat", dict(kind="flat")), ("flat, 4 shards", dict(kind="flat", shards=4))]
    for nprobe in (16, 64, 128, 256):
        out.append((f"ivf nlist={nlist} nprobe={nprobe}", dict(kind="ivf", nlist=nlist, nprobe=nprobe)))
    out.append((f"ivf nprobe=64, 4 shards", dict(kind="ivf", shards=4, nlist=nlist // 4, nprobe=16)))
    for ef in (512, 1024):
        out.append((f"hnsw M=32 ef={ef}", dict(kind="hnsw", hnsw_m=32, ef_search=ef)))
    out.append(("hnsw ef=512, 4 shards", dict(kind="hnsw", shards=4, hnsw_m=32, ef_search=512)))
    return out


def summarize(name, ids, exact_ids, batch_seconds, n_queries, latency, end_to_end, build, memory, targets):
    row = {"method": name, "build_s": round(build, 1), "memory_mb": round(memory / 1e6),
           "queries_per_s": round(n_queries / batch_seconds), "search_recall": float(search_recall(ids, exact_ids).mean()),
           "latency_p50_ms": float(np.percentile(latency, 50)), "latency_p99_ms": float(np.percentile(latency, 99)),
           "end_to_end_p50_ms": float(np.percentile(end_to_end, 50)), "end_to_end_p99_ms": float(np.percentile(end_to_end, 99))}
    if targets is not None:
        row["true_song_in_top_k"] = float((ids == targets[:, None]).any(axis=1).mean())
    return row


def main():
    ap = argparse.ArgumentParser(description="Exact search vs FAISS, with and without sharding.")
    ap.add_argument("mode", choices=["synthetic", "real"])
    ap.add_argument("--songs", type=int, default=1_053_328, help="synthetic: catalog size")
    ap.add_argument("--config", help="real: retriever config")
    ap.add_argument("--seed", type=int, default=1, help="real: retriever training seed")
    ap.add_argument("--queries", type=int, default=5000, help="queries for batch speed and recall")
    ap.add_argument("--latency-queries", type=int, default=300, help="queries timed one at a time")
    ap.add_argument("--k", type=int, default=500)
    ap.add_argument("--only", nargs="*", help="run only methods whose name contains one of these")
    args = ap.parse_args()
    device, rng, targets = pick_device(), np.random.RandomState(0), None

    if args.mode == "synthetic":
        vectors = rng.randn(args.songs, 65).astype(np.float32)
        queries = rng.randn(args.queries, 65).astype(np.float32)
        label = f"synthetic_{args.songs}"
    else:
        from recsys.config import load_config
        from recsys.data import build_dataset
        from recsys.model import build_model
        from recsys.train import run_dir_for
        cfg = load_config(args.config)
        ds = build_dataset(cfg)
        model = build_model(cfg, ds).to(device)
        model.load_state_dict(torch.load(run_dir_for(args.config, args.seed) / "best.pt", weights_only=True)["model"])
        pick = np.sort(rng.choice(len(ds.Y_test), args.queries, replace=False))  # a fixed sample of held-out windows
        vectors = song_vectors(model)
        queries = query_vectors(model, ds.XI_test[pick], ds.N_test[pick], device, lengths=ds.LI_test[pick])
        targets = ds.Y_test[pick]
        label = f"real_{Path(args.config).stem}_{len(vectors)}"
    latency_queries = queries[:args.latency_queries]
    print(f"{label}: {len(vectors):,} songs x {vectors.shape[1]} numbers | {len(queries):,} queries | top {args.k}",
          flush=True)

    rows = []
    exact = ExactSearch(vectors, device)
    exact.search(queries[:100], args.k)  # warm up
    start = time.perf_counter()
    exact_ids, _ = exact.search(queries, args.k)
    batch = time.perf_counter() - start
    lat = exact.latency_ms(latency_queries, args.k)
    rows.append(summarize(f"exact ({device.type} GPU)" if device.type != "cpu" else "exact (CPU)", exact_ids, exact_ids,
                          batch, len(queries), lat, lat, 0.0, exact.memory_bytes(), targets))
    print(json.dumps(rows[-1]), flush=True)

    with SearchWorker() as worker:
        for name, params in methods(len(vectors)):
            if args.only and not any(o in name for o in args.only):
                continue
            built = worker.build(name, vectors, **params)
            worker.search(name, queries[:100], args.k)  # warm up
            start = time.perf_counter()
            ids, _ = worker.search(name, queries, args.k)
            batch = time.perf_counter() - start
            rows.append(summarize(name, ids, exact_ids, batch, len(queries), worker.latency_ms(name, latency_queries, args.k),
                                  worker.end_to_end_ms(name, latency_queries, args.k), built["build_seconds"],
                                  built["memory_bytes"], targets))
            worker.drop(name)
            print(json.dumps(rows[-1]), flush=True)

    out = ROOT / "runs" / "search_benchmark" / f"{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n{'method':32}{'build s':>8}{'MB':>7}{'queries/s':>11}{'recall':>8}"
          f"{'p50 ms':>8}{'p99 ms':>8}{'e2e p50':>9}{'e2e p99':>9}" + (f"{'true@k':>8}" if targets is not None else ""))
    for r in rows:
        print(f"{r['method']:32}{r['build_s']:>8.1f}{r['memory_mb']:>7}{r['queries_per_s']:>11,}{r['search_recall']:>8.3f}"
              f"{r['latency_p50_ms']:>8.2f}{r['latency_p99_ms']:>8.2f}{r['end_to_end_p50_ms']:>9.2f}"
              f"{r['end_to_end_p99_ms']:>9.2f}" + (f"{r['true_song_in_top_k']:>8.3f}" if targets is not None else ""))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
