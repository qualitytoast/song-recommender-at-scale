"""What storing the song vectors in fewer bits costs: today's 32-bit floats vs 16-bit floats vs 8 bits.

    # 1. variant bundles (FAISS only: no PyTorch in this process)
    python scripts/quantize_check.py build --bundle bundles/<bundle> --out <scratch folder>
    # 2. the held-out cases through each (the serving code: recsys.recommender)
    python scripts/quantize_check.py score --bundle bundles/<bundle> --variants <scratch folder> \\
        --cases runs/listenbrainz/compare --out runs/quantize

build: a copy of the bundle per precision (unchanged files as APFS clones, no extra disk), with the two
per-song input-vector tables rounded to that precision (8 bits: each of the 64 dimensions split into 256 even
steps between its lowest and highest value; 16 bits: half-precision floats) and the search index rebuilt with
FAISS's scalar quantizer at that precision on the same IVF clusters, so only the precision changes. The rounded
tables are stored as 32-bit floats here (the accuracy test); their size at that precision is reported.
score: the comparison's 5,000 held-out ListenBrainz cases ("all"), answered with the bundle's own settings by
each variant: next-song NDCG@10 and hit@10, the shortlist's recall (the next song among the reranked
candidates), and agreement with the 32-bit answers (songs shared in the top 10, identical top 10s).
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INPUTS = ["retriever_song_inputs.npy", "ranker_song_inputs.npy"]
VARIANTS = {"16": "QT_fp16", "8": "QT_8bit"}


def rounded(a, bits):
    """a (songs, d) at the given precision, back as float32: 16 = half floats; 8 = per-dimension min-max steps."""
    if bits == "16":
        return a.astype(np.float16).astype(np.float32)
    lo, hi = a.min(0), a.max(0)
    step = np.where(hi > lo, (hi - lo) / 255, 1.0)
    return (np.round((a - lo) / step).clip(0, 255) * step + lo).astype(np.float32)


def build(args):
    import faiss

    from recsys.shard_slices import cluster_centres

    bundle, out = Path(args.bundle), Path(args.out)
    centres = cluster_centres(bundle / "search_index.faiss")
    nlist, d = centres.shape
    vectors = np.ascontiguousarray(np.load(bundle / "search_vectors.npy"), dtype=np.float32)
    sizes = {"32": {f: (bundle / f).stat().st_size for f in INPUTS + ["search_index.faiss"]}}
    for bits, qtype in VARIANTS.items():
        start, folder = time.perf_counter(), out / f"bits{bits}"
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        for f in bundle.iterdir():
            if f.is_file() and f.name not in INPUTS + ["search_index.faiss"]:
                subprocess.run(["cp", "-c", str(f), str(folder / f.name)], check=True)  # an APFS clone
        sizes[bits] = {}
        for f in INPUTS:
            a = np.load(bundle / f)
            np.save(folder / f, rounded(a, bits))
            sizes[bits][f] = a.size * int(bits) // 8 + (2 * a.shape[1] * 4 if bits == "8" else 0)
        quantizer = faiss.IndexFlatIP(d)
        quantizer.add(centres)  # the same clusters: only the precision changes
        index = faiss.IndexIVFScalarQuantizer(quantizer, d, nlist, getattr(faiss.ScalarQuantizer, qtype),
                                              faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)
        index.add(vectors)
        faiss.write_index(index, str(folder / "search_index.faiss"))
        sizes[bits]["search_index.faiss"] = (folder / "search_index.faiss").stat().st_size
        print(f"{bits}-bit bundle in {time.perf_counter() - start:.0f} s: {folder}", flush=True)
    (out / "sizes.json").write_text(json.dumps(sizes, indent=2))
    print(json.dumps(sizes, indent=2))


def score(args):
    from recsys.recommender import Recommender, pick_top

    cases = json.loads((Path(args.cases) / "cases.json").read_text())["all"]
    sizes = json.loads((Path(args.variants) / "sizes.json").read_text())
    paths = {"32": Path(args.bundle), **{b: Path(args.variants) / f"bits{b}" for b in VARIANTS}}
    answers, results, began = {}, {"cases": len(cases)}, time.perf_counter()
    for bits, path in paths.items():
        start = time.perf_counter()
        rec = Recommender(path, "cpu")
        rows, tops = [], []
        for c in cases:
            ids = [rec.song_id[u] for u in c["lb"]["songs"]]
            ranked, scores, _, _ = rec.recommend_ids(ids, rec.encode_name(""), rec.mix)
            top = pick_top(ranked, scores, rec.song_artist, rec.max_per_artist)[0].tolist()
            target = rec.song_id[c["lb"]["target"]]
            rows.append([1 / np.log2(top.index(target) + 2) if target in top else 0.0, target in top,
                         target in ranked.tolist()])
            tops.append(top)
        rec.close()
        answers[bits] = tops
        m = np.array(rows, dtype=np.float64).mean(0)
        results[bits] = {"next_ndcg@10": round(m[0], 4), "next_hit@10": round(m[1], 4),
                         "shortlist_recall": round(m[2], 4), "bytes": sizes[bits],
                         "seconds": round(time.perf_counter() - start)}
        if bits != "32":
            shared = [len(set(a) & set(b)) / 10 for a, b in zip(answers["32"], tops)]
            results[bits]["same_songs_in_top_10"] = round(float(np.mean(shared)), 4)
            results[bits]["identical_top_10"] = round(float(np.mean([a == b for a, b in zip(answers["32"], tops)])), 4)
        print(json.dumps({bits: results[bits]}), flush=True)
    results["seconds"] = round(time.perf_counter() - began)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "check.json").write_text(json.dumps(results, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("build")
    b.add_argument("--bundle", required=True)
    b.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--bundle", required=True)
    s.add_argument("--variants", required=True)
    s.add_argument("--cases", default="runs/listenbrainz/compare")
    s.add_argument("--out", default="runs/quantize")
    args = ap.parse_args()
    (build if args.mode == "build" else score)(args)


if __name__ == "__main__":
    main()
