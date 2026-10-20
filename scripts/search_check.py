"""Check the find-songs menu of a running server: what a list of searches finds, and how fast it answers
someone typing.

    python scripts/search_check.py show --url http://127.0.0.1:8000 --out runs/serving/search/show_new.json
    python scripts/search_check.py typing --url http://127.0.0.1:8000 --bundle bundles/<bundle> \\
        --out runs/serving/search/typing_docker.json

show: the top 5 for each query in QUERIES (or --queries a,b,c).
typing: songs drawn by popularity from the bundle, each typed as "track artist" one letter at a time; every
prefix (each one once, so the shared cache can't answer it) is a /songs request, timed end to end, with the
server's /health search counts before and after (how many the search answered, and fell back on).
Needs only the standard library and NumPy (it runs on the server too).
"""
import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

QUERIES = ["one more tpme", "one more ti", "daft punk one more", "discovery daft punk", "radiohed creep", "creep",
           "bts", "love story taylor", "taylor swift", "nightcall kavinsky", "smells like teen", "mr brightside",
           "bohemian rapsody", "blinding lights", "lose yourself eminem", "kendrik lamar humble", "billie eilish",
           "the weeknd", "love", "a"]


def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def songs(url, query, limit):
    return get(f"{url}/songs?{urllib.parse.urlencode({'q': query, 'limit': limit})}")["songs"]


def show(args):
    queries = args.queries.split(",") if args.queries else QUERIES
    out = {}
    for q in queries:
        out[q] = [f"{s['track']} - {s['artist']}" for s in songs(args.url, q, 5)]
        print(f"{q!r}:\n    " + "\n    ".join(out[q] or ["(nothing)"]))
    return out


def typing(args):
    bundle = Path(args.bundle)
    names = json.loads((bundle / "songs.json").read_text())
    artist = np.load(bundle / "song_artist.npy")
    popularity = np.load(bundle / "song_playlists.npy").astype(np.float64)
    rng = np.random.RandomState(args.seed)
    picked = rng.choice(len(popularity), args.songs, replace=False, p=popularity / popularity.sum())
    prefixes = []
    for j in picked:
        text = f"{names['tracks'][j]} {names['artist_names'][artist[j]]}"
        prefixes += [text[:k] for k in range(1, len(text) + 1) if text[k - 1] != " "]
    prefixes = list(dict.fromkeys(p.strip() for p in prefixes if p.strip()))
    before = get(f"{args.url}/health")["search"]
    times = []
    for p in prefixes:
        start = time.perf_counter()
        songs(args.url, p, 8)
        times.append((time.perf_counter() - start) * 1000)
    after = get(f"{args.url}/health")["search"]
    counts = {k: after[k] - before[k] for k in ("answered", "errors", "skipped")} if before and after else None
    out = {"songs": args.songs, "requests": len(prefixes), "p50_ms": round(float(np.percentile(times, 50)), 2),
           "p99_ms": round(float(np.percentile(times, 99)), 2), "max_ms": round(max(times), 2), "search": counts}
    print(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("show")
    s.add_argument("--queries", help="comma-separated (default: QUERIES)")
    t = sub.add_parser("typing")
    t.add_argument("--bundle", required=True)
    t.add_argument("--songs", type=int, default=200)
    t.add_argument("--seed", type=int, default=0)
    for p in (s, t):
        p.add_argument("--url", default="http://127.0.0.1:8000")
        p.add_argument("--out", help="save the result here (JSON)")
    args = ap.parse_args()
    result = show(args) if args.mode == "show" else typing(args)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
