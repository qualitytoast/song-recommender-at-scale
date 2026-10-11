"""Score served recommendations on held-out playlists: the next song, and the rest of the playlist.

    # 1. test cases from held-out playlists (no server or models needed)
    python scripts/playlist_eval.py cases --config configs/retriever_1m.toml --bundle bundles/<bundle> \\
        --out runs/serving/playlist_eval
    # 2. score the recommender in this process with each serving setting
    python scripts/playlist_eval.py score --cases runs/serving/playlist_eval --bundle bundles/<bundle>

cases: --cases cases sampled at random from every held-out any-length case (a known song with a known
song right before it; the input is up to input_length known songs before it, as in the offline
evaluation), each with its songs and the rest of its playlist as Spotify URIs and its playlist name.
score: the top 10 for each case and setting (Catalog.recommend's own code), and per setting:
  next hit@10     the case's next song is in the top 10 (the offline evaluation's question)
  later p@10      share of the 10 that the playlist goes on to contain (the next song or any later one):
                  do the recommendations belong in the playlist, not only right after its last song?
  artists         different artists in the top 10
  last artist     share of the top 10 by the last input song's artist
for every case and for "mixed" cases (3+ input songs, every one by a different artist).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_cases(args):
    from recsys.config import load_config
    from recsys.data import concat_playlists, split_playlists
    from recsys.mpd_store import load_store

    d = load_config(args.config).data
    store = load_store(d.folder)
    lengths = np.diff(store.playlist_offsets)
    kept = np.flatnonzero(lengths >= d.min_playlist_len)[:d.max_playlists]
    _, _, held_out = split_playlists(list(range(len(kept))), d.test_split, d.val_split,
                                     load_config(args.config).data_seed)
    playlists = kept[np.array(held_out, dtype=np.int64)]
    tracks, offsets = concat_playlists(store, playlists)
    # store track ID -> the bundle's song ID (-1: not in the catalog)
    catalog = json.loads((Path(args.bundle) / "songs.json").read_text())["uris"]
    in_catalog = {uri: j for j, uri in enumerate(catalog)}
    to_bundle = np.array([in_catalog.get(uri, -1) for uri in store.track_uris], dtype=np.int64)
    songs = to_bundle[tracks]
    playlist = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    t = np.arange(1, len(songs))
    t = t[(songs[t] >= 0) & (songs[t - 1] >= 0) & (playlist[t] == playlist[t - 1])]  # as any_length_cases
    t = np.sort(np.random.RandomState(args.seed).choice(t, args.cases, replace=False))
    last_unknown = np.maximum.accumulate(np.where(songs < 0, np.arange(len(songs)), -1))
    cases = []
    for target in t.tolist():
        p = playlist[target]
        start = max(offsets[p], last_unknown[target - 1] + 1, target - args.input_length)
        rest = songs[target:offsets[p + 1]]
        cases.append({"songs": [catalog[j] for j in songs[start:target]],
                      "name": store.playlist_names[playlists[p]],
                      "target": catalog[songs[target]],
                      "later": [catalog[j] for j in dict.fromkeys(rest[rest >= 0].tolist())]})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.json").write_text(json.dumps({"config": args.config, "seed": args.seed, "cases": cases}))
    print(f"{len(cases):,} cases from {len(playlists):,} held-out playlists -> {out / 'cases.json'}")


SETTINGS = [{"max_per_artist": 0, "mix": 0.0},   # the models alone, as trained and evaluated offline
            {"max_per_artist": 2, "mix": 0.0},
            {"max_per_artist": 0, "mix": 0.5},
            {"max_per_artist": 2, "mix": 0.5}]   # the served default


def score_cases(args):
    from recsys.recommender import Recommender

    cases = json.loads((Path(args.cases) / "cases.json").read_text())["cases"]
    rec = Recommender(args.bundle, args.device)
    artist = {uri: rec.song_artist[j] for uri, j in rec.song_id.items()}.get
    results, start = [], time.perf_counter()
    for setting in SETTINGS:
        rows = {"all": [], "mixed": []}
        for c in cases:
            top = [r["uri"] for r in rec.recommend(c["songs"], c["name"], **setting)["recommendations"]]
            later = set(c["later"])
            row = [c["target"] in top, np.mean([u in later for u in top]), len({artist(u) for u in top}),
                   np.mean([artist(u) == artist(c["songs"][-1]) for u in top])]
            rows["all"].append(row)
            if len(c["songs"]) >= 3 and len({artist(u) for u in c["songs"]}) == len(c["songs"]):
                rows["mixed"].append(row)
        results.append({**setting, **{group: dict(zip(["next_hit@10", "later_p@10", "artists", "last_artist"],
                                                      np.mean(r, axis=0).tolist()), cases=len(r))
                                      for group, r in rows.items()}})
    rec.close()
    (Path(args.cases) / f"scores_{rec.version}.json").write_text(json.dumps(results, indent=2))
    print(f"{len(cases):,} cases ({results[0]['mixed']['cases']:,} mixed) in {time.perf_counter() - start:.0f}s")
    print(f"{'max/artist':>10} {'mix':>4} | {'next hit@10':>11} {'later p@10':>10} {'artists':>7} {'last artist':>11}"
          f" | mixed: {'next hit@10':>11} {'later p@10':>10} {'last artist':>11}")
    for r in results:
        a, m = r["all"], r["mixed"]
        print(f"{r['max_per_artist']:>10} {r['mix']:>4} | {a['next_hit@10']:11.1%} {a['later_p@10']:10.1%} "
              f"{a['artists']:7.1f} {a['last_artist']:11.1%} | mixed: {m['next_hit@10']:11.1%} "
              f"{m['later_p@10']:10.1%} {m['last_artist']:11.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("cases")
    c.add_argument("--config", required=True, help="the retriever config whose held-out playlists to use")
    c.add_argument("--bundle", required=True, help="a serving bundle: its catalog decides which songs are known")
    c.add_argument("--out", required=True)
    c.add_argument("--cases", type=int, default=5000)
    c.add_argument("--input-length", type=int, default=10)
    c.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--cases", required=True)
    s.add_argument("--bundle", required=True)
    s.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.mode == "cases":
        make_cases(args)
    else:
        score_cases(args)
