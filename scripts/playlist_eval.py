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
    args = ap.parse_args()
    if args.mode == "cases":
        make_cases(args)
