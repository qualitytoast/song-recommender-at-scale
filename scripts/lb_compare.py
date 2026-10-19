"""Step 7 of the ListenBrainz plan: today's MPD model vs the ListenBrainz model, on ListenBrainz's held-out users.

    # 1. cases (no models needed)
    python scripts/lb_compare.py cases --store data/lb_store --lb-bundle bundles/<listenbrainz> \\
        --mpd-store data/mpd_store --mpd-bundle bundles/<mpd> --out runs/listenbrainz/compare
    # 2. both bundles answer them in this process (the serving code: recsys.recommender)
    python scripts/lb_compare.py score --cases runs/listenbrainz/compare --lb-bundle ... --mpd-bundle ...
    # 3. the ListenBrainz bundle on the "all" cases with each serving setting in GRID (choosing its defaults)
    python scripts/lb_compare.py tune --cases runs/listenbrainz/compare --lb-bundle ...

cases: two samples of the held-out users' any-length cases (a song with a known song right before it in the same
session; the input: the up to 10 known songs before it), each with the rest of its session:
  shared  cases whose next song both catalogs know (an MPD song of the same artist + title: the store's
          track_mpd). Each case holds its songs as both models know them: the MPD model's input leaves out songs it
          doesn't know, as the site does (cases where it knows none of them are counted, then left out).
  all     any held-out case (the ListenBrainz model only), for the BTS check.
score: per bundle and setting (the models' own answers: max_per_artist 0, mix 0; the site's: 2, 0.5):
  next NDCG@10 and hit@10, later p@10 (share of the top 10 that the session goes on to contain), artists (different
  ones in the top 10), last artist (share by the last input song's artist), no Spotify link (share of the top 10
  without a Spotify URI: the webpage can't play those); and on "all" for the ListenBrainz model: among cases with
  no BTS song in their input, the share of top-10 songs by BTS, and of next songs by BTS.
tune: the same scores for the ListenBrainz model on "all", per setting in GRID, and on its "mixed" cases (3+ input
  songs, every one by a different artist: closer to a list typed into the webpage than a session that plays an
  album through); saved to <cases>/tune.json.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HELD_OUT = 3
SETTINGS = [{"max_per_artist": 0, "mix": 0.0}, {"max_per_artist": 2, "mix": 0.5}]
GRID = [{"max_per_artist": a, "mix": m} for a in (0, 1, 2, 3) for m in (0.0, 0.25, 0.5)]
WATCHED_ARTIST = "BTS"


def case_positions(offsets, known, input_length):
    """For sessions given by offsets (in concatenated song arrays) and which songs a model knows: every position
    t whose song and the song before it are known and in the same session, with the start of its input (the
    known songs right before t, up to input_length)."""
    n = offsets[-1]
    session = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    t = np.arange(1, n)
    t = t[known[t] & known[t - 1] & (session[t] == session[t - 1])]
    last_unknown = np.maximum.accumulate(np.where(known, -1, np.arange(n)))
    starts = np.maximum.reduce([offsets[session[t]], last_unknown[t - 1] + 1, t - input_length])
    return t, starts, session


def build_cases(lb_uris, mpd_uri_of, tracks, offsets, n_cases, seed, input_length=10):
    """(shared, all, counts): case dicts from held-out sessions (tracks: store track IDs, offsets per session).
    lb_uris[i]: the ListenBrainz bundle URI of store track i (None if not in its catalog); mpd_uri_of[i]: the MPD
    bundle URI of the same song (None if none)."""
    lb = np.array([u is not None for u in lb_uris])[tracks]
    mpd = np.array([u is not None for u in mpd_uri_of])[tracks]
    t, starts, session = case_positions(offsets, lb, input_length)
    rng = np.random.RandomState(seed)

    def make(i):
        rest = tracks[t[i]:offsets[session[t[i]] + 1]]
        songs = tracks[starts[i]:t[i]]
        return {"lb": {"songs": [lb_uris[j] for j in songs], "target": lb_uris[tracks[t[i]]],
                       "later": list(dict.fromkeys(lb_uris[j] for j in rest if lb_uris[j]))},
                "mpd": {"songs": [mpd_uri_of[j] for j in songs if mpd_uri_of[j]], "target": mpd_uri_of[tracks[t[i]]],
                        "later": list(dict.fromkeys(mpd_uri_of[j] for j in rest if mpd_uri_of[j]))}}

    shared_pool = np.flatnonzero(mpd[t])
    counts = {"held_out_sessions": len(offsets) - 1, "cases": int(len(t)),
              "cases_whose_next_song_both_know": int(len(shared_pool))}
    shared = [make(i) for i in np.sort(rng.choice(shared_pool, min(n_cases, len(shared_pool)), replace=False))]
    counts["shared_sampled"] = len(shared)
    shared = [c for c in shared if c["mpd"]["songs"]]
    counts["shared_with_an_input_song_the_mpd_model_knows"] = len(shared)
    every = [make(i) for i in np.sort(rng.choice(len(t), min(n_cases, len(t)), replace=False))]
    return shared, every, counts


def make_cases(args):
    from recsys.data import concat_playlists
    from recsys.mpd_store import load_store

    store = load_store(args.store)
    side = np.load(Path(args.store) / "playlist_split.npy")
    held = np.flatnonzero(side == HELD_OUT)
    tracks, offsets = concat_playlists(store, held)
    lb_catalog = set(json.loads((Path(args.lb_bundle) / "songs.json").read_text())["uris"])
    lb_uris = [u if u in lb_catalog else None for u in store.track_uris]
    mpd_catalog = set(json.loads((Path(args.mpd_bundle) / "songs.json").read_text())["uris"])
    mpd_uris = load_store(args.mpd_store).track_uris
    track_mpd = np.load(Path(args.store) / "track_mpd.npy")
    mpd_uri_of = [mpd_uris[m] if m >= 0 and mpd_uris[m] in mpd_catalog else None for m in track_mpd]
    shared, every, counts = build_cases(lb_uris, mpd_uri_of, np.asarray(tracks), np.asarray(offsets),
                                        args.cases, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.json").write_text(json.dumps({"store": args.store, "seed": args.seed, "counts": counts,
                                                "shared": shared, "all": every}))
    print(json.dumps(counts, indent=2))


def ndcg_at_10(top, target):
    return 1 / math.log2(top.index(target) + 2) if target in top else 0.0


def answer(rec, case, setting):
    response = rec.recommend(case["songs"], None, **setting)["recommendations"]
    return [r["uri"] for r in response], [r["artist"] for r in response]


def score(rec, cases, side, setting, artist_of):
    rows = []
    for c in cases:
        top, artists = answer(rec, c[side], setting)
        later = set(c[side]["later"])
        rows.append([ndcg_at_10(top, c[side]["target"]), c[side]["target"] in top, np.mean([u in later for u in top]),
                     len(set(artists)), np.mean([a == artist_of(c[side]["songs"][-1]) for a in artists]),
                     np.mean([not u.startswith("spotify:track:") for u in top])])
    return dict(zip(["next_ndcg@10", "next_hit@10", "later_p@10", "artists", "last_artist", "no_spotify_link"],
                    np.mean(rows, axis=0).round(4).tolist()))


def score_cases(args):
    from recsys.recommender import Recommender

    data = json.loads((Path(args.cases) / "cases.json").read_text())
    results, began = {"counts": data["counts"]}, time.perf_counter()
    for side, bundle in (("mpd", args.mpd_bundle), ("lb", args.lb_bundle)):
        rec = Recommender(bundle, args.device)
        artist_of = lambda uri: rec.artists[rec.song_id[uri]] if uri in rec.song_id else None
        results[side] = {"bundle": Path(bundle).name,
                         "shared": {f"max_per_artist={s['max_per_artist']}, mix={s['mix']}":
                                    score(rec, data["shared"], side, s, artist_of) for s in SETTINGS}}
        if side == "lb":
            results[side]["all"] = {f"max_per_artist={s['max_per_artist']}, mix={s['mix']}":
                                    score(rec, data["all"], side, s, artist_of) for s in SETTINGS}
            no_watched = [c for c in data["all"] if all(artist_of(u) != WATCHED_ARTIST for u in c["lb"]["songs"])]
            watched = {"cases_without_it_in_the_input": len(no_watched),
                       "share_of_next_songs_by_it": round(float(np.mean(
                           [artist_of(c["lb"]["target"]) == WATCHED_ARTIST for c in no_watched])), 4)}
            for s in SETTINGS:
                tops = [answer(rec, c["lb"], s)[1] for c in no_watched]
                watched[f"share_of_top_10_by_it, max_per_artist={s['max_per_artist']}, mix={s['mix']}"] = round(
                    float(np.mean([a == WATCHED_ARTIST for top in tops for a in top])), 4)
                watched[f"cases_with_it_in_the_top_10, max_per_artist={s['max_per_artist']}, mix={s['mix']}"] = \
                    round(float(np.mean([WATCHED_ARTIST in top for top in tops])), 4)
            results[side][WATCHED_ARTIST] = watched
        rec.close()
    results["seconds"] = round(time.perf_counter() - began)
    (Path(args.cases) / "scores.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def tune(args):
    from recsys.recommender import Recommender

    data = json.loads((Path(args.cases) / "cases.json").read_text())
    rec, began = Recommender(args.lb_bundle, args.device), time.perf_counter()
    artist_of = lambda uri: rec.artists[rec.song_id[uri]] if uri in rec.song_id else None
    songs = lambda c: c["lb"]["songs"]
    mixed = [c for c in data["all"] if len(songs(c)) >= 3 and len({artist_of(u) for u in songs(c)}) == len(songs(c))]
    results = {"bundle": Path(args.lb_bundle).name, "cases": len(data["all"]), "mixed_cases": len(mixed),
               "settings": []}
    for s in GRID:
        results["settings"].append({**s, "all": score(rec, data["all"], "lb", s, artist_of),
                                    "mixed": score(rec, mixed, "lb", s, artist_of)})
        print(json.dumps(results["settings"][-1]), flush=True)
    rec.close()
    results["seconds"] = round(time.perf_counter() - began)
    (Path(args.cases) / "tune.json").write_text(json.dumps(results, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("cases")
    c.add_argument("--store", default="data/lb_store")
    c.add_argument("--lb-bundle", required=True)
    c.add_argument("--mpd-store", default="data/mpd_store")
    c.add_argument("--mpd-bundle", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--cases", type=int, default=5000)
    c.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--cases", required=True)
    s.add_argument("--lb-bundle", required=True)
    s.add_argument("--mpd-bundle", required=True)
    s.add_argument("--device", default="cpu")
    t = sub.add_parser("tune")
    t.add_argument("--cases", required=True)
    t.add_argument("--lb-bundle", required=True)
    t.add_argument("--device", default="cpu")
    args = ap.parse_args()
    {"cases": make_cases, "score": score_cases, "tune": tune}[args.mode](args)


if __name__ == "__main__":
    main()
