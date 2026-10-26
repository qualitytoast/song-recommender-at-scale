"""Step 7 of the ListenBrainz plan: today's MPD model vs the ListenBrainz model, on ListenBrainz's held-out users.

    # 1. cases (no models needed)
    python scripts/lb_compare.py cases --store data/lb_store --lb-bundle bundles/<listenbrainz> \\
        --mpd-store data/mpd_store --mpd-bundle bundles/<mpd> --out runs/listenbrainz/compare
    # 2. both bundles answer them in this process (the serving code: recsys.recommender)
    python scripts/lb_compare.py score --cases runs/listenbrainz/compare --lb-bundle ... --mpd-bundle ...
    # 3. the ListenBrainz bundle on the "all" cases with each serving setting in GRID (choosing its defaults)
    python scripts/lb_compare.py tune --cases runs/listenbrainz/compare --lb-bundle ...
    # 4. ListenBrainz pairs trained on different stores (catalogs) on the same held-out listens
    python scripts/lb_compare.py catalog-cases --store data/lb_store_n2 --out runs/listenbrainz/compare_catalogs
    python scripts/lb_compare.py catalog-score --cases runs/listenbrainz/compare_catalogs \\
        --pair today bundles/<bundle> data/lb_store --pair n2 bundles/<bundle> data/lb_store_n2 ...

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
catalog-cases: a sample of one store's held-out any-length cases (the store with the biggest catalog keeps the most of
  what was played), each song as its song key (recsys.listenbrainz.song_key: artist + title, cleaned), which is the
  same in every store; Spotify URIs aren't (each goes to one song per store, the most played first).
catalog-score: each pair (a bundle and the store it was trained on) answers every case. Its input is the case's songs
  it knows, as on the site; a next song it doesn't know is a miss, and so is a case with no input song it knows (no
  answer). Next NDCG@10, hit@10 and later p@10 over every case; artists, last artist and no Spotify link over the
  cases it answers; the shares of cases it answers and whose next song it knows. With its own answers (0, 0) and
  the site's settings (3, 0); saved to <cases>/scores.json.
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
CATALOG_SETTINGS = [{"max_per_artist": 0, "mix": 0.0}, {"max_per_artist": 3, "mix": 0.0}]  # own; the site's
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


def store_song_keys(store):
    """Each store track's song key: its artist's key (a ListenBrainz store's artist URIs are
    listenbrainz:artist:<key>) and its cleaned title, as recsys.listenbrainz.song_key made it."""
    from recsys.listenbrainz import clean_title

    artist_keys = [uri.split(":", 2)[2] for uri in store.artist_uris]
    return [f"{artist_keys[a]}|{clean_title(name)}" for a, name in zip(store.track_artist, store.track_names)]


def build_catalog_cases(keys, tracks, offsets, n_cases, seed, input_length=10):
    """(cases, counts): case dicts {songs, target, later} as song keys, from held-out sessions (tracks: store
    track IDs, offsets per session). Every song of the store is known to it, so every position with a song before
    it in the same session is a case."""
    t, starts, session = case_positions(offsets, np.ones(offsets[-1], dtype=bool), input_length)
    picked = np.sort(np.random.RandomState(seed).choice(len(t), min(n_cases, len(t)), replace=False))
    cases = [{"songs": [keys[j] for j in tracks[starts[i]:t[i]]], "target": keys[tracks[t[i]]],
              "later": list(dict.fromkeys(keys[j] for j in tracks[t[i]:offsets[session[t[i]] + 1]]))}
             for i in picked]
    return cases, {"held_out_sessions": len(offsets) - 1, "cases": int(len(t)), "sampled": len(cases)}


def as_known(cases, uri_of):
    """The cases as one pair sees them (uri_of: song key -> its URI, for the songs it knows): input and later
    songs it doesn't know left out, a target it doesn't know as None."""
    return [{"songs": [uri_of[k] for k in c["songs"] if k in uri_of], "target": uri_of.get(c["target"]),
             "later": [uri_of[k] for k in c["later"] if k in uri_of]} for c in cases]


def catalog_scores(cases, answer_songs, artist_of):
    """One pair's scores on its as_known cases; answer_songs(songs) -> (top URIs, their artists)."""
    every, answered = [], []
    for c in cases:
        if not c["songs"]:                        # no input song it knows: no answer, a miss
            every.append([0.0, 0.0, 0.0])
            continue
        top, artists = answer_songs(c["songs"])
        later = set(c["later"])
        every.append([ndcg_at_10(top, c["target"]), c["target"] in top, np.mean([u in later for u in top])])
        answered.append([len(set(artists)), np.mean([a == artist_of(c["songs"][-1]) for a in artists]),
                         np.mean([not u.startswith("spotify:track:") for u in top])])
    out = dict(zip(["next_ndcg@10", "next_hit@10", "later_p@10"], np.mean(every, axis=0).round(4).tolist()))
    if answered:
        out.update(zip(["artists", "last_artist", "no_spotify_link"], np.mean(answered, axis=0).round(4).tolist()))
    out["answered"] = round(len(answered) / len(cases), 4)
    out["knows_the_next_song"] = round(float(np.mean([c["target"] is not None for c in cases])), 4)
    return out


def make_catalog_cases(args):
    from recsys.data import concat_playlists
    from recsys.mpd_store import load_store

    store = load_store(args.store)
    held = np.flatnonzero(np.load(Path(args.store) / "playlist_split.npy") == HELD_OUT)
    tracks, offsets = concat_playlists(store, held)
    cases, counts = build_catalog_cases(store_song_keys(store), np.asarray(tracks), np.asarray(offsets),
                                        args.cases, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.json").write_text(json.dumps({"store": args.store, "seed": args.seed, "counts": counts,
                                                "cases": cases}))
    print(json.dumps(counts, indent=2))


def score_catalogs(args):
    from recsys.mpd_store import load_store
    from recsys.recommender import Recommender

    data = json.loads((Path(args.cases) / "cases.json").read_text())
    results, began = {"case_store": data["store"], "counts": data["counts"]}, time.perf_counter()
    for name, bundle, store_dir in args.pair:
        store = load_store(store_dir)
        catalog = set(json.loads((Path(bundle) / "songs.json").read_text())["uris"])
        uri_of = {k: u for k, u in zip(store_song_keys(store), store.track_uris) if u in catalog}
        cases = as_known(data["cases"], uri_of)
        rec = Recommender(bundle, args.device)
        artist_of = lambda uri: rec.artists[rec.song_id[uri]] if uri in rec.song_id else None
        results[name] = {"bundle": Path(bundle).name, "store": store_dir}
        for s in CATALOG_SETTINGS:
            answer_songs = lambda songs, s=s: answer(rec, {"songs": songs}, s)
            results[name][f"max_per_artist={s['max_per_artist']}, mix={s['mix']}"] = catalog_scores(
                cases, answer_songs, artist_of)
        rec.close()
        print(json.dumps({name: results[name]}), flush=True)
    results["seconds"] = round(time.perf_counter() - began)
    (Path(args.cases) / "scores.json").write_text(json.dumps(results, indent=2))


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
    cc = sub.add_parser("catalog-cases")
    cc.add_argument("--store", required=True)
    cc.add_argument("--out", required=True)
    cc.add_argument("--cases", type=int, default=5000)
    cc.add_argument("--seed", type=int, default=0)
    cs = sub.add_parser("catalog-score")
    cs.add_argument("--cases", required=True)
    cs.add_argument("--pair", nargs=3, action="append", required=True, metavar=("NAME", "BUNDLE", "STORE"),
                    help="a name, a bundle and the store its models were trained on (repeat per pair)")
    cs.add_argument("--device", default="cpu")
    args = ap.parse_args()
    {"cases": make_cases, "score": score_cases, "tune": tune, "catalog-cases": make_catalog_cases,
     "catalog-score": score_catalogs}[args.mode](args)


if __name__ == "__main__":
    main()
