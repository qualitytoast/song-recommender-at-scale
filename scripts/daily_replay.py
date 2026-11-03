"""Part A of the daily-update plan: is a daily update worth it? Replays real ListenBrainz daily dumps onto the served
pair, then scores the results on listening the model has never seen.

    # 1. the replays up to --until, one bundle per variant, and the cases (no PyTorch: FAISS extends the index)
    python scripts/daily_replay.py build --bundle bundles/<served> --store data/lb_store_n3_1.5x \\
        --dumps data/lb_daily/dumps --until 2026-10-29
    # 2. every variant answers the cases (the one-process recommender)
    python scripts/daily_replay.py score

build: the dumps (one folder each: a .tar.zst and its .sha256) chained by window (recsys.daily.chain); each used
dump's fresh listens as an extract (data/lb_daily/extracts/<dump>, reused once made); then the replays, each starting
from today's catalog, through every dump whose window ends by the replay's end (recsys.daily.apply_dump):
  no new songs     max_new 0: listens and pairs of today's songs only
  3, 5, 10         new songs at that many listeners (3 is the catalog's own bar)
  3, a week late   the same as 3, ending 7 days before --until (what a weekly update loses)
and the variants: bundles in data/lb_daily/replay/<variant>, the served bundle's files hard-linked where unchanged,
each with song_keys.json (every song's key), so cases can be matched to songs:
  today                   the served pair as it is
  old_pairs               today's songs, their pairs counted on (the "no new songs" replay)
  new_songs               new songs (3 listeners) with stand-in vectors, and only the pairs with a new song in them
  new_songs_artist, new_songs_artist_norm   the same with the plain or the lengthened artist average (daily.standins)
  both                    new songs and every pair
  new_songs_5, new_songs_10, new_songs_week the other replays, as new_songs
Cases: next-song cases (scripts/lb_compare.py build_catalog_cases) from sessions played after --until, in dumps after
the last one a replay used, by listeners the model never trained on (the store's held-out users, or users not in it).
Each song is its song key, so a next song a variant doesn't know is a miss for it.

score: per variant and setting (the models' own answers: max_per_artist 0, mix 0; the site's: 3, 0): next NDCG@10,
hit@10 and later p@10 over every case (as scripts/lb_compare.py catalog-score), NDCG@10 split by what the next song is
(one of today's songs, a song only an update added, a song nothing knows), the share of cases whose next song it
knows and of top 10s with a new song in them, and the difference from today case by case, with its standard error.
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recsys.daily import (DAY, MAX_NEW, MIN_USERS, Seen, Served, Update, apply_dump, chain, fresh_extract,  # noqa: E402
                          involving_new, read_window, ruled, standins, with_pairs)
from recsys.lb_extract import utc_seconds  # noqa: E402
from recsys.lb_store import HELD_OUT, store_song_keys  # noqa: E402
from recsys.mpd_store import load_store  # noqa: E402
from scripts.lb_compare import CATALOG_SETTINGS, answer, as_known, build_catalog_cases, ndcg_at_10  # noqa: E402

TABLES = ("retriever_song_inputs", "ranker_song_inputs", "search_vectors")
REPLAYS = {"no new songs": (0, MIN_USERS, 0), "3": (0, 3, MAX_NEW), "5": (0, 5, MAX_NEW), "10": (0, 10, MAX_NEW),
           "3, a week late": (7, 3, MAX_NEW)}  # label: (days before --until it ends, listeners, songs per dump)
VARIANTS = {"today": (None, None, None), "old_pairs": ("no new songs", "all", None),
            "new_songs": ("3", "new", "standin"), "new_songs_artist": ("3", "new", "artist"),
            "new_songs_artist_norm": ("3", "new", "artist_norm"), "both": ("3", "all", "standin"),
            "new_songs_5": ("5", "new", "standin"), "new_songs_10": ("10", "new", "standin"),
            "new_songs_week": ("3, a week late", "new", "standin")}  # name: (replay, pairs added, stand-in mode)


def log(message):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def bundle_keys(bundle, store):
    """Every bundle song's key, from the store it was trained on (by its URI)."""
    uris = json.loads((Path(bundle) / "songs.json").read_text())["uris"]
    key_of = dict(zip(store.track_uris, store_song_keys(store)))
    return [key_of[u] for u in uris]


def write_variant(base, folder, served, update, pairs, mode, tables):
    """The base bundle with an update (served: its songs; pairs: "all" of the update's pairs, "new" those with a new
    song, None: none), as a bundle the one-process recommender loads. Returns its song count."""
    import faiss  # here: build's process never imports PyTorch

    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    n, grew = len(served), len(served) > served.base
    written = {"settings.json", "song_playlists.npy", "song_keys.json", "manifest.json"}
    if pairs:
        written |= {"pair_keys.npy", "pair_counts.npy"}
    if grew:
        written |= {*(f"{t}.npy" for t in TABLES), "search_index.faiss", "song_artist.npy", "song_album.npy",
                    "song_genres.npy", "songs.json", "song_albums.json", "name_keys.json", "songs_by_track.npy",
                    "songs_by_artist.npy"}
    for p in base.iterdir():
        if p.is_file() and p.name not in written:
            os.link(p, folder / p.name)
    settings = json.loads((base / "settings.json").read_text())
    (folder / "settings.json").write_text(json.dumps({**settings, "songs": n}, indent=2))
    if pairs:
        keep = involving_new(update.pairs, served.base) if pairs == "new" else slice(None)
        keys, counts = with_pairs(np.load(base / "pair_keys.npy"), np.load(base / "pair_counts.npy"), served.base,
                                  update.pairs[keep], update.counts[keep], n)
        np.save(folder / "pair_keys.npy", keys)
        np.save(folder / "pair_counts.npy", counts.astype(np.int16 if counts.max() < 2 ** 15 else np.int32))
    listens = np.concatenate([served.listens.astype(np.int64), np.zeros(n - served.base, dtype=np.int64)])
    listens[:len(update.listens)] += update.listens
    np.save(folder / "song_playlists.npy", listens.astype(np.int32))
    (folder / "song_keys.json").write_text(json.dumps(served.keys))
    if grew:
        rows = standins(served, update, tables, mode)
        for name in TABLES:
            np.save(folder / f"{name}.npy", np.concatenate([tables[name], rows[name].astype(tables[name].dtype)]))
        index = faiss.read_index(str(base / "search_index.faiss"))
        assert index.ntotal == served.base
        index.add(np.ascontiguousarray(rows["search_vectors"].astype(tables["search_vectors"].dtype), dtype=np.float32))
        faiss.write_index(index, str(folder / "search_index.faiss"))
        song_artist = served.song_artist()
        np.save(folder / "song_artist.npy", song_artist)
        np.save(folder / "song_album.npy", served.song_album())
        np.save(folder / "song_genres.npy", served.song_genres())
        (folder / "songs.json").write_text(json.dumps({"uris": served.uris, "tracks": served.tracks,
                                                       "artist_names": served.artist_names}))
        (folder / "song_albums.json").write_text(json.dumps(served.album_names + [s.album for s in served.added]))
        from recsys.bundle import name_index
        by_track, by_artist, name_keys = name_index(served.tracks, served.artist_names, song_artist, listens)
        np.save(folder / "songs_by_track.npy", by_track)
        np.save(folder / "songs_by_artist.npy", by_artist)
        (folder / "name_keys.json").write_text(json.dumps(name_keys))
    from recsys.bundle import write_manifest
    write_manifest(folder)
    return n


def test_listens(extracts, names, until):
    """The listens played after until in the given dumps, as one set: (users, seconds, song codes, session numbers)
    after the store's rules, and the song keys the codes stand for."""
    code, columns = {}, ([], [], [])
    for name in names:
        ext, _ = extracts[name]
        local = np.array([code.setdefault(row[0], len(code)) for row in ext.songs()], dtype=np.int64)
        after = np.asarray(ext.timestamp) >= until
        columns[0].append(np.asarray(ext.user)[after])
        columns[1].append(np.asarray(ext.timestamp)[after])
        columns[2].append(local[np.asarray(ext.song)[after]] if len(local) else np.zeros(0, dtype=np.int64))
    return ruled(*(np.concatenate(c) for c in columns), until), list(code)


def build(args):
    began = time.perf_counter()
    until = utc_seconds(args.until)
    files = {d.name: next(d.glob("*.tar.zst")) for d in sorted(Path(args.dumps).iterdir())
             if d.is_dir() and any(d.glob("*.tar.zst"))}
    windows = {name: read_window(path) for name, path in files.items()}
    used, skipped, gaps = chain(windows)
    log(f"{len(files)} dumps: {len(used)} used, skipped (a window already used) {skipped}, "
        f"gaps {[(iso(a), iso(b)) for a, b in gaps]}")
    extracts = {}
    for name in used:
        t = time.perf_counter()
        extracts[name] = fresh_extract(files[name], Path(args.extracts) / name, args.workers)
        log(f"{name}: window {iso(windows[name][0])} to {iso(windows[name][1])}, "
            f"{extracts[name][0].meta['listens']:,} fresh listens ({time.perf_counter() - t:.0f} s)")
    base = Path(args.bundle)
    served0 = Served(base, bundle_keys(base, load_store(args.store)))
    log(f"today's catalog: {len(served0):,} songs")
    results, report = {}, {"until": args.until, "dumps": len(files), "used": used, "skipped": skipped,
                           "gaps": [[iso(a), iso(b)] for a, b in gaps], "replays": {}, "variants": {}}
    for label, (days_early, min_users, max_new) in REPLAYS.items():
        t = time.perf_counter()
        end = until - days_early * DAY
        served, update, seen = served0.copy(), Update(), Seen()
        for name in used:
            if windows[name][1] <= end + 3600:
                apply_dump(served, update, seen, extracts[name][0], extracts[name][1], name, min_users, max_new)
        results[label] = (served, update)
        report["replays"][label] = {"through": iso(end), "dumps": len(update.dumps), "new_songs": len(served.added),
                                    "pairs": int(len(update.pairs)),
                                    "pairs_with_a_new_song": int(involving_new(update.pairs, served.base).sum()),
                                    "listens": int(update.listens.sum()), "seconds": round(time.perf_counter() - t),
                                    "per_dump": update.dumps}
        log(f"replay {label!r} through {iso(end)}: {len(update.dumps)} dumps, {len(served.added):,} new songs, "
            f"{len(update.pairs):,} pairs, {int(update.listens.sum()):,} listens ({time.perf_counter() - t:.0f} s)")
    added = results["3"][0].added
    report["new_songs_sample"] = [[s.key, s.uri, s.listeners, iso(s.joined)] for s in
                                  sorted(added, key=lambda s: -s.listeners)[:40]]
    tables = {name: np.load(base / f"{name}.npy", mmap_mode="r") for name in TABLES}
    out = Path(args.out)
    for variant, (replay, pairs, mode) in VARIANTS.items():
        t = time.perf_counter()
        served, update = results[replay] if replay else (served0, Update())
        songs = write_variant(base, out / variant, served, update, pairs, mode, tables)
        report["variants"][variant] = {"replay": replay, "pairs": pairs, "standin": mode, "songs": songs,
                                       "seconds": round(time.perf_counter() - t)}
        log(f"variant {variant}: {songs:,} songs ({time.perf_counter() - t:.0f} s)")
    # the cases: sessions played after until, in dumps after every replay's, by listeners the model never trained on
    last = max(windows[d["dump"]][1] for d in results["3"][1].dumps)
    later = [n for n in used if windows[n][1] > last + 1]
    (users, seconds, codes, session), keys = test_listens(extracts, later, until)
    store_users = np.load(Path(args.store) / "playlist_users.npy")
    side = np.load(Path(args.store) / "playlist_split.npy")
    unseen = np.isin(users, store_users[side == HELD_OUT]) | ~np.isin(users, store_users)
    codes, session = codes[unseen], session[unseen]
    starts = np.flatnonzero(np.append(True, session[1:] != session[:-1])) if len(codes) else np.zeros(0, np.int64)
    offsets = np.append(starts, len(codes)).astype(np.int64)
    cases, counts = build_catalog_cases(keys, codes, offsets, args.cases, args.seed)
    known = served0.ids(keys) >= 0
    counts.update({"dumps": later, "listeners": int(len(np.unique(users[unseen]))), "listens": int(len(codes)),
                   "listens_of_todays_songs": round(float(known[codes].mean()), 4)})
    report["cases"] = counts
    cases_dir = Path(args.cases_out)
    cases_dir.mkdir(parents=True, exist_ok=True)
    (cases_dir / "cases.json").write_text(json.dumps({"until": args.until, "seed": args.seed, "counts": counts,
                                                      "cases": cases}))
    report["seconds"] = round(time.perf_counter() - began)
    (cases_dir / "build.json").write_text(json.dumps(report, indent=2))
    log(f"cases: {json.dumps(counts)}")
    log(f"build done in {report['seconds']} s")


def case_rows(rec, cases, setting, base_songs):
    """Per case: NDCG@10, hit@10, later p@10, answered, a new song (ID >= base_songs) in the top 10."""
    rows = np.zeros((len(cases), 5))
    for i, c in enumerate(cases):
        if not c["songs"]:  # no input song it knows: no answer, a miss
            continue
        top, _ = answer(rec, {"songs": c["songs"]}, setting)
        later = set(c["later"])
        rows[i] = [ndcg_at_10(top, c["target"]), c["target"] in top, np.mean([u in later for u in top]), 1,
                   any(rec.song_id[u] >= base_songs for u in top)]
    return rows


def mean_se(x):
    return [round(float(x.mean()), 5), round(float(x.std(ddof=1) / math.sqrt(len(x))), 5)] if len(x) > 1 else [None, None]


def score(args):
    from recsys.recommender import Recommender  # PyTorch: only in this command's process

    began = time.perf_counter()
    cases_dir, replay = Path(args.cases_out), Path(args.out)
    data = json.loads((cases_dir / "cases.json").read_text())
    names = [v for v in VARIANTS if (replay / v).is_dir()]
    today_keys = set(json.loads((replay / "today" / "song_keys.json").read_text()))
    base_songs = len(today_keys)
    results = {"until": data["until"], "counts": data["counts"], "variants": {}}
    per_case = {}
    for name in names:
        t = time.perf_counter()
        folder = replay / name
        rec = Recommender(folder, args.device)
        uri_of = dict(zip(json.loads((folder / "song_keys.json").read_text()), rec.uris))
        cases = as_known(data["cases"], uri_of)
        kind = np.array([0 if c["target"] in today_keys else 1 if c["target"] in uri_of else 2 for c in data["cases"]])
        out = {"songs": len(rec.uris), "knows_the_next_song": round(float(np.mean(kind < 2) if name != "today"
                                                                           else np.mean(kind == 0)), 4)}
        for setting in CATALOG_SETTINGS:
            label = f"max_per_artist={setting['max_per_artist']}, mix={setting['mix']}"
            rows = case_rows(rec, cases, setting, base_songs)
            per_case[f"{name}|{label}"] = rows
            s = {"next_ndcg@10": round(float(rows[:, 0].mean()), 4), "next_hit@10": round(float(rows[:, 1].mean()), 4),
                 "later_p@10": round(float(rows[:, 2].mean()), 4), "answered": round(float(rows[:, 3].mean()), 4),
                 "top_10s_with_a_new_song": round(float(rows[:, 4].mean()), 4),
                 "ndcg@10_next_is_todays_song": round(float(rows[kind == 0, 0].mean()), 4),
                 "ndcg@10_next_is_a_new_song": round(float(rows[kind == 1, 0].mean()), 4) if (kind == 1).any() else None,
                 "cases_next_is_a_new_song": int((kind == 1).sum())}
            if name != "today":
                d = rows[:, 0] - per_case[f"today|{label}"][:, 0]
                s["ndcg@10_minus_today"] = mean_se(d)
                s["ndcg@10_minus_today_next_is_todays_song"] = mean_se(d[kind == 0])
            out[label] = s
        rec.close()
        out["seconds"] = round(time.perf_counter() - t)
        results["variants"][name] = out
        log(f"{name}: {json.dumps(out)}")
    results["seconds"] = round(time.perf_counter() - began)
    (cases_dir / "scores.json").write_text(json.dumps(results, indent=2))
    np.savez(cases_dir / "per_case.npz", **per_case)
    log(f"score done in {results['seconds']} s")


def iso(seconds):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(seconds))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("build", "score"):
        p = sub.add_parser(name)
        p.add_argument("--out", default="data/lb_daily/replay", help="the variants' bundles")
        p.add_argument("--cases-out", default="runs/listenbrainz/daily_replay", help="cases, report and scores")
    b = sub.choices["build"]
    b.add_argument("--bundle", required=True, help="the served bundle (all its files, as exported)")
    b.add_argument("--store", required=True, help="the store it was trained on")
    b.add_argument("--dumps", default="data/lb_daily/dumps")
    b.add_argument("--extracts", default="data/lb_daily/extracts")
    b.add_argument("--until", required=True, help="the replays' end and the cases' start, YYYY-MM-DD (UTC)")
    b.add_argument("--cases", type=int, default=20000)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--workers", type=int, default=6)
    sub.choices["score"].add_argument("--device", default="cpu")
    args = ap.parse_args()
    {"build": build, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
