"""Step 2 of the ListenBrainz plan: the numbers behind the decisions left after extracting the listens.

Reads the extract (recsys.lb_extract) and reports:
  1. users: listens per user, each user's busiest day, listens every few seconds, listens at the same second
     as the previous one; the users these flag as bots, and how many listens they hold
  2. sessions: what the session rules keep (recsys.lb_sessions)
  3. per-user caps: total listens kept (after the session rules, bots out) if each user keeps at most their
     K most recent listens
  4. two ways to a sample the MPD's size (66.3M listens): (a) random users, each capped at K, until the sample
     is big enough; (b) every user, capped at the K that adds up to it
  5. per sample: the catalog at each minimum number of users per song (N), and how many of those songs
     have a Spotify ID (a listen's link, or an MPD song of the same artist + title) and genres (the
     MusicBrainz genre file of the MPD's artists, by Spotify or MusicBrainz artist ID)

    python scripts/lb_report.py --extract data/lb_extract --out runs/listenbrainz/report.json
"""
import argparse
import calendar
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys.lb_extract import load_extract  # noqa: E402
from recsys.lb_sessions import by_user_and_time, sessions, user_starts_from, users_per_song  # noqa: E402
from recsys.mpd_store import load_store  # noqa: E402
from scripts.lb_first_look import mpd_lookup  # noqa: E402

MPD_SIZE = 66_346_428   # song entries in the MPD's 1M playlists: the sample's target
CAPS = (500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000)
SAMPLE_CAPS = (2_000, 5_000, 10_000, None)
MINIMUMS = (1, 2, 3, 5, 10, 20, 50)
COVERAGE_MINIMUMS = (2, 5, 10)
DAY = 86_400
CHUNK = 50_000_000


def log(message):
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def share(a, b):
    return round(float(a) / float(b), 4) if b else None


def percentiles(values, ps=(50, 90, 99, 99.9)):
    values = np.asarray(values)
    return {f"p{p:g}": int(np.percentile(values, p)) for p in ps} | {"max": int(values.max())}


def user_activity(users, seconds, n_users):
    """Per user: listens, gaps between listens, gaps under 30 s, gaps of 0 s, and the busiest day's listens."""
    listens = np.bincount(users, minlength=n_users)
    first = user_starts_from(users)
    same = ~first[1:]
    gap_users, gaps = users[1:][same], np.diff(seconds)[same]
    n_gaps = np.bincount(gap_users, minlength=n_users)
    rapid = np.bincount(gap_users[gaps < 30], minlength=n_users)
    repeat_second = np.bincount(gap_users[gaps == 0], minlength=n_users)
    del gap_users, gaps
    day = seconds // DAY
    new_day = first.copy()
    new_day[1:] |= day[1:] != day[:-1]
    del day
    starts = np.flatnonzero(new_day)
    busiest = np.zeros(n_users, dtype=np.int64)
    np.maximum.at(busiest, users[starts], np.diff(np.append(starts, len(users))))
    return listens, n_gaps, rapid, repeat_second, busiest


def rank_from_end(users):
    """0 for each user's most recent listen, 1 for the one before, ... (listens sorted by user, then time)."""
    first = np.flatnonzero(user_starts_from(users))
    ends = np.append(first[1:], len(users))
    return (np.repeat(ends, np.diff(np.append(first, len(users)))) - np.arange(len(users)) - 1).astype(np.int64)


def song_flags(extract, n_songs):
    """Per song: True if any of its listens carries a Spotify track ID (read in chunks)."""
    linked = np.zeros(n_songs, dtype=bool)
    for start in range(0, extract.meta["listens"], CHUNK):
        song = np.asarray(extract.song[start:start + CHUNK])
        spotify = np.asarray(extract.spotify[start:start + CHUNK])
        linked[np.unique(song[spotify >= 0])] = True
    return linked


def genre_lookups(path):
    """(Spotify artist URIs with genres, MusicBrainz artist IDs with genres) from the MPD genre file."""
    uris, mbids = set(), set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["genres"]:
                uris.add(row["artist_uri"])
                mbids.update(row["mbids"])
    return uris, mbids


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--extract", default="data/lb_extract")
    ap.add_argument("--mpd-store", default="data/mpd_store")
    ap.add_argument("--genres", default="data/genres/musicbrainz_all_mpd_artists.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    began = time.time()
    extract = load_extract(args.extract)
    n_songs = extract.meta["songs"]
    start = calendar.timegm(datetime.fromisoformat(extract.meta["window"]["start"]).utctimetuple())
    report = {"extract": {k: extract.meta[k] for k in ("listens", "songs", "users", "window")}}

    log(f"sorting {extract.meta['listens']:,} listens by user, then time")
    users, seconds, songs, user_ids = by_user_and_time(extract.user, extract.timestamp, extract.song, start)
    n_users = len(user_ids)

    log("users")
    listens, n_gaps, rapid, repeat_second, busiest = user_activity(users, seconds, n_users)
    top = np.sort(listens)[::-1]
    flags = {"over 1,000 listens in one day (~58 h of music)": busiest > 1_000,
             "most gaps under 30 s (1,000+ listens)": (listens >= 1_000) & (rapid * 2 > n_gaps),
             "most listens at the same second as the previous (100+ listens)": (listens >= 100) & (repeat_second * 2 > listens)}
    bot = np.zeros(n_users, dtype=bool)
    for flag in flags.values():
        bot |= flag
    report["users"] = {
        "listens_per_user": percentiles(listens),
        "share_of_listens_from_top_1pct": share(top[:max(1, n_users // 100)].sum(), listens.sum()),
        "share_of_listens_from_top_10pct": share(top[:max(1, n_users // 10)].sum(), listens.sum()),
        "busiest_day_listens": percentiles(busiest),
        "listens_at_the_same_second_as_the_previous": share(repeat_second.sum(), listens.sum()),
        "bot_flags": {name: {"users": int(f.sum()), "share_of_listens": share(listens[f].sum(), listens.sum())}
                      for name, f in flags.items()} | {
            "any flag": {"users": int(bot.sum()), "share_of_listens": share(listens[bot].sum(), listens.sum())}},
    }

    log("sessions")
    kept, session = sessions(users, seconds, songs)
    del seconds
    human = ~bot[users]
    kept &= human
    kept_listens = int(kept.sum())
    session_lengths = np.bincount(session[kept])
    session_lengths = session_lengths[session_lengths > 0]
    del session
    kept_per_user = np.bincount(users[kept], minlength=n_users)
    report["sessions"] = {"listens_of_non_bot_users": int(human.sum()), "kept": kept_listens,
                          "share_kept": share(kept_listens, human.sum()), "sessions": int(len(session_lengths)),
                          "session_length": percentiles(session_lengths)}
    del human

    log("caps")
    report["caps"] = {f"K={k:,}": {"listens": int(np.minimum(kept_per_user, k).sum()),
                                   "users_over_the_cap": int((kept_per_user > k).sum())} for k in CAPS}

    log("samples")
    rng = np.random.default_rng(args.seed)
    candidates = rng.permutation(np.flatnonzero(kept_per_user > 0))
    samples = {}
    for k in SAMPLE_CAPS:
        capped = kept_per_user[candidates] if k is None else np.minimum(kept_per_user[candidates], k)
        cum = np.cumsum(capped)
        n = int(np.searchsorted(cum, MPD_SIZE)) + 1
        chosen = np.zeros(n_users, dtype=bool)
        chosen[candidates[:n]] = True
        samples[f"random users, cap {'none' if k is None else f'{k:,}'}"] = (chosen, k)
    lo, hi = 1, int(kept_per_user.max())
    while lo < hi:  # the smallest cap at which every user together reaches the target
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if np.minimum(kept_per_user, mid).sum() >= MPD_SIZE else (mid + 1, hi)
    samples[f"every user, cap {lo:,}"] = (kept_per_user > 0, lo)

    rank = np.full(len(users), np.iinfo(np.int32).max, dtype=np.int32)
    rank[kept] = rank_from_end(users[kept])
    linked = song_flags(extract, n_songs)
    catalogs, needed = {}, np.zeros(n_songs, dtype=bool)
    report["samples"] = {}
    for name, (chosen, k) in samples.items():
        mask = kept & chosen[users]
        if k is not None:
            mask &= rank < k
        per_song_users = users_per_song(users, songs, n_songs, mask)
        per_song_listens = np.bincount(songs[mask], minlength=n_songs)
        total = int(mask.sum())
        report["samples"][name] = {
            "users": int(chosen.sum()), "share_of_users": share(chosen.sum(), (kept_per_user > 0).sum()),
            "listens": total, "listens_per_user": percentiles(np.bincount(users[mask], minlength=n_users)[chosen]),
            "catalog": {f"N={n}": {"songs": int((per_song_users >= n).sum()),
                                   "share_of_listens": share(per_song_listens[per_song_users >= n].sum(), total)}
                        for n in MINIMUMS}}
        catalogs[name] = per_song_users
        needed |= per_song_users >= min(COVERAGE_MINIMUMS)
        log(f"  {name}: {chosen.sum():,} users, {total:,} listens")
    del rank, mask

    log("Spotify IDs and genres for the catalog songs")
    store = load_store(args.mpd_store)
    _, _, mpd_by_name, _ = mpd_lookup(store)
    genre_uris, genre_mbids = genre_lookups(args.genres)
    mpd_match = np.zeros(n_songs, dtype=bool)
    genres = np.zeros(n_songs, dtype=bool)
    with open(Path(args.extract) / "songs.jsonl", encoding="utf-8") as f:
        for code, line in enumerate(f):
            if needed[code]:
                track = mpd_by_name.get(json.loads(line)[0])
                if track is not None:
                    mpd_match[code] = True
                    genres[code] = store.artist_uris[store.track_artist[track]] in genre_uris
    spotify_artist = np.asarray(extract.song_spotify_artist)
    artist_mbid = np.asarray(extract.song_artist_mbid)
    for code in np.flatnonzero(needed & ~genres):
        if spotify_artist[code] >= 0 and f"spotify:artist:{extract.spotify_artists[spotify_artist[code]]}" in genre_uris:
            genres[code] = True
        elif artist_mbid[code] >= 0 and extract.artist_mbids[artist_mbid[code]] in genre_mbids:
            genres[code] = True
    has_mbid = artist_mbid >= 0
    for name, per_song_users in catalogs.items():
        for n in COVERAGE_MINIMUMS:
            songs_n = per_song_users >= n
            count = songs_n.sum()
            report["samples"][name]["catalog"][f"N={n}"] |= {
                "with_a_spotify_id": share((songs_n & (linked | mpd_match)).sum(), count),
                "an_mpd_song": share((songs_n & mpd_match).sum(), count),
                "with_genres_now": share((songs_n & genres).sum(), count),
                "no_genres_but_an_artist_musicbrainz_id": share((songs_n & ~genres & has_mbid).sum(), count)}
    report["seconds"] = round(time.time() - began)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    log(f"report: {args.out} ({report['seconds']} s)")


if __name__ == "__main__":
    main()
