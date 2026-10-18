"""Step 0 of the ListenBrainz plan: a first look at real listens before streaming the full dump.

Reads a few recent daily ("incremental") dumps and reports:
  1. what the listens carry (Spotify links, MusicBrainz IDs, album names, lengths) and which apps sent them
  2. how many listens are MPD songs: by Spotify link (exact) and by cleaned-up artist + title (song_key), how
     often the two agree, and samples of name matches, disagreements and misses to check by hand
  3. how listening is spread over users: listens per user, the busiest user-days (bots?), repeated timestamps
  4. how big the catalog gets if a song needs at least N different users, and how many of its songs have a
     Spotify ID (from a listen's link or an MPD match), so the webpage can play them
  5. how many listens survive the session rules (a new session after 30 min without a listen, back-to-back
     repeats dropped, sessions of at least 4 songs)
  6. how fast one process reads listens (for the full dump's time)

    python scripts/lb_first_look.py --dumps a.tar.zst b.tar.zst --mpd-store data/mpd_store \
        --out runs/listenbrainz/first_look.json
"""
import argparse
import array
import json
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys.listenbrainz import listen_files, mpd_lookup, parse_listen, song_key  # noqa: E402
from recsys.mpd_store import load_store  # noqa: E402

SESSION_GAP = 30 * 60   # seconds without a listen that start a new session
MIN_SESSION = 4         # songs a session needs to be kept
CATALOG_MINIMUMS = (1, 2, 3, 5, 10, 20)
SEPARATORS = re.compile(r", | & | x | / | feat\. | ft\. ", re.IGNORECASE)  # several artists in one credit
SAMPLES = 25


class Reservoir:
    """A fixed-size random sample of a stream (every item equally likely to be kept)."""

    def __init__(self, size, rng):
        self.size, self.rng, self.seen, self.items = size, rng, 0, []

    def add(self, item):
        self.seen += 1
        if len(self.items) < self.size:
            self.items.append(item)
        elif (j := self.rng.randrange(self.seen)) < self.size:
            self.items[j] = item


def share(a, b):
    return round(a / b, 4) if b else None


def percentiles(values, ps=(50, 90, 99)):
    return {f"p{p}": int(np.percentile(values, p)) for p in ps} | {"max": int(values.max())}


def sessions(users, times, keys):
    """Listens kept and sessions made by the session rules, over listens sorted by (user, time)."""
    new = np.ones(len(users), dtype=bool)
    new[1:] = (users[1:] != users[:-1]) | (times[1:] - times[:-1] > SESSION_GAP)
    session = np.cumsum(new) - 1
    keep = np.ones(len(users), dtype=bool)
    keep[1:] = ~((session[1:] == session[:-1]) & (keys[1:] == keys[:-1]))  # back-to-back repeats
    lengths = np.bincount(session[keep], minlength=session[-1] + 1)
    long_enough = lengths >= MIN_SESSION
    kept = int(lengths[long_enough].sum())
    return {"sessions": int(long_enough.sum()), "listens_kept": kept, "share_of_listens_kept": share(kept, len(users)),
            "repeats_dropped": int((~keep).sum()),
            "session_length": percentiles(lengths[long_enough]) if long_enough.any() else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dumps", nargs="+", required=True, help="ListenBrainz .tar.zst listen dumps")
    ap.add_argument("--mpd-store", default="data/mpd_store")
    ap.add_argument("--out", required=True, help="report (JSON)")
    args = ap.parse_args()
    rng = random.Random(0)

    start = time.perf_counter()
    store = load_store(args.mpd_store)
    counts, by_spotify, by_name, mpd_keys = mpd_lookup(store)
    mpd_name = lambda i: f"{store.artist_names[store.track_artist[i]]} - {store.track_names[i]}"
    print(f"MPD lookups: {len(by_spotify):,} tracks, {len(by_name):,} song keys "
          f"({time.perf_counter() - start:.0f}s)", flush=True)

    users, times, key_ids = array.array("q"), array.array("q"), array.array("q")
    by_spotify_hit, by_name_hit, links = array.array("q"), array.array("q"), array.array("b")
    key_index, months = {}, Counter()
    has = Counter()
    clients = Counter()
    fallback_hits = 0
    samples = {name: Reservoir(SAMPLES, rng) for name in
               ("name_only_matches", "spotify_and_name_disagree", "spotify_match_name_missed", "no_match")}
    read_start, lines = time.perf_counter(), 0
    for path in args.dumps:
        with open(path, "rb") as f:
            for year, month, listens in listen_files(f):
                for line in listens:
                    lines += 1
                    listen = parse_listen(line)
                    if not listen.artist or not listen.title:
                        has["no artist or title"] += 1
                        continue
                    months[f"{year}-{month:02d}"] += 1
                    if lines % 50 == 0:  # where listens come from, on a sample
                        info = json.loads(line)["track_metadata"].get("additional_info") or {}
                        clients[str(info.get("music_service") or info.get("submission_client") or "?")] += 1
                    has["spotify link"] += bool(listen.spotify_id)
                    has["recording MusicBrainz ID"] += bool(listen.recording_mbid)
                    has["album name"] += bool(listen.release)
                    has["length"] += bool(listen.duration_ms)
                    has["artist list (main artist known)"] += listen.first_artist != listen.artist
                    key = song_key(listen.first_artist, listen.title)
                    k = key_index.setdefault(key, len(key_index))
                    s = by_spotify.get(listen.spotify_id, -1) if listen.spotify_id else -1
                    n = by_name.get(key, -1)
                    users.append(listen.user_id)
                    times.append(listen.timestamp)
                    key_ids.append(k)
                    by_spotify_hit.append(s)
                    by_name_hit.append(n)
                    links.append(bool(listen.spotify_id))
                    shown = f"{listen.artist} - {listen.title}"
                    if s < 0 and n >= 0:
                        samples["name_only_matches"].add([shown, mpd_name(n)])
                    elif s >= 0 and n >= 0 and mpd_keys[s] != key:
                        samples["spotify_and_name_disagree"].add([shown, mpd_name(s), mpd_name(n)])
                    elif s >= 0 and n < 0:
                        samples["spotify_match_name_missed"].add([shown, mpd_name(s)])
                    elif s < 0 and n < 0:
                        samples["no_match"].add(shown)
                        if listen.first_artist == listen.artist and SEPARATORS.search(listen.artist):
                            first = SEPARATORS.split(listen.artist)[0]
                            fallback_hits += by_name.get(song_key(first, listen.title), -1) >= 0
    read_seconds = time.perf_counter() - read_start

    users, times, key_ids = (np.frombuffer(a, dtype=np.int64) for a in (users, times, key_ids))
    s_hit, n_hit = np.frombuffer(by_spotify_hit, dtype=np.int64), np.frombuffer(by_name_hit, dtype=np.int64)
    total = len(users)
    either = np.where(s_hit >= 0, s_hit, n_hit)
    both = (s_hit >= 0) & (n_hit >= 0)

    # Users
    _, user_listens = np.unique(users, return_counts=True)
    top = np.sort(user_listens)[::-1]
    _, user_day_listens = np.unique(users * 100_000 + times // 86_400, return_counts=True)
    order = np.lexsort((times, users))
    u, t, k = users[order], times[order], key_ids[order]
    same_second = int(((u[1:] == u[:-1]) & (t[1:] == t[:-1])).sum())

    # Catalog: songs played by at least N different users, and the share of listens they hold
    pairs = np.unique(key_ids * (int(users.max()) + 1) + users)
    users_per_song = np.bincount(pairs // (int(users.max()) + 1), minlength=len(key_index))
    listens_per_song = np.bincount(key_ids, minlength=len(key_index))
    # A song is playable on the webpage if some listen of it has a Spotify link or it is an MPD track
    playable = (np.bincount(key_ids, weights=np.frombuffer(links, dtype=np.int8), minlength=len(key_index)) > 0) | (
        np.bincount(key_ids, weights=either >= 0, minlength=len(key_index)) > 0)
    catalog = {}
    for n in CATALOG_MINIMUMS:
        songs = users_per_song >= n
        catalog[f"N={n}"] = {"songs": int(songs.sum()),
                             "share_of_listens": share(int(listens_per_song[songs].sum()), total),
                             "share_with_an_mpd_match": share(
                                 len(np.unique(key_ids[(either >= 0) & songs[key_ids]])), int(songs.sum())),
                             "share_with_a_spotify_id": share(int((playable & songs).sum()), int(songs.sum()))}

    report = {
        "dumps": [Path(p).name for p in args.dumps],
        "listens": total, "skipped_without_artist_or_title": has.pop("no artist or title", 0),
        "users": int(len(user_listens)),
        "listened_at": {"first": datetime.fromtimestamp(int(times.min()), timezone.utc).isoformat(),
                        "last": datetime.fromtimestamp(int(times.max()), timezone.utc).isoformat(),
                        "share_in_last_30_days": share(int((times > times.max() - 30 * 86_400).sum()), total),
                        "months": dict(sorted(months.items())[-6:])},
        "fields": {name: share(c, total) for name, c in has.items()},
        "sources_sampled": {name: share(c, sum(clients.values())) for name, c in clients.most_common(12)},
        "mpd_matches": {
            "by_spotify_link": share(int((s_hit >= 0).sum()), total),
            "by_artist_and_title": share(int((n_hit >= 0).sum()), total),
            "by_either": share(int((either >= 0).sum()), total),
            "by_either_and_in_2_plus_mpd_playlists": share(int(((either >= 0) & (counts[np.maximum(either, 0)] >= 2)).sum()), total),
            "both_ways_same_song_key": share(int((both & (np.array(mpd_keys, dtype=object)[np.maximum(s_hit, 0)]
                                                         == np.array(mpd_keys, dtype=object)[np.maximum(n_hit, 0)])).sum()),
                                              int(both.sum())),
            "both_ways_same_spotify_track": share(int((both & (s_hit == n_hit)).sum()), int(both.sum())),
            "spotify_link_but_name_missed": share(int(((s_hit >= 0) & (n_hit < 0)).sum()), int((s_hit >= 0).sum())),
            "unmatched_found_by_first_artist_of_credit": share(fallback_hits, int((either < 0).sum())),
        },
        "users_detail": {"listens_per_user": percentiles(user_listens),
                         "share_of_listens_from_top_1pct_users": share(int(top[:max(1, len(top) // 100)].sum()), total),
                         "listens_per_user_day": percentiles(user_day_listens),
                         "user_days_over_300": int((user_day_listens > 300).sum()),
                         "user_days_over_1000": int((user_day_listens > 1000).sum()),
                         "listens_at_the_same_second_as_the_users_previous": same_second},
        "distinct_songs": len(key_index),
        "catalog_by_minimum_users": catalog,
        "sessions_all_songs": sessions(u, t, k),
        "sessions_songs_with_5_plus_users": None,
        "reading": {"listens_per_second_one_process": round(lines / read_seconds), "seconds": round(read_seconds)},
        "samples": {name: r.items for name, r in samples.items()},
    }
    in_catalog = users_per_song[k] >= 5
    report["sessions_songs_with_5_plus_users"] = sessions(u[in_catalog], t[in_catalog], k[in_catalog])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({key: v for key, v in report.items() if key != "samples"}, indent=2, ensure_ascii=False))
    print(f"report: {args.out} ({time.perf_counter() - start:.0f}s)")


if __name__ == "__main__":
    main()
