"""Step 3 of the ListenBrainz plan: the training data, laid out like the MPD store (recsys.mpd_store).

From the extract (recsys.lb_extract), in order:
  1. listens sorted by user, then time; dropped: listens on impossible days (more than max_day that day),
     listens at the same second as the user's previous one (duplicates), listens followed by the user's next
     within `skip` seconds (skipped: the song didn't play)
  2. the session rules (recsys.lb_sessions): a new session after 30 min, back-to-back repeats dropped, >= 4 songs
  3. users split at random: 80% training (20% of them for the ranker), 10% validation, 10% held-out; all of a
     user's sessions land on the same side
  4. each user keeps their K most recent listens; the catalog is the songs at least `min_users` different
     training users played in them; other songs' listens are dropped and the session rules applied again. K is
     the smallest cap at which the listens left reach `target` (the MPD's 66.3M song entries)
  5. per catalog song: its name and artist (its first listen's), its main artist (the cleaned name in its key),
     its most common album, its length (the median of its listens' lengths, else its MPD song's, else 0 =
     unknown), its Spotify URI (its most linked Spotify track, else its MPD song's; each URI used once; else
     "listenbrainz:track:<key>"); per artist: genres from MusicBrainz, by the artist's MusicBrainz ID, Spotify
     page or MPD songs

Writes the MPD store's files (sessions as playlists, with no names) plus:
  playlist_users.npy (ListenBrainz user ID), playlist_split.npy (0 retriever, 1 ranker, 2 validation,
  3 held-out), playlist_starts.npy (the session's first listen, Unix seconds), track_mpd.npy (the MPD store's
  track ID of each song, -1 none), genres.jsonl (per artist URI, the format of data/genres/*.jsonl).

    MallocLargeCache=0 python -m recsys.lb_store --extract data/lb_extract --out data/lb_store \\
        --mb-genres data/genres/musicbrainz_lb_artists.jsonl

Memory: at most ~13 GB in use for the 465M-listen extract (the noise filter over every listen). On macOS,
MallocLargeCache=0 hands freed memory back to the system at once; by default macOS keeps big freed blocks
for reuse, and the process holds ~24 GB.
"""
import argparse
import calendar
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from recsys.lb_extract import load_extract
from recsys.lb_sessions import by_user_and_time, rank_from_end, sessions, user_starts_from, users_per_song
from recsys.listenbrainz import clean_text
from recsys.mpd_store import MARKER

MPD_SIZE = 66_346_428      # song entries in the MPD's 1M playlists
RETRIEVER, RANKER, VALIDATION, HELD_OUT = 0, 1, 2, 3
CHUNK = 50_000_000


def log(message):
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def noise(users, seconds, max_day=1000, skip=30):
    """True for listens to drop: on a day with more than max_day of the user's listens, at the same second as
    the user's previous listen, or followed by the user's next one within `skip` seconds."""
    first = user_starts_from(users)
    day = seconds // 86_400
    new_day = first.copy()
    new_day[1:] |= day[1:] != day[:-1]
    del day
    run = np.cumsum(new_day, dtype=np.int32)
    run -= 1
    drop = (np.bincount(run) > max_day)[run]       # per day first: one byte per listen, not eight
    del run, new_day
    same_user = ~first[1:]
    gap = np.diff(seconds)
    drop[1:] |= same_user & (gap == 0)               # the later of two listens at the same second
    drop[:-1] |= same_user & (gap > 0) & (gap < skip)  # the next listen came too soon: this one was skipped
    return drop


def split_users(n_users, seed, validation=0.1, held_out=0.1, ranker=0.2):
    """Each user's side (RETRIEVER, RANKER, VALIDATION, HELD_OUT), at random."""
    order = np.random.default_rng(seed).permutation(n_users)
    n_val, n_test = round(n_users * validation), round(n_users * held_out)
    n_train = n_users - n_val - n_test
    n_ranker = round(n_train * ranker)
    split = np.empty(n_users, dtype=np.int8)
    split[order[:n_train - n_ranker]] = RETRIEVER
    split[order[n_train - n_ranker:n_train]] = RANKER
    split[order[n_train:n_train + n_val]] = VALIDATION
    split[order[n_train + n_val:]] = HELD_OUT
    return split


def capped_catalog(users, seconds, songs, training, rank, n_songs, k, min_users):
    """With each user capped at their k most recent listens (rank: rank_from_end(users)): (listens kept, as
    indices, the catalog, and each kept listen's session number). The catalog is the songs at least min_users
    training users played; other songs' listens are dropped and the session rules applied again (dropping songs
    can leave repeats and short sessions). The session numbers are the rules' own: dropping a repeat can open
    a gap over 30 min between the listens left, which must not split their session."""
    capped = rank < k
    catalog = users_per_song(users, songs, n_songs, capped & training) >= min_users
    idx = np.flatnonzero(capped & catalog[songs])
    kept, session = sessions(users[idx], seconds[idx], songs[idx])
    return idx[kept], catalog, session[kept]


def choose_cap(users, seconds, songs, training, n_songs, min_users, target, report=log):
    """The smallest per-user cap k whose listens (after the catalog filter) reach target: doubling from 4,096
    until it's enough, then a binary search; every listen if even no cap falls short. Returns
    (k or None, kept indices, catalog, session numbers)."""
    rank = rank_from_end(users)
    most = int(np.bincount(users).max())
    trial = lambda k: capped_catalog(users, seconds, songs, training, rank, n_songs, k, min_users)
    lo, hi = 1, min(4096, most)
    best = trial(hi)
    report(f"  cap {hi:,}: {len(best[0]):,} listens, {int(best[1].sum()):,} songs")
    while len(best[0]) < target and hi < most:
        lo, hi = hi + 1, min(hi * 2, most)
        best = trial(hi)
        report(f"  cap {hi:,}: {len(best[0]):,} listens, {int(best[1].sum()):,} songs")
    if len(best[0]) < target:
        report(f"  no cap: {len(best[0]):,} listens, below the target {target:,}")
        return None, *best
    while lo < hi:
        mid = (lo + hi) // 2
        found = trial(mid)
        report(f"  cap {mid:,}: {len(found[0]):,} listens, {int(found[1].sum()):,} songs")
        if len(found[0]) >= target:
            hi, best = mid, found
        else:
            lo = mid + 1
    return hi, *best


def value_counts(songs, values):
    """The distinct (song, value) pairs of parallel arrays as sorted keys song << 32 | value (values >= 0),
    and how many times each occurs."""
    return np.unique(songs.astype(np.int64) << 32 | values.astype(np.int64), return_counts=True)


def merged_counts(parts):
    """Several value_counts results (one per chunk of listens) as one: each key once, its counts added."""
    keys = np.concatenate([k for k, _ in parts])
    counts = np.concatenate([c for _, c in parts])
    if not len(keys):
        return keys, counts
    order = np.argsort(keys, kind="stable")
    keys, counts = keys[order], counts[order]
    first = np.flatnonzero(np.append(True, keys[1:] != keys[:-1]))
    return keys[first], np.add.reduceat(counts, first)


def mode_from_counts(keys, counts, n_songs):
    """The most common value per song (-1 if none), from value_counts keys and counts; ties: the smaller value."""
    out = np.full(n_songs, -1, dtype=np.int64)
    if not len(keys):
        return out
    song = keys >> 32
    order = np.lexsort((-counts, song))          # per song, the highest count first
    first = np.ones(len(order), dtype=bool)
    first[1:] = song[order][1:] != song[order][:-1]
    winners = order[first]
    out[song[winners]] = keys[winners] & 0xFFFFFFFF
    return out


def median_from_counts(keys, counts, n_songs):
    """The median value per song (0 if none; of an even number, the lower middle one), from value_counts keys
    and counts: the value the song's middle listen has, with its listens in value order."""
    out = np.zeros(n_songs, dtype=np.int64)
    if not len(keys):
        return out
    song = keys >> 32
    starts = np.flatnonzero(np.append(True, song[1:] != song[:-1]))
    total = np.add.reduceat(counts, starts)      # each song's listens
    end = np.cumsum(counts)                      # one past each key's last listen, all songs' listens in a row
    middle = end[starts] - counts[starts] + (total - 1) // 2
    out[song[starts]] = keys[np.searchsorted(end, middle, side="right")] & 0xFFFFFFFF
    return out


def mode_per_song(songs, values, n_songs):
    """The most common value per song (-1 if none), from parallel arrays; ties: the smaller value."""
    return mode_from_counts(*value_counts(songs, values), n_songs)


def median_per_song(songs, values, n_songs):
    """The median value per song (0 if none)."""
    return median_from_counts(*value_counts(songs, values), n_songs)


def per_song_columns(extract, catalog):
    """For catalog songs, from every listen in the extract: the most common Spotify track (code), album (code)
    and the median length (ms, plausible lengths only). Read in chunks; each chunk's (song, value) pairs are
    counted before the next is read, so memory holds the distinct pairs, not one pair per listen."""
    n_songs = len(catalog)
    spotify_counts, release_counts, length_counts = [], [], []
    for start in range(0, extract.meta["listens"], CHUNK):
        song = np.asarray(extract.song[start:start + CHUNK])
        keep = catalog[song]
        song = song[keep]
        spotify = np.asarray(extract.spotify[start:start + CHUNK])[keep]
        release = np.asarray(extract.release[start:start + CHUNK])[keep]
        length = np.asarray(extract.duration_ms[start:start + CHUNK])[keep]
        spotify_counts.append(value_counts(song[spotify >= 0], spotify[spotify >= 0]))
        release_counts.append(value_counts(song[release >= 0], release[release >= 0]))
        plausible = (length >= 20_000) & (length <= 30 * 60_000)
        length_counts.append(value_counts(song[plausible], length[plausible]))
    return (mode_from_counts(*merged_counts(spotify_counts), n_songs),
            mode_from_counts(*merged_counts(release_counts), n_songs),
            median_from_counts(*merged_counts(length_counts), n_songs))


def read_genre_files(mpd_genres, mb_genres):
    """Genre lookups: {Spotify artist URI: genres} and {MusicBrainz ID: genres} (non-empty ones only), from
    the MPD artists' file (artist_uri, mbids) and the ListenBrainz artists' MusicBrainz file (mbid,
    spotify_ids)."""
    by_uri, by_mbid = {}, {}
    if mpd_genres and Path(mpd_genres).exists():
        for line in open(mpd_genres, encoding="utf-8"):
            row = json.loads(line)
            if row["genres"]:
                by_uri.setdefault(row["artist_uri"], row["genres"])
                for mbid in row["mbids"]:
                    by_mbid.setdefault(mbid, row["genres"])
    if mb_genres and Path(mb_genres).exists():
        for line in open(mb_genres, encoding="utf-8"):
            row = json.loads(line)
            if row["genres"]:
                by_mbid[row["mbid"]] = row["genres"]
                for sid in row["spotify_ids"]:
                    by_uri.setdefault(f"spotify:artist:{sid}", row["genres"])
    return by_uri, by_mbid


def build_store(extract_dir, out, target=MPD_SIZE, min_users=5, seed=42, max_day=1000, skip=30,
                mpd_store=None, mpd_genres=None, mb_genres=None):
    """Build the store in `out` (see the module docstring); returns its store.json contents."""
    began = time.time()
    out = Path(out)
    if (out / MARKER).exists():
        raise FileExistsError(f"{out} already holds a store")
    out.mkdir(parents=True, exist_ok=True)
    extract = load_extract(extract_dir)
    n_songs = extract.meta["songs"]
    start = calendar.timegm(datetime.fromisoformat(extract.meta["window"]["start"]).utctimetuple())

    log(f"sorting {extract.meta['listens']:,} listens by user, then time")
    users, seconds, songs, user_ids = by_user_and_time(extract.user, extract.timestamp, extract.song, start)
    counts = {"listens": len(users)}
    keep = ~noise(users, seconds, max_day, skip)
    counts["after the noise filter"] = int(keep.sum())
    users = users[keep]                          # one column at a time: each original is freed before the next
    seconds = seconds[keep]                      # copy (a true/false mask, not a list of 8-byte positions)
    songs = songs[keep]
    del keep
    kept = sessions(users, seconds, songs)[0]
    users = users[kept]
    seconds = seconds[kept]
    songs = songs[kept]
    counts["after the session rules"] = len(users)
    del kept

    split = split_users(len(user_ids), seed)
    training = split[users] <= RANKER
    log(f"choosing the cap: {len(users):,} listens, target {target:,}")
    cap, final, catalog, session = choose_cap(users, seconds, songs, training, n_songs, min_users, target)
    held_out = split[users] == HELD_OUT          # what share of the held-out users' listening the catalog holds
    coverage = float(catalog[songs[held_out]].mean()) if held_out.any() else None
    del held_out
    users, seconds, songs = users[final], seconds[final], songs[final]
    counts["final"] = len(users)
    del final, training

    # sessions -> playlists; catalog songs -> track IDs (ordered by extract song code)
    codes = np.flatnonzero(catalog)
    track_of = np.full(n_songs, -1, dtype=np.int64)
    track_of[codes] = np.arange(len(codes))
    starts = np.flatnonzero(np.append(True, session[1:] != session[:-1]))  # the session rules' own sessions
    offsets = np.append(starts, len(users)).astype(np.int64)
    session_users = users[starts]
    log(f"{len(starts):,} sessions, {len(codes):,} songs; reading the songs' details")

    spotify_code, release_code, length = per_song_columns(extract, catalog)
    song_rows = {}
    with open(Path(extract_dir) / "songs.jsonl", encoding="utf-8") as f:
        for code, line in enumerate(f):
            if catalog[code]:
                song_rows[code] = json.loads(line)
    mpd_track = np.full(len(codes), -1, dtype=np.int64)
    store_mpd = None
    if mpd_store:
        from recsys.listenbrainz import mpd_lookup
        from recsys.mpd_store import load_store
        store_mpd = load_store(mpd_store)
        _, _, mpd_by_name, _ = mpd_lookup(store_mpd)
        for i, code in enumerate(codes):
            mpd_track[i] = mpd_by_name.get(song_rows[code][0], -1)

    # Spotify URIs, each used once: songs with more listens choose first
    listens_per_track = np.bincount(track_of[songs], minlength=len(codes))
    uris, taken = [None] * len(codes), set()
    for i in np.argsort(-listens_per_track, kind="stable"):
        code = codes[i]
        options = []
        if spotify_code[code] >= 0:
            options.append(f"spotify:track:{extract.spotify_tracks[spotify_code[code]]}")
        if mpd_track[i] >= 0:
            options.append(store_mpd.track_uris[mpd_track[i]])
        uri = next((u for u in options if u not in taken), f"listenbrainz:track:{song_rows[code][0]}")
        taken.add(uri)
        uris[i] = uri
    durations = length[codes].copy()
    if store_mpd is not None:
        missing = (durations == 0) & (mpd_track >= 0)
        durations[missing] = np.asarray(store_mpd.track_duration_ms)[mpd_track[missing]]

    # artists (by cleaned main-artist name) and albums (artist + album name)
    artist_index, artist_uris, artist_names, track_artist = {}, [], [], np.empty(len(codes), dtype=np.int32)
    album_index, album_uris, album_names, track_album = {}, [], [], np.empty(len(codes), dtype=np.int32)
    for i, code in enumerate(codes):
        key, artist, title = song_rows[code]
        artist_key = key.split("|", 1)[0]
        a = artist_index.get(artist_key)
        if a is None:
            a = artist_index[artist_key] = len(artist_uris)
            artist_uris.append(f"listenbrainz:artist:{artist_key}")
            artist_names.append(artist)
        track_artist[i] = a
        release = extract.releases[release_code[code]] if release_code[code] >= 0 else ""
        album_key = f"{artist_key}|{clean_text(release)}"
        b = album_index.get(album_key)
        if b is None:
            b = album_index[album_key] = len(album_uris)
            album_uris.append(f"listenbrainz:album:{album_key}")
            album_names.append(release)
        track_album[i] = b

    # genres per artist: its songs' most common MusicBrainz ID, else Spotify page, else MPD songs' artist
    by_uri, by_mbid = read_genre_files(mpd_genres, mb_genres)
    def artist_mode(per_song):  # each artist's most common ID among its songs that have one
        values = np.asarray(per_song)[codes].astype(np.int64)
        have = values >= 0
        return mode_per_song(track_artist[have].astype(np.int64), values[have], len(artist_uris))
    mbid_code, spotify_artist_code = artist_mode(extract.song_artist_mbid), artist_mode(extract.song_spotify_artist)
    mpd_artist_uri = {}
    if store_mpd is not None:
        for i in np.flatnonzero(mpd_track >= 0):
            mpd_artist_uri.setdefault(int(track_artist[i]),
                                      store_mpd.artist_uris[store_mpd.track_artist[mpd_track[i]]])
    with_genres = 0
    with open(out / "genres.jsonl", "w", encoding="utf-8") as f:
        for a, (uri, name) in enumerate(zip(artist_uris, artist_names)):
            mbid = extract.artist_mbids[mbid_code[a]] if mbid_code[a] >= 0 else None
            spotify = f"spotify:artist:{extract.spotify_artists[spotify_artist_code[a]]}" \
                if spotify_artist_code[a] >= 0 else None
            genres = (by_mbid.get(mbid) if mbid else None) or (by_uri.get(spotify) if spotify else None) \
                or by_uri.get(mpd_artist_uri.get(a)) or []
            with_genres += bool(genres)
            f.write(json.dumps({"artist_uri": uri, "artist_name": name, "mbids": [mbid] if mbid else [],
                                "genres": genres}, ensure_ascii=False) + "\n")

    write_json = lambda name, value: (out / name).write_text(json.dumps(value, ensure_ascii=False))
    write_json("track_uris.json", uris)
    write_json("track_names.json", [song_rows[c][2] for c in codes])
    np.save(out / "track_artist.npy", track_artist)
    np.save(out / "track_album.npy", track_album)
    np.save(out / "track_duration_ms.npy", durations.astype(np.int32))
    np.save(out / "track_mpd.npy", mpd_track.astype(np.int32))
    write_json("artist_uris.json", artist_uris)
    write_json("artist_names.json", artist_names)
    write_json("album_uris.json", album_uris)
    write_json("album_names.json", album_names)
    write_json("playlist_names.json", [""] * len(starts))
    np.save(out / "playlist_pids.npy", np.arange(len(starts), dtype=np.int32))
    np.save(out / "playlist_offsets.npy", offsets)
    np.save(out / "playlist_tracks.npy", track_of[songs].astype(np.int32))
    np.save(out / "playlist_users.npy", user_ids[session_users])
    np.save(out / "playlist_split.npy", split[session_users])
    np.save(out / "playlist_starts.npy", (seconds[starts].astype(np.int64) + start))
    sides = np.bincount(split[session_users], weights=np.diff(offsets), minlength=4)
    meta = {"source": str(Path(extract_dir).resolve()), "window": extract.meta["window"],
            "rules": {"max_listens_per_day": max_day, "skip_seconds": skip, "session_gap_seconds": 1800,
                      "min_session_songs": 4, "min_users_per_song": min_users, "target_listens": target,
                      "split": "users: 80% training (20% of them ranker), 10% validation, 10% held-out",
                      "seed": seed},
            "cap_per_user": cap, "listens": counts, "playlists": len(starts), "track_entries": int(len(users)),
            "tracks": len(codes), "artists": len(artist_uris), "albums": len(album_uris),
            "held_out_listens_in_catalog": round(coverage, 4) if coverage is not None else None,
            "users": int(len(np.unique(session_users))),
            "listens_by_side": dict(zip(("retriever", "ranker", "validation", "held_out"), map(int, sides))),
            "tracks_with_spotify_uri": sum(u.startswith("spotify:track:") for u in uris),
            "tracks_with_length": int((durations > 0).sum()), "tracks_in_mpd": int((mpd_track >= 0).sum()),
            "artists_with_genres": with_genres, "build_seconds": round(time.time() - began),
            "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / MARKER).write_text(json.dumps(meta, indent=2))  # last: marks the store as finished
    log(f"done in {meta['build_seconds']} s")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--extract", default="data/lb_extract")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=MPD_SIZE)
    ap.add_argument("--min-users", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mpd-store", default="data/mpd_store")
    ap.add_argument("--mpd-genres", default="data/genres/musicbrainz_all_mpd_artists.jsonl")
    ap.add_argument("--mb-genres", default="data/genres/musicbrainz_lb_artists.jsonl")
    args = ap.parse_args()
    meta = build_store(args.extract, args.out, args.target, args.min_users, args.seed, mpd_store=args.mpd_store,
                       mpd_genres=args.mpd_genres, mb_genres=args.mb_genres)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
