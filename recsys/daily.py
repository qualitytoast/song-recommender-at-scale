"""Daily updates from ListenBrainz, the data side: what each day's dump adds to the served catalog, without
retraining the models.

ListenBrainz publishes a dump of each day's submitted listens (recsys.listenbrainz.dump_window). Per dump, oldest
first:
  1. chain: a dump is used only if its window starts where the last one used ended, or later, so two dumps of the
     same window (there were two on 2026-10-19) are never both counted
  2. its fresh listens, played at most FRESH_DAYS before the window's end, as a small extract (recsys.lb_extract);
     the rest, most of a dump (68-76% in late October 2026), is listening history imported from elsewhere
  3. the store's rules (recsys.lb_store): the noise filter, then the session rules (recsys.lb_sessions)
  4. new songs: a song outside the catalog joins once MIN_USERS different listeners have played it within NEW_DAYS,
     if the catalog has its artist (so it can get vectors: standins) and its Spotify track isn't already a catalog
     song's (that would be the same song under another title); most listeners first, at most MAX_NEW a dump
  5. the catalog songs' sessions (the other songs dropped and the session rules applied again, as in the store):
     each song's listens, and how often each song came 1 to 5 songs after another (rank_features.pair_counts)

A new song gets the next song ID, so every ID keeps its meaning (and its shard, ID % shards). The models never saw
it: it gets stand-in vectors made from the bundle's finished ones (standins). Imports numpy only, never PyTorch, so a
process with FAISS can use it (the two can't share a process, see recsys/ann.py).
"""
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recsys.lb_extract import MARKER, extract, load_extract
from recsys.lb_sessions import by_user_and_time, sessions
from recsys.lb_store import merged_counts, mode_per_song, noise
from recsys.listenbrainz import clean_text, dump_window
from recsys.pair_table import pair_counts

DAY = 86_400
FRESH_DAYS = 30   # listens played longer ago than this before their dump's end are imported history, not counted
NEW_DAYS = 30     # a new song's listeners are counted over this many days
MIN_USERS = 3     # different listeners a song needs to join (the catalog's own bar: N = 3)
MAX_NEW = 2000    # songs added per dump at most
TOLERANCE = 1.0   # seconds a dump's window may start before the last one's end (clock rounding)
AB = 1 << 32      # an update's pair keys are a * AB + b: they don't depend on the catalog's size


def chain(windows, after=None):
    """Which dumps to use, in order. windows: {name: (start, end)}; after: where the last dump used ended (None:
    start from the earliest). Each dump's window starts where the one before ended, so the dumps are followed in
    order of their start: one is used if it starts at or after the end of the last one used (within TOLERANCE),
    and another dump of a window already used is skipped. (The nearly empty dump seconds after each day's can end
    a fraction of a second before it starts, and the next day's then starts before it: ordered by their ends, a
    whole day would look like a repeat.) Returns (used, skipped, gaps): names in order, and the stretches
    [(from, to)] of time that no used dump covers."""
    used, skipped, gaps = [], [], []
    for name, (start, end) in sorted(windows.items(), key=lambda item: (item[1][0], item[1][1], item[0])):
        if after is not None and start < after - TOLERANCE:
            skipped.append(name)
            continue
        if after is not None and start > after + TOLERANCE:
            gaps.append((after, start))
        used.append(name)
        after = end
    return used, skipped, gaps


def read_window(path):
    with open(path, "rb") as f:
        return dump_window(f)


def fresh_extract(dump, out, workers=4):
    """The dump's fresh listens (played within FRESH_DAYS before its window's end, or up to an hour after it) as an
    extract in out (recsys.lb_extract; reused if one is there already). Refuses a dump whose checksum doesn't match
    its .sha256 file. Returns (the extract, the dump's window)."""
    start, end = read_window(dump)
    out = Path(out)
    if not (out / MARKER).exists():
        shutil.rmtree(out, ignore_errors=True)  # an unfinished one
        meta = extract(str(dump), out, int(end) - FRESH_DAYS * DAY, int(end) + 3600, workers)
        if meta["sha256_ok"] is not True:
            shutil.rmtree(out)
            raise ValueError(f"{dump}: checksum {'missing' if meta['sha256_ok'] is None else 'wrong'}")
    return load_extract(out), (start, end)


def ruled(user, timestamp, song, start):
    """Listens after the store's rules (the noise filter, then the session rules), sorted by user, then time:
    (user IDs, Unix seconds, songs, session numbers) of the listens kept. start: no listen is earlier."""
    users, seconds, songs, user_ids = by_user_and_time(user, timestamp, song, start)
    keep = ~noise(users, seconds)
    users, seconds, songs = users[keep], seconds[keep], songs[keep]
    kept, session = sessions(users, seconds, songs)
    return user_ids[users[kept]], seconds[kept].astype(np.int64) + start, songs[kept], session[kept]


def catalog_sessions(users, seconds, ids):
    """The catalog songs' sessions, from ruled listens (ids: each one's song ID, -1 outside the catalog): the other
    songs dropped and the session rules applied again, as the store does. Returns (song IDs, offsets per session)."""
    idx = np.flatnonzero(ids >= 0)
    kept, session = sessions(users[idx], seconds[idx], ids[idx])
    songs, session = ids[idx][kept].astype(np.int64), session[kept]
    starts = np.flatnonzero(np.append(True, session[1:] != session[:-1])) if len(songs) else np.zeros(0, np.int64)
    return songs, np.append(starts, len(songs)).astype(np.int64)


def key_hash(key):
    """A song key as a 64-bit number (the same in every run, unlike Python's hash)."""
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little", signed=True)


class Seen:
    """Who played songs outside the catalog, per dump: for the new-song rule. Each dump's part is its distinct
    (song key hash, listener) pairs."""

    def __init__(self):
        self.days = []  # (the dump's end, (n, 2) int64 pairs)

    def add(self, end, hashes, users):
        self.days.append((end, np.unique(np.stack([hashes, users]).T.astype(np.int64), axis=0)))

    def listeners(self, hashes, now):
        """How many different listeners each song (by key hash) had within NEW_DAYS before now."""
        recent = [p for end, p in self.days if end > now - NEW_DAYS * DAY]
        if not recent:
            return np.zeros(len(hashes), dtype=np.int64)
        h, n = np.unique(np.unique(np.concatenate(recent), axis=0)[:, 0], return_counts=True)
        at = np.minimum(np.searchsorted(h, hashes), len(h) - 1)
        return np.where(h[at] == hashes, n[at], 0)


@dataclass
class NewSong:
    key: str        # recsys.listenbrainz.song_key
    track: str      # its name, as its first listen in the dump gave it
    album: str      # its most common album name in the dump ("" if none)
    uri: str        # spotify:track:<ID> (its most common Spotify track), else listenbrainz:track:<key>
    artist: int     # the bundle's artist ID
    album_id: int   # the bundle's album ID for the artist's album of that name, else a new one
    listeners: int  # different listeners within NEW_DAYS when it joined
    joined: float   # the end of the dump it joined in


class Served:
    """The served songs as an update sees them, by song ID: the bundle's (base) and the ones updates added. keys:
    every song's key (recsys.listenbrainz.song_key); the rest from the bundle's files (recsys/export.py)."""

    def __init__(self, bundle, keys):
        bundle = Path(bundle)
        songs = json.loads((bundle / "songs.json").read_text())
        self.uris, self.tracks, self.artist_names = list(songs["uris"]), list(songs["tracks"]), songs["artist_names"]
        self.artist, self.album = np.load(bundle / "song_artist.npy"), np.load(bundle / "song_album.npy")
        self.genres, self.listens = np.load(bundle / "song_genres.npy"), np.load(bundle / "song_playlists.npy")
        self.album_names = json.loads((bundle / "song_albums.json").read_text())
        self.base = len(self.uris)
        if len(keys) != self.base:
            raise ValueError(f"{len(keys):,} song keys for a bundle of {self.base:,} songs")
        self.keys = list(keys)
        self.id_of = {k: i for i, k in enumerate(self.keys)}
        self.artist_of = {}  # artist key (the part of a song key before "|") -> the bundle's artist ID
        for k, a in zip(self.keys, self.artist.tolist()):
            self.artist_of.setdefault(k.split("|", 1)[0], a)
        self.spotify = {u.rsplit(":", 1)[1] for u in self.uris if u.startswith("spotify:track:")}
        self.added = []      # NewSong, in ID order from base on
        self._albums = None  # (artist ID, cleaned album name) -> album ID, made when first needed

    def copy(self):
        """A copy that adding songs to leaves this one as it is (the bundle's arrays are shared, never changed)."""
        other = object.__new__(Served)
        other.__dict__.update(self.__dict__)
        other.uris, other.tracks, other.keys, other.added = list(self.uris), list(self.tracks), list(self.keys), list(self.added)
        other.id_of, other.spotify = dict(self.id_of), set(self.spotify)
        other._albums = None if self._albums is None else dict(self._albums)
        return other

    def __len__(self):
        return len(self.keys)

    def ids(self, keys):
        """Each key's song ID, -1 for songs not served."""
        return np.array([self.id_of.get(k, -1) for k in keys], dtype=np.int64)

    def album_id(self, artist, name):
        """The artist's album of that name (cleaned, as the store matches albums), or a new album ID."""
        if self._albums is None:
            self._albums = {(a, clean_text(n)): b for a, n, b in zip(self.artist.tolist(), self.album_names,
                                                                   self.album.tolist())}
            self._next_album = int(self.album.max()) + 1
        key = (artist, clean_text(name))
        if key not in self._albums:
            self._albums[key] = self._next_album
            self._next_album += 1
        return self._albums[key]

    def add(self, key, track, album, spotify_id, listeners, joined):
        artist = self.artist_of[key.split("|", 1)[0]]
        uri = f"spotify:track:{spotify_id}" if spotify_id else f"listenbrainz:track:{key}"
        song = NewSong(key, track, album, uri, artist, self.album_id(artist, album), int(listeners), joined)
        self.id_of[key] = len(self.keys)
        self.keys.append(key)
        self.uris.append(uri)
        self.tracks.append(track)
        if spotify_id:
            self.spotify.add(spotify_id)
        self.added.append(song)
        return song

    def song_artist(self):
        return np.concatenate([self.artist, np.array([s.artist for s in self.added], dtype=self.artist.dtype)])

    def song_album(self):
        return np.concatenate([self.album, np.array([s.album_id for s in self.added], dtype=self.album.dtype)])

    def song_genres(self):
        """Each song's genre IDs: a new song has its artist's (the bundle's songs of an artist all have the same)."""
        artists, at = np.unique(self.artist, return_index=True)
        first = dict(zip(artists.tolist(), at.tolist()))
        rows = [self.genres[first[s.artist]] for s in self.added]
        return np.concatenate([self.genres, np.array(rows, dtype=self.genres.dtype).reshape(-1, self.genres.shape[1])])


class Update:
    """What the dumps used so far add to the bundle: per song ID, its listens; per pair (a, b), how often b came 1 to
    5 songs after a (keys a * AB + b, sorted); and each dump's report."""

    def __init__(self):
        self.pairs, self.counts = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        self.listens = np.zeros(0, dtype=np.int64)
        self.dumps = []

    def copy(self):
        other = Update()
        other.pairs, other.counts, other.listens, other.dumps = self.pairs, self.counts, self.listens, list(self.dumps)
        return other

    def add(self, songs, offsets, n_songs):
        keys, counts = pair_counts(songs, offsets, AB)
        self.pairs, self.counts = merged_counts([(self.pairs, self.counts), (keys, counts)])
        listens = np.bincount(songs, minlength=n_songs)
        listens[:len(self.listens)] += self.listens
        self.listens = listens
        return keys, counts


def apply_dump(served, update, seen, ext, window, name, min_users=MIN_USERS, max_new=MAX_NEW):
    """One dump's fresh listens (ext, from fresh_extract) into served (new songs) and update (listens and pairs);
    see the module docstring. Returns the dump's report, also kept in update.dumps."""
    start, end = window
    rows = list(ext.songs())  # per song of the extract: [key, artist, title] of its first listen
    users, seconds, codes, _ = ruled(ext.user, ext.timestamp, ext.song, int(end) - FRESH_DAYS * DAY)
    ids = served.ids([r[0] for r in rows])
    # who played the songs outside the catalog; those with enough listeners join
    outside = ids[codes] < 0
    hashes = np.zeros(len(rows), dtype=np.int64)
    out_codes = np.unique(codes[outside])
    hashes[out_codes] = [key_hash(rows[c][0]) for c in out_codes]
    seen.add(end, hashes[codes[outside]], users[outside])
    listeners = seen.listeners(hashes[out_codes], end)
    song_all = np.asarray(ext.song, dtype=np.int64)
    spotify = mode_per_song(*_valid(song_all, ext.spotify), len(rows))
    release = mode_per_song(*_valid(song_all, ext.release), len(rows))
    known_artist = np.array([rows[c][0].split("|", 1)[0] in served.artist_of for c in out_codes], dtype=bool)
    sid = [ext.spotify_tracks[spotify[c]] if spotify[c] >= 0 else "" for c in out_codes]
    free = np.array([not s or s not in served.spotify for s in sid], dtype=bool)
    ok = np.flatnonzero((listeners >= min_users) & known_artist & free)
    ok = sorted(ok, key=lambda i: (-listeners[i], rows[out_codes[i]][0]))[:max_new]
    joined = []
    for i in ok:
        c = out_codes[i]
        if sid[i] and sid[i] in served.spotify:  # two of today's songs with the same Spotify track: the first wins
            continue
        album = ext.releases[release[c]] if release[c] >= 0 else ""
        joined.append(served.add(rows[c][0], rows[c][2], album, sid[i], listeners[i], end))
        ids[c] = len(served) - 1
    # the catalog songs' sessions (new songs included): listens and pairs
    songs, offsets = catalog_sessions(users, seconds, ids[codes])
    keys, counts = update.add(songs, offsets, len(served))
    per_song = np.bincount(songs) if len(songs) else np.zeros(1, dtype=np.int64)
    per_user = np.unique(users, return_counts=True)[1] if len(users) else np.zeros(1, dtype=np.int64)
    report = {"dump": name, "start": start, "end": end, "fresh_listens": int(ext.meta["listens"]),
              "after_rules": int(len(users)), "listeners": int(len(np.unique(users))),
              "catalog_listens": int(len(songs)), "sessions": int(len(offsets) - 1),
              "songs_outside": int(len(out_codes)), "new_songs": len(joined),
              "songs_served": len(served), "pairs": int(len(keys)), "pair_instances": int(counts.sum()),
              "top_song_share": round(float(per_song.max() / max(len(songs), 1)), 4),
              "top_user_listens": int(per_user.max())}
    update.dumps.append(report)
    return report


def _valid(songs, values):
    values = np.asarray(values, dtype=np.int64)
    keep = values >= 0
    return songs[keep], values[keep]


def with_pairs(keys, counts, songs, pairs, pair_counts_, new_songs):
    """A pair table (keys a * songs + b, sorted and distinct; counts) with an update's pairs (keys a * AB + b, sorted
    and distinct; counts) added, as keys a * new_songs + b for the catalog grown to new_songs songs: the counts of a
    pair in both are summed. Returns (keys, counts as int64)."""
    if new_songs != songs:
        keys = keys // songs * new_songs + keys % songs  # ordered by a, then b: still sorted
    more = pairs // AB * new_songs + pairs % AB
    at = np.searchsorted(keys, more)
    old = at < len(keys)
    old[old] = keys[at[old]] == more[old]
    counts = counts.astype(np.int64)
    counts[at[old]] += pair_counts_[old]
    return np.insert(keys, at[~old], more[~old]), np.insert(counts, at[~old], pair_counts_[~old])


def involving_new(pairs, base):
    """Which of an update's pairs (keys a * AB + b) have a song an update added (ID >= base)."""
    return (pairs // AB >= base) | (pairs % AB >= base)


STANDIN_MODES = ("artist", "artist_norm", "standin")


def standins(served, update, tables, mode="standin", search="search_vectors"):
    """Stand-in rows for the songs updates added, for each of the bundle's per-song vector tables (tables: {name:
    (base songs, d) array}; in tables[search] the last column is the song's bias, averaged but never lengthened).
    mode:
      "artist"       the plain average of the artist's songs
      "artist_norm"  the same, lengthened to their average length
      "standin"      half the artist's songs (weighted by their listens) and half the songs played next to it (its
                     pairs, either way, weighted by how often), lengthened to their average length; the neighbours
                     alone if the artist has fewer than 3 songs, the artist alone if there are no neighbours
    Averaging shortens vectors (to ~0.77 of their length for the search vectors, ~0.62 for the input vectors, on the
    served pair), and a short vector rarely scores high enough to be shortlisted: hence the lengthening.
    Returns {name: (added songs, d) float32}."""
    if mode not in STANDIN_MODES:
        raise ValueError(f"unknown stand-in mode {mode!r}; known: {STANDIN_MODES}")
    base, added = served.base, served.added
    order = np.argsort(served.artist, kind="stable")
    bounds = np.searchsorted(served.artist[order], np.arange(served.artist.max() + 2))
    a, b = update.pairs // AB, update.pairs % AB
    pick = ((a >= base) & (b < base)), ((b >= base) & (a < base))
    new = np.concatenate([a[pick[0]], b[pick[1]]])
    other = np.concatenate([b[pick[0]], a[pick[1]]])
    times = np.concatenate([update.counts[pick[0]], update.counts[pick[1]]]).astype(np.float64)
    by_new = np.argsort(new, kind="stable")
    new, other, times = new[by_new], other[by_new], times[by_new]
    out = {name: np.zeros((len(added), t.shape[1]), dtype=np.float32) for name, t in tables.items()}
    for i, song in enumerate(added):
        art = order[bounds[song.artist]:bounds[song.artist + 1]]
        lo, hi = np.searchsorted(new, [base + i, base + i + 1])
        ids, w = source_weights(mode, art, served.listens[art].astype(np.float64), other[lo:hi], times[lo:hi])
        for name, table in tables.items():
            out[name][i] = blend(np.asarray(table[ids], dtype=np.float32), w, name == search, mode != "artist")
    return out


def source_weights(mode, artist_songs, artist_listens, neighbours, times):
    """(song IDs, weights summing to 1) a stand-in averages (see standins)."""
    if mode != "standin":
        return artist_songs, np.full(len(artist_songs), 1 / len(artist_songs))
    use_artist, use_neighbours = len(artist_songs) >= 3 or not len(neighbours), len(neighbours) > 0
    share = 0.5 if use_artist and use_neighbours else 1.0
    parts = []
    if use_artist:
        weight = artist_listens + 1
        parts.append((artist_songs, share * weight / weight.sum()))
    if use_neighbours:
        parts.append((neighbours, share * times / times.sum()))
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def blend(rows, weights, has_bias, lengthen):
    """The weighted average of rows; lengthened to their weighted average length if lengthen (the bias column, the
    last one if has_bias, is only averaged)."""
    vectors = rows[:, :-1] if has_bias else rows
    v = weights @ vectors
    if lengthen:
        length = np.linalg.norm(v)
        if length > 0:
            v = v * (weights @ np.linalg.norm(vectors, axis=1)) / length
    return np.append(v, weights @ rows[:, -1]) if has_bias else v
