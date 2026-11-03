"""A shard's appeared-together counts in about 5 bytes a pair: recsys/shard_slices.py writes them and
recsys/shard.py reads them. Each pair (a, b), song b after song a, is stored without its earlier song:
a's pairs sit next to each other, so a contents list says where they begin.

  pair_starts.npy                        (songs + 1,) int32: a's pairs are rows starts[a] to starts[a + 1]
  pair_later.npy                         (pairs,) int32: each pair's later song b, sorted within a's rows
  pair_small.npy                         (pairs,) uint8: each pair's count, BIG if it's BIG or more
  pair_big_keys.npy, pair_big_counts.npy the exact count of each of those few pairs, sorted by key a * songs + b

Every count stays exact: the ranker reads them, so they must equal what it trained on. pair_counts counts them
(recsys/rank_features.py, recsys/export.py and recsys/daily.py use it). Imports numpy only.
"""
from pathlib import Path

import numpy as np

BIG = 255  # the largest number a byte holds; only 0.013% of the served pairs reach it
PAIR_WINDOW = 5   # a pair (a, b): song b comes 1 to 5 songs after song a in a playlist
AB = 1 << 32      # an update's pair keys are a * AB + b: they don't depend on the catalog's size


def pair_counts(songs, offsets, vocab_size, window=PAIR_WINDOW):
    """(keys, counts), keys sorted: key a * vocab_size + b; counts[i] is how often song b comes
    1 to `window` songs after song a in the same playlist. songs: playlists concatenated (vocab
    IDs, -1 = outside the vocab, never counted); offsets: where each starts, plus the end."""
    playlist = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    keys = []
    for d in range(1, window + 1):
        a, b = songs[:-d], songs[d:]
        ok = (a >= 0) & (b >= 0) & (playlist[:-d] == playlist[d:])
        keys.append(a[ok] * vocab_size + b[ok])
    return np.unique(np.concatenate(keys), return_counts=True)


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


def save_pairs(folder, keys, counts, songs):
    """Write the table to folder. keys: sorted a * songs + b, as from rank_features.pair_counts (a subset
    of them too); counts: their counts."""
    folder = Path(folder)
    big = counts >= BIG
    np.save(folder / "pair_starts.npy", np.searchsorted(keys // songs, np.arange(songs + 1)).astype(np.int32))
    np.save(folder / "pair_later.npy", (keys % songs).astype(np.int32))
    np.save(folder / "pair_small.npy", np.minimum(counts, BIG).astype(np.uint8))
    np.save(folder / "pair_big_keys.npy", keys[big])
    np.save(folder / "pair_big_counts.npy", counts[big].astype(np.int32))


def read_pairs(folder):
    """The table in folder back as (keys a * songs + b, sorted; int64 counts), songs being the catalog it was
    saved for (one less than pair_starts' length): what save_pairs was given."""
    folder = Path(folder)
    starts = np.load(folder / "pair_starts.npy")
    later = np.load(folder / "pair_later.npy").astype(np.int64)
    songs = len(starts) - 1
    keys = np.repeat(np.arange(songs, dtype=np.int64), np.diff(starts)) * songs + later
    counts = np.load(folder / "pair_small.npy").astype(np.int64)
    big_keys = np.load(folder / "pair_big_keys.npy")
    counts[np.searchsorted(keys, big_keys)] = np.load(folder / "pair_big_counts.npy")
    return keys, counts


class PairTable:
    def __init__(self, folder, songs):
        folder = Path(folder)
        self.songs = songs
        self.starts, self.later = np.load(folder / "pair_starts.npy"), np.load(folder / "pair_later.npy")
        self.small = np.load(folder / "pair_small.npy")
        self.big_keys, self.big_counts = np.load(folder / "pair_big_keys.npy"), np.load(folder / "pair_big_counts.npy")

    def __len__(self):
        return len(self.later)

    def lookup(self, keys):
        """(n,) int32: the count of each key a * songs + b, 0 for pairs never seen."""
        earlier, later = keys // self.songs, (keys % self.songs).astype(np.int32)
        out = np.zeros(len(keys), dtype=np.int32)
        order = np.argsort(earlier)
        for mine in np.split(order, np.flatnonzero(np.diff(earlier[order])) + 1) if len(keys) else []:
            song = earlier[mine[0]]  # one earlier song's keys at a time: a request has at most 10 of them
            begin, end = self.starts[song], self.starts[song + 1]
            if begin < end:
                rows = self.later[begin:end]
                at = np.minimum(np.searchsorted(rows, later[mine]), len(rows) - 1)
                found = rows[at] == later[mine]
                out[mine[found]] = self.small[begin + at[found]]
        big = out == BIG
        if big.any():
            out[big] = self.big_counts[np.searchsorted(self.big_keys, keys[big])]
        return out
