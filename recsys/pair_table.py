"""A shard's appeared-together counts in about 5 bytes a pair: recsys/shard_slices.py writes them and
recsys/shard.py reads them. Each pair (a, b), song b after song a, is stored without its earlier song:
a's pairs sit next to each other, so a contents list says where they begin.

  pair_starts.npy                        (songs + 1,) int32: a's pairs are rows starts[a] to starts[a + 1]
  pair_later.npy                         (pairs,) int32: each pair's later song b, sorted within a's rows
  pair_small.npy                         (pairs,) uint8: each pair's count, BIG if it's BIG or more
  pair_big_keys.npy, pair_big_counts.npy the exact count of each of those few pairs, sorted by key a * songs + b

Every count stays exact: the ranker reads them, so they must equal what it trained on. Imports numpy only.
"""
from pathlib import Path

import numpy as np

BIG = 255  # the largest number a byte holds; only 0.013% of the served pairs reach it


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
