"""Turns MPD playlists into (10-song window -> next song) examples.

Reproduces v1's data.py exactly: same vocab, same windows, same split. These
v1 quirks are kept on purpose so Phase 1 results compare to v1's:
  - songs are identified by track name, so different songs sharing a title merge
  - the vocab is counted over all playlists, including held-out ones
  - the validation subset is the first val_size held-out windows, so it is
    also part of the held-out set the final numbers are reported on
"""
import json
import os
from collections import Counter
from dataclasses import dataclass

import numpy as np


def load_playlists(folder, max_playlists, min_playlist_len):
    """Track names for the first max_playlists playlists.

    Files are read in alphabetical order, not numeric: mpd.slice.10000-10999.json
    comes before mpd.slice.2000-2999.json. That is the order v1 read them in.
    """
    playlists = []
    for name in sorted(f for f in os.listdir(folder) if f.endswith(".json")):
        with open(os.path.join(folder, name), encoding="utf-8") as f:
            for playlist in json.load(f)["playlists"]:
                tracks = [t["track_name"] for t in playlist["tracks"]]
                if len(tracks) >= min_playlist_len:
                    playlists.append(tracks)
        if len(playlists) >= max_playlists:
            break
    return playlists[:max_playlists]


def build_vocab(playlists, min_freq):
    """List of kept song names; a song's ID is its index in the list.

    A song is kept if it appears at least min_freq times. IDs follow the order
    songs first appear in, so the same playlists always give the same IDs.
    """
    counts = Counter(t for p in playlists for t in p)
    first_seen = dict.fromkeys(t for p in playlists for t in p)  # keeps insertion order
    return [t for t in first_seen if counts[t] >= min_freq]


def split_playlists(playlists, test_split, seed):
    """Shuffle playlists, then cut them into (train, held_out).

    v1 did np.random.seed(seed) then np.random.shuffle(playlists) as its first
    random call. RandomState(seed) is that same generator without the global
    state, so it gives the identical shuffle. The input list is not modified.
    """
    shuffled = list(playlists)
    np.random.RandomState(seed).shuffle(shuffled)
    cut = int(len(shuffled) * (1 - test_split))
    return shuffled[:cut], shuffled[cut:]


def make_windows(playlists, track_to_id, context_length):
    """Slide a window along each playlist: X = context_length songs, Y = the next one.

    Windows touching a song outside the vocab (in the input or as the target)
    are skipped, so every kept example is a true run of consecutive songs.
    """
    X, Y = [], []
    for playlist in playlists:
        ids = [track_to_id.get(t) for t in playlist]
        for i in range(len(ids) - context_length):
            window, target = ids[i:i + context_length], ids[i + context_length]
            if target is None or None in window:
                continue
            X.append(window)
            Y.append(target)
    X = np.array(X, dtype=np.int64).reshape(-1, context_length)  # keeps 2D shape when empty
    return X, np.array(Y, dtype=np.int64)


@dataclass
class Dataset:
    vocab: list           # vocab[i] is the name of song i
    X_train: np.ndarray   # (n, context_length) song IDs
    Y_train: np.ndarray   # (n,) next-song IDs
    X_test: np.ndarray    # full held-out set
    Y_test: np.ndarray
    X_val: np.ndarray     # first val_size rows of the held-out set
    Y_val: np.ndarray


def build_dataset(cfg):
    d = cfg.data
    playlists = load_playlists(d.folder, d.max_playlists, d.min_playlist_len)
    vocab = build_vocab(playlists, d.min_freq)
    track_to_id = {t: i for i, t in enumerate(vocab)}
    train, held_out = split_playlists(playlists, d.test_split, cfg.seed)
    X_train, Y_train = make_windows(train, track_to_id, d.context_length)
    X_test, Y_test = make_windows(held_out, track_to_id, d.context_length)
    return Dataset(vocab, X_train, Y_train, X_test, Y_test,
                   X_test[:d.val_size], Y_test[:d.val_size])


if __name__ == "__main__":
    import argparse
    from recsys.config import load_config

    ap = argparse.ArgumentParser(description="Build the dataset and print its sizes.")
    ap.add_argument("--config", required=True)
    ds = build_dataset(load_config(ap.parse_args().config))
    print(f"songs {len(ds.vocab):,} | train {len(ds.X_train):,} | "
          f"held-out {len(ds.X_test):,} | validation {len(ds.X_val):,}")
