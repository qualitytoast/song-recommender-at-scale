"""Turns MPD playlists into (10-song window -> next song) examples.

With v1's settings (configs/v1_*.toml) this reproduces v1's data.py exactly:
same vocab, windows and split. v1's quirks are config choices so those runs
keep reproducing; configs/v1_fixed.toml switches them off:
  - song_key "track_name": songs are identified by title, so different songs
    sharing a title merge ("track_uri" keeps them apart)
  - vocab_from "all": song counts include held-out playlists ("train": training only)
  - validation "held_out_prefix": the validation set is the first val_size
    held-out windows, so it is also part of the reported held-out set
    ("separate_playlists": its own playlists, never held out)
"""
import json
import os
from collections import Counter
from dataclasses import dataclass

import numpy as np


def load_playlists(folder, max_playlists, min_playlist_len, song_key):
    """Songs of the first max_playlists playlists, identified by song_key.

    Returns (playlists, names): each playlist is a list of song keys, and
    names maps every key to its track name for display.

    Files are read in alphabetical order, not numeric: mpd.slice.10000-10999.json
    comes before mpd.slice.2000-2999.json. That is the order v1 read them in.
    """
    playlists, names = [], {}
    for file_name in sorted(f for f in os.listdir(folder) if f.endswith(".json")):
        with open(os.path.join(folder, file_name), encoding="utf-8") as f:
            for playlist in json.load(f)["playlists"]:
                tracks = playlist["tracks"]
                if len(tracks) >= min_playlist_len:
                    playlists.append([t[song_key] for t in tracks])
                    names.update((t[song_key], t["track_name"]) for t in tracks)
        if len(playlists) >= max_playlists:
            break
    return playlists[:max_playlists], names


def build_vocab(playlists, min_freq):
    """List of kept song keys; a song's ID is its index in the list.

    A song is kept if it appears at least min_freq times. IDs follow the order
    songs first appear in, so the same playlists always give the same IDs.
    """
    counts = Counter(t for p in playlists for t in p)
    first_seen = dict.fromkeys(t for p in playlists for t in p)  # keeps insertion order
    return [t for t in first_seen if counts[t] >= min_freq]


def split_playlists(playlists, test_split, val_split, seed):
    """Shuffle playlists, then cut them into (train, validation, held_out).

    v1 did np.random.seed(seed) then np.random.shuffle(playlists) as its first
    random call. RandomState(seed) is that same generator without the global
    state, so it gives the identical shuffle. The input list is not modified.

    The held-out cut is v1's, so the held-out playlists are the same whatever
    val_split is; validation playlists come out of the training side.
    """
    shuffled = list(playlists)
    np.random.RandomState(seed).shuffle(shuffled)
    test_cut = int(len(shuffled) * (1 - test_split))
    val_cut = test_cut - int(len(shuffled) * val_split)
    return shuffled[:val_cut], shuffled[val_cut:test_cut], shuffled[test_cut:]


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
    vocab: list           # vocab[i] is the key (title or URI) of song i
    names: list           # names[i] is the track name of song i, for display
    X_train: np.ndarray   # (n, context_length) song IDs
    Y_train: np.ndarray   # (n,) next-song IDs
    X_val: np.ndarray     # used for early stopping
    Y_val: np.ndarray
    X_test: np.ndarray    # held-out set the final numbers are reported on
    Y_test: np.ndarray


def build_dataset(cfg):
    d = cfg.data
    playlists, names = load_playlists(d.folder, d.max_playlists, d.min_playlist_len, d.song_key)
    train, val, held_out = split_playlists(playlists, d.test_split, d.val_split, cfg.seed)
    vocab = build_vocab(playlists if d.vocab_from == "all" else train, d.min_freq)
    track_to_id = {t: i for i, t in enumerate(vocab)}
    X_train, Y_train = make_windows(train, track_to_id, d.context_length)
    X_test, Y_test = make_windows(held_out, track_to_id, d.context_length)
    if d.validation == "held_out_prefix":
        X_val, Y_val = X_test[:d.val_size], Y_test[:d.val_size]
    else:
        X_val, Y_val = make_windows(val, track_to_id, d.context_length)
    return Dataset(vocab=vocab, names=[names[k] for k in vocab],
                   X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val,
                   X_test=X_test, Y_test=Y_test)


if __name__ == "__main__":
    import argparse
    from recsys.config import load_config

    ap = argparse.ArgumentParser(description="Build the dataset and print its sizes.")
    ap.add_argument("--config", required=True)
    ds = build_dataset(load_config(ap.parse_args().config))
    print(f"songs {len(ds.vocab):,} | train {len(ds.X_train):,} | "
          f"validation {len(ds.X_val):,} | held-out {len(ds.X_test):,}")
