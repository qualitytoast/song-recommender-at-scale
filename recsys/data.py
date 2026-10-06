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
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

DURATION_BUCKETS = 10  # song lengths are grouped into this many equal-sized buckets
NAME_WORD_MIN_COUNT = 2  # playlist-name words used in fewer training names are dropped
GENRE_MIN_ARTISTS = 5    # genres listed for fewer vocab artists are dropped
GENRES_PER_ARTIST = 5    # an artist keeps at most this many genres (most votes first)
GENRES_FILE = "data/genres/musicbrainz_artists.jsonl"  # from scripts/fetch_genres.py or genres_from_dump.py


def load_playlists(folder, max_playlists, min_playlist_len, song_key):
    """Songs of the first max_playlists playlists, identified by song_key.

    Returns (playlists, tracks, playlist_names): each playlist is a list of
    song keys; tracks maps every key to its MPD track record (track_name,
    artist_uri, album_uri, duration_ms, ...), and with song_key "track_name"
    same-titled songs share one record (the last one read); playlist_names[i]
    is playlist i's name.

    Files are read in alphabetical order, not numeric: mpd.slice.10000-10999.json
    comes before mpd.slice.2000-2999.json. That is the order v1 read them in.
    """
    playlists, tracks, playlist_names = [], {}, []
    for file_name in sorted(f for f in os.listdir(folder) if f.endswith(".json")):
        with open(os.path.join(folder, file_name), encoding="utf-8") as f:
            for playlist in json.load(f)["playlists"]:
                if len(playlist["tracks"]) >= min_playlist_len:
                    playlists.append([t[song_key] for t in playlist["tracks"]])
                    tracks.update((t[song_key], t) for t in playlist["tracks"])
                    playlist_names.append(playlist["name"])
        if len(playlists) >= max_playlists:
            break
    return playlists[:max_playlists], tracks, playlist_names[:max_playlists]


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
    Returns (X, Y, P): P[w] is the index (in `playlists`) of window w's playlist.
    """
    X, Y, P = [], [], []
    for p, playlist in enumerate(playlists):
        ids = [track_to_id.get(t) for t in playlist]
        for i in range(len(ids) - context_length):
            window, target = ids[i:i + context_length], ids[i + context_length]
            if target is None or None in window:
                continue
            X.append(window)
            Y.append(target)
            P.append(p)
    X = np.array(X, dtype=np.int64).reshape(-1, context_length)  # keeps 2D shape when empty
    return X, np.array(Y, dtype=np.int64), np.array(P, dtype=np.int64)


def song_feature_ids(vocab, tracks, field):
    """For each song in the vocab, the ID of its value of `field` (e.g. its artist).

    IDs are numbered in order of first appearance along the vocab. Returns
    (ids, count): ids[i] is song i's ID, count is how many distinct values.
    """
    values = [tracks[key][field] for key in vocab]
    value_to_id = {v: i for i, v in enumerate(dict.fromkeys(values))}
    return np.array([value_to_id[v] for v in values], dtype=np.int64), len(value_to_id)


def duration_buckets(vocab, tracks, n_buckets):
    """Each vocab song's length bucket, 0 (shortest) to n_buckets - 1 (longest).

    Cut points are quantiles of the vocab songs' durations, so each bucket holds
    about 1/n_buckets of the songs (here ~2:50, 3:10, ... 5:04 for 10 buckets).
    Returns (ids, count) like song_feature_ids.
    """
    ms = np.array([tracks[key]["duration_ms"] for key in vocab], dtype=np.float64)
    cuts = np.quantile(ms, np.linspace(0, 1, n_buckets + 1)[1:-1])
    return np.searchsorted(cuts, ms, side="right").astype(np.int64), n_buckets


def name_words(name):
    """Lowercased words of a playlist name: "Old Country " -> ["old", "country"]."""
    return re.findall(r"\w+", name.lower())


def build_word_vocab(names, min_count):
    """Words used in at least min_count of the names, in order of first appearance.
    Word i gets ID i + 1: ID 0 is padding."""
    counts = Counter(w for n in names for w in set(name_words(n)))
    first_seen = dict.fromkeys(w for n in names for w in name_words(n))
    return [w for w in first_seen if counts[w] >= min_count]


def encode_names(names, word_to_id):
    """(len(names), width) array of word IDs, one row per name, padded with 0.

    Unknown words are dropped, so a name with no known words is all padding.
    width is the longest name's known-word count (at least 1).
    """
    ids = [[word_to_id[w] for w in name_words(n) if w in word_to_id] for n in names]
    out = np.zeros((len(ids), max(1, max(map(len, ids), default=0))), dtype=np.int64)
    for row, word_ids in enumerate(ids):
        out[row, :len(word_ids)] = word_ids
    return out


def load_artist_genres(path):
    """{artist_uri: [genre names, most votes first]} from a genres JSONL file
    (written by scripts/fetch_genres.py or scripts/genres_from_dump.py).
    Ties in votes are in alphabetical order, as the scripts write them."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["artist_uri"]: [g["name"] for g in r["genres"]] for r in rows}


def build_genre_vocab(artist_genres, artist_uris, min_artists):
    """Genres listed for at least min_artists of the given artists, most widely
    used first (ties alphabetical). Genre i gets ID i + 1: ID 0 is padding."""
    counts = Counter(g for uri in artist_uris for g in artist_genres.get(uri, []))
    return sorted((g for g, n in counts.items() if n >= min_artists), key=lambda g: (-counts[g], g))


def artist_genre_ids(artist_genres, artist_uris, genre_to_id, per_artist):
    """(len(artist_uris), per_artist) array of genre IDs per artist, padded with 0.

    Rare genres (not in genre_to_id) are dropped first, then each artist keeps
    its per_artist most-voted remaining genres, so a common genre is never
    crowded out by rare ones. Artists with no genres left are all padding.
    """
    out = np.zeros((len(artist_uris), per_artist), dtype=np.int64)
    for row, uri in enumerate(artist_uris):
        kept = [genre_to_id[g] for g in artist_genres.get(uri, []) if g in genre_to_id][:per_artist]
        out[row, :len(kept)] = kept
    return out


@dataclass
class Dataset:
    vocab: list           # vocab[i] is the key (title or URI) of song i
    names: list           # names[i] is the track name of song i, for display
    song_features: dict   # {"artist": (ids, count), ...}: per-song feature IDs, see song_feature_ids
    artist_uris: list     # artist_uris[a] is the Spotify URI of artist ID a
    artist_names: list    # artist_names[a] is its name, for display
    X_train: np.ndarray   # (n, context_length) song IDs
    Y_train: np.ndarray   # (n,) next-song IDs
    X_val: np.ndarray     # used for early stopping
    Y_val: np.ndarray
    X_test: np.ndarray    # held-out set the final numbers are reported on
    Y_test: np.ndarray
    name_words: list      # name_words[i] is the word with ID i + 1 (0 is padding)
    N_train: np.ndarray   # (n, width) word IDs of each window's playlist name
    N_val: np.ndarray
    N_test: np.ndarray
    genre_names: list = None      # genre_names[i] is the genre with ID i + 1 (0 is padding)
    song_genres: np.ndarray = None  # (vocab_size, GENRES_PER_ARTIST) genre IDs of each song's artist


def build_dataset(cfg):
    d = cfg.data
    playlists, tracks, playlist_names = load_playlists(d.folder, d.max_playlists,
                                                       d.min_playlist_len, d.song_key)
    # Split playlist indices, so playlists and their names stay paired. Same shuffle
    # as splitting the playlists themselves: it only depends on the seed and length.
    train_idx, val_idx, held_out_idx = split_playlists(list(range(len(playlists))),
                                                       d.test_split, d.val_split, cfg.data_seed)
    train, val, held_out = ([playlists[i] for i in idx] for idx in (train_idx, val_idx, held_out_idx))
    vocab = build_vocab(playlists if d.vocab_from == "all" else train, d.min_freq)
    track_to_id = {t: i for i, t in enumerate(vocab)}

    # Playlist names: word vocab from training names only; each window gets its playlist's words.
    words = build_word_vocab([playlist_names[i] for i in train_idx], NAME_WORD_MIN_COUNT)
    encoded = encode_names(playlist_names, {w: i + 1 for i, w in enumerate(words)})

    X_train, Y_train, P_train = make_windows(train, track_to_id, d.context_length)
    X_test, Y_test, P_test = make_windows(held_out, track_to_id, d.context_length)
    N_train = encoded[train_idx][P_train]
    N_test = encoded[held_out_idx][P_test]
    if d.validation == "held_out_prefix":
        X_val, Y_val, N_val = X_test[:d.val_size], Y_test[:d.val_size], N_test[:d.val_size]
    else:
        X_val, Y_val, P_val = make_windows(val, track_to_id, d.context_length)
        N_val = encoded[val_idx][P_val]
    song_features = {"artist": song_feature_ids(vocab, tracks, "artist_uri"),
                     "album": song_feature_ids(vocab, tracks, "album_uri"),
                     "duration": duration_buckets(vocab, tracks, DURATION_BUCKETS)}
    artist_uris = list(dict.fromkeys(tracks[k]["artist_uri"] for k in vocab))  # same order as artist IDs
    artist_name = {tracks[k]["artist_uri"]: tracks[k]["artist_name"] for k in vocab}

    # Genres (only when used: needs the fetched genres file). Each song gets its artist's genres.
    genre_names = song_genres = None
    if "genre" in cfg.model.features:
        artist_genres = load_artist_genres(GENRES_FILE)
        missing = [u for u in artist_uris if u not in artist_genres]
        if missing:
            raise ValueError(f"{len(missing):,} vocab artists are missing from {GENRES_FILE}; fetch them first")
        genre_names = build_genre_vocab(artist_genres, artist_uris, GENRE_MIN_ARTISTS)
        per_artist = artist_genre_ids(artist_genres, artist_uris,
                                      {g: i + 1 for i, g in enumerate(genre_names)}, GENRES_PER_ARTIST)
        song_genres = per_artist[song_features["artist"][0]]

    return Dataset(vocab=vocab, names=[tracks[k]["track_name"] for k in vocab],
                   song_features=song_features, artist_uris=artist_uris,
                   artist_names=[artist_name[u] for u in artist_uris],
                   X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val,
                   X_test=X_test, Y_test=Y_test,
                   name_words=words, N_train=N_train, N_val=N_val, N_test=N_test,
                   genre_names=genre_names, song_genres=song_genres)


if __name__ == "__main__":
    import argparse
    from recsys.config import load_config

    ap = argparse.ArgumentParser(description="Build the dataset and print its sizes.")
    ap.add_argument("--config", required=True)
    ds = build_dataset(load_config(ap.parse_args().config))
    print(f"songs {len(ds.vocab):,} | train {len(ds.X_train):,} | "
          f"validation {len(ds.X_val):,} | held-out {len(ds.X_test):,}")
