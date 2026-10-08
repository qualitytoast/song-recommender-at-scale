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

from recsys.mpd_store import is_store, load_store, playlists_from_store

DURATION_BUCKETS = 10  # song lengths are grouped into this many equal-sized buckets
NAME_WORD_MIN_COUNT = 2  # playlist-name words used in fewer training names are dropped
GENRE_MIN_ARTISTS = 5    # genres listed for fewer vocab artists are dropped
GENRES_PER_ARTIST = 5    # an artist keeps at most this many genres (most votes first)


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


def make_chunks(playlists, track_to_id, context_length):
    """Training examples for the every-position objective.

    Each playlist is split at songs outside the vocab; each run of vocab songs is
    cut into chunks of up to context_length + 1 songs, consecutive chunks sharing
    one song, so every transition (song -> next song) appears exactly once. A
    chunk of k songs gives k - 1 predictions: inputs are its first k - 1 songs,
    targets its last k - 1 (target t is the song after input t).

    Returns (inputs, targets, P), each row one chunk: inputs (n, context_length)
    padded with 0, targets (n, context_length) padded with -100 (which the loss
    ignores), P[c] = index of chunk c's playlist. Padding sits after the real
    songs, so with causal attention no real position can see it.
    """
    L = context_length
    inputs, targets, P = [], [], []
    for p, playlist in enumerate(playlists):
        run = []
        for key in playlist + [None]:  # None ends the last run
            song = track_to_id.get(key) if key is not None else None
            if song is not None:
                run.append(song)
                continue
            for start in range(0, len(run) - 1, L):
                chunk = run[start:start + L + 1]
                pad = L + 1 - len(chunk)
                inputs.append(chunk[:-1] + [0] * pad)
                targets.append(chunk[1:] + [-100] * pad)
                P.append(p)
            run = []
    as_array = lambda rows: np.array(rows, dtype=np.int64).reshape(-1, L)
    return as_array(inputs), as_array(targets), np.array(P, dtype=np.int64)


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
    return buckets_from_durations([tracks[key]["duration_ms"] for key in vocab], n_buckets)


def buckets_from_durations(durations_ms, n_buckets):
    """duration_buckets for an array of durations (one per vocab song)."""
    ms = np.asarray(durations_ms, dtype=np.float64)
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
    Xc_train: np.ndarray = None   # every-position training chunks, see make_chunks: inputs,
    Yc_train: np.ndarray = None   # targets (-100 = padding),
    Nc_train: np.ndarray = None   # and name word IDs of each chunk's playlist
    X_rank: np.ndarray = None     # windows from the ranker's playlists (data.ranker_split),
    Y_rank: np.ndarray = None     # which the retriever never trains on
    N_rank: np.ndarray = None
    genre_names: list = None      # genre_names[i] is the genre with ID i + 1 (0 is padding)
    song_genres: np.ndarray = None  # (vocab_size, GENRES_PER_ARTIST) genre IDs of each song's artist
    fit_songs: np.ndarray = None    # the retriever's (part-A) playlists, concatenated: vocab IDs, -1 = outside
    fit_offsets: np.ndarray = None  # the vocab; where each playlist starts in fit_songs, plus the end
    H_rank: tuple = None  # (songs, begin, end) for the X_rank windows: window w's playlist up to and
    H_val: tuple = None   # including its context songs is songs[begin[w]:end[w]] (window_histories);
    H_test: tuple = None  # used by the ranker's whole-playlist inputs


def build_dataset(cfg):
    d = cfg.data
    if is_store(d.folder) and d.song_key == "track_uri":
        return build_dataset_from_store(cfg)  # integer arrays all the way; same result
    if is_store(d.folder):  # compact store from recsys.mpd_store, else raw MPD JSON slices
        playlists, tracks, playlist_names = playlists_from_store(load_store(d.folder), d.max_playlists,
                                                                 d.min_playlist_len, d.song_key)
    else:
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

    fit_idx, rank_idx = split_off_ranker(train_idx, d.ranker_split)
    fit = [playlists[i] for i in fit_idx]
    rank = [playlists[i] for i in rank_idx]
    X_train, Y_train, P_train = make_windows(fit, track_to_id, d.context_length)
    Xc_train, Yc_train, Pc_train = make_chunks(fit, track_to_id, d.context_length)
    X_rank, Y_rank, P_rank = make_windows(rank, track_to_id, d.context_length)
    X_test, Y_test, P_test = make_windows(held_out, track_to_id, d.context_length)
    N_train = encoded[fit_idx][P_train]
    N_test = encoded[held_out_idx][P_test]
    H_test = window_histories(*playlist_arrays(held_out, track_to_id), d.context_length)
    if d.validation == "held_out_prefix":
        X_val, Y_val, N_val = X_test[:d.val_size], Y_test[:d.val_size], N_test[:d.val_size]
        H_val = keep_histories(H_test, np.arange(len(X_val)))
    else:
        X_val, Y_val, P_val = make_windows(val, track_to_id, d.context_length)
        N_val = encoded[val_idx][P_val]
        H_val = window_histories(*playlist_arrays(val, track_to_id), d.context_length)
    keep = validation_keep(len(X_val), d.val_max_windows, cfg.data_seed)
    X_val, Y_val, N_val = sample_validation(X_val, Y_val, N_val, d.val_max_windows, cfg.data_seed)
    H_val = H_val if keep is None else keep_histories(H_val, keep)
    fit_songs, fit_offsets = playlist_arrays(fit, track_to_id)
    song_features = {"artist": song_feature_ids(vocab, tracks, "artist_uri"),
                     "album": song_feature_ids(vocab, tracks, "album_uri"),
                     "duration": duration_buckets(vocab, tracks, DURATION_BUCKETS)}
    artist_uris = list(dict.fromkeys(tracks[k]["artist_uri"] for k in vocab))  # same order as artist IDs
    artist_name = {tracks[k]["artist_uri"]: tracks[k]["artist_name"] for k in vocab}

    # Genres (only when used: needs the fetched genres file). Each song gets its artist's genres.
    genre_names = song_genres = None
    if "genre" in cfg.model.features:
        genre_names, song_genres = song_genre_ids(d.genres_file, artist_uris, song_features["artist"][0])

    return Dataset(vocab=vocab, names=[tracks[k]["track_name"] for k in vocab],
                   song_features=song_features, artist_uris=artist_uris,
                   artist_names=[artist_name[u] for u in artist_uris],
                   X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val,
                   X_test=X_test, Y_test=Y_test,
                   name_words=words, N_train=N_train, N_val=N_val, N_test=N_test,
                   Xc_train=Xc_train, Yc_train=Yc_train, Nc_train=encoded[fit_idx][Pc_train],
                   X_rank=X_rank, Y_rank=Y_rank, N_rank=encoded[rank_idx][P_rank],
                   genre_names=genre_names, song_genres=song_genres, fit_songs=fit_songs, fit_offsets=fit_offsets,
                   H_rank=window_histories(*playlist_arrays(rank, track_to_id), d.context_length),
                   H_val=H_val, H_test=H_test)


def validation_keep(n, max_windows, seed):
    """The indices sample_validation keeps out of n validation windows, or None for all."""
    if not max_windows or n <= max_windows:
        return None
    return np.sort(np.random.RandomState(seed).choice(n, max_windows, replace=False))


def sample_validation(X_val, Y_val, N_val, max_windows, seed):
    """At most max_windows validation windows (0 = all), a fixed random sample in the
    original order. Early stopping then costs less per check; held-out stays complete."""
    keep = validation_keep(len(X_val), max_windows, seed)
    if keep is None:
        return X_val, Y_val, N_val
    return X_val[keep], Y_val[keep], N_val[keep]


def playlist_arrays(playlists, track_to_id):
    """(songs, offsets) for playlists given as lists of song keys: their vocab IDs concatenated
    (-1 = outside the vocab), and where each playlist starts, plus the end."""
    songs = np.array([track_to_id.get(t, -1) for p in playlists for t in p], dtype=np.int64)
    return songs, np.concatenate([[0], np.cumsum([len(p) for p in playlists])]).astype(np.int64)


def window_histories(songs, offsets, context_length):
    """(songs, begin, end) for the windows windows_from_arrays cuts from these playlists, in the
    same order: window w's playlist up to and including its context songs is songs[begin[w]:end[w]],
    so songs[end[w] - context_length:end[w]] is its context."""
    _, _, playlist, start = windows_from_arrays(songs, offsets, context_length, with_starts=True)
    return songs, offsets[playlist], start + context_length


def keep_histories(histories, keep):
    """window_histories for the windows at indices keep only."""
    songs, begin, end = histories
    return songs, begin[keep], end[keep]


def split_off_ranker(train_idx, fraction):
    """(retriever playlists, ranker playlists): the last `fraction` of the (already
    shuffled) training playlists are kept for the second-stage ranker. The vocab and
    everything else are still built from all training playlists, so the held-out
    windows are the same with or without the split."""
    cut = len(train_idx) - int(len(train_idx) * fraction)
    return train_idx[:cut], train_idx[cut:]


def song_genre_ids(genres_file, artist_uris, song_artist):
    """(genre_names, song_genres): the genre vocabulary and each song's artist's genre IDs."""
    artist_genres = load_artist_genres(genres_file)
    missing = [u for u in artist_uris if u not in artist_genres]
    if missing:
        raise ValueError(f"{len(missing):,} vocab artists are missing from {genres_file}; fetch them first")
    genre_names = build_genre_vocab(artist_genres, artist_uris, GENRE_MIN_ARTISTS)
    per_artist = artist_genre_ids(artist_genres, artist_uris,
                                  {g: i + 1 for i, g in enumerate(genre_names)}, GENRES_PER_ARTIST)
    return genre_names, per_artist[song_artist]


# --- the same dataset, built from a store's integer arrays (scales to the full MPD) ---

def first_appearance_ids(values):
    """(unique values in order of first appearance, each value's index in that order):
    the array version of dict.fromkeys, as used for vocab, artist and album IDs."""
    unique, first, inverse = np.unique(values, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    position = np.empty(len(order), dtype=np.int64)
    position[order] = np.arange(len(order))
    return unique[order], position[inverse.reshape(-1)]


def concat_playlists(store, playlist_ids):
    """(track IDs of the given store playlists, concatenated; start offset of each, plus the end)."""
    starts, ends = store.playlist_offsets[playlist_ids], store.playlist_offsets[np.asarray(playlist_ids) + 1]
    lengths = ends - starts
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    positions = np.repeat(starts - offsets[:-1], lengths) + np.arange(offsets[-1])
    return np.asarray(store.playlist_tracks[positions], dtype=np.int64), offsets


def windows_from_arrays(songs, offsets, context_length, with_starts=False):
    """make_windows for concatenated vocab IDs (-1 = outside the vocab) with playlist offsets.
    with_starts: also return each window's first song's index in songs."""
    L, n = context_length, len(songs)
    playlist = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    end = offsets[1:][playlist]
    unknown_before = np.concatenate([[0], np.cumsum(songs < 0)])  # unknown songs before each position
    start = np.arange(n)
    start = start[start + L < end]  # room for L inputs and a target in the same playlist
    start = start[unknown_before[start + L + 1] - unknown_before[start] == 0]
    X = songs[start[:, None] + np.arange(L)].reshape(-1, L)
    if with_starts:
        return X, songs[start + L], playlist[start], start
    return X, songs[start + L], playlist[start]


def chunks_from_arrays(songs, offsets, context_length):
    """make_chunks for concatenated vocab IDs (-1 = outside the vocab) with playlist offsets."""
    L, n = context_length, len(songs)
    playlist = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    pos = np.arange(n - 1)
    # A transition (song at g -> song at g + 1) is usable if both are vocab songs of one playlist.
    usable = (songs[:-1] >= 0) & (songs[1:] >= 0) & (playlist[:-1] == playlist[1:])
    g = pos[usable]
    run_start = np.ones(len(g), dtype=bool)
    run_start[1:] = g[1:] != g[:-1] + 1  # a run of consecutive transitions breaks at a gap
    run = np.cumsum(run_start) - 1
    k = np.arange(len(g)) - np.flatnonzero(run_start)[run]  # transition's index within its run
    run_length = np.bincount(run)
    chunk_offset = np.concatenate([[0], np.cumsum((run_length + L - 1) // L)])
    chunk = chunk_offset[run] + k // L
    n_chunks = int(chunk_offset[-1])
    inputs = np.zeros((n_chunks, L), dtype=np.int64)
    targets = np.full((n_chunks, L), -100, dtype=np.int64)
    inputs[chunk, k % L] = songs[g]
    targets[chunk, k % L] = songs[g + 1]
    chunk_playlist = np.zeros(n_chunks, dtype=np.int64)
    chunk_playlist[chunk] = playlist[g]
    return inputs, targets, chunk_playlist


def build_dataset_from_store(cfg):
    """build_dataset for a store with track-URI song keys, without per-track Python objects.
    Gives the same Dataset as reading the same playlists from the JSON slices."""
    d = cfg.data
    store = load_store(d.folder)
    lengths = np.diff(store.playlist_offsets)
    kept = np.flatnonzero(lengths >= d.min_playlist_len)[:d.max_playlists]
    train_idx, val_idx, held_out_idx = (np.array(i, dtype=np.int64) for i in split_playlists(
        list(range(len(kept))), d.test_split, d.val_split, cfg.data_seed))

    counted, _ = concat_playlists(store, kept if d.vocab_from == "all" else kept[train_idx])
    counts = np.bincount(counted, minlength=len(store.track_uris))
    in_order, _ = first_appearance_ids(counted)
    vocab_tracks = in_order[counts[in_order] >= d.min_freq]  # store track ID of each vocab song
    to_vocab = np.full(len(store.track_uris), -1, dtype=np.int64)
    to_vocab[vocab_tracks] = np.arange(len(vocab_tracks))

    names_kept = [store.playlist_names[i] for i in kept]
    words = build_word_vocab([names_kept[i] for i in train_idx], NAME_WORD_MIN_COUNT)
    encoded = encode_names(names_kept, {w: i + 1 for i, w in enumerate(words)})

    def split_arrays(idx):
        tracks, offsets = concat_playlists(store, kept[idx])
        return to_vocab[tracks], offsets
    fit_idx, rank_idx = split_off_ranker(train_idx, d.ranker_split)
    train_songs, train_offsets = split_arrays(fit_idx)
    test_songs, test_offsets = split_arrays(held_out_idx)
    X_train, Y_train, P_train = windows_from_arrays(train_songs, train_offsets, d.context_length)
    Xc_train, Yc_train, Pc_train = chunks_from_arrays(train_songs, train_offsets, d.context_length)
    X_test, Y_test, P_test = windows_from_arrays(test_songs, test_offsets, d.context_length)
    N_train, N_test = encoded[fit_idx][P_train], encoded[held_out_idx][P_test]
    rank_arrays = split_arrays(rank_idx)
    X_rank, Y_rank, P_rank = windows_from_arrays(*rank_arrays, d.context_length)
    H_test = window_histories(test_songs, test_offsets, d.context_length)
    if d.validation == "held_out_prefix":
        X_val, Y_val, N_val = X_test[:d.val_size], Y_test[:d.val_size], N_test[:d.val_size]
        H_val = keep_histories(H_test, np.arange(len(X_val)))
    else:
        val_arrays = split_arrays(val_idx)
        X_val, Y_val, P_val = windows_from_arrays(*val_arrays, d.context_length)
        N_val = encoded[val_idx][P_val]
        H_val = window_histories(*val_arrays, d.context_length)
    keep = validation_keep(len(X_val), d.val_max_windows, cfg.data_seed)
    X_val, Y_val, N_val = sample_validation(X_val, Y_val, N_val, d.val_max_windows, cfg.data_seed)
    H_val = H_val if keep is None else keep_histories(H_val, keep)

    artists, song_artist = first_appearance_ids(store.track_artist[vocab_tracks])
    albums, song_album = first_appearance_ids(store.track_album[vocab_tracks])
    song_features = {"artist": (song_artist, len(artists)), "album": (song_album, len(albums)),
                     "duration": buckets_from_durations(store.track_duration_ms[vocab_tracks], DURATION_BUCKETS)}
    artist_uris = [store.artist_uris[a] for a in artists]
    genre_names = song_genres = None
    if "genre" in cfg.model.features:
        genre_names, song_genres = song_genre_ids(d.genres_file, artist_uris, song_artist)
    return Dataset(vocab=[store.track_uris[j] for j in vocab_tracks],
                   names=[store.track_names[j] for j in vocab_tracks],
                   song_features=song_features, artist_uris=artist_uris,
                   artist_names=[store.artist_names[a] for a in artists],
                   X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, X_test=X_test, Y_test=Y_test,
                   name_words=words, N_train=N_train, N_val=N_val, N_test=N_test,
                   Xc_train=Xc_train, Yc_train=Yc_train, Nc_train=encoded[fit_idx][Pc_train],
                   X_rank=X_rank, Y_rank=Y_rank, N_rank=encoded[rank_idx][P_rank],
                   genre_names=genre_names, song_genres=song_genres, fit_songs=train_songs,
                   fit_offsets=train_offsets, H_rank=window_histories(*rank_arrays, d.context_length),
                   H_val=H_val, H_test=H_test)


if __name__ == "__main__":
    import argparse
    from recsys.config import load_config

    ap = argparse.ArgumentParser(description="Build the dataset and print its sizes.")
    ap.add_argument("--config", required=True)
    ds = build_dataset(load_config(ap.parse_args().config))
    print(f"songs {len(ds.vocab):,} | train {len(ds.X_train):,} | "
          f"validation {len(ds.X_val):,} | held-out {len(ds.X_test):,}")
