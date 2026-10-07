import json

import numpy as np

from recsys.config import Config, DataConfig, ModelConfig
from recsys.data import (artist_genre_ids, sample_validation, build_dataset, build_genre_vocab, build_vocab,
                         build_word_vocab, duration_buckets, encode_names, load_artist_genres,
                         load_playlists, make_chunks, make_windows, name_words, song_feature_ids,
                         split_playlists)


def write_slice(folder, name, playlists, names=None):
    """Fake MPD slice. A track is a title (URI and artist made from it),
    a (title, uri) pair, or a (title, uri, artist) triple."""
    def track(t):
        t = t if isinstance(t, tuple) else (t,)
        title, uri, artist = t + (f"spotify:track:{t[0]}", f"artist:{t[0]}")[len(t) - 1:]
        return {"track_name": title, "track_uri": uri, "artist_uri": artist, "artist_name": artist,
                "album_uri": f"album:{artist}", "duration_ms": 200_000}  # one album per artist
    names = names or [f"playlist {i}" for i in range(len(playlists))]
    body = {"playlists": [{"name": n, "tracks": [track(t) for t in p]} for n, p in zip(names, playlists)]}
    (folder / name).write_text(json.dumps(body))


def load(folder, max_playlists=10, song_key="track_name"):
    return load_playlists(folder, max_playlists, min_playlist_len=4, song_key=song_key)[:2]


# --- load_playlists ---

def test_files_read_in_alphabetical_order(tmp_path):
    # Alphabetically "10-11" sorts before "2-3", even though 2 < 10 numerically.
    write_slice(tmp_path, "mpd.slice.2-3.json", [["two"] * 4])
    write_slice(tmp_path, "mpd.slice.10-11.json", [["ten"] * 4])
    write_slice(tmp_path, "mpd.slice.0-1.json", [["zero"] * 4])
    playlists, _ = load(tmp_path)
    assert [p[0] for p in playlists] == ["zero", "ten", "two"]


def test_short_playlists_skipped_and_non_json_ignored(tmp_path):
    write_slice(tmp_path, "mpd.slice.0-1.json", [["a", "b", "c"], ["a", "b", "c", "d"]])
    (tmp_path / "README.md").write_text("not a slice")
    playlists, _ = load(tmp_path)
    assert playlists == [["a", "b", "c", "d"]]


def test_stops_at_max_playlists(tmp_path):
    write_slice(tmp_path, "mpd.slice.0-1.json", [["a"] * 4, ["b"] * 4, ["c"] * 4])
    write_slice(tmp_path, "mpd.slice.1-2.json", [["d"] * 4])
    playlists, _ = load(tmp_path, max_playlists=2)
    assert [p[0] for p in playlists] == ["a", "b"]


def test_track_name_merges_same_titled_songs_and_uri_keeps_them_apart(tmp_path):
    two_homes = [("Home", "uri:1"), ("Home", "uri:2"), "x", "y"]
    write_slice(tmp_path, "mpd.slice.0-1.json", [two_homes])
    by_name, _ = load(tmp_path, song_key="track_name")
    by_uri, tracks = load(tmp_path, song_key="track_uri")
    assert by_name[0][:2] == ["Home", "Home"]
    assert by_uri[0][:2] == ["uri:1", "uri:2"]
    assert tracks["uri:1"]["track_name"] == tracks["uri:2"]["track_name"] == "Home"


# --- song_feature_ids ---

def test_song_feature_ids_follow_vocab_order_and_share_ids():
    tracks = {"s1": {"artist_uri": "A"}, "s2": {"artist_uri": "B"}, "s3": {"artist_uri": "A"}}
    ids, count = song_feature_ids(["s2", "s1", "s3"], tracks, "artist_uri")
    np.testing.assert_array_equal(ids, [0, 1, 1])  # B first seen -> 0; s1 and s3 share A
    assert count == 2


def test_duration_buckets_split_songs_into_equal_groups():
    tracks = {f"s{i}": {"duration_ms": ms} for i, ms in enumerate([400, 100, 300, 200])}
    vocab = ["s0", "s1", "s2", "s3"]
    # 2 buckets: cut at the median (250) -> short = {100, 200}, long = {300, 400}
    ids, count = duration_buckets(vocab, tracks, 2)
    np.testing.assert_array_equal(ids, [1, 0, 1, 0])
    assert count == 2
    # 4 buckets: cuts at 175, 250, 325 -> one song each, in order of length
    np.testing.assert_array_equal(duration_buckets(vocab, tracks, 4)[0], [3, 0, 2, 1])


# --- build_vocab ---

def test_vocab_drops_rare_songs():
    # counts: a=1, b=2, c=3, d=1
    playlists = [["a", "b", "c"], ["b", "c", "d"], ["c"]]
    assert build_vocab(playlists, min_freq=2) == ["b", "c"]


def test_vocab_ids_follow_first_appearance_not_alphabet():
    assert build_vocab([["z", "a"], ["a", "z"]], min_freq=2) == ["z", "a"]


def test_vocab_counts_repeats_within_one_playlist():
    assert build_vocab([["x", "x"]], min_freq=2) == ["x"]


# --- make_windows ---

def test_windows_slide_one_step():
    ids = {"a": 0, "b": 1, "c": 2}
    X, Y, _ = make_windows([["a", "b", "c", "a"]], ids, context_length=2)
    np.testing.assert_array_equal(X, [[0, 1], [1, 2]])
    np.testing.assert_array_equal(Y, [2, 0])


def test_windows_touching_unknown_song_skipped():
    # "?" is not in the vocab. With context 2:
    #   [b, c] -> ?  skip (target)   [c, ?] -> b  skip (input)
    #   [?, b] -> c  skip (input)    [b, c] -> b  keep
    ids = {"b": 0, "c": 1}
    X, Y, _ = make_windows([["b", "c", "?", "b", "c", "b"]], ids, context_length=2)
    np.testing.assert_array_equal(X, [[0, 1]])
    np.testing.assert_array_equal(Y, [0])


def test_playlist_not_longer_than_context_gives_no_windows():
    X, Y, _ = make_windows([["a", "a"]], {"a": 0}, context_length=2)
    assert X.shape == (0, 2) and Y.shape == (0,)


# --- split_playlists ---

PLAYLISTS = [[str(i)] for i in range(10)]


def test_two_way_split_sizes_and_no_overlap():
    train, val, held_out = split_playlists(PLAYLISTS, test_split=0.1, val_split=0.0, seed=42)
    assert len(train) == 9 and val == [] and len(held_out) == 1
    assert sorted(train + held_out) == sorted(PLAYLISTS)


def test_split_matches_v1_global_seed_shuffle():
    # v1: np.random.seed(seed); np.random.shuffle(playlists); first 90% train.
    playlists = [[str(i)] for i in range(50)]
    expected = list(playlists)
    np.random.seed(42)
    np.random.shuffle(expected)
    train, _, held_out = split_playlists(playlists, test_split=0.1, val_split=0.0, seed=42)
    assert train == expected[:45] and held_out == expected[45:]


def test_three_way_split_80_10_10():
    train, val, held_out = split_playlists(PLAYLISTS, test_split=0.1, val_split=0.1, seed=42)
    assert (len(train), len(val), len(held_out)) == (8, 1, 1)
    assert sorted(train + val + held_out) == sorted(PLAYLISTS)


def test_held_out_does_not_depend_on_val_split():
    _, _, two_way = split_playlists(PLAYLISTS, test_split=0.1, val_split=0.0, seed=42)
    _, _, three_way = split_playlists(PLAYLISTS, test_split=0.1, val_split=0.1, seed=42)
    assert two_way == three_way


def test_split_does_not_modify_input():
    original = list(PLAYLISTS)
    split_playlists(PLAYLISTS, test_split=0.1, val_split=0.1, seed=42)
    assert PLAYLISTS == original


# --- build_dataset: how the settings are wired together ---

def tiny_config(folder, vocab_from, validation, val_split, val_size=2, song_key="track_name", features=()):
    data = DataConfig(folder=str(folder), max_playlists=20, min_playlist_len=4, min_freq=2,
                      song_key=song_key, vocab_from=vocab_from, context_length=2,
                      test_split=0.1, validation=validation, val_size=val_size, val_split=val_split,
                      genres_file=str(folder / "genres.jsonl"), val_max_windows=0)
    model = ModelConfig(embed_dim=4, num_layers=1, dropout=0.0, scale_attention=True,
                        init="pytorch", features=list(features))
    return Config(data_seed=42, train_seeds=[1], data=data, model=model, train=None)


def write_tiny_mpd(folder):
    # 20 playlists over songs s0..s9; each song is in a few playlists.
    rng = np.random.RandomState(0)
    write_slice(folder, "mpd.slice.0-19.json",
                [[f"s{j}" for j in rng.randint(0, 10, size=6)] for _ in range(20)])


def test_vocab_from_train_counts_only_training_playlists(tmp_path):
    write_tiny_mpd(tmp_path)
    cfg = tiny_config(tmp_path, "train", "separate_playlists", 0.1)
    ds = build_dataset(cfg)
    playlists, _ = load(tmp_path, max_playlists=20)
    train, _, _ = split_playlists(playlists, 0.1, 0.1, seed=42)
    assert ds.vocab == build_vocab(train, min_freq=2)


def test_separate_validation_comes_from_its_own_playlists(tmp_path):
    write_tiny_mpd(tmp_path)
    ds = build_dataset(tiny_config(tmp_path, "train", "separate_playlists", 0.1))
    playlists, _ = load(tmp_path, max_playlists=20)
    _, val, held_out = split_playlists(playlists, 0.1, 0.1, seed=42)
    ids = {t: i for i, t in enumerate(ds.vocab)}
    np.testing.assert_array_equal(ds.X_val, make_windows(val, ids, 2)[0])
    np.testing.assert_array_equal(ds.X_test, make_windows(held_out, ids, 2)[0])


def test_song_features_map_each_vocab_song_to_its_artist_and_album(tmp_path):
    write_tiny_mpd(tmp_path)
    ds = build_dataset(tiny_config(tmp_path, "train", "separate_playlists", 0.1))
    _, tracks = load(tmp_path, max_playlists=20)
    for field, name in [("artist_uri", "artist"), ("album_uri", "album")]:
        ids, count = ds.song_features[name]
        assert len(ids) == len(ds.vocab) and count == len({tracks[k][field] for k in ds.vocab})
        for i in range(len(ds.vocab)):  # same ID <=> same value
            for j in range(len(ds.vocab)):
                same_value = tracks[ds.vocab[i]][field] == tracks[ds.vocab[j]][field]
                assert (ids[i] == ids[j]) == same_value


def test_held_out_prefix_validation_is_first_held_out_windows(tmp_path):
    write_tiny_mpd(tmp_path)
    ds = build_dataset(tiny_config(tmp_path, "all", "held_out_prefix", 0.0, val_size=2))
    np.testing.assert_array_equal(ds.X_val, ds.X_test[:2])


# --- playlist names ---

def test_load_returns_names_of_kept_playlists(tmp_path):
    write_slice(tmp_path, "mpd.slice.0-1.json", [["a"] * 3, ["b"] * 4], names=["too short", "Gym"])
    playlists, _, names = load_playlists(tmp_path, 10, min_playlist_len=4, song_key="track_name")
    assert playlists == [["b"] * 4] and names == ["Gym"]


def test_name_words_lowercase_and_split_on_non_letters():
    assert name_words("Old Country ") == ["old", "country"]
    assert name_words("90's R&B!!") == ["90", "s", "r", "b"]
    assert name_words("🔥🔥") == []


def test_word_vocab_counts_names_not_repeats():
    # "rap" is in 1 name (twice), "chill" in 2 names
    assert build_word_vocab(["rap rap", "chill", "Chill vibes"], min_count=2) == ["chill"]


def test_encode_names_pads_with_zero_and_drops_unknown_words():
    encoded = encode_names(["chill vibes", "rap", "chill"], {"chill": 1, "vibes": 2})
    np.testing.assert_array_equal(encoded, [[1, 2], [0, 0], [1, 0]])


def test_windows_record_their_playlist():
    _, _, P = make_windows([["a", "b"], ["a", "b", "a", "b"]], {"a": 0, "b": 1}, context_length=2)
    np.testing.assert_array_equal(P, [1, 1])  # first playlist is too short for a window


def test_each_window_gets_its_own_playlists_name(tmp_path):
    rng = np.random.RandomState(0)
    playlists = [[f"s{j}" for j in rng.randint(0, 10, size=6)] for _ in range(20)]
    names = [["chill", "gym", "party"][i % 3] + f" mix" for i in range(20)]
    write_slice(tmp_path, "mpd.slice.0-19.json", playlists, names)
    ds = build_dataset(tiny_config(tmp_path, "train", "separate_playlists", 0.1))
    ids = {t: i for i, t in enumerate(ds.vocab)}
    for X, Y, N in [(ds.X_train, ds.Y_train, ds.N_train), (ds.X_test, ds.Y_test, ds.N_test)]:
        for x, y, n in zip(X, Y, N):
            run = list(x) + [y]  # window + target: 3 consecutive songs in its playlist
            homes = [k for k, p in enumerate(playlists)
                     if any([ids.get(t) for t in p[i:i + 3]] == run for i in range(len(p) - 2))]
            words = {ds.name_words[w - 1] for w in n if w}
            assert any(words <= set(name_words(names[k])) and words for k in homes)


# --- genres ---

ARTIST_GENRES = {  # most votes first, as the fetch scripts write them
    "A": ["rare", "rock", "pop", "indie"],
    "B": ["rock", "pop"],
    "C": ["rock", "folk"],
    "D": ["rare2"],
    # "E" was never matched
}


def test_genre_vocab_keeps_genres_used_by_enough_artists_most_used_first():
    # rock: 3 artists, pop: 2, everything else: 1
    assert build_genre_vocab(ARTIST_GENRES, ["A", "B", "C", "D", "E"], min_artists=2) == ["rock", "pop"]


def test_genre_vocab_only_counts_the_given_artists():
    assert build_genre_vocab(ARTIST_GENRES, ["A", "B"], min_artists=2) == ["pop", "rock"]  # tie: alphabetical


def test_rare_genres_dropped_before_the_cap():
    ids = artist_genre_ids(ARTIST_GENRES, ["A", "B", "C", "D", "E"], {"rock": 1, "pop": 2, "indie": 3},
                           per_artist=2)
    # A: "rare" is dropped first, so A keeps rock and pop (capping first would keep only rock).
    # D lost its only genre to the cutoff and E was never matched: both all padding.
    np.testing.assert_array_equal(ids, [[1, 2], [1, 2], [1, 0], [0, 0], [0, 0]])


def test_load_artist_genres_keeps_vote_order(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text('{"artist_uri": "A", "artist_name": "a", "mbids": ["m"], '
                    '"genres": [{"name": "rock", "count": 5}, {"name": "pop", "count": 2}]}\n'
                    '{"artist_uri": "E", "artist_name": "e", "mbids": [], "genres": []}\n')
    assert load_artist_genres(path) == {"A": ["rock", "pop"], "E": []}


def test_each_song_gets_its_artists_genres(tmp_path):
    write_tiny_mpd(tmp_path)  # songs s0..s9, artist of sJ is "artist:sJ"
    # Every artist is "rock"; artists of s0-s5 are also "pop" (6 artists); s0's also "rare" (1 artist).
    genres_file = tmp_path / "genres.jsonl"
    with open(genres_file, "w") as f:
        for j in range(10):
            names = (["rare"] if j == 0 else []) + (["pop"] if j < 6 else []) + ["rock"]
            f.write(json.dumps({"artist_uri": f"artist:s{j}", "artist_name": f"s{j}", "mbids": ["m"],
                                "genres": [{"name": n, "count": 1} for n in names]}) + "\n")
    ds = build_dataset(tiny_config(tmp_path, "train", "separate_playlists", 0.1,
                                   song_key="track_uri", features=["genre"]))
    vocab_artists = {f"artist:{k.rsplit(':', 1)[-1]}" for k in ds.vocab}
    expected_vocab = ["rock", "pop"] if sum(a in vocab_artists for a in [f"artist:s{j}" for j in range(6)]) >= 5 else ["rock"]
    assert ds.genre_names == expected_vocab  # "rare" (1 artist) never makes it
    for song, row in zip(ds.vocab, ds.song_genres):
        j = int(song.rsplit("s", 1)[-1])
        wanted = [g for g in (["pop"] if j < 6 else []) + ["rock"] if g in ds.genre_names]
        assert [ds.genre_names[i - 1] for i in row if i] == wanted


# --- every-position chunks ---

def test_chunks_split_at_unknown_songs_and_cover_each_transition_once():
    ids = {k: i for i, k in enumerate("abcdefgh")}  # "?" is not in the vocab
    # runs: [a, b] and [c, d, e, f, g, h]; with context 3 the long run gives chunks
    # [c d e f] (3 targets) and [f g h] (2 targets, padded): 1 + 3 + 2 = 6 transitions
    X, Y, P = make_chunks([["a", "b", "?", "c", "d", "e", "f", "g", "h"]], ids, context_length=3)
    np.testing.assert_array_equal(X, [[0, 0, 0], [2, 3, 4], [5, 6, 0]])
    np.testing.assert_array_equal(Y, [[1, -100, -100], [3, 4, 5], [6, 7, -100]])
    np.testing.assert_array_equal(P, [0, 0, 0])
    pairs = [(x, y) for xr, yr in zip(X, Y) for x, y in zip(xr, yr) if y != -100]
    assert pairs == [(0, 1), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]  # every transition, once, in order


def test_chunks_skip_single_songs_and_record_playlists():
    ids = {"a": 0, "b": 1}
    X, Y, P = make_chunks([["a"], ["a", "?", "b"], ["b", "a"]], ids, context_length=3)
    np.testing.assert_array_equal(Y, [[0, -100, -100]])  # only playlist 2 has two neighbours in the vocab
    np.testing.assert_array_equal(P, [2])


def test_chunks_contain_every_full_window_target(tmp_path):
    # Every (window -> target) example the last-position objective trains on is also
    # predicted somewhere in the chunks, from the same last song.
    write_tiny_mpd(tmp_path)
    ds = build_dataset(tiny_config(tmp_path, "train", "separate_playlists", 0.1))
    chunk_pairs = {(x, y) for xr, yr in zip(ds.Xc_train, ds.Yc_train) for x, y in zip(xr, yr) if y != -100}
    window_pairs = set(zip(ds.X_train[:, -1], ds.Y_train))
    assert window_pairs <= chunk_pairs


def test_validation_sample_is_a_fixed_subset_in_order():
    X, Y, N = np.arange(100).reshape(50, 2), np.arange(50), np.arange(50)[:, None]
    a, b = sample_validation(X, Y, N, 10, seed=42), sample_validation(X, Y, N, 10, seed=42)
    assert len(a[1]) == 10 and all(np.array_equal(x, y) for x, y in zip(a, b))  # same sample every run
    assert np.all(np.diff(a[1]) > 0)                                            # original order kept
    np.testing.assert_array_equal(a[0], X[a[1]])                                 # rows stay together
    assert sample_validation(X, Y, N, 0, seed=42)[1] is Y                        # 0 = all
