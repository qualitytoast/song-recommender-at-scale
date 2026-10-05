import json

import numpy as np

from recsys.config import Config, DataConfig
from recsys.data import build_dataset, build_vocab, load_playlists, make_windows, split_playlists


def write_slice(folder, name, playlists):
    """Fake MPD slice. A track is a title (URI made from it) or a (title, uri) pair."""
    def track(t):
        title, uri = t if isinstance(t, tuple) else (t, f"spotify:track:{t}")
        return {"track_name": title, "track_uri": uri}
    body = {"playlists": [{"tracks": [track(t) for t in p]} for p in playlists]}
    (folder / name).write_text(json.dumps(body))


def load(folder, max_playlists=10, song_key="track_name"):
    return load_playlists(folder, max_playlists, min_playlist_len=4, song_key=song_key)


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
    by_uri, names = load(tmp_path, song_key="track_uri")
    assert by_name[0][:2] == ["Home", "Home"]
    assert by_uri[0][:2] == ["uri:1", "uri:2"]
    assert names["uri:1"] == names["uri:2"] == "Home"


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
    X, Y = make_windows([["a", "b", "c", "a"]], ids, context_length=2)
    np.testing.assert_array_equal(X, [[0, 1], [1, 2]])
    np.testing.assert_array_equal(Y, [2, 0])


def test_windows_touching_unknown_song_skipped():
    # "?" is not in the vocab. With context 2:
    #   [b, c] -> ?  skip (target)   [c, ?] -> b  skip (input)
    #   [?, b] -> c  skip (input)    [b, c] -> b  keep
    ids = {"b": 0, "c": 1}
    X, Y = make_windows([["b", "c", "?", "b", "c", "b"]], ids, context_length=2)
    np.testing.assert_array_equal(X, [[0, 1]])
    np.testing.assert_array_equal(Y, [0])


def test_playlist_not_longer_than_context_gives_no_windows():
    X, Y = make_windows([["a", "a"]], {"a": 0}, context_length=2)
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

def tiny_config(folder, vocab_from, validation, val_split, val_size=2):
    data = DataConfig(folder=str(folder), max_playlists=20, min_playlist_len=4, min_freq=2,
                      song_key="track_name", vocab_from=vocab_from, context_length=2,
                      test_split=0.1, validation=validation, val_size=val_size, val_split=val_split)
    return Config(data_seed=42, train_seeds=[1], data=data, model=None, train=None)


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


def test_held_out_prefix_validation_is_first_held_out_windows(tmp_path):
    write_tiny_mpd(tmp_path)
    ds = build_dataset(tiny_config(tmp_path, "all", "held_out_prefix", 0.0, val_size=2))
    np.testing.assert_array_equal(ds.X_val, ds.X_test[:2])
