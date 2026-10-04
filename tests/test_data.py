import json

import numpy as np

from recsys.data import build_vocab, load_playlists, make_windows, split_playlists


def write_slice(folder, name, playlists):
    body = {"playlists": [{"tracks": [{"track_name": t} for t in p]} for p in playlists]}
    (folder / name).write_text(json.dumps(body))


# --- load_playlists ---

def test_files_read_in_alphabetical_order(tmp_path):
    # Alphabetically "10-11" sorts before "2-3", even though 2 < 10 numerically.
    write_slice(tmp_path, "mpd.slice.2-3.json", [["two"] * 4])
    write_slice(tmp_path, "mpd.slice.10-11.json", [["ten"] * 4])
    write_slice(tmp_path, "mpd.slice.0-1.json", [["zero"] * 4])
    assert [p[0] for p in load_playlists(tmp_path, 10, 4)] == ["zero", "ten", "two"]


def test_short_playlists_skipped_and_non_json_ignored(tmp_path):
    write_slice(tmp_path, "mpd.slice.0-1.json", [["a", "b", "c"], ["a", "b", "c", "d"]])
    (tmp_path / "README.md").write_text("not a slice")
    assert load_playlists(tmp_path, 10, 4) == [["a", "b", "c", "d"]]


def test_stops_at_max_playlists(tmp_path):
    write_slice(tmp_path, "mpd.slice.0-1.json", [["a"] * 4, ["b"] * 4, ["c"] * 4])
    write_slice(tmp_path, "mpd.slice.1-2.json", [["d"] * 4])
    assert [p[0] for p in load_playlists(tmp_path, 2, 4)] == ["a", "b"]


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

def test_split_sizes_and_no_overlap():
    playlists = [[str(i)] for i in range(10)]
    train, held_out = split_playlists(playlists, test_split=0.1, seed=42)
    assert len(train) == 9 and len(held_out) == 1
    assert sorted(train + held_out) == sorted(playlists)


def test_split_matches_v1_global_seed_shuffle():
    # v1: np.random.seed(seed); np.random.shuffle(playlists); first 90% train.
    playlists = [[str(i)] for i in range(50)]
    expected = list(playlists)
    np.random.seed(42)
    np.random.shuffle(expected)
    train, held_out = split_playlists(playlists, test_split=0.1, seed=42)
    assert train == expected[:45] and held_out == expected[45:]


def test_split_does_not_modify_input():
    playlists = [[str(i)] for i in range(10)]
    original = list(playlists)
    split_playlists(playlists, test_split=0.1, seed=42)
    assert playlists == original
