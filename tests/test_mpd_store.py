import dataclasses
import json

import numpy as np

from recsys.config import Config, DataConfig, ModelConfig
from recsys.data import (build_dataset, chunks_from_arrays, first_appearance_ids, load_playlists,
                         make_chunks, make_windows, windows_from_arrays)
from recsys.mpd_store import build_store, is_store, load_store, playlists_from_store


def track(title, uri=None, artist="X", album="Y", ms=200_000):
    return {"pos": 0, "track_name": title, "track_uri": uri or f"spotify:track:{title}",
            "artist_name": artist, "artist_uri": f"artist:{artist}", "album_name": album,
            "album_uri": f"album:{album}", "duration_ms": ms}


def write_slice(folder, name, playlists, first_pid):
    body = {"playlists": [{"name": n, "pid": first_pid + i, "tracks": ts} for i, (n, ts) in enumerate(playlists)]}
    (folder / name).write_text(json.dumps(body))


def make_raw(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    # Alphabetical order puts "10-11" before "2-3": that's the order playlists are stored in.
    write_slice(raw, "mpd.slice.2-3.json",
                [("third", [track("c"), track("a"), track("d", artist="Z", album="W")])], 2)
    write_slice(raw, "mpd.slice.10-11.json",
                [("first", [track("a"), track("b"), track("a"), track("b")]),
                 ("second", [track("Home", uri="uri:h1"), track("Home", uri="uri:h2", artist="Q"),
                             track("b"), track("c")])], 10)
    return raw


def test_store_layout(tmp_path):
    meta = build_store(make_raw(tmp_path), tmp_path / "store")
    s = load_store(tmp_path / "store")
    assert is_store(tmp_path / "store") and not (tmp_path / "store.tmp").exists()
    assert s.playlist_names == ["first", "second", "third"]
    np.testing.assert_array_equal(s.playlist_pids, [10, 11, 2])
    # tracks numbered by first appearance: a=0, b=1, h1=2, h2=3, c=4, d=5
    assert s.track_uris == ["spotify:track:a", "spotify:track:b", "uri:h1", "uri:h2", "spotify:track:c", "spotify:track:d"]
    np.testing.assert_array_equal(s.playlist_offsets, [0, 4, 8, 11])
    np.testing.assert_array_equal(s.playlist(0), [0, 1, 0, 1])
    np.testing.assert_array_equal(s.playlist(2), [4, 0, 5])
    assert [s.artist_uris[a] for a in s.track_artist] == ["artist:X", "artist:X", "artist:X", "artist:Q", "artist:X", "artist:Z"]
    assert s.album_names[s.track_album[5]] == "W"
    assert meta["playlists"] == 3 and meta["track_entries"] == 11 and meta["tracks"] == 6
    assert meta["tracks_with_conflicting_metadata"] == 0


def test_conflicting_metadata_is_counted(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_slice(raw, "mpd.slice.0-1.json", [("p", [track("a", ms=1000), track("a", ms=2000), track("b")])], 0)
    assert build_store(raw, tmp_path / "store")["tracks_with_conflicting_metadata"] == 1


def test_store_gives_exactly_what_the_json_loader_gives(tmp_path):
    raw = make_raw(tmp_path)
    build_store(raw, tmp_path / "store")
    s = load_store(tmp_path / "store")
    fields = ["track_uri", "track_name", "artist_uri", "artist_name", "album_uri", "duration_ms"]
    for song_key in ("track_uri", "track_name"):
        for max_playlists, min_len in [(10, 4), (10, 3), (2, 3)]:  # 2 = a whole slice, as with 5,000
            from_json = load_playlists(raw, max_playlists, min_len, song_key)
            from_store = playlists_from_store(s, max_playlists, min_len, song_key)
            assert from_store[0] == from_json[0] and from_store[2] == from_json[2]
            assert {k: {f: v[f] for f in fields} for k, v in from_store[1].items()} == \
                   {k: {f: v[f] for f in fields} for k, v in from_json[1].items()}


# --- datasets built from the store's arrays equal those built from the JSON ---

def random_mpd(tmp_path, n_playlists=120, seed=0):
    rng = np.random.RandomState(seed)
    raw = tmp_path / "raw"
    raw.mkdir()
    songs = [(f"s{i}", f"a{i % 9}", f"al{i % 13}", int(rng.randint(60, 400)) * 1000) for i in range(60)]
    for f in range(3):
        playlists = []
        for _ in range(n_playlists // 3):
            n = rng.randint(2, 25)
            ids = rng.zipf(1.6, n) % 60  # some songs common, many rare
            playlists.append((f"{rng.choice(['chill', 'gym', 'rap', 'Party mix', 'x'])} {rng.randint(3)}",
                              [track(songs[i][0], artist=songs[i][1], album=songs[i][2], ms=songs[i][3]) for i in ids]))
        write_slice(raw, f"mpd.slice.{f}.json", playlists, f * 1000)
    with open(tmp_path / "genres.jsonl", "w") as g:
        for a in range(9):
            names = ["rock"] * (a % 2) + ["pop"] * (a % 3 > 0) + [f"rare{a}"]
            g.write(json.dumps({"artist_uri": f"artist:a{a}", "artist_name": f"a{a}", "mbids": ["m"],
                                "genres": [{"name": n, "count": 1} for n in names]}) + "\n")
    return raw


def config(folder, genres, max_playlists=1000, validation="separate_playlists", val_split=0.1, val_size=0):
    data = DataConfig(folder=str(folder), max_playlists=max_playlists, min_playlist_len=4, min_freq=2,
                      song_key="track_uri", vocab_from="train", context_length=3, test_split=0.1,
                      validation=validation, val_size=val_size, val_split=val_split, genres_file=str(genres))
    model = ModelConfig(embed_dim=4, num_layers=1, dropout=0.0, scale_attention=True, init="pytorch",
                        features=["artist", "album", "duration", "playlist_name", "genre"])
    return Config(data_seed=42, train_seeds=[1], data=data, model=model, train=None)


def assert_same_dataset(a, b):
    for field in dataclasses.fields(a):
        x, y = getattr(a, field.name), getattr(b, field.name)
        if isinstance(x, dict):
            assert x.keys() == y.keys()
            for k in x:
                np.testing.assert_array_equal(x[k][0], y[k][0], err_msg=f"{field.name}[{k}]")
                assert x[k][1] == y[k][1], f"{field.name}[{k}] count"
        elif isinstance(x, np.ndarray):
            assert x.dtype == y.dtype, field.name
            np.testing.assert_array_equal(x, y, err_msg=field.name)
        else:
            assert x == y, field.name


def test_store_dataset_equals_json_dataset(tmp_path):
    raw = random_mpd(tmp_path)
    build_store(raw, tmp_path / "store")
    genres = tmp_path / "genres.jsonl"
    for kwargs in [{}, {"max_playlists": 40}, {"validation": "held_out_prefix", "val_split": 0.0, "val_size": 7}]:
        from_json = build_dataset(config(raw, genres, **kwargs))
        from_store = build_dataset(config(tmp_path / "store", genres, **kwargs))
        assert len(from_json.Y_train) > 50 and len(from_json.Yc_train) > 20  # a real test, not empty data
        assert_same_dataset(from_json, from_store)


def test_window_and_chunk_arrays_equal_the_python_versions():
    rng = np.random.RandomState(1)
    playlists = [list(rng.randint(-1, 6, rng.randint(1, 30))) for _ in range(40)]  # -1 = unknown song
    songs = np.concatenate([np.array(p, dtype=np.int64) for p in playlists])
    offsets = np.concatenate([[0], np.cumsum([len(p) for p in playlists])])
    ids = {i: i for i in range(6)}  # make_windows/make_chunks look songs up; -1 isn't in the vocab
    for got, expected in [(windows_from_arrays(songs, offsets, 4), make_windows(playlists, ids, 4)),
                          (chunks_from_arrays(songs, offsets, 4), make_chunks(playlists, ids, 4))]:
        for g, e in zip(got, expected):
            np.testing.assert_array_equal(g, e)


def test_first_appearance_ids_match_dict_fromkeys():
    values = np.array([5, 2, 5, 9, 2, 7, 9])
    unique, ids = first_appearance_ids(values)
    assert unique.tolist() == list(dict.fromkeys(values.tolist())) == [5, 2, 9, 7]
    assert ids.tolist() == [0, 1, 0, 2, 1, 3, 2]
