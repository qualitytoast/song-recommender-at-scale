import json

import numpy as np

from recsys.data import load_playlists
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
