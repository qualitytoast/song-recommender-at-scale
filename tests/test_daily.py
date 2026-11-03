import calendar
import hashlib
import io
import json

import numpy as np
import pytest

from recsys.daily import (AB, DAY, NewSong, Seen, Served, Update, apply_dump, catalog_sessions, chain,
                          fresh_extract, involving_new, standins, with_pairs)
from recsys.listenbrainz import dump_window, song_key
from recsys.rank_features import pair_counts
from tests.test_listenbrainz import dump_bytes, listen_line

END = calendar.timegm((2026, 11, 2, 0, 0, 0))
SPOTIFY = [f"{i:022d}" for i in range(6)]  # the tiny bundle's Spotify track IDs
B11_SPOTIFY = "B" * 22


def test_dump_window_reads_the_timestamp_files_before_the_listens():
    data = dump_bytes({"d/SCHEMA_SEQUENCE": b"1", "d/START_TIMESTAMP": b"2026-10-18 00:00:03.293870+00:00\n",
                       "d/END_TIMESTAMP": b"2026-10-19 00:00:03.024117+00:00\n",
                       "d/listens/2026/10.listens": listen_line() + b"\n"})
    start, end = dump_window(io.BytesIO(data))
    assert start == calendar.timegm((2026, 10, 18, 0, 0, 3)) + 0.29387
    assert end == calendar.timegm((2026, 10, 19, 0, 0, 3)) + 0.024117
    with pytest.raises(ValueError):
        dump_window(io.BytesIO(dump_bytes({"d/listens/2026/10.listens": listen_line() + b"\n"})))


def test_chain_skips_a_second_dump_of_a_window_and_reports_gaps():
    windows = {"d1": (0, 100), "d1-again": (0, 101), "d1-tail": (100, 100.3), "d2": (100.3, 200), "d4": (300, 400)}
    assert chain(windows) == (["d1", "d1-tail", "d2", "d4"], ["d1-again"], [(200, 300)])
    assert chain(windows, after=200) == (["d4"], ["d1", "d1-again", "d1-tail", "d2"], [(200, 300)])
    assert chain(windows, after=400) == ([], ["d1", "d1-again", "d1-tail", "d2", "d4"], [])


def test_chain_follows_a_tail_dump_that_ends_before_it_starts():
    # dumps 2752-2754 (2026-10-29/30): the tail dump's end is 0.16 s before its start, and the next day starts there
    windows = {"2752": (1793232003.372690, 1793318403.434693), "2753": (1793318403.434693, 1793318403.272376),
               "2754": (1793318403.272376, 1793404803.115939)}
    assert chain(windows) == (["2752", "2754"], ["2753"], [])


def test_catalog_sessions_apply_the_session_rules_again():
    users, seconds = np.zeros(6, dtype=np.int64), np.arange(6) * 120
    # dropping the unknown songs leaves 3 songs: too short a session
    assert len(catalog_sessions(users[:5], seconds[:5], np.array([0, -1, 1, 2, -1]))[0]) == 0
    # it leaves 1 next to 1: a back-to-back repeat, dropped
    songs, offsets = catalog_sessions(users, seconds, np.array([0, 1, -1, 1, 2, 3]))
    assert list(songs) == [0, 1, 2, 3] and list(offsets) == [0, 4]


def test_with_pairs_equals_counting_everything_at_once_after_the_catalog_grows():
    rng = np.random.default_rng(0)
    old, grown = 30, 34
    a_songs, a_offsets = rng.integers(0, old, 400), np.arange(0, 401, 8)
    b_songs, b_offsets = rng.integers(0, grown, 300), np.arange(0, 301, 6)
    keys, counts = pair_counts(a_songs, a_offsets, old)
    more, more_counts = pair_counts(b_songs, b_offsets, AB)
    got = with_pairs(keys, counts, old, more, more_counts, grown)
    want = pair_counts(np.concatenate([a_songs, b_songs]), np.concatenate([a_offsets, b_offsets[1:] + 400]), grown)
    assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])
    same = with_pairs(keys, counts, old, *pair_counts(a_songs[:16], a_offsets[:3], AB), old)  # no growth
    assert same[1].sum() == counts.sum() + pair_counts(a_songs[:16], a_offsets[:3], old)[1].sum()


def test_seen_counts_listeners_within_the_window_only():
    seen = Seen()
    seen.add(END - 40 * DAY, np.array([7, 7]), np.array([1, 2]))   # too long ago
    seen.add(END - DAY, np.array([7, 7, 8]), np.array([2, 3, 3]))
    seen.add(END, np.array([7, 9]), np.array([3, 4]))
    assert list(seen.listeners(np.array([7, 8, 9, 10]), END)) == [2, 1, 1, 0]


def write_tiny_bundle(folder):
    """The per-song files an update reads (recsys/export.py's names): 6 songs by 3 artists."""
    folder.mkdir()
    (folder / "songs.json").write_text(json.dumps({
        "uris": [f"spotify:track:{s}" for s in SPOTIFY],
        "tracks": ["Song 1", "Song 2", "Song 3", "Song 4", "Song 5", "Song 6"],
        "artist_names": ["Artist A", "Artist B", "Artist C"]}))
    np.save(folder / "song_artist.npy", np.array([0, 0, 0, 1, 1, 2], dtype=np.int32))
    np.save(folder / "song_album.npy", np.array([0, 0, 1, 2, 2, 3], dtype=np.int32))
    np.save(folder / "song_genres.npy", np.array([[1, 2]] * 3 + [[3, 0]] * 2 + [[0, 0]], dtype=np.int16))
    np.save(folder / "song_playlists.npy", np.array([10, 5, 1, 7, 3, 2], dtype=np.int32))
    (folder / "song_albums.json").write_text(json.dumps(["Album X", "Album X", "Album Y", "B Side", "B Side", "C"]))
    return Served(folder, [song_key(a, t) for a, t in [("Artist A", "Song 1"), ("Artist A", "Song 2"),
                                                        ("Artist A", "Song 3"), ("Artist B", "Song 4"),
                                                        ("Artist B", "Song 5"), ("Artist C", "Song 6")]])


A = lambda n, release="Album X", **info: dict(artist="Artist A", title=f"Song {n}", release=release, info=info)
B = lambda n, release="B Side", **info: dict(artist="Artist B", title=f"Song {n}", release=release, info=info)
ALIAS = dict(artist="Artist C", title="Song Six (Live)", release="",  # song 6's Spotify track
             info={"spotify_id": f"https://open.spotify.com/track/{SPOTIFY[5]}"})
Z1 = dict(artist="Artist Z", title="Zed", release="", info={})        # an artist the catalog doesn't have
B11 = B(11, release="New Album", spotify_id=f"https://open.spotify.com/track/{B11_SPOTIFY}")
SESSIONS = {1: [A(1), A(2), A(9), B(4), B(5), A(10), ALIAS, Z1],
            2: [A(1), A(2), A(9), B(4), B(5), A(10), ALIAS, Z1, B11],
            3: [A(1), A(2), A(9), B(4), B(5), ALIAS, Z1, B11],
            4: [A(1), A(2), A(3), B11],
            5: [B11, A(1), A(2), A(3)]}


@pytest.fixture
def day(tmp_path):
    """A tiny bundle and one dump of five listeners' sessions, plus an old (imported) listen."""
    served = write_tiny_bundle(tmp_path / "bundle")
    listens = [listen_line(user=u, ts=END - DAY + u * 3600 + i * 120, **s) for u, songs in SESSIONS.items()
               for i, s in enumerate(songs)]
    listens.append(listen_line(user=6, ts=END - 40 * DAY, **A(9)))
    data = dump_bytes({"d/START_TIMESTAMP": b"2026-11-01 00:00:03+00:00", "d/END_TIMESTAMP": b"2026-11-02 00:00:00+00:00",
                       "d/listens/2026/11.listens": b"\n".join(listens) + b"\n"})
    dump = tmp_path / "dump.tar.zst"
    dump.write_bytes(data)
    (tmp_path / "dump.tar.zst.sha256").write_text(hashlib.sha256(data).hexdigest())
    ext, window = fresh_extract(dump, tmp_path / "extract", workers=1)
    return served, ext, window


def test_new_songs_join_by_the_rules(day):
    served, ext, window = day
    assert ext.meta["listens"] == sum(map(len, SESSIONS.values()))  # the old listen isn't fresh
    report = apply_dump(served, Update(), Seen(), ext, window, "d")
    # B11 (4 listeners) and A9 (3) join, most listeners first; A10 has 2 listeners, Zed an unknown artist, and
    # "Song Six (Live)" is song 6's Spotify track under another title
    assert [s.key for s in served.added] == ["artist b|song 11", "artist a|song 9"] and report["new_songs"] == 2
    b11, a9 = served.added
    assert (b11.uri, b11.artist, b11.album_id, b11.listeners) == (f"spotify:track:{B11_SPOTIFY}", 1, 4, 4)
    assert (a9.uri, a9.artist, a9.album_id, a9.listeners) == ("listenbrainz:track:artist a|song 9", 0, 0, 3)
    assert served.id_of["artist b|song 11"] == 6 and served.uris[7] == a9.uri and len(served) == 8
    assert served.song_genres()[6:].tolist() == [[3, 0], [1, 2]]   # their artists' genres
    capped = write_tiny_bundle(ext.folder.parent / "bundle2")
    apply_dump(capped, Update(), Seen(), ext, window, "d", max_new=1)
    assert [s.key for s in capped.added] == ["artist b|song 11"]


def test_listens_and_pairs_come_from_the_catalog_songs_sessions(day):
    served, ext, window = day
    update = Update()
    apply_dump(served, update, Seen(), ext, window, "d")
    # sessions as the catalog sees them (B11 = 6, A9 = 7): u1 [0 1 7 3 4], u2 and u3 [0 1 7 3 4 6], u4 [0 1 2 6],
    # u5 [6 0 1 2]
    assert list(update.listens) == [5, 5, 2, 3, 3, 0, 4, 3]
    count = dict(zip(update.pairs.tolist(), update.counts.tolist()))
    assert count[1 * AB + 7] == 3 and count[7 * AB + 3] == 3 and count[0 * AB + 6] == 3 and count[6 * AB + 0] == 1
    assert sum(count.values()) == 10 + 15 * 2 + 6 + 6
    assert involving_new(update.pairs, served.base).sum() == sum(1 for k in count if k // AB >= 6 or k % AB >= 6)


def test_standins_average_lengthen_and_fall_back_to_neighbours(tmp_path):
    served = write_tiny_bundle(tmp_path / "bundle")
    served.add("artist a|song 9", "Song 9", "", "", 3, END)    # artist A: 3 songs
    served.add("artist c|song 7", "Song 7", "", "", 3, END)    # artist C: 1 song
    rng = np.random.default_rng(0)
    search, inputs = rng.normal(size=(6, 4)).astype(np.float16), rng.normal(size=(6, 3)).astype(np.float16)
    tables = {"search_vectors": search, "inputs": inputs}
    update = Update()
    update.pairs = np.array([3 * AB + 6, 6 * AB + 4, 7 * AB + 1], dtype=np.int64)  # song 9: after 4, before 5
    update.counts = np.array([1, 3, 2], dtype=np.int64)
    s32, i32 = search.astype(np.float32), inputs.astype(np.float32)
    plain = standins(served, update, tables, "artist")
    assert np.allclose(plain["search_vectors"][0], s32[:3].mean(0), atol=1e-6)
    assert np.allclose(plain["inputs"][1], i32[5], atol=1e-6)       # artist C's only song
    longer = standins(served, update, tables, "artist_norm")["inputs"][0]
    assert np.isclose(np.linalg.norm(longer), np.linalg.norm(i32[:3], axis=1).mean(), atol=1e-5)
    assert np.allclose(longer / np.linalg.norm(longer), plain["inputs"][0] / np.linalg.norm(plain["inputs"][0]), atol=1e-5)
    got = standins(served, update, tables)
    # song 9: half artist A's songs by listens (11, 6, 2 with one added), half its neighbours 3 and 4 (1 and 3 times)
    weights = np.zeros(6)
    weights[:3] = 0.5 * np.array([11, 6, 2]) / 19
    weights[3], weights[4] = 0.5 * 1 / 4, 0.5 * 3 / 4
    v = weights @ s32[:, :3]
    assert np.allclose(got["search_vectors"][0, :3], v * (weights @ np.linalg.norm(s32[:, :3], axis=1)) / np.linalg.norm(v),
                       atol=1e-5)
    assert np.isclose(got["search_vectors"][0, 3], weights @ s32[:, 3], atol=1e-6)  # the bias: averaged only
    # song 7: artist C has one song, so its neighbour (song 1) alone
    assert np.allclose(got["inputs"][1], i32[1], atol=1e-5)
    assert isinstance(served.added[0], NewSong)
