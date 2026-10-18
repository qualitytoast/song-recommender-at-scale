import calendar
import json

import numpy as np

from recsys.lb_extract import extract
from recsys.lb_sessions import rank_from_end
from recsys.lb_store import (HELD_OUT, RANKER, RETRIEVER, VALIDATION, build_store, capped_catalog, choose_cap,
                             median_per_song, mode_per_song, noise, split_users)
from recsys.mpd_store import load_store
from tests.test_listenbrainz import dump_bytes, listen_line

JAN = calendar.timegm((2026, 1, 1, 0, 0, 0))
FEB = calendar.timegm((2026, 2, 1, 0, 0, 0))


def test_noise_drops_impossible_days_duplicates_and_skips():
    users = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    seconds = np.array([0, 200, 200, 210, 600,                     # user 0: 5 listens on day 0
                        *(86_400 + np.arange(6) * 1000), 2 * 86_400])  # user 1: 6 on day 1, 1 on day 2
    drop = noise(users, seconds, max_day=5, skip=30)
    # user 0: the second listen at 200 is a duplicate (dropped); the first is followed by it 0 s later, which
    # isn't a skip; 210 comes 390 s before the next, so it played. User 1: day 1 is over 5 listens.
    assert list(drop) == [False, False, True, False, False, True, True, True, True, True, True, False]


def test_split_users_is_random_and_has_the_planned_shares():
    split = split_users(1000, seed=1)
    counts = np.bincount(split, minlength=4)
    assert list(counts) == [640, 160, 100, 100]
    assert not (split_users(1000, seed=2) == split).all()
    assert (split_users(1000, seed=1) == split).all()


def test_mode_and_median_per_song():
    songs = np.array([0, 0, 0, 2, 2, 2, 2])
    values = np.array([7, 5, 7, 9, 1, 1, 9])
    assert list(mode_per_song(songs, values, 4)) == [7, -1, 1, -1]      # ties: the smaller value
    assert list(median_per_song(songs, np.array([10, 30, 20, 4, 1, 3, 2]), 4)) == [20, 0, 2, 0]


def test_choose_cap_finds_the_smallest_cap_reaching_the_target():
    users = np.repeat(np.arange(3), 8)
    seconds = np.tile(np.arange(8) * 200, 3)            # one session per user
    songs = np.array([0, 1, 2, 3, 4, 5, 6, 7,           # user 0
                      0, 1, 2, 3, 4, 5, 6, 6,           # user 1 (the last is a repeat)
                      8, 1, 2, 3, 4, 5, 6, 9])          # user 2 (8 and 9: nobody else)
    training = np.ones(len(users), dtype=bool)
    rank = rank_from_end(users)
    quiet = lambda message: None
    kept_at = lambda k: len(capped_catalog(users, seconds, songs, training, rank, 10, k, 2)[0])
    for target in (8, 15, 18, kept_at(8)):
        cap, kept, catalog, _ = choose_cap(users, seconds, songs, training, 10, 2, target, report=quiet)
        assert cap == min(k for k in range(1, 9) if kept_at(k) >= target) and len(kept) == kept_at(cap)
    cap, kept, catalog, _ = choose_cap(users, seconds, songs, training, 10, 2, 8, report=quiet)
    assert not catalog[8] and not catalog[9]            # one user each
    none, everything, catalog, _ = choose_cap(users, seconds, songs, training, 10, 2, 10**6, report=quiet)
    assert none is None and len(everything) == kept_at(8) and catalog[:7].all() and not catalog[7:].any()


def test_dropping_a_repeat_never_splits_its_session():
    # one user: X at 0 s, Y at 1,000, Y again at 2,500 (a repeat), Z at 3,000, W at 3,500. Without the repeat,
    # Y -> Z is a 2,000 s gap (over 30 min), but the user was listening the whole time: one session of 4.
    users = np.zeros(5, dtype=np.int32)
    seconds = np.array([0, 1000, 2500, 3000, 3500])
    songs = np.array([0, 1, 1, 2, 3])
    _, kept, _, session = choose_cap(users, seconds, songs, np.ones(5, dtype=bool), 4, 1, 4, report=lambda m: None)
    assert list(songs[kept]) == [0, 1, 2, 3] and len(set(session)) == 1


def test_build_store_end_to_end(tmp_path):
    lines = []
    for user in range(20):
        t = JAN + user * 10_000
        for i in range(12):                              # one session of 12 songs, 200 s apart
            song = i % 6                                 # 6 popular songs ...
            title = f"Hit {song}" if i < 10 else f"Rare {user}-{i}"   # ... and 2 rare songs per user
            info = {"duration_ms": 180_000 + song}
            if song == 0:
                info["spotify_id"] = "https://open.spotify.com/track/1rrgWMXGCGHru5bIRxGFV0"
            if song == 1:                                # the same Spotify link: only one song may keep it
                info["spotify_id"] = "https://open.spotify.com/track/1rrgWMXGCGHru5bIRxGFV0"
            info["artist_mbids"] = ["7fa7fc04-1011-4876-8095-ecd232edea87"]
            lines.append(listen_line(user=user, ts=t, artist="Band", title=title,
                                     release="Album A" if user % 3 else "Album B", info=info))
            t += 200
    source = tmp_path / "dump.tar.zst"
    source.write_bytes(dump_bytes({"d/listens/2026/1.listens": b"\n".join(lines) + b"\n"}))
    extract(str(source), tmp_path / "extract", JAN, FEB, workers=1)
    mb = tmp_path / "mb.jsonl"
    mb.write_text(json.dumps({"mbid": "7fa7fc04-1011-4876-8095-ecd232edea87", "name": "Band", "spotify_ids": [],
                              "genres": [{"name": "rock", "count": 3}]}) + "\n")

    meta = build_store(tmp_path / "extract", tmp_path / "store", target=150, min_users=3, seed=0, mb_genres=mb)
    store = load_store(tmp_path / "store")
    split = np.load(tmp_path / "store" / "playlist_split.npy")
    users = np.load(tmp_path / "store" / "playlist_users.npy")
    lengths = np.diff(store.playlist_offsets)
    assert meta["tracks"] == 6 and sorted(store.track_names) == [f"Hit {i}" for i in range(6)]  # rare songs out
    assert (lengths >= 4).all() and lengths.sum() == meta["track_entries"] >= 150
    for p in range(len(lengths)):                        # no back-to-back repeats inside a session
        s = store.playlist(p)
        assert (s[1:] != s[:-1]).all()
    for u in np.unique(users):                           # a user's sessions all on one side
        assert len(np.unique(split[users == u])) == 1
    assert set(np.unique(split)) <= {RETRIEVER, RANKER, VALIDATION, HELD_OUT}
    uris = store.track_uris
    assert len(set(uris)) == 6 and sum(u.startswith("spotify:track:") for u in uris) == 1
    assert sorted(set(np.asarray(store.track_duration_ms))) == [180_000 + i for i in range(6)]
    assert store.album_names[store.track_album[0]] == "Album A"   # the most common album
    genres = [json.loads(line) for line in (tmp_path / "store" / "genres.jsonl").read_text().splitlines()]
    assert genres == [{"artist_uri": "listenbrainz:artist:band", "artist_name": "Band",
                       "mbids": ["7fa7fc04-1011-4876-8095-ecd232edea87"], "genres": [{"name": "rock", "count": 3}]}]
