import numpy as np
import pytest

from recsys.lb_sessions import (by_user_and_time, most_recent, rank_from_end, session_starts, sessions,
                                 users_per_song)


def test_by_user_and_time_sorts_and_compacts_users():
    user = np.array([900, 7, 900, 7, 42], dtype=np.int32)
    timestamp = np.array([1005, 1003, 1001, 1009, 1000], dtype=np.int64)
    song = np.array([10, 11, 12, 13, 14], dtype=np.int32)
    users, seconds, songs, user_ids = by_user_and_time(user, timestamp, song, start=1000)
    assert list(user_ids) == [7, 42, 900]
    assert list(users) == [0, 0, 1, 2, 2]
    assert list(seconds) == [3, 9, 0, 1, 5]
    assert list(songs) == [11, 13, 14, 12, 10]
    assert users.dtype == seconds.dtype == songs.dtype == np.int32


def test_by_user_and_time_rejects_times_outside_the_window():
    with pytest.raises(ValueError):
        by_user_and_time(np.array([1]), np.array([5]), np.array([0]), start=10)


def test_sessions_split_on_gaps_and_users_drop_repeats_and_short_ones():
    #             user 0: session A (5 listens, one repeat) | gap > 30 min | B (3)  || user 1: C (4)
    users = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    seconds = np.array([0, 60, 120, 180, 240, 5000, 5060, 5120, 5130, 5190, 5250, 5310])
    songs = np.array([1, 2, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4])
    assert list(np.flatnonzero(session_starts(users, seconds))) == [0, 5, 8]
    kept, session = sessions(users, seconds, songs)
    assert list(session) == [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    assert list(kept) == [True, True, False, True, True, False, False, False, True, True, True, True]


def test_a_session_too_short_after_dropping_repeats_is_dropped():
    users = np.zeros(5, dtype=int)
    seconds = np.arange(5) * 60
    songs = np.array([1, 1, 2, 2, 3])   # 3 different songs in a row: below 4
    kept, _ = sessions(users, seconds, songs)
    assert not kept.any()


def test_most_recent_keeps_each_users_last_k():
    users = np.array([0, 0, 0, 0, 1, 1, 2])
    assert list(most_recent(users, 2)) == [False, False, True, True, True, True, True]
    assert most_recent(users, 10).all()


def test_users_per_song_counts_distinct_users():
    users = np.array([0, 0, 1, 2, 2, 2])
    songs = np.array([5, 5, 5, 1, 1, 0])
    assert list(users_per_song(users, songs, 7)) == [1, 1, 0, 0, 0, 2, 0]
    mask = np.array([True, True, False, True, True, True])
    assert list(users_per_song(users, songs, 7, mask)) == [1, 1, 0, 0, 0, 1, 0]


def test_rank_from_end_counts_back_from_each_users_last_listen():
    assert list(rank_from_end(np.array([0, 0, 0, 1, 2, 2]))) == [2, 1, 0, 0, 1, 0]
