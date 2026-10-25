"""Extracted ListenBrainz listens (recsys.lb_extract) as sessions, and the counts that decide the catalog.

Every function works on listens sorted by user, then time (by_user_and_time), as plain numpy arrays, so
hundreds of millions of listens need no Python loop:
  - a session ends after SESSION_GAP seconds without a listen (or at the user's last listen);
  - a song right after itself in the same session is a repeat and is dropped;
  - a session needs MIN_SESSION songs (after dropping repeats) to be kept.
"""
import numpy as np

SESSION_GAP = 30 * 60   # seconds without a listen that end a session
MIN_SESSION = 4         # songs a session needs to be kept (as a playlist needs 4)
TIME_BITS = 26          # seconds since the window start fit in 26 bits: windows up to ~776 days


def by_user_and_time(user, timestamp, song, start):
    """The listens sorted by user, then time, as (users, seconds, songs, user_ids):
    users: int32, 0..U-1 (compact: the i-th distinct user), seconds: int32 since `start`, songs: int32,
    user_ids: the original user ID of each compact user. Sorts one int64 key (user << 26 | seconds) and
    the song column alongside, which keeps memory to a few copies of one column."""
    key = np.array(user, dtype=np.int64)            # copies (and computes) in place, one column at a time
    key <<= TIME_BITS
    seconds = np.array(timestamp, dtype=np.int64)
    seconds -= start
    if len(seconds) and (seconds.min() < 0 or seconds.max() >= 1 << TIME_BITS):
        raise ValueError("timestamps must be in [start, start + 2^26 s)")
    key |= seconds
    del seconds
    order = np.argsort(key)
    key = key[order]
    songs = np.asarray(song)[order].astype(np.int32, copy=False)
    del order
    original = (key >> TIME_BITS).astype(np.int32)
    first = user_starts_from(original)
    user_ids = original[first]
    users = np.cumsum(first, dtype=np.int32)
    users -= 1
    del original, first
    key &= (1 << TIME_BITS) - 1
    seconds = key.astype(np.int32)
    return users, seconds, songs, user_ids


def user_starts_from(users):
    """True at each user's first listen (listens sorted by user)."""
    first = np.ones(len(users), dtype=bool)
    first[1:] = users[1:] != users[:-1]
    return first


def session_starts(users, seconds, gap=SESSION_GAP):
    """True at the first listen of each session."""
    start = user_starts_from(users)
    start[1:] |= (seconds[1:] - seconds[:-1]) > gap
    return start


def sessions(users, seconds, songs, gap=SESSION_GAP, min_len=MIN_SESSION):
    """The session rules: (kept, session) where kept marks the listens that stay (not a back-to-back
    repeat, in a session of at least min_len songs after repeats are dropped) and session numbers each
    listen's session (0, 1, ... in order)."""
    start = session_starts(users, seconds, gap)
    session = np.cumsum(start, dtype=np.int32)
    session -= 1
    repeat = np.zeros(len(songs), dtype=bool)
    repeat[1:] = ~start[1:] & (songs[1:] == songs[:-1])
    lengths = np.bincount(session[~repeat], minlength=session[-1] + 1 if len(session) else 0)
    kept = ~repeat & (lengths >= min_len)[session]  # per session first: one byte per listen, not eight
    return kept, session


def most_recent(users, k):
    """True for each user's k most recent listens (listens sorted by user, then time)."""
    first = np.flatnonzero(user_starts_from(users))
    ends = np.append(first[1:], len(users))       # one past each user's last listen
    end_of = np.repeat(ends, np.diff(np.append(first, len(users))))
    return (end_of - np.arange(len(users))) <= k


def rank_from_end(users):
    """0 for each user's most recent listen, 1 for the one before, ... (listens sorted by user, then time).
    Computed in int32 (fewer than 2^31 listens), in place."""
    first = np.flatnonzero(user_starts_from(users))
    ends = np.append(first[1:], len(users)).astype(np.int32)
    rank = np.repeat(ends, np.diff(np.append(first, len(users))))   # one past the user's last listen
    rank -= np.arange(len(users), dtype=np.int32)
    rank -= 1
    return rank


def users_per_song(users, songs, n_songs, mask=None):
    """How many different users listened to each song (only the masked listens, if a mask is given)."""
    if mask is not None:
        users, songs = users[mask], songs[mask]
    n_users = int(users.max()) + 1 if len(users) else 1
    pairs = np.unique(songs.astype(np.int64) * n_users + users)
    return np.bincount((pairs // n_users).astype(np.int64), minlength=n_songs).astype(np.int32)
