import numpy as np

from recsys.config import load_config
from recsys.data import (DURATION_BUCKETS, build_dataset, buckets_from_durations, buckets_with_unknown,
                         load_artist_genres)
from recsys.mpd_store import load_store
from tests.test_lb_store import tiny_store


def lb_config(tmp_path):
    """configs/retriever_1m_lb.toml, pointed at the small store and sized for it."""
    text = open("configs/retriever_1m_lb.toml").read()
    text = text.replace('folder = "data/lb_store"', f'folder = "{tmp_path / "store"}"')
    text = text.replace('genres_file = "data/lb_store/genres.jsonl"', f'genres_file = "{tmp_path / "store" / "genres.jsonl"}"')
    for setting in ("val_max_windows", "test_max_windows"):
        text = "\n".join(f"{setting} = 0" if line.startswith(f"{setting} =") else line for line in text.split("\n"))
    text = "\n".join("eval_every_targets = 100" if line.startswith("eval_every_targets =") else line
                     for line in text.split("\n"))
    path = tmp_path / "retriever_1m_lb.toml"
    path.write_text(text)
    return load_config(path)


def test_buckets_with_unknown_keeps_unknown_lengths_apart():
    ms = np.array([0, 100, 200, 300, 400, 0, 500, 600, 700, 800, 900, 1000])
    ids, n = buckets_with_unknown(ms, 4)
    assert n == 5 and list(ids[ms == 0]) == [4, 4]
    known_ids, _ = buckets_from_durations(ms[ms > 0], 4)
    assert list(ids[ms > 0]) == list(known_ids)        # the known lengths bucket as if alone


def test_a_listenbrainz_store_loads_with_its_own_user_split(tmp_path):
    tiny_store(tmp_path)
    ds = build_dataset(lb_config(tmp_path))
    store = load_store(tmp_path / "store")
    side = np.load(tmp_path / "store" / "playlist_split.npy")
    lengths = np.diff(store.playlist_offsets)
    assert len(ds.fit_songs) == lengths[side == 0].sum()       # the retriever's users' sessions only
    assert ds.fit_offsets[-1] == len(ds.fit_songs)
    assert len(ds.O_val) - 1 == (side == 2).sum() and len(ds.O_test) - 1 == (side == 3).sum()
    assert len(ds.O_rank) - 1 == (side == 1).sum()
    assert set(ds.vocab) <= set(store.track_uris) and len(ds.vocab) == 6
    assert ds.song_features["duration"][1] == DURATION_BUCKETS + 1
    # every vocab artist is in the store's genre file (loading fails otherwise); a genre only becomes an input once
    # GENRE_MIN_ARTISTS artists have it, which one artist can't reach
    assert load_artist_genres(tmp_path / "store" / "genres.jsonl")[ds.artist_uris[0]] == ["rock"]
    assert not ds.name_words                                   # sessions have no names
