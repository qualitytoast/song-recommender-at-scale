from types import SimpleNamespace

import numpy as np

from scripts.lb_compare import (as_known, build_cases, build_catalog_cases, case_positions, catalog_scores,
                                ndcg_at_10, store_song_keys)


def test_case_positions_need_a_known_song_before_in_the_same_session():
    offsets = np.array([0, 4, 7])                     # sessions: tracks 0-3 and 4-6
    known = np.array([True, True, False, True, True, True, True])
    t, starts, _ = case_positions(offsets, known, input_length=2)
    assert list(t) == [1, 5, 6]                       # 2 and 3: unknown song at or right before; 4: starts a session
    assert list(starts) == [0, 4, 4]                  # 6's input: 4 and 5 (two songs)


def test_build_cases_maps_each_case_to_both_models():
    lb_uris = ["lb:a", "lb:b", "lb:c", "lb:d", None]        # store track 4: not in the ListenBrainz catalog
    mpd_uri_of = ["mpd:a", None, "mpd:c", "mpd:d", None]    # track 1: the MPD model doesn't know it
    tracks = np.array([0, 1, 2, 3, 0, 4, 2])                # sessions: [a b c d], [a ? c]
    offsets = np.array([0, 4, 7])
    shared, every, counts = build_cases(lb_uris, mpd_uri_of, tracks, offsets, n_cases=10, seed=0)
    assert counts["cases"] == 3 and counts["cases_whose_next_song_both_know"] == 2   # targets b, c, d; b: MPD no
    by_target = {c["lb"]["target"]: c for c in shared}
    assert set(by_target) == {"lb:c", "lb:d"}
    assert by_target["lb:d"]["lb"]["songs"] == ["lb:a", "lb:b", "lb:c"]
    assert by_target["lb:d"]["mpd"] == {"songs": ["mpd:a", "mpd:c"], "target": "mpd:d", "later": ["mpd:d"]}
    assert by_target["lb:c"]["lb"]["later"] == ["lb:c", "lb:d"]
    assert len(every) == 3


def test_ndcg_at_10():
    assert ndcg_at_10(["x", "y"], "x") == 1.0
    assert abs(ndcg_at_10(["x", "y"], "y") - 1 / np.log2(3)) < 1e-12
    assert ndcg_at_10(["x"], "z") == 0.0


def test_store_song_keys_are_artist_key_and_cleaned_title():
    store = SimpleNamespace(artist_uris=["listenbrainz:artist:daft punk", "listenbrainz:artist:bts"],
                            track_artist=[0, 1], track_names=["One More Time (Radio Edit)", "Dynamite"])
    assert store_song_keys(store) == ["daft punk|one more time", "bts|dynamite"]


def test_catalog_cases_as_each_pair_sees_them():
    keys = ["a|1", "b|2", "c|3", "d|4"]
    tracks = np.array([0, 1, 2, 3, 0, 3, 2])           # sessions: [a b c d], [a d c]
    offsets = np.array([0, 4, 7])
    cases, counts = build_catalog_cases(keys, tracks, offsets, n_cases=10, seed=0, input_length=2)
    assert counts == {"held_out_sessions": 2, "cases": 5, "sampled": 5}
    assert cases[2] == {"songs": ["b|2", "c|3"], "target": "d|4", "later": ["d|4"]}      # input: up to 2 songs
    assert cases[0] == {"songs": ["a|1"], "target": "b|2", "later": ["b|2", "c|3", "d|4"]}
    small = as_known(cases, {"a|1": "spotify:track:A", "c|3": "spotify:track:C", "d|4": "lb:D"})  # no b
    assert small[0] == {"songs": ["spotify:track:A"], "target": None, "later": ["spotify:track:C", "lb:D"]}
    assert small[2] == {"songs": ["spotify:track:C"], "target": "lb:D", "later": ["lb:D"]}


def test_catalog_scores_count_unknown_songs_as_misses():
    cases = [{"songs": ["x"], "target": "t", "later": ["t"]},     # hit at rank 1
             {"songs": ["x"], "target": None, "later": []},        # next song unknown: a miss
             {"songs": [], "target": "t", "later": ["t"]}]         # no input song known: no answer, a miss
    answer = lambda songs: (["t", "spotify:track:u"], ["A", "B"])
    out = catalog_scores(cases, answer, artist_of=lambda uri: "A")
    assert out["next_ndcg@10"] == round(1 / 3, 4) and out["next_hit@10"] == round(1 / 3, 4)
    assert out["answered"] == round(2 / 3, 4) and out["knows_the_next_song"] == round(2 / 3, 4)
    assert out["artists"] == 2 and out["last_artist"] == 0.5 and out["no_spotify_link"] == 0.5
