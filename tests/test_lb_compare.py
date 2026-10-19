import numpy as np

from scripts.lb_compare import build_cases, case_positions, ndcg_at_10


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
