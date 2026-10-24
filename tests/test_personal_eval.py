import numpy as np

from scripts.personal_eval import METRICS, bootstrap, make_cases, score


def test_cases_and_the_new_and_liked_songs_of_a_period():
    plays = {"kept": np.ones(7, dtype=bool), "listened": np.array([1, 1, 1, 1, 1, 1, 0], dtype=bool),
             "skipped": np.array([0, 0, 0, 0, 1, 0, 0], dtype=bool), "song": np.array([1, 2, 3, 4, 5, 6, 7]),
             "session": np.array([0, 0, 1, 1, 1, 1, 1]), "seconds": np.array([0, 10, 100, 110, 120, 130, 140])}
    cases, known, liked = make_cases(plays, start=50, stop=200, input_length=2)
    assert [(c[1], c[2]) for c in cases] == [([3], 4), ([3, 4], 5), ([4, 5], 6), ([5, 6], 7)]
    assert cases[0][3] == {4, 5, 6, 7} and known == {1, 2}       # before the period: songs 1 and 2
    assert liked == {3, 4, 6}                                     # 5 skipped; 7 under 30 s


def test_the_owners_score_weighs_finds_70_and_next_song_30():
    base = {"finds_per_100": 2.0, "next_ndcg@10": 0.2}
    assert score(base, base) == 1.0
    assert abs(score({"finds_per_100": 3.0, "next_ndcg@10": 0.2}, base) - 1.35) < 1e-12  # 0.7 x 1.5 + 0.3


def test_bootstrap_ranges_cover_a_clear_gain():
    rng = np.random.default_rng(0)
    base = rng.random((200, len(METRICS)))
    best = base.copy()
    best[:, METRICS.index("finds_per_100")] += 1.0
    r = bootstrap(base, best, np.repeat(np.arange(40), 5), n=200)
    assert r["score"][0] > 1.0 and r["finds_per_100 change"][0] > 0.9
