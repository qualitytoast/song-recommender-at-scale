import numpy as np

from recsys.taste import adjust, blend, load_profile, make_profile, save_profile


def test_blend_moves_queries_toward_the_taste_and_keeps_the_last_1():
    h = np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0]], dtype=np.float32)
    taste = np.array([0.0, 4.0, 1.0])
    assert blend(h, taste, 0.0) is h                                  # no profile weight: untouched
    out = blend(h, taste, np.array([0.5, 0.0]))
    np.testing.assert_allclose(out, [[1.0, 2.0, 1.0], [0.0, 2.0, 1.0]])
    assert out.dtype == h.dtype


def test_adjust_adds_the_artist_bonus_and_skip_penalty_then_resorts():
    p = make_profile("me", "v", np.ones(3), artist_affinity=[0.0, 1.0], skip_rate=[0.0, 0.0, 0.9, 0.0])
    ids, scores, song_artist = np.array([0, 1, 2, 3]), np.array([3.0, 2.5, 2.0, 1.0]), np.array([0, 1, 1, 0])
    assert adjust(p, ids, scores, song_artist)[0] is ids             # alpha = beta = 0: unchanged
    got_ids, got_scores = adjust(p.with_settings(0, 1.0, 1.0), ids, scores, song_artist)
    assert got_ids.tolist() == [1, 0, 2, 3]                           # 3.5, 3.0, 2.1, 1.0
    np.testing.assert_allclose(got_scores, [3.5, 3.0, 2.1, 1.0])


def test_a_profile_saves_and_loads_with_its_settings(tmp_path):
    p = make_profile("me", "abc", np.arange(3.0), [0.5], [0.0, 1.0], w=0.2, alpha=1.0, beta=0.5)
    save_profile(tmp_path / "me.npz", p)
    q = load_profile(tmp_path / "me.npz")
    assert (q.key, q.bundle, q.w, q.alpha, q.beta, q.digest) == ("me", "abc", 0.2, 1.0, 0.5, p.digest)
    assert q.cache_id() == p.cache_id() and p.with_settings(0, 0, 0).cache_id() != p.cache_id()
