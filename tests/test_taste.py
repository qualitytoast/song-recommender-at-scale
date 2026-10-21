import numpy as np
import pytest

from recsys.recommender import Recommender
from recsys.search import mixed_query_vectors
from recsys.taste import adjust, blend, load_profile, make_profile, save_profile
from tests.serving_helpers import ARTIST_OF, V, write_tiny_bundle


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


URIS = ["spotify:track:3", "spotify:track:17", "spotify:track:9"]


@pytest.fixture(scope="module")
def rec(tmp_path_factory):
    r = Recommender(write_tiny_bundle(tmp_path_factory.mktemp("b"))[0], "cpu")
    yield r
    r.close()


def top(rec, **kw):
    return [x["uri"] for x in rec.recommend(URIS, max_per_artist=0, mix=0, **kw)["recommendations"]]


def query_of(rec, uris):
    X = np.array([[int(u.split(":")[-1]) for u in uris]])
    return mixed_query_vectors(rec.retriever, X, rec.encode_name(""), rec.device, 0.0)[0]


def flat(rec, **settings):
    return make_profile("me", rec.version, query_of(rec, URIS), np.zeros(len(ARTIST_OF) + 10), np.zeros(V), **settings)


def test_no_profile_an_unknown_key_and_zero_settings_all_give_the_usual_answer(rec):
    usual = top(rec)
    rec.profiles = {"me": flat(rec)}
    assert top(rec, profile="nobody") == usual and not rec.recommend(URIS, profile="nobody")["personalized"]
    assert top(rec, profile="me") == usual and rec.recommend(URIS, profile="me")["personalized"]


def test_the_taste_vector_pulls_the_shortlist(rec):
    usual = top(rec)
    rec.profiles = {"me": flat(rec, w=1.0)}                     # the request's own query: nothing moves
    assert top(rec, profile="me") == usual
    other = query_of(rec, ["spotify:track:30", "spotify:track:31"])
    rec.profiles = {"me": make_profile("me", rec.version, other, np.zeros(10), np.zeros(V), w=1.0)}
    assert top(rec, profile="me") != usual


def test_the_artist_bonus_and_the_skip_penalty_reorder_the_answer(rec):
    usual = top(rec)
    first = int(usual[0].split(":")[-1])
    skip = np.zeros(V)
    skip[first] = 1.0
    rec.profiles = {"me": make_profile("me", rec.version, query_of(rec, URIS), np.zeros(10), skip, beta=100.0)}
    assert usual[0] not in top(rec, profile="me")               # always skipped: out of the top 10
    loved = rec.song_artist[int(usual[-1].split(":")[-1])]
    affinity = np.zeros(int(rec.song_artist.max()) + 1)
    affinity[loved] = 1.0
    rec.profiles = {"me": make_profile("me", rec.version, query_of(rec, URIS), affinity, np.zeros(V), alpha=100.0)}
    assert rec.song_artist[int(top(rec, profile="me")[0].split(":")[-1])] == loved


def test_personalized_answers_have_their_own_cache_entries(rec):
    class DictCache:  # not a dict: an empty one would count as no cache
        def __init__(self):
            self.store = {}

        def get(self, key):
            return self.store.get(key)

        def set(self, key, value):
            self.store[key] = value
    rec.profiles, rec.cache = {"me": flat(rec, alpha=1.0)}, DictCache()
    try:
        rec.recommend(URIS)
        rec.recommend(URIS, profile="me")
        assert len(rec.cache.store) == 2 and rec.recommend(URIS, profile="me")["cached"]
    finally:
        rec.cache = None


def test_profiles_load_only_for_their_bundle_and_the_http_api_takes_a_key(rec, tmp_path):
    from fastapi.testclient import TestClient

    from recsys.serve import create_app
    save_profile(tmp_path / "mine.npz", flat(rec, alpha=1.0))
    save_profile(tmp_path / "old.npz", make_profile("old", "another-version", np.ones(3), np.zeros(1), np.zeros(1)))
    with TestClient(create_app(rec.path, "cpu", profiles=tmp_path)) as client:
        assert client.get("/health").json()["profiles"] == 1
        body = client.post("/recommend", json={"songs": URIS, "profile": "me"}).json()
        assert body["personalized"] and not client.post("/recommend", json={"songs": URIS}).json()["personalized"]
