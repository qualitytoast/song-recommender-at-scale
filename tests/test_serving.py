import json
import shutil

import numpy as np
import pytest
import torch

from recsys.export import write_manifest
from recsys.rank_features import CandidateFeatures
from recsys.ranker import build_shortlists, feature_fn, rerank
from recsys.recommender import NoKnownSongs, Recommender, pick_top
from recsys.search import ExactSearch, mixed_query_vectors, query_vectors, song_vectors
from serving_helpers import ARTISTS, CPU, SETTINGS, TRACKS, V, write_tiny_bundle

@pytest.fixture(scope="module")
def served(tmp_path_factory):
    path, retriever, ranker, pairs = write_tiny_bundle(tmp_path_factory.mktemp("bundles"))
    rec = Recommender(path, "cpu")
    yield rec, retriever, ranker, pairs
    rec.close()


def offline_order(retriever, ranker, pairs, songs, name_ids):
    """The reranked shortlist the offline evaluation's code computes for one input (exact search)."""
    X, N = np.array([songs]), np.array([name_ids])
    vectors = song_vectors(retriever)
    ids, scores, _ = build_shortlists(retriever, X, N, np.array([0]), CPU, SETTINGS["shortlist"],
                                      ExactSearch(vectors, CPU).search, vectors, exclude_input=True)
    featurizer = CandidateFeatures.from_retriever(SETTINGS["features"], retriever, None, pairs, CPU)
    features = feature_fn(featurizer, ids, scores, (X[0], np.array([0]), np.array([len(songs)])), np.array([0]), CPU)
    reranked = rerank(ranker, X, N, ids, scores, CPU, features=features)[0]
    order = np.argsort(-reranked, kind="stable")
    return ids[0, order], reranked[order]


def test_the_serving_models_compute_what_the_trained_ones_do(served):
    rec, retriever, ranker, pairs = served
    X, N = torch.tensor([[3, 17, 9]]), torch.tensor([[2, 4]])
    torch.testing.assert_close(rec.retriever.hidden_states(X, N), retriever.hidden_states(X, N))
    cands, base = torch.tensor([[5, 11, 30, 2]]), torch.randn(1, 4)
    ranks, top = torch.arange(1, 5)[None, :], base[:, 0]
    trained = CandidateFeatures.from_retriever(SETTINGS["features"], retriever, None, pairs, CPU)
    f = rec.features(X, cands, base, ranks, top)
    torch.testing.assert_close(f, trained(X, cands, base, ranks, top))
    torch.testing.assert_close(rec.ranker(X, N, cands, base, f), ranker(X, N, cands, base, f))


@pytest.mark.parametrize("songs", [[7], [7, 21], [3, 17, 9], [1, 2, 3, 4, 5]])
def test_recommendations_match_the_offline_pipeline(served, songs):
    rec, retriever, ranker, pairs = served
    name_ids = rec.encode_name("Road trip mix!")[0]
    ids, scores, times, notes = rec.recommend_ids(songs, name_ids, mix=0)  # the offline pipeline's playlist vector
    expected_ids, expected_scores = offline_order(retriever, ranker, pairs, songs[-3:], name_ids)
    np.testing.assert_array_equal(ids, expected_ids)
    np.testing.assert_allclose(scores, expected_scores, rtol=1e-5, atol=1e-5)
    assert not set(ids) & set(songs[-3:])  # the input songs are never recommended
    assert set(times) == {"retriever", "search", "ranker inputs", "ranker"}


def test_mixing_averages_the_guesses_made_after_each_song(served):
    retriever = served[1]
    X, N = np.array([[3, 17, 9], [5, 8, 0]]), np.array([[2, 4], [1, 0]])  # the tiny model reads 3 songs
    lengths = np.array([3, 2])
    np.testing.assert_array_equal(mixed_query_vectors(retriever, X, N, CPU, 0.0, lengths),
                                  query_vectors(retriever, X, N, CPU, lengths=lengths))  # 0: the last guess alone
    for mix in (0.5, 1.0):
        got = mixed_query_vectors(retriever, X, N, CPU, mix, lengths)
        for r in range(len(X)):  # each prefix on its own, weighted: the last 1, then mix, mix^2, ...
            n = lengths[r]
            prefixes = np.vstack([query_vectors(retriever, X[r:r + 1, :i + 1], N[r:r + 1], CPU) for i in range(n)])
            w = mix ** np.arange(n - 1, -1, -1)
            np.testing.assert_allclose(got[r], w @ prefixes / w.sum(), rtol=1e-5, atol=1e-5)
    per_row = mixed_query_vectors(retriever, X, N, CPU, np.array([0.0, 0.5]), lengths)  # one mix per row
    np.testing.assert_array_equal(per_row[0], query_vectors(retriever, X, N, CPU, lengths=lengths)[0])
    np.testing.assert_allclose(per_row[1], mixed_query_vectors(retriever, X, N, CPU, 0.5, lengths)[1], rtol=1e-6)


def test_pick_top_keeps_at_most_max_per_artist_songs_by_one_artist():
    ids, scores = np.arange(8), -np.arange(8.0)
    artist = np.array([0, 0, 0, 1, 0, 1, 1, 2])
    np.testing.assert_array_equal(pick_top(ids, scores, artist, 2, k=4)[0], [0, 1, 3, 5])
    np.testing.assert_array_equal(pick_top(ids, scores, artist, 2, k=4)[1], [0, -1, -3, -5])
    np.testing.assert_array_equal(pick_top(ids, scores, artist, 0, k=4)[0], [0, 1, 2, 3])  # 0: no limit
    np.testing.assert_array_equal(pick_top(ids, scores, artist, 1, k=4)[0], [0, 1, 3, 7])  # too few artists:
    # the best skipped song fills the last place


def test_a_request_gets_the_artist_limit_it_asks_for(served):
    rec = served[0]
    name_ids = rec.encode_name("")
    ranked, scores, _, _ = rec.recommend_ids([3, 17, 9], name_ids)
    uris = ["spotify:track:3", "spotify:track:17", "spotify:track:9"]
    for limit in (0, 1, 2, 10):
        top = [int(r["uri"].split(":")[-1]) for r in rec.recommend(uris, max_per_artist=limit)["recommendations"]]
        assert top == pick_top(ranked, scores, rec.song_artist, limit)[0].tolist()
    assert rec.recommend(uris)["recommendations"] == rec.recommend(uris, max_per_artist=2)["recommendations"]
    assert [int(r["uri"].split(":")[-1]) for r in rec.recommend(uris, max_per_artist=0)["recommendations"]] == \
        ranked[:10].tolist()


def test_a_request_skips_unknown_songs_and_reads_the_last_known_ones(served):
    rec = served[0]
    uris = ["spotify:track:5", "spotify:track:nope", "spotify:track:6", "spotify:track:7", "spotify:track:8", "x"]
    out = rec.recommend(uris, name="chill")
    assert [u["uri"] for u in out["used"]] == ["spotify:track:6", "spotify:track:7", "spotify:track:8"]  # last 3 known
    assert out["used"][0] == {"uri": "spotify:track:6", "track": "Song 6", "artist": rec.artists[6]}
    assert out["ignored"] == ["spotify:track:nope", "x"]
    assert len(out["recommendations"]) == 10 and out["version"] == rec.version
    first = out["recommendations"][0]
    assert set(first) == {"uri", "track", "artist", "score"}
    j = int(first["uri"].split(":")[-1])
    assert first["track"] == TRACKS[j] and first["artist"] in ARTISTS
    assert out["recommendations"] == rec.recommend(uris[2:5], name="chill")["recommendations"]
    assert len(rec.recommend(["spotify:track:9"])["recommendations"]) == 10  # no name: fine
    with pytest.raises(NoKnownSongs) as error:
        rec.recommend(["a", "b"])
    assert error.value.ignored == ["a", "b"]


def test_unknown_name_words_are_left_out_like_in_training(served):
    rec = served[0]
    np.testing.assert_array_equal(rec.encode_name("Road TRIP zzzz"), [[3, 4]])
    np.testing.assert_array_equal(rec.encode_name("zzzz"), [[0]])
    np.testing.assert_array_equal(rec.encode_name(None), [[0]])


def test_the_version_comes_from_every_file(served, tmp_path):
    rec = served[0]
    copy = tmp_path / "copy"
    shutil.copytree(rec.path, copy)
    assert write_manifest(copy) == rec.version  # the same files give the same version
    settings = json.loads((copy / "settings.json").read_text())
    (copy / "settings.json").write_text(json.dumps({**settings, "shortlist": 16}))
    assert write_manifest(copy) != rec.version
    assert rec.path.name == f"tiny-{rec.version}"


def test_the_http_api(served, tmp_path):
    from fastapi.testclient import TestClient
    from recsys.serve import create_app
    rec = served[0]
    with TestClient(create_app(rec.path, "cpu")) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["version"] == rec.version and health["songs"] == V
        assert health["defaults"] == {"max_per_artist": 2, "mix": 0.5}  # a bundle without its own
        out = client.post("/recommend", json={"songs": ["spotify:track:6", "nope", "spotify:track:7"],
                                              "name": "road trip"})
        assert out.status_code == 200
        body = out.json()
        assert body == {**rec.recommend(["spotify:track:6", "nope", "spotify:track:7"], "road trip"),
                        "timings_ms": body["timings_ms"]}
        assert client.post("/recommend", json={"songs": ["spotify:track:6"]}).status_code == 200  # no name
        empty = client.post("/recommend", json={"songs": []})
        assert empty.status_code == 400 and "no songs" in empty.json()["detail"]["error"]
        unknown = client.post("/recommend", json={"songs": ["a", "b"]})
        assert unknown.status_code == 400 and unknown.json()["detail"]["ignored"] == ["a", "b"]
        assert client.post("/recommend", json={"name": "x"}).status_code == 422  # malformed: no songs field
        menu = client.get("/songs", params={"q": "nightc", "limit": 2}).json()["songs"]
        assert [(s["track"], s["artist"]) for s in menu] == [("Nightcall", "Kavinsky"), ("Nightcall", "London Grammar")]
        assert client.get("/songs", params={"q": "x", "limit": 500}).status_code == 422  # at most 50
        any_artist = client.post("/recommend", json={"songs": ["spotify:track:6"], "max_per_artist": 0}).json()
        assert any_artist["recommendations"] == rec.recommend(["spotify:track:6"], max_per_artist=0)["recommendations"]
        assert client.post("/recommend", json={"songs": ["spotify:track:6"], "max_per_artist": 11}).status_code == 422
        last_only = client.post("/recommend", json={"songs": ["spotify:track:6", "spotify:track:7"], "mix": 0}).json()
        assert last_only["recommendations"] == rec.recommend(["spotify:track:6", "spotify:track:7"], mix=0)["recommendations"]
        assert client.post("/recommend", json={"songs": ["spotify:track:6"], "mix": 1.5}).status_code == 422


def test_finding_songs_by_name_puts_the_most_played_first(served):
    rec = served[0]
    found = lambda q, **k: [(s["track"], s["artist"]) for s in rec.find_songs(q, **k)]
    assert found("nightcall") == [("Nightcall", "Kavinsky"), ("Nightcall", "London Grammar"),
                                  ("Nightcall - Breakbot Remix", "Kavinsky")]
    assert found("NIGHT")[:2] == [("Nightcall", "Kavinsky"), ("Night Drive", "Chromatics")]
    kav = rec.find_songs("kav")
    assert [s["track"] for s in kav[:2]] == ["Nightcall", "Odd Look"] and {s["artist"] for s in kav} == {"Kavinsky"}
    assert [s["playlists"] for s in kav] == sorted((s["playlists"] for s in kav), reverse=True)
    assert found("nightcall - kav") == [("Nightcall", "Kavinsky"), ("Nightcall - Breakbot Remix", "Kavinsky")]
    assert found("nightcall - breakbot") == [("Nightcall - Breakbot Remix", "Kavinsky")]
    assert found("beyonce")[0] == ("Halo", "Beyoncé")  # accents and case don't matter
    assert found("night", limit=1) == [("Nightcall", "Kavinsky")]
    assert found("  ") == [] and found("zzz") == []
    assert rec.find_songs("halo")[0] == {"uri": "spotify:track:4", "track": "Halo", "artist": "Beyoncé", "playlists": 30}


def test_songs_can_be_given_as_track_and_artist_text(served):
    rec = served[0]
    assert rec.resolve("Nightcall - Kavinsky") == 0
    assert rec.resolve("nightcall - london grammar") == 1
    assert rec.resolve("Nightcall") == 0  # a track name alone: the most-played song with that name
    assert rec.resolve("Nightcall - Breakbot Remix - Kavinsky") == 2  # a " - " inside the track name
    assert rec.resolve("Halo - Beyonce") == 4
    assert rec.resolve("spotify:track:7") == 7
    assert rec.resolve("Nightcall - Nobody") is None and rec.resolve("spotify:track:999") is None
    out = rec.recommend(["Nightcall - Kavinsky", "nope", "spotify:track:3"])
    assert [u["uri"] for u in out["used"]] == ["spotify:track:0", "spotify:track:3"] and out["ignored"] == ["nope"]
    assert out["recommendations"] == rec.recommend(["spotify:track:0", "spotify:track:3"])["recommendations"]


def test_the_menu_names_the_bundles_count(tmp_path):
    path = write_tiny_bundle(tmp_path, settings={"popularity": "listens"})[0]
    rec = Recommender(path, "cpu")
    try:
        assert set(rec.find_songs("nightcall")[0]) == {"uri", "track", "artist", "listens"}
        assert set(rec.recommend(["spotify:track:0"])["used"][0]) == {"uri", "track", "artist"}
    finally:
        rec.close()


def test_a_bundle_sets_its_own_artist_limit_and_mix(tmp_path):
    path = write_tiny_bundle(tmp_path, settings={"max_per_artist": 1, "mix": 0.0})[0]
    rec = Recommender(path, "cpu")
    try:
        uris = ["spotify:track:3", "spotify:track:17", "spotify:track:9"]
        assert (rec.max_per_artist, rec.mix) == (1, 0.0)
        assert rec.recommend(uris)["recommendations"] == rec.recommend(uris, max_per_artist=1, mix=0.0)["recommendations"]
    finally:
        rec.close()


def test_a_bundle_without_playlist_names_loads_and_answers(tmp_path):
    """A model trained without the playlist-name input (ListenBrainz sessions have no names) exports "no name
    words", and serving builds its model without a name table; a request's name is then ignored."""
    path = write_tiny_bundle(tmp_path, names=False)[0]
    assert json.loads((path / "settings.json").read_text())["name_word_count"] is None
    rec = Recommender(path, "cpu")
    try:
        with_name = rec.recommend(["spotify:track:0", "spotify:track:3"], "chill mix", max_per_artist=0, mix=0.0)
        without = rec.recommend(["spotify:track:0", "spotify:track:3"], None, max_per_artist=0, mix=0.0)
    finally:
        rec.close()
    assert [r["uri"] for r in with_name["recommendations"]] == [r["uri"] for r in without["recommendations"]]
    assert len(without["recommendations"]) == 10
