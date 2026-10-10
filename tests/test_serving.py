import json
import shutil

import numpy as np
import pytest
import torch

from recsys.export import write_bundle, write_manifest
from recsys.model import SongRecommender
from recsys.rank_features import CandidateFeatures, pair_counts
from recsys.ranker import CandidateRanker, build_shortlists, feature_fn, rerank
from recsys.recommender import NoKnownSongs, Recommender
from recsys.search import ExactSearch, song_vectors

V, CPU = 40, torch.device("cpu")
SETTINGS = {"shortlist": 15, "exclude_input": True, "ivf_nlist": 2, "ivf_nprobe": 2,  # every cluster: exact
            "features": ["overlap", "cooccurrence"], "correction_hidden": 4, "windows": "any_length", "sources": {}}
WORDS = ["chill", "gym", "road", "trip", "mix"]
ARTISTS = ["Kavinsky", "London Grammar", "Chromatics", "Beyoncé", "Artist 4", "Artist 5"]
# A few named songs (the rest "Song i"): song i's artist is ARTIST_OF[i]; PLAYLISTS[i] its playlist count.
NAMED = {0: "Nightcall", 1: "Nightcall", 2: "Nightcall - Breakbot Remix", 3: "Night Drive", 4: "Halo", 5: "Odd Look"}
ARTIST_OF = [0, 1, 0, 2, 3, 0]
PLAYLISTS = np.array([50, 5, 3, 20, 30, 10] + [1 + i % 4 for i in range(V - 6)])
TRACKS = [NAMED.get(i, f"Song {i}") for i in range(V)]


def tiny_pair():
    """A retriever with artist, album, length and genre features and name words, and a ranker on it,
    every weight random (so each table matters), the ranker's song tables fine-tuned apart."""
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    genres = rng.randint(0, 5, (V, 2))
    genres[:4] = 0  # some songs with no genres
    artist = rng.randint(0, 6, V)
    artist[:6] = ARTIST_OF
    features = {"artist": artist, "album": rng.randint(0, 12, V), "duration": rng.randint(0, 4, V)}
    retriever = SongRecommender(V, 8, 3, 2, 0.0, True, "pytorch", features, name_word_count=len(WORDS),
                                song_genres=genres, causal=True).eval()
    with torch.no_grad():
        for table in [*retriever.input_features.values(), *retriever.output_features.values(), retriever.input_genres,
                      retriever.output_genres, retriever.name_words]:
            table.weight[table.padding_idx is not None:].normal_()
    ranker = CandidateRanker(retriever, n_features=7, hidden=4).eval()
    with torch.no_grad():
        for p in [ranker.candidate_marker, ranker.candidate_position, *ranker.correction.parameters(),
                  ranker.retriever.song_embedding.weight, ranker.retriever.input_features["artist"].weight]:
            p.add_(torch.randn_like(p))
        ranker.feature_mean.normal_()
        ranker.feature_std.uniform_(0.5, 2.0)
    pairs = pair_counts(rng.randint(0, V, 300), np.arange(0, 301, 10), V)
    return retriever, ranker, pairs


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    retriever, ranker, pairs = tiny_pair()
    path = write_bundle(tmp_path_factory.mktemp("bundles"), "tiny", retriever, ranker, SETTINGS,
                        [f"spotify:track:{i}" for i in range(V)], TRACKS, ARTISTS, WORDS, pairs, PLAYLISTS, CPU)
    rec = Recommender(path, "cpu")
    yield rec, retriever, ranker, pairs
    rec.close()


def offline_top10(retriever, ranker, pairs, songs, name_ids):
    """The top 10 the offline evaluation's code computes for one input (exact search)."""
    X, N = np.array([songs]), np.array([name_ids])
    vectors = song_vectors(retriever)
    ids, scores, _ = build_shortlists(retriever, X, N, np.array([0]), CPU, SETTINGS["shortlist"],
                                      ExactSearch(vectors, CPU).search, vectors, exclude_input=True)
    featurizer = CandidateFeatures.from_retriever(SETTINGS["features"], retriever, None, pairs, CPU)
    features = feature_fn(featurizer, ids, scores, (X[0], np.array([0]), np.array([len(songs)])), np.array([0]), CPU)
    reranked = rerank(ranker, X, N, ids, scores, CPU, features=features)[0]
    order = np.argsort(-reranked, kind="stable")[:10]
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
    ids, scores, times = rec.recommend_ids(songs, name_ids)
    expected_ids, expected_scores = offline_top10(retriever, ranker, pairs, songs[-3:], name_ids)
    np.testing.assert_array_equal(ids, expected_ids)
    np.testing.assert_allclose(scores, expected_scores, rtol=1e-5, atol=1e-5)
    assert not set(ids) & set(songs[-3:])  # the input songs are never recommended
    assert set(times) == {"retriever", "search", "ranker inputs", "ranker"}


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
