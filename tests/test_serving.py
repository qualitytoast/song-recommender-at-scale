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


def tiny_pair():
    """A retriever with artist, album, length and genre features and name words, and a ranker on it,
    every weight random (so each table matters), the ranker's song tables fine-tuned apart."""
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    genres = rng.randint(0, 5, (V, 2))
    genres[:4] = 0  # some songs with no genres
    features = {"artist": rng.randint(0, 6, V), "album": rng.randint(0, 12, V), "duration": rng.randint(0, 4, V)}
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
                        [f"spotify:track:{i}" for i in range(V)], [f"Song {i}" for i in range(V)],
                        [f"Artist {a}" for a in range(6)], WORDS, pairs, CPU)
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
    assert out["used"] == ["spotify:track:6", "spotify:track:7", "spotify:track:8"]  # the last 3 known
    assert out["ignored"] == ["spotify:track:nope", "x"]
    assert len(out["recommendations"]) == 10 and out["version"] == rec.version
    first = out["recommendations"][0]
    assert set(first) == {"uri", "track", "artist", "score"}
    j = int(first["uri"].split(":")[-1])
    assert first["track"] == f"Song {j}" and first["artist"].startswith("Artist ")
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
