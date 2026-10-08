"""FAISS runs only inside the search worker process (it can't share a process with
PyTorch, see recsys/ann.py), so every FAISS check here goes through SearchWorker."""
import numpy as np
import pytest
import torch

from recsys.model import SongRecommender
from recsys.search import ExactSearch, SearchWorker, query_vectors, search_recall, song_vectors

ARTIST = [0, 0, 1, 2, 2, 2, 1]
GENRES = [[1, 2], [1, 0], [2, 0], [0, 0], [1, 2], [2, 0], [1, 0]]


@pytest.fixture(scope="module")
def worker():
    with SearchWorker() as w:
        yield w


def random_catalog(n=3000, d=9, queries=50, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, d).astype(np.float32), rng.randn(queries, d).astype(np.float32)


def exact(vectors, queries, k):
    return ExactSearch(vectors, torch.device("cpu")).search(queries, k)


def test_song_and_query_vectors_reproduce_the_retrievers_scores():
    torch.manual_seed(0)
    model = SongRecommender(vocab_size=7, embed_dim=4, context_length=3, num_layers=2, dropout=0.0,
                            scale_attention=True, init="pytorch", song_features={"artist": ARTIST},
                            name_word_count=3, song_genres=GENRES, causal=True).eval()
    with torch.no_grad():  # nonzero feature vectors, so they're part of the test
        model.output_features["artist"].weight.normal_()
        model.output_genres.weight[1:].normal_()
    X, N = np.array([[0, 3, 6], [2, 5, 1]]), np.array([[1, 2], [0, 0]])
    scores = query_vectors(model, X, N, torch.device("cpu")) @ song_vectors(model).T
    torch.testing.assert_close(torch.from_numpy(scores), model(torch.from_numpy(X), torch.from_numpy(N)))


def test_flat_in_the_worker_finds_exactly_the_exact_top_k(worker):
    vectors, queries = random_catalog()
    worker.build("flat", vectors, kind="flat")
    ids, scores = worker.search("flat", queries, 20)
    exact_ids, exact_scores = exact(vectors, queries, 20)
    np.testing.assert_array_equal(ids, exact_ids)
    np.testing.assert_allclose(scores, exact_scores, rtol=1e-5, atol=1e-5)


def test_sharding_doesnt_change_results(worker):
    vectors, queries = random_catalog()
    worker.build("flat", vectors, kind="flat")
    worker.build("flat4", vectors, kind="flat", shards=4)
    np.testing.assert_array_equal(worker.search("flat4", queries, 20)[0], worker.search("flat", queries, 20)[0])


def test_ivf_searching_every_cluster_is_exact_and_fewer_clusters_can_miss(worker):
    vectors, queries = random_catalog()
    exact_ids, _ = exact(vectors, queries, 20)
    worker.build("ivf_all", vectors, kind="ivf", nlist=16, nprobe=16)
    np.testing.assert_array_equal(worker.search("ivf_all", queries, 20)[0], exact_ids)
    worker.build("ivf_one", vectors, kind="ivf", nlist=16, nprobe=1)
    assert search_recall(worker.search("ivf_one", queries, 20)[0], exact_ids).mean() < 1.0


def test_hnsw_with_a_wide_search_finds_almost_everything(worker):
    vectors, queries = random_catalog()
    worker.build("hnsw", vectors, kind="hnsw", ef_search=256)
    assert search_recall(worker.search("hnsw", queries, 20)[0], exact(vectors, queries, 20)[0]).mean() > 0.95


def test_worker_reports_errors_and_keeps_running(worker):
    with pytest.raises(RuntimeError, match="unknown index kind"):
        worker.build("bad", np.zeros((10, 3), dtype=np.float32), kind="nope")
    vectors, queries = random_catalog(n=100)
    worker.build("small", vectors, kind="flat")
    assert worker.search("small", queries, 5)[0].shape == (50, 5)


def test_search_recall_by_hand():
    exact_ids = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    approx = np.array([[4, 3, 9, 1], [5, 6, 7, 8]])
    np.testing.assert_allclose(search_recall(approx, exact_ids), [0.75, 1.0])
