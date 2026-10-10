from types import SimpleNamespace

import numpy as np
import pytest
import torch

from recsys.model import SongRecommender
from recsys.ranker import (MISSED, CandidateRanker, build_shortlists, check_vocab, final_ranks, sample_negatives,
                           song_search, with_negatives)
from recsys.search import song_vectors

ARTIST = [0, 0, 1, 2, 2, 2, 1]
GENRES = [[1, 2], [1, 0], [2, 0], [0, 0], [1, 2], [2, 0], [1, 0]]


def retriever(causal=True, scale=True):
    torch.manual_seed(0)
    return SongRecommender(vocab_size=7, embed_dim=4, context_length=3, num_layers=2, dropout=0.0,
                           scale_attention=scale, init="pytorch", song_features={"artist": ARTIST},
                           name_word_count=3, song_genres=GENRES, causal=causal).eval()


def trained_ranker(causal=True, scale=True):
    """A ranker whose new parameters and feature tables aren't zero, so their effects are visible."""
    ranker = CandidateRanker(retriever(causal, scale)).eval()
    r = ranker.retriever
    with torch.no_grad():
        for p in (ranker.candidate_marker, ranker.candidate_position, ranker.correction.weight,
                  r.input_features["artist"].weight, r.input_genres.weight[1:], r.name_words.weight[1:]):
            p.normal_()
    return ranker


def ranker_mask(L, C):
    """(L + C, L + C) bool, True where attention is blocked: context position i sees
    context positions <= i; candidate k sees the context and itself only."""
    i, j = torch.arange(L + C)[:, None], torch.arange(L + C)[None, :]
    return ~torch.where(i < L, j <= i, (j < L) | (j == i))


def masked_reference(ranker, ids, names, candidates, base_scores):
    """The ranker computed the plain way: full (L + C) x (L + C) attention with the
    blocked scores masked out. CandidateRanker computes only the allowed scores and
    must give the same result."""
    r = ranker.retriever
    L = ids.shape[1]
    context = r.embed_songs(ids) + r.position_embedding(torch.arange(L))
    cand = r.embed_songs(candidates) + ranker.candidate_position + ranker.candidate_marker
    x = torch.cat([context, cand], dim=1) + r.mean_vector(r.name_words, names)[:, None, :]
    blocked = ranker_mask(L, candidates.shape[1])
    for block in r.blocks:
        x = block(x, blocked)
    return base_scores + ranker.correction(x[:, L:]).squeeze(-1)


CONTEXT, NAMES = torch.tensor([[0, 3, 6]]), torch.tensor([[1, 2]])


def test_mask_by_hand():
    # 2 context songs (c0, c1), 2 candidates (k0, k1); True = blocked
    assert ranker_mask(2, 2).int().tolist() == [
        [0, 1, 1, 1],   # c0 sees c0
        [0, 0, 1, 1],   # c1 sees c0, c1
        [0, 0, 0, 1],   # k0 sees the context and itself
        [0, 0, 1, 0]]   # k1 sees the context and itself, not k0


def test_computing_only_the_allowed_scores_matches_full_masked_attention():
    torch.manual_seed(1)
    ids, names = torch.randint(0, 7, (5, 3)), torch.randint(0, 4, (5, 2))
    candidates, base = torch.randint(0, 7, (5, 6)), torch.randn(5, 6)
    for causal in (True, False):      # the context is causal in the ranker either way
        for scale in (True, False):   # v1's unscaled attention too
            ranker = trained_ranker(causal, scale)
            torch.testing.assert_close(ranker(ids, names, candidates, base),
                                       masked_reference(ranker, ids, names, candidates, base))


def test_before_training_the_ranker_returns_the_retriever_scores():
    ranker = CandidateRanker(retriever()).eval()
    base = torch.tensor([[2.5, -1.0, 0.3]])
    torch.testing.assert_close(ranker(CONTEXT, NAMES, torch.tensor([[4, 1, 2]]), base), base)


def test_each_candidate_is_scored_independently_of_the_others():
    ranker = trained_ranker()
    zeros = torch.zeros(1, 4)
    alone = ranker(CONTEXT, NAMES, torch.tensor([[4]]), zeros[:, :1])
    with_others = ranker(CONTEXT, NAMES, torch.tensor([[4, 1, 2, 5]]), zeros)
    with_others_swapped = ranker(CONTEXT, NAMES, torch.tensor([[5, 2, 1, 4]]), zeros)
    torch.testing.assert_close(with_others[0, 0], alone[0, 0])
    torch.testing.assert_close(with_others_swapped[0, 3], alone[0, 0])  # order doesn't matter either


def test_candidate_scores_depend_on_the_context():
    ranker = trained_ranker()
    cand, zeros = torch.tensor([[4, 1]]), torch.zeros(1, 2)
    a = ranker(CONTEXT, NAMES, cand, zeros)
    b = ranker(torch.tensor([[2, 2, 5]]), NAMES, cand, zeros)
    assert not torch.allclose(a, b)


def test_negatives_are_distinct_shortlist_songs_other_than_the_true_one():
    short = torch.tensor([[5, 3, 9, 1, 7], [2, 4, 6, 8, 0]])
    y = torch.tensor([9, 1])  # row 0's true song is shortlisted, row 1's isn't
    for seed in range(20):
        pos = sample_negatives(short, y, 4, torch.Generator().manual_seed(seed))
        for row in range(2):
            songs = short[row, pos[row]].tolist()
            assert len(set(pos[row].tolist())) == 4 and y[row].item() not in songs


def test_examples_put_the_true_song_first_and_keep_each_songs_own_score():
    short = torch.tensor([[5, 3, 9, 1, 7], [2, 4, 6, 8, 0]], dtype=torch.int32)
    short_scores = short.float() / 10  # song s scores s / 10, so a score shows which song it belongs to
    y, true_scores = torch.tensor([9, 1]), torch.tensor([0.9, 0.1])
    candidates, base = with_negatives(short, short_scores, true_scores, y, 3, torch.Generator().manual_seed(0))
    pos = sample_negatives(short, y, 3, torch.Generator().manual_seed(0))  # the same draws
    assert candidates.dtype == torch.long and candidates.shape == base.shape == (2, 4)
    assert candidates[:, 0].tolist() == [9, 1] and base[:, 0].tolist() == pytest.approx([0.9, 0.1])
    assert torch.equal(candidates[:, 1:], short.gather(1, pos).long())
    assert torch.allclose(base[:, 1:], candidates[:, 1:].float() / 10)


def test_final_ranks():
    short = np.array([[5, 3, 9], [2, 4, 6], [1, 2, 3]])
    scores = np.array([[0.1, 0.9, 0.5], [3.0, 1.0, 2.0], [1.0, 1.0, 0.0]])
    y = np.array([9, 7, 2])
    # row 0: song 9 scores 0.5, one song higher -> 2; row 1: song 7 not shortlisted;
    # row 2: song 2 ties with song 1 -> ties go to the true song -> 1
    np.testing.assert_array_equal(final_ranks(scores, short, y), [2, MISSED, 1])


def big_retriever(vocab=400):
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    m = SongRecommender(vocab_size=vocab, embed_dim=8, context_length=3, num_layers=1, dropout=0.0,
                        scale_attention=True, init="pytorch", song_features={"artist": rng.randint(0, 20, vocab)},
                        name_word_count=3, song_genres=rng.randint(0, 4, (vocab, 2)), causal=True).eval()
    with torch.no_grad():
        m.output_features["artist"].weight.normal_()
        m.output_genres.weight[1:].normal_()
    rng = np.random.RandomState(1)
    return m, rng.randint(0, vocab, (60, 3)), rng.randint(0, 4, (60, 2)), rng.randint(0, vocab, 60)


def shortlists(search, nlist=0, nprobe=0, k=20):
    model, X, N, Y = big_retriever()
    vectors = song_vectors(model)
    rc = SimpleNamespace(search=search, ivf_nlist=nlist, ivf_nprobe=nprobe)
    with song_search(rc, vectors, torch.device("cpu")) as find:
        return build_shortlists(model, X, N, Y, torch.device("cpu"), k, find, vectors, chunk=25), (model, X, N, Y)


def test_exact_shortlists_are_the_retrievers_top_k_and_true_scores():
    (ids, scores, true), (model, X, N, Y) = shortlists("exact")
    logits = model(torch.from_numpy(X), torch.from_numpy(N))
    top = torch.topk(logits, 20, dim=1)
    np.testing.assert_array_equal(ids, top.indices.numpy())
    np.testing.assert_allclose(scores, top.values.detach().numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(true, logits[torch.arange(len(Y)), torch.from_numpy(Y)].detach().numpy(),
                               rtol=1e-5, atol=1e-5)
    assert ids.dtype == np.int32 and scores.dtype == np.float32


def test_ivf_shortlists_searching_every_cluster_match_exact_and_fewer_clusters_can_miss():
    (exact_ids, exact_scores, exact_true), _ = shortlists("exact")
    (ids, scores, true), _ = shortlists("ivf", nlist=8, nprobe=8)
    np.testing.assert_array_equal(ids, exact_ids)
    np.testing.assert_allclose(scores, exact_scores, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(true, exact_true, rtol=1e-5, atol=1e-5)  # true scores never depend on the search
    (ids, _, _), _ = shortlists("ivf", nlist=8, nprobe=2)
    assert not np.array_equal(ids, exact_ids)


def test_every_retriever_input_feature_reaches_the_ranker_scores():
    # The ranker reads songs through the retriever's input side, so changing any input
    # table's row for a context song or for a candidate must change that candidate's
    # score. (Output-side tables enter through the retriever's frozen score instead.)
    context, names, cand, zeros = torch.tensor([[0, 3, 6]]), torch.tensor([[1, 2]]), torch.tensor([[4]]), torch.zeros(1, 1)
    for side, song in (("context", 3), ("candidate", 4)):
        for table, row in (("song_embedding", song), ("input_features.artist", ARTIST[song]),
                           ("input_genres", GENRES[song][0]), ("name_words", 1), ("position_embedding", 1)):
            if side == "candidate" and table == "position_embedding":
                continue  # a candidate has its own learned position
            ranker = trained_ranker()
            before = ranker(context, names, cand, zeros)
            with torch.no_grad():
                ranker.retriever.get_submodule(table).weight[row] += 1.0
            assert not torch.allclose(ranker(context, names, cand, zeros), before), (side, table)


def test_check_vocab_refuses_a_checkpoint_trained_on_other_songs():
    check_vocab({"vocab": ["a", "b"]}, ["a", "b"], "best.pt")   # same songs, same order: fine
    check_vocab({}, ["a", "b"], "best.pt")                      # older ranker checkpoint without a vocab
    for other in (["b", "a"], ["a", "b", "c"]):                 # reordered or different songs
        with pytest.raises(ValueError, match="different vocab"):
            check_vocab({"vocab": ["a", "b"]}, other, "best.pt")


def test_negatives_can_be_drawn_from_the_top_of_the_shortlist_only():
    short = torch.tensor([[5, 3, 9, 1, 7, 8], [2, 4, 6, 8, 0, 1]])
    y = torch.tensor([3, 8])  # true songs at positions 1 and 3
    for seed in range(20):
        g1, g2 = torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed)
        pos = sample_negatives(short, y, 2, g1, top=3)
        assert ((pos < 3) & (short.gather(1, pos) != y[:, None])).all()  # first 3 positions, never the true song
        # top = the whole shortlist draws exactly as before
        assert torch.equal(sample_negatives(short, y, 2, g2, top=6),
                           sample_negatives(short, y, 2, torch.Generator().manual_seed(seed)))


def test_score_reports_recall_at_any_shortlist_size():
    from recsys.evaluate import score
    result = score(np.array([1, 120, 300, 600]), (250,))
    assert result["recall@250"] == 0.5 and result["recall@100"] == 0.25 and result["recall@500"] == 0.75


def test_shortlists_are_cached_and_reused_without_searching(tmp_path, monkeypatch):
    import recsys.ranker as ranker_module
    from recsys.ranker import cached_shortlists
    model, X, N, Y = big_retriever()
    vectors = song_vectors(model)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"retriever weights")
    rc = SimpleNamespace(search="exact", ivf_nlist=0, ivf_nprobe=0, shortlist=20, exclude_input=False,
                         search_name=lambda: "exact")
    sets = [(X, N, Y, None), (X[:30], N[:30], Y[:30], None)]
    built, hits = cached_shortlists(rc, checkpoint, model, vectors, torch.device("cpu"), sets, tmp_path / "cache")
    assert hits == 0 and len(list((tmp_path / "cache").glob("*.npz"))) == 2

    def no_search(*args, **kwargs):
        raise AssertionError("searched although every set was cached")
    monkeypatch.setattr(ranker_module, "song_search", no_search)
    loaded, hits = cached_shortlists(rc, checkpoint, model, vectors, torch.device("cpu"), sets, tmp_path / "cache")
    assert hits == 2
    for b, l in zip(built, loaded):
        for x, y in zip(b, l):
            np.testing.assert_array_equal(x, y)


def test_shortlist_cache_key_changes_with_anything_that_decides_the_shortlists():
    from recsys.ranker import shortlist_key
    _, X, N, Y = big_retriever()
    rc = lambda **c: SimpleNamespace(**{"shortlist": 20, "exclude_input": False, "search_name": lambda: "exact", **c})
    base = shortlist_key(rc(), "abc", X, N, Y, None)
    assert shortlist_key(rc(), "abc", X, N, Y, None) == base
    for other in (shortlist_key(rc(), "abd", X, N, Y, None), shortlist_key(rc(shortlist=10), "abc", X, N, Y, None),
                  shortlist_key(rc(exclude_input=True), "abc", X, N, Y, None),
                  shortlist_key(rc(search_name=lambda: "ivf"), "abc", X, N, Y, None),
                  shortlist_key(rc(), "abc", X[1:], N[1:], Y[1:], None),
                  shortlist_key(rc(), "abc", X, N, Y, np.full(len(Y), 3))):
        assert other != base


def test_the_retrievers_own_exact_score_is_reused_only_when_it_fits(tmp_path, monkeypatch):
    import json
    import os
    import recsys.ranker as ranker_module
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "runs" / "retriever_ranker_runs" / "5k" / "retriever_5k_x" / "seed1"
    run.mkdir(parents=True)
    (run / "best.pt").write_bytes(b"w")
    saved = {"n": 5, "ndcg@10": 0.5, "recall@100": 0.9}
    (run / "eval.json").write_text(json.dumps({"full held-out": saved}))
    monkeypatch.setattr(ranker_module, "rank_and_loss", lambda *a, **k: (np.ones(5, dtype=int), None))
    rc = SimpleNamespace(retriever="configs/retriever_5k_x.toml", shortlist=100)
    ds = SimpleNamespace(Y_test=np.zeros(5), XI_test=None, N_test=None, LI_test=None)
    scored = lambda ranks: {"computed": True}
    got = ranker_module.retriever_exact_score(rc, 1, ds, None, "cpu", scored)
    assert got["ndcg@10"] == 0.5 and got["from"].endswith("eval.json")      # reused
    ds.Y_test = np.zeros(6)                                                  # different windows
    assert ranker_module.retriever_exact_score(rc, 1, ds, None, "cpu", scored) == {"computed": True}
    ds.Y_test = np.zeros(5)
    os.utime(run / "eval.json", (1, 1))                                      # older than the checkpoint
    assert ranker_module.retriever_exact_score(rc, 1, ds, None, "cpu", scored) == {"computed": True}


def test_a_full_held_out_evaluation_reads_the_retrievers_full_held_out_score(tmp_path, monkeypatch):
    import json
    import recsys.ranker as ranker_module
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "runs" / "retriever_ranker_runs" / "5k" / "retriever_5k_x" / "seed1"
    run.mkdir(parents=True)
    (run / "best.pt").write_bytes(b"w")
    (run / "eval.json").write_text(json.dumps({"full held-out": {"n": 5, "ndcg@10": 0.5, "recall@100": 0.9}}))
    (run / "eval_all.json").write_text(json.dumps({"full held-out": {"n": 8, "ndcg@10": 0.4, "recall@100": 0.8}}))
    monkeypatch.setattr(ranker_module, "rank_and_loss", lambda *a, **k: (np.ones(8, dtype=int), None))
    rc = SimpleNamespace(retriever="configs/retriever_5k_x.toml", shortlist=100)
    ds = SimpleNamespace(Y_test=np.zeros(8), XI_test=None, N_test=None, LI_test=None)
    got = ranker_module.retriever_exact_score(rc, 1, ds, None, "cpu", lambda r: {"computed": True}, all_held_out=True)
    assert got["ndcg@10"] == 0.4 and got["from"].endswith("eval_all.json")


def test_all_held_out_keeps_every_window_in_a_copy_of_the_config(tmp_path):
    from recsys.config import load_config
    from recsys.evaluate import eval_file, with_all_held_out
    cfg = load_config("configs/retriever_1m.toml")
    full = with_all_held_out(cfg, True)
    assert full.data.test_max_windows == 0 and cfg.data.test_max_windows == 200000
    assert with_all_held_out(cfg, False) is cfg
    assert (eval_file(False), eval_file(True)) == ("eval.json", "eval_all.json")
