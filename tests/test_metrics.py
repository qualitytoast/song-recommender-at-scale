import numpy as np

from recsys.metrics import gains_at_k, hits_at_k, ndcg_at_k, target_ranks

# Three examples over a 4-song catalog.
LOGITS = np.array([
    [0.9, 0.1, 0.2, 0.3],  # true song 0 has the top score      -> rank 1
    [0.9, 0.1, 0.2, 0.3],  # true song 2: only 0.9, 0.3 beat it  -> rank 3
    [0.5, 0.5, 0.5, 0.1],  # true song 1 ties with 0 and 2       -> rank 1
])
TARGETS = np.array([0, 2, 1])


def test_ranks_count_strictly_higher_scores():
    np.testing.assert_array_equal(target_ranks(LOGITS, TARGETS), [1, 3, 1])


def test_gain_is_one_over_log2_rank_plus_one():
    # rank 1 -> 1/log2(2) = 1.0, rank 3 -> 1/log2(4) = 0.5
    np.testing.assert_allclose(gains_at_k(LOGITS, TARGETS, k=10), [1.0, 0.5, 1.0])


def test_gain_is_zero_past_k():
    np.testing.assert_allclose(gains_at_k(LOGITS, TARGETS, k=2), [1.0, 0.0, 1.0])


def test_ndcg_is_mean_gain():
    assert ndcg_at_k(LOGITS, TARGETS, k=10) == (1.0 + 0.5 + 1.0) / 3


def test_hits_at_k():
    np.testing.assert_array_equal(hits_at_k(LOGITS, TARGETS, k=2), [True, False, True])
    np.testing.assert_array_equal(hits_at_k(LOGITS, TARGETS, k=3), [True, True, True])
