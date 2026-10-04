import numpy as np

from recsys.baselines import popularity_scores
from recsys.metrics import target_ranks


def test_counts_next_songs():
    _, counts = popularity_scores([2, 2, 0, 2, 0, 3], vocab_size=5)
    np.testing.assert_array_equal(counts, [2, 0, 3, 1, 0])


def test_ranks_by_count_then_id():
    # counts: song 0 = 2, song 1 = 0, song 2 = 3, song 3 = 1, song 4 = 0
    # order:  2 (3x), 0 (2x), 3 (1x), then 1 and 4 tie at 0 -> lower ID first
    scores, _ = popularity_scores([2, 2, 0, 2, 0, 3], vocab_size=5)
    assert list(np.argsort(-scores)) == [2, 0, 3, 1, 4]


def test_ties_do_not_flatter_baseline():
    # Songs 1 and 4 both have count 0. If they shared a score, both would get
    # rank 4. With ID tie-breaking, song 4 gets rank 5.
    scores, _ = popularity_scores([2, 2, 0, 2, 0, 3], vocab_size=5)
    logits = np.broadcast_to(scores, (2, 5))
    np.testing.assert_array_equal(target_ranks(logits, [1, 4]), [4, 5])
