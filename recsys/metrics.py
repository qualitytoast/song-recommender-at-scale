"""Ranking metrics, identical in definition to v1's metrics.py.

Every function takes logits of shape (n_examples, vocab_size), one score per
song in the catalog, and the true next-song IDs, shape (n_examples,). Inputs
are numpy arrays: move model output to the CPU first.
"""
import numpy as np


def target_ranks(logits, targets):
    """Rank of each true song among all songs, 1 = the model's top pick.

    rank = 1 + the number of songs scored strictly higher than the true one,
    so ties resolve in the true song's favour (v1's convention).
    """
    targets = np.asarray(targets)
    true_scores = logits[np.arange(len(targets)), targets][:, None]
    return np.sum(logits > true_scores, axis=1) + 1


def gains_at_k(logits, targets, k=10):
    """Per-example NDCG: 1/log2(1 + rank) if rank <= k, else 0.

    There is one relevant song per example, so the ideal DCG is 1 and NDCG is
    just this gain.
    """
    ranks = target_ranks(logits, targets)
    gains = 1.0 / np.log2(ranks + 1)
    gains[ranks > k] = 0.0
    return gains


def ndcg_at_k(logits, targets, k=10):
    """Mean NDCG@k. This is the number training early-stops on."""
    return float(np.mean(gains_at_k(logits, targets, k)))


def hits_at_k(logits, targets, k=10):
    """Per-example bool: did the true song land in the top k?"""
    return target_ranks(logits, targets) <= k
