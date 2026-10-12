"""Ranking metrics, identical in definition to v1's metrics.py.

The *_at_k functions take logits of shape (n_examples, vocab_size), one score
per song in the catalog, and the true next-song IDs, shape (n_examples,), as
numpy arrays. The *_from_ranks functions take ranks already computed (e.g. on
the GPU by recsys.model.rank_and_loss, for catalogs too big to hold all logits).
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
    return gains_from_ranks(target_ranks(logits, targets), k)


def gains_from_ranks(ranks, k=10):
    """Per-example NDCG gain from target ranks: 1/log2(1 + rank) if rank <= k, else 0."""
    ranks = np.asarray(ranks)
    gains = 1.0 / np.log2(ranks + 1)
    gains[ranks > k] = 0.0
    return gains


def ndcg_at_k(logits, targets, k=10):
    """Mean NDCG@k. This is the number training early-stops on."""
    return float(np.mean(gains_at_k(logits, targets, k)))


def hits_at_k(logits, targets, k=10):
    """Per-example bool: did the true song land in the top k?"""
    return target_ranks(logits, targets) <= k


def target_weights(n, decay):
    """Weight of each of n target songs: decay ** (steps ahead - 1), so 1 for the next song."""
    return decay ** np.arange(n, dtype=np.float64)


def multi_target_gains(ranks, weights, k=10):
    """Per-case NDCG@k when several songs are right, each as relevant as its weight: ranks (cases, n) of each
    case's target songs (0 = no such target: past the playlist's end, outside the vocab or a repeat). The sum of
    weight / log2(1 + rank) over targets in the top k, over the best possible sum (the targets' weights, largest
    first, at ranks 1, 2, ...). With one target per case it is gains_from_ranks."""
    ranks = np.asarray(ranks)
    valid = ranks > 0
    w = np.where(valid, np.broadcast_to(np.asarray(weights, dtype=np.float64), ranks.shape), 0.0)
    found = valid & (ranks <= k)
    dcg = np.where(found, w / np.log2(np.where(found, ranks, 1) + 1.0), 0.0).sum(axis=1)
    best = -np.sort(-w, axis=1)[:, :k]
    idcg = (best / np.log2(np.arange(2, best.shape[1] + 2))).sum(axis=1)
    return np.where(idcg > 0, dcg / np.where(idcg > 0, idcg, 1.0), 0.0)


def later_precision(top, later, n_songs):
    """Per case: the share of its top songs (cases, k) that its playlist goes on to contain (later: sorted keys
    case * n_songs + song, data.later_keys)."""
    keys = np.arange(len(top))[:, None] * n_songs + np.asarray(top)
    return np.isin(keys, later).mean(axis=1)


def artist_variety(top, song_artist, last_songs):
    """Per case: (how many different artists its top songs (cases, k) have, the share of them by the artist of
    its last input song)."""
    artists = np.asarray(song_artist)[np.asarray(top)]
    ordered = np.sort(artists, axis=1)
    distinct = (ordered[:, 1:] != ordered[:, :-1]).sum(axis=1) + 1
    return distinct, (artists == np.asarray(song_artist)[last_songs][:, None]).mean(axis=1)
