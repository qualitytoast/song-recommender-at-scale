"""Non-ML baselines the model has to beat."""
import numpy as np


def popularity_scores(Y_train, vocab_size):
    """One fixed score per song: how often it is the next song in training.

    Ignores the playlist entirely and always suggests the most common next
    songs. Same as v1's baseline.

    Ties are broken by song ID (lower ID ranks higher) so every song gets a
    distinct score. Raw counts tie thousands of songs, and the ranking rule in
    metrics.py resolves ties in the true song's favour, which would flatter
    this baseline for no real reason.

    Returns (scores, counts).
    """
    counts = np.bincount(np.asarray(Y_train, dtype=int), minlength=vocab_size)
    order = np.lexsort((np.arange(vocab_size), -counts))  # count desc, then ID asc
    scores = np.empty(vocab_size, dtype=np.float64)
    scores[order] = -np.arange(vocab_size, dtype=np.float64)  # most popular scores highest
    return scores, counts


def popularity_ranks(scores, targets):
    """Rank of each target under fixed, all-distinct scores (popularity_scores):
    a song's rank is its position in the score order, so no scores matrix is needed."""
    position = np.empty(len(scores), dtype=np.int64)
    position[np.argsort(-scores, kind="stable")] = np.arange(len(scores))
    return position[np.asarray(targets)] + 1
