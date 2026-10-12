"""Sampled softmax: train against a small candidate set instead of every song.

With a large catalog, scoring every song for every prediction is too slow and
too big (a million songs x 320 predictions per batch is 1.3 GB of scores). Each
training batch instead scores its predictions against a fixed-size candidate set:
  - every target slot of the batch: each true next song is a wrong answer for
    the other predictions ("in-batch negatives"), and
  - n_random songs drawn from the whole catalog: uniformly, or (negative_power
    > 0) in proportion to (how often the song is a training target)^power, which
    favours popular songs, the usual confusions.
Each prediction's label is its own slot. A column holding the same song as a
prediction's answer, other than its own slot, is masked out (an "accidental
hit": it's not a wrong answer), as are padding slots.

Popular songs are more often some prediction's true next song, so they appear
as candidates (and get pushed down as wrong answers) more often than their
popularity deserves. The logQ correction subtracts log(expected times a song is
in the candidate set) from its score, which undoes that bias.

Every shape is fixed (no sorting or de-duplicating), so the GPU never has to
stop and report a size to the CPU mid-step; that kept the CPU and GPU from
overlapping their work and made a first version slower than the full softmax.
Validation and evaluation still rank the full catalog.
"""
import torch
from torch import nn


def random_probs(target_freq, power):
    """Probability of drawing each song as a random negative: target_freq**power, normalized.
    power 0 is uniform over the whole catalog."""
    weights = torch.ones_like(target_freq) if power == 0 else target_freq ** power
    return weights / weights.sum()


def draw_random_songs(n_random, vocab_size, generator, probs=None):
    """n_random songs, uniform or drawn from probs, on the CPU from `generator`, so they
    don't disturb other random streams (and the CPU knows which songs a batch uses)."""
    if probs is None:
        return torch.randint(0, vocab_size, (n_random,), generator=generator)
    return torch.multinomial(probs, n_random, replacement=True, generator=generator)


def candidate_set(Y, n_random, vocab_size, generator, probs=None, random_songs=None):
    """(candidates, real): every target slot of Y, flattened (padding slots hold song 0),
    then n_random random songs (uniform, or drawn from probs; or random_songs if given,
    from draw_random_songs); real marks the columns that aren't padding."""
    slots = Y.reshape(-1)
    if random_songs is None:
        random_songs = draw_random_songs(n_random, vocab_size, generator, probs)
    random_songs = random_songs.to(Y.device)
    candidates = torch.cat([slots.clamp(min=0), random_songs])
    real = torch.cat([slots != -100, torch.ones(n_random, dtype=torch.bool, device=Y.device)])
    return candidates, real


def log_q(candidates, n_targets, target_freq, n_random, probs):
    """log(expected number of times each candidate is in the candidate set):
    as a true target (n_targets draws from the target frequencies) plus as a
    random song (n_random draws from probs, see random_probs)."""
    return torch.log((n_targets * target_freq[candidates] + n_random * probs[candidates]).clamp(min=1e-12))


def sampled_softmax_loss(logits, Y, candidates, real, correction):
    """Cross-entropy over the candidate set from candidate_set().

    logits: (..., len(candidates)) candidate scores for each target slot of Y
    Y: true song IDs (-100 = padding); correction: log_q of the candidates."""
    targets = Y.reshape(-1)
    n = len(targets)
    scores = logits.reshape(n, len(candidates)) - correction
    own = torch.arange(n, device=Y.device)
    columns = torch.arange(len(candidates), device=Y.device)
    accidental = (targets[:, None] == candidates[None, :]) & (own[:, None] != columns[None, :])
    scores = scores.masked_fill(accidental | ~real[None, :], float("-inf"))
    labels = own.masked_fill(targets == -100, -100)
    return nn.functional.cross_entropy(scores, labels, ignore_index=-100)


def multi_target_loss(logits, positions, T, weights, candidates, real, correction):
    """Sampled softmax when each position has several right answers: its next len(weights) songs.

    The candidates are T (chunks, W) flattened (each chunk's songs after its first, data.chunk_targets: the
    first L columns are its positions' own next songs) plus random songs, as candidate_set builds from T.
    logits (chunks, L, len(candidates)); positions (chunks, L) True for real positions. Position k of chunk c
    has right answers T[c, k], T[c, k + 1], ...: column c * W + k + d for the song d + 1 steps ahead, unless it
    is -100 (past the playlist's end, unknown) or repeats a nearer one. Each right answer is scored against the
    position's wrong answers only (every real candidate that isn't one of its right songs), never against the
    others: -log(e^s / (e^s + sum over wrong answers)); a position's loss is the weighted mean over its right
    answers, the batch's the mean over positions. With one weight it equals sampled_softmax_loss."""
    B, L = positions.shape
    W, n = T.shape[1], len(weights)
    assert W >= L + n - 1, f"T needs the {L + n - 1} songs after each chunk's first, has {W}"
    P = B * L
    scores = logits.reshape(P, len(candidates)) - correction
    k = torch.arange(L, device=T.device)
    cols = (torch.arange(B, device=T.device)[:, None, None] * W + k[None, :, None]
            + torch.arange(n, device=T.device)[None, None, :]).reshape(P, n)         # (P, n) right-answer columns
    songs = T.reshape(-1)[cols]                                                      # their songs, -100 if none
    valid = (songs != -100) & positions.reshape(P, 1)
    for d in range(1, n):  # a song repeated within the next n counts once, at its nearest place
        valid[:, d] &= ~(songs[:, d:d + 1] == songs[:, :d]).any(dim=1)
    right = torch.zeros_like(scores, dtype=torch.bool)
    for d in range(n):
        right |= (songs[:, d:d + 1] == candidates[None, :]) & valid[:, d:d + 1]
    wrong = scores.masked_fill(right | ~real[None, :], float("-inf"))
    lse_wrong = torch.logsumexp(wrong, dim=1, keepdim=True)
    s = scores.gather(1, cols)
    losses = torch.logaddexp(s, lse_wrong) - s                                       # (P, n)
    w = weights.to(scores.dtype)[None, :] * valid
    per_position = (losses * w).sum(dim=1) / w.sum(dim=1).clamp(min=1e-12)
    used = valid.any(dim=1)
    return (per_position * used).sum() / used.sum().clamp(min=1)
