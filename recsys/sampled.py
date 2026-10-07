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


def candidate_set(Y, n_random, vocab_size, generator, probs=None):
    """(candidates, real): every target slot of Y, flattened (padding slots hold song 0),
    then n_random random songs (uniform, or drawn from probs); real marks the columns
    that aren't padding. Random draws come from `generator` (CPU), so they don't
    disturb other random streams."""
    slots = Y.reshape(-1)
    if probs is None:
        random_songs = torch.randint(0, vocab_size, (n_random,), generator=generator)
    else:
        random_songs = torch.multinomial(probs, n_random, replacement=True, generator=generator)
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
