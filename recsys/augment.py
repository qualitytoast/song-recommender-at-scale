"""Training-time window augmentation, so the model can't rely on exact sequences.

For each window in a batch, independently:
  - reorder: with probability reorder_prob, a random run of 3-5 consecutive
    songs is shuffled
  - crop: the first 0..crop_max songs are hidden, as if the context were shorter
  - mask: each song is hidden with probability mask_prob
Hidden songs are shown to the model as a learned [MASK] vector with their
features switched off (see SongRecommender.forward). Only training batches are
augmented; validation and held-out windows never are.
"""
import torch

REORDER_SPAN = (3, 5)  # shortest and longest run of songs that gets shuffled


def augment_plan(batch, length, generator, mask_prob, crop_max, reorder_prob):
    """Random augmentation for a batch of windows, drawn from `generator` (CPU).

    Returns (order, hidden):
      order:  (batch, length) positions to read from; apply with X.gather(1, order)
      hidden: (batch, length) bool, True where the song is hidden
    Reordering is applied first; hiding then picks positions, not songs.
    """
    positions = torch.arange(length)

    # Reorder: give positions inside the chosen span random sort keys within the
    # span's own range, so sorting by key shuffles the span and leaves the rest in place.
    span = torch.randint(REORDER_SPAN[0], REORDER_SPAN[1] + 1, (batch,), generator=generator)
    start = (torch.rand(batch, generator=generator) * (length - span + 1)).long()
    chosen = torch.rand(batch, generator=generator) < reorder_prob
    in_span = (positions >= start[:, None]) & (positions < (start + span)[:, None]) & chosen[:, None]
    random_key = start[:, None] + torch.rand(batch, length, generator=generator) * span[:, None]
    order = torch.argsort(torch.where(in_span, random_key, positions.float().expand(batch, length)), dim=1)

    # Crop, then mask.
    crop = torch.randint(0, crop_max + 1, (batch,), generator=generator)
    hidden = positions < crop[:, None]
    hidden |= torch.rand(batch, length, generator=generator) < mask_prob
    return order, hidden
