# Results

All runs report NDCG@10 and Hits@10 on held-out data, using v1's metric
definitions (rank = 1 + songs scored strictly higher than the true one).
"Most-popular" always suggests the most common next songs in training.

| Date | Run | What changed | Data | NDCG@10 | Hits@10 | Most-popular NDCG@10 / Hits@10 | Best epoch | Train time | Commit |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-05 | v1 (NumPy, reference) | — | MPD, first 5,000 playlists: 33,770 songs, 110,333 train / 14,844 held-out windows | 0.0330 | 5.9% | 0.0049 / 1.1% | 22 (of 33) | 20.2 min (CPU) | v1 `cab8a1d` |
| 2026-10-03 | `v1_baseline` | Rebuilt v1 in PyTorch: same data, split, architecture, init, SGD settings, early stopping | same | 0.0373 | 6.3% | 0.0049 / 1.1% | 27 (of 38) | 5.0 min (mps) | `1afdcf8` |
| 2026-10-03 | `v1_adam` | `v1_baseline` with Adam (lr 1e-3, L2 weight decay 1e-3) instead of SGD (lr 0.05) | same | 0.0339 | 6.6% | 0.0049 / 1.1% | 38 (of 40, hit the cap) | 9.8 min (mps) | `650a7bb` |

## Notes

**Held-out numbers include the validation subset.** Like v1, the validation
set used for early stopping is the first 3,000 held-out windows, and the
reported numbers cover all 14,844. Split out for `v1_baseline`:

| | n | NDCG@10 | Hits@10 |
|---|---|---|---|
| validation subset | 3,000 | 0.0301 (v1: 0.0292) | 5.2% (v1: 5.5%) |
| held-out minus validation | 11,844 | 0.0392 | 6.5% |
| full held-out | 14,844 | 0.0373 | 6.3% |

**Why `v1_baseline` beats v1 by 0.0043.** The model and evaluation code match
v1 exactly: v1's trained weights loaded into the PyTorch model give the same
logits (max difference 5e-6) and score exactly 0.0292 / 0.0330 / 5.9% with
this repo's evaluation code. The gap is therefore from training randomness
(different initial weights, dropout masks and batch order give a different
best checkpoint), not a mismatch. One run per side can't measure how large
that randomness is.

**`v1_adam` vs `v1_baseline`: no clear winner on quality.**

| | NDCG@10 | Hits@1 | Hits@5 | Hits@10 | Val NDCG@10 | Lowest val loss | Final train loss |
|---|---|---|---|---|---|---|---|
| SGD (`v1_baseline`) | 0.0373 | 1.8% | 4.5% | 6.3% | 0.0301 | 8.47 | 6.08 |
| Adam (`v1_adam`) | 0.0339 | 1.2% | 3.9% | 6.6% | 0.0319 | 8.24 | 7.46 |

- Adam puts the right song in the top 10 slightly more often but ranks it
  lower within the top 10, so NDCG@10 is lower. Both gaps are about the size
  of the seed-to-seed difference between v1 and `v1_baseline` (0.0043), so
  one run each can't call it.
- Adam learns much faster per epoch: no early plateau (beats most-popular
  from epoch 0), and reaches validation NDCG@10 0.0265 at epoch 7 (114 s) vs
  epoch 19 (157 s) for SGD. Each Adam epoch is slower (~14 s vs ~8 s), since
  it updates two extra running averages for each of the 4.4M weights.
- Adam's training loss stalls at ~7.45 with a small train/val gap, while
  SGD's keeps falling to ~6.1 and overfits. That looks like much stronger
  regularization. Likely cause: L2 weight decay is added to the gradient,
  and Adam rescales each weight's gradient, so decay on weights with small
  data gradients (most song-embedding rows in any batch) becomes relatively
  much larger. AdamW (decay applied separately) avoids that. Not tested.
- It never early-stopped: patience only counts from epoch 20, and new bests
  kept arriving, with the best at epoch 38 of the 40-epoch cap.
