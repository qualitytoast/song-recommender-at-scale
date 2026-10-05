# Results

All runs report NDCG@10 and Hits@10 on held-out data, using v1's metric
definitions (rank = 1 + songs scored strictly higher than the true one).
"Most-popular" always suggests the most common next songs in training.

| Date | Run | What changed | Data | NDCG@10 | Hits@10 | Most-popular NDCG@10 / Hits@10 | Best epoch | Train time | Commit |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-05 | v1 (NumPy, reference) | — | MPD, first 5,000 playlists: 33,770 songs, 110,333 train / 14,844 held-out windows | 0.0330 | 5.9% | 0.0049 / 1.1% | 22 (of 33) | 20.2 min (CPU) | v1 `cab8a1d` |
| 2026-10-03 | `v1_baseline` | Rebuilt v1 in PyTorch: same data, split, architecture, init, SGD settings, early stopping | same | 0.0373 | 6.3% | 0.0049 / 1.1% | 27 (of 38) | 5.0 min (mps) | `1afdcf8` |
| 2026-10-03 | `v1_adam` | `v1_baseline` with Adam (lr 1e-3, L2 weight decay 1e-3) instead of SGD (lr 0.05) | same | 0.0339 | 6.6% | 0.0049 / 1.1% | 38 (of 40, hit the cap) | 9.8 min (mps) | `650a7bb` |
| 2026-10-05 | `v1_adamw` | `v1_adam` with AdamW (decoupled weight decay, wd 0.05) instead of Adam (L2, wd 1e-3) | same | 0.0460 | 6.9% | 0.0049 / 1.1% | 11 (of 30) | 7.2 min (mps) | `be55be7` |
| 2026-10-05 | `v1_fixed` | `v1_adamw` with all five v1 quirks fixed at once: track URIs, vocab from training playlists only, separate 80/10/10 validation playlists, scaled attention, PyTorch default init | MPD, first 5,000 playlists, 4,000 / 500 / 500: 30,587 songs, 74,115 train / 6,109 val / 7,722 held-out windows | 0.0539 | 7.8% | 0.0068 / 1.5% | 5 (of 30) | 4.3 min (mps) | `815d067` |

## Notes

**Held-out numbers include the validation subset (v1 through `v1_adamw`).** Like v1, the validation
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
  SGD's keeps falling to ~6.1 and overfits: Adam is over-regularized. Its
  learned song vectors end 30-40% smaller than SGD's for songs of every
  popularity (see the weight-size table under `v1_adamw`), not mainly rare
  songs as first guessed. The L2 decay is added to the gradient and goes
  through Adam's per-weight rescaling; for the 7,502 songs that never appear
  in a training window, decay is the whole gradient, and Adam drives their
  vectors to exactly 0. `v1_adamw` tests the fix.
- It never early-stopped: patience only counts from epoch 20, and new bests
  kept arriving, with the best at epoch 38 of the 40-epoch cap.

**`v1_adamw`: best run so far, by a margin larger than the seed noise seen.**

| | NDCG@10 | Hits@1 | Hits@5 | Hits@10 | Held-out minus val NDCG@10 | Val NDCG@10 | Final train / val loss |
|---|---|---|---|---|---|---|---|
| SGD (`v1_baseline`) | 0.0373 | 1.8% | 4.5% | 6.3% | 0.0392 | 0.0301 | 6.08 / 9.22 |
| Adam (`v1_adam`) | 0.0339 | 1.2% | 3.9% | 6.6% | 0.0343 | 0.0319 | 7.46 / 8.27 |
| AdamW (`v1_adamw`) | **0.0460** | **2.9%** | **5.2%** | **6.9%** | **0.0482** | **0.0373** | 2.97 / 11.44 |

- On the 11,844 held-out windows not used for early stopping, AdamW beats
  SGD by 0.009 NDCG@10. The only seed-to-seed gap measured so far (v1 vs
  `v1_baseline` on the same windows: 0.0340 vs 0.0392) is about half that.
  Biggest gain is at the top: Hits@1 1.8% -> 2.9%.
- It learns fast and overfits hard: best epoch 11, validation loss rises
  from epoch 3 while NDCG@10 keeps improving until epoch 7-11, and training
  loss falls to 2.97. Early stopping on NDCG@10 (not loss) is what keeps the
  good checkpoint. Validation loss gets worse because the model grows
  overconfident in wrong answers even as its ranking of the right one improves.
- wd 0.05 was picked so AdamW shrinks weights by lr x wd = 5e-5 per step, the
  same as SGD. The decay did match: unseen songs' vectors (decay only) end at
  0.101 after 12 epochs, exactly 0.8 x (1 - 5e-5)^41,376. But regularization
  overall is much weaker than SGD's, because Adam's rescaling moves every
  weight that gets a gradient by ~lr per step, which outweighs the decay,
  where SGD's steps on rarely-seen songs are tiny and decay wins.

Average size of each song's learned vector at the best epoch, by how often
the song appears in training windows (started at ~0.80 for embeddings, ~1.41
for output rows):

| Times seen | Songs | Embedding SGD / Adam / AdamW | Output row SGD / Adam / AdamW |
|---|---|---|---|
| never | 7,502 | 0.006 / 0.000 / 0.101 | 0.022 / 0.048 / 0.324 |
| 1-10 | 9,412 | 0.082 / 0.071 / 0.599 | 0.093 / 0.066 / 0.435 |
| 11-100 | 13,868 | 0.208 / 0.150 / 0.735 | 0.228 / 0.130 / 0.571 |
| over 100 | 2,988 | 0.400 / 0.254 / 0.721 | 0.368 / 0.235 / 0.625 |

22% of the vocab (7,502 songs) never appears in a training window: v1 counts
song frequency over held-out playlists too, and windows touching a dropped
song are skipped. The model can never learn these songs.

**`v1_fixed`: the fixes make the evaluation honest without costing accuracy.**

Its held-out set is different from every earlier row, so its 0.0539 can't be
compared with them directly. The held-out *playlists* are the same 500, but
with URIs and a training-only vocab, windows containing a song seen fewer
than twice in training are skipped, leaving 7,722 of them (all learnable
songs, so easier: most-popular rises from 0.0049 to 0.0068). Where the
windows went:

| Data settings | Songs | Train | Val | Held-out |
|---|---|---|---|---|
| v1 | 33,770 | 110,333 | 3,000 | 14,844 |
| + 80/10/10 split | 33,770 | 97,482 | 12,851 | 14,844 |
| + track URI | 36,636 | 81,737 | 10,448 | 12,473 |
| + vocab from train (= `v1_fixed`) | 30,587 | 74,115 | 6,109 | 7,722 |

Every song in those 7,722 windows is also in the title-based vocab of the
earlier runs, and none of the earlier runs trained on the held-out playlists,
so all runs can be scored on exactly the same windows:

| Run, scored on `v1_fixed`'s 7,722 held-out windows | NDCG@10 | Hits@1 | Hits@10 |
|---|---|---|---|
| `v1_baseline` (SGD) | 0.0462 | 2.1% | 7.8% |
| `v1_adam` | 0.0422 | 1.4% | 8.2% |
| `v1_adamw` | 0.0544 | 3.3% | 8.4% |
| `v1_fixed` | 0.0539 | 3.4% | 7.8% |

- `v1_fixed` ties `v1_adamw` while training on 12% fewer playlists (33%
  fewer windows), with no held-out data used for early stopping and no
  same-titled songs merged (both of which favour `v1_adamw` here). Five
  changes were bundled, so the effect of each one is unknown.
- AdamW's lead over SGD holds on these windows (0.054 vs 0.046).
- It overfits even faster than `v1_adamw`: best epoch 5, then validation
  NDCG@10 falls every epoch. `min_epochs = 20` (tuned for SGD's slow start)
  made it run 24 epochs past its best.

**End of Phase 1.** The PyTorch rebuild matches v1 (same outputs from v1's
weights; same data, split and metrics), runs ~4x faster, and with AdamW and
the fixes reaches 0.054 NDCG@10 / 7.8% Hits@10 on an honest held-out set,
7.9x the most-popular baseline.

# Phase 2: adding song and playlist features

Each row adds one feature on top of the row above, so "Change vs previous" is
what that feature added given the ones before it. Every row uses the
`v1_fixed` data (same 7,722 held-out windows) and training setup, with
`min_epochs = 10`, and is run with training seeds 1, 2 and 3 (`data_seed` 42
for all, so the split never changes). Numbers are on the held-out set; a
change only counts if it is clearly larger than the seed spread.

| Config | Seeds | NDCG@10 mean (min–max) | Change vs previous | Hits@10 | Hits@1 | Best epochs | Params | Min / seed |
|---|---|---|---|---|---|---|---|---|
| `p2_base` | 3 | 0.0556 (0.0548–0.0570) | — | 8.3% (8.1%–8.6%) | 3.4% (3.4%–3.4%) | 5, 7, 8 | 4,038,011 | 3.1 |

Most-popular baseline on the same held-out windows: NDCG@10 0.0068, Hits@10 1.5%.

Runs (table made with `python -m recsys.summarize <configs>`):
- `p2_base` (2026-10-05, `3910bde`): `v1_fixed` with `min_epochs` 20 -> 10 and
  3 training seeds. Seed spread 0.0022 NDCG@10; `v1_fixed` (seed 42, 0.0539)
  sits just below it, so seed-to-seed noise is about +-0.002.
