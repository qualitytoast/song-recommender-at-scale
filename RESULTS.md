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
| `p2_artist` | 3 | 0.0657 (0.0646–0.0680) | +0.0101 | 10.6% (10.5%–10.9%) | 3.5% (3.3%–3.8%) | 3, 4, 4 | 4,999,035 | 4.0 |
| `p2_album` | 3 | 0.0671 (0.0657–0.0697) | +0.0013 | 10.9% (10.7%–11.3%) | 3.5% (3.4%–3.6%) | 3, 4, 4 | 7,110,139 | 4.7 |
| `p2_duration` | 3 | 0.0674 (0.0667–0.0685) | +0.0003 | 11.0% (10.9%–11.2%) | 3.5% (3.3%–3.7%) | 3, 3, 4 | 7,111,419 | 5.1 |
| `p2_name` | 3 | 0.0669 (0.0652–0.0685) | -0.0004 | 10.9% (10.5%–11.2%) | 3.5% (3.3%–3.7%) | 3, 3, 4 | 7,149,499 | 5.3 |
| `p2_genre` | 3 | 0.0692 (0.0669–0.0715) | +0.0022 | 11.6% (11.4%–11.8%) | 3.4% (3.1%–3.7%) | 3, 3, 2 | 7,202,491 | 9.7 |

Most-popular baseline on the same held-out windows: NDCG@10 0.0068, Hits@10 1.5%.

Runs (table made with `python -m recsys.summarize <configs>`):
- `p2_base` (2026-10-05, `3910bde`): `v1_fixed` with `min_epochs` 20 -> 10 and
  3 training seeds. Seed spread 0.0022 NDCG@10; `v1_fixed` (seed 42, 0.0539)
  sits just below it, so seed-to-seed noise is about +-0.002.
- `p2_artist` (2026-10-05, `2081525`): + artist (7,508 artists; an input and
  an output vector each, starting at zero; +961,024 params). Clear gain: every
  seed improves over the same seed without artist (0.0548 -> 0.0680, 0.0570 ->
  0.0646, 0.0550 -> 0.0647), about 5x the seed noise. The gain is in getting
  the right song into the top 10 (Hits@10 8.3% -> 10.6%), not to #1 (Hits@1
  3.4% -> 3.5%): artist tells the model which neighbourhood of songs comes
  next, less which exact song. It also learns faster (validation NDCG@10 at
  epoch 2 is 0.049-0.056 vs 0.019-0.029) and peaks earlier (epoch 3-4), then
  overfits as before. ~30% slower per epoch: every batch builds all 30,587
  candidates' artist vectors.
- `p2_album` (2026-10-05, `11eb612`): + album (16,493 albums; +2,111,104
  params, the most of any feature). Marginal gain: each seed is up by a
  similar small amount over the same seed with artist only (+0.0017, +0.0012,
  +0.0010 on held-out), but best validation NDCG@10 barely moves (0.0632 ->
  0.0635, 0.0599 -> 0.0600, 0.0602 -> 0.0598). Most of what album says,
  artist already said: albums mostly sit inside one artist. It learns faster
  still (epoch-1 validation 0.037-0.048 vs 0.023-0.034) and overfits faster
  (epoch-4 training loss ~4.8 vs ~5.2, validation declines sooner after the
  peak), as expected from adding 2.1M per-album parameters.
- `p2_duration` (2026-10-05, `a4b0c3d`): + duration, as 10 equal-sized length
  buckets (cut points 2:50, 3:10, 3:23, 3:34, 3:44, 3:56, 4:10, 4:30, 5:04;
  +1,280 params). No measurable effect: against the same seed with album, one
  seed goes down and two go up (-0.0012, +0.0010, +0.0010), well inside seed
  noise. A song's length says little about what comes next once artist and
  album are known.
- `p2_name` (2026-10-05, `10846ba`): + playlist name: words used in at least
  2 training playlist names (594 words, e.g. "wedding", "gym", "rap",
  "summer"), averaged into one vector added at every position; 80% of windows
  have at least one known word (+38,080 params). No gain: against the same
  seed with duration, 0.0000, +0.0002, -0.0015. The model does use the name
  (its word vectors end up larger than its artist vectors, 0.70 vs 0.55), but
  what it learns doesn't transfer: hiding the names from the trained model
  *raises* held-out NDCG@10 on every seed (0.0685 -> 0.0694, 0.0670 -> 0.0680,
  0.0652 -> 0.0664). Likely cause: with 4,000 training playlists, many name
  words belong to only a few playlists, so a name works like a playlist ID
  the model memorizes songs against. Ten songs of context already say most of
  what a name like "rap" would.

**Phase 2, steps 1-4 summary.** What each feature added, on top of the ones
before it (3 seeds each; seed noise about +-0.002):

| Feature | NDCG@10 change | Verdict |
|---|---|---|
| artist | +0.0101 (0.0556 -> 0.0657) | clear gain, every seed |
| album | +0.0013 | marginal: every seed up a little on held-out, validation flat |
| duration | +0.0003 | none |
| playlist name | -0.0004 | none; learned but doesn't generalize (see above) |

Artist accounts for nearly all of the 0.0556 -> 0.0669 gain. Album, duration
and name together add about +0.001 and 2.15M parameters. Every run still
peaks by epoch 3-4 and then overfits; none of the features changes that.
Genre (step 5) is still to come.
- `p2_genre` (2026-10-06, `072b0c4`): + genre, from MusicBrainz (genre and tag
  data CC BY-NC-SA, credit MusicBrainz). Genres listed for at least 5 vocab
  artists (413 of 781), then each artist's top 5 by votes; a song's genre
  vector is the average over its artist's genres, on the input and output
  side, starting at zero; 14.9% of songs have no genre (+52,992 params).
  Small but consistent gain: against the same seed with playlist name,
  held-out NDCG@10 +0.0030, -0.0001, +0.0039; validation NDCG@10 up on every
  seed (+0.0023, +0.0022, +0.0019) and Hits@10 up on every seed (11.1% ->
  11.8%, 11.2% -> 11.4%, 10.5% -> 11.7%). The model leans on genre heavily:
  hiding every song's genres from the trained model roughly halves held-out
  NDCG@10 (0.0715 -> 0.0373, 0.0669 -> 0.0349, 0.0691 -> 0.0376). That shows how
  much the trained model depends on genre, not how much genre adds (the
  training-time comparison above does that). Cost: ~21 s per epoch vs ~16 s,
  from averaging genre vectors for all 30,587 candidates every batch (seed 1
  ran at ~38 s per epoch, most likely other load on the machine).

**Phase 2 final: what each feature added**, on top of the ones before it
(3 paired seeds each; seed-to-seed noise about +-0.002):

| Feature added | NDCG@10 | Change | Hits@10 | Verdict |
|---|---|---|---|---|
| (none, `p2_base`) | 0.0556 | — | 8.3% | |
| artist | 0.0657 | **+0.0101** | 10.6% | clear gain, every seed |
| album | 0.0671 | +0.0013 | 10.9% | marginal: held-out up a little on every seed, validation flat |
| duration | 0.0674 | +0.0003 | 11.0% | none |
| playlist name | 0.0669 | -0.0004 | 10.9% | none; learned but doesn't generalize |
| genre | 0.0692 | **+0.0022** | 11.6% | small, consistent gain: validation and Hits@10 up on every seed |

All five features together: 0.0556 -> 0.0692 NDCG@10 (+24%), Hits@10 8.3% ->
11.6%, 10.2x the most-popular baseline, for +3.16M parameters (+78%). Artist
and genre, the two features shared across many songs, give nearly all of it;
both mostly help get the right song into the top 10 rather than to #1 (Hits@1
stays at 3.4-3.5%). Every run still peaks by epoch 2-4 and then overfits;
no feature changed that, which points to more training data as the next lever.

# Overfitting experiments

Every Phase 2 run peaks by epoch 2-4. At its best epoch, `p2_genre` already
scores NDCG@10 0.46 on training windows vs 0.066 on validation (Hits@10 64%
vs 11%): it memorizes training playlists. 95% of training transitions (last
song -> next song) occur exactly once, and only 10.6% of validation
transitions ever occur in training. Each experiment below changes one thing
on top of `p2_genre` and is compared to it, same seeds, same held-out windows.

| Config | Seeds | NDCG@10 mean (min–max) | Change vs `p2_genre` | Hits@10 | Hits@1 | Best epochs | Params | Min / seed |
|---|---|---|---|---|---|---|---|---|
| `p2_genre` | 3 | 0.0692 (0.0669–0.0715) | — | 11.6% (11.4%–11.8%) | 3.4% (3.1%–3.7%) | 3, 3, 2 | 7,202,491 | 9.7 |
| `p2_augment` | 3 | 0.0689 (0.0686–0.0693) | -0.0002 | 11.6% (11.3%–11.7%) | 3.4% (3.3%–3.5%) | 4, 5, 5 | 7,202,555 | 12.0 |
| `p2_causal` | 3 | 0.0739 (0.0736–0.0744) | **+0.0047** | 12.1% (12.0%–12.2%) | 3.8% (3.8%–3.8%) | 5, 5, 4 | 7,202,491 | 5.2 |
| `p2_causal_augment` | 3 | 0.0714 (0.0704–0.0733) | +0.0022 (-0.0025 vs `p2_causal`) | 12.3% (12.0%–12.7%) | 3.3% (3.2%–3.4%) | 7, 7, 8 | 7,202,555 | 9.4 |

- `p2_augment` (2026-10-06, `f424206`): training windows only are altered
  (`recsys/augment.py`): with 50% chance a run of 3-5 songs is shuffled, the
  first 0-5 songs are hidden, and each song is hidden with 20% chance; a hidden
  song becomes a learned [MASK] vector with its features switched off. About
  40% of context songs end up hidden. Result: overfitting slows, the peak
  doesn't rise. Per seed, held-out NDCG@10 -0.0022, +0.0017, -0.0003 and best
  validation NDCG@10 -0.0015, +0.0033, -0.0010: no gain. But the peak moves
  from epoch 2-3 to 4-5, validation NDCG declines more slowly after it (0.048-
  0.052 at epoch 11 vs 0.042-0.046), training loss falls more slowly (4.1 vs
  2.5 at epoch 11), and the seed spread narrows (0.0007 vs 0.0046). Early
  stopping already caught `p2_genre` before memorization took over, so slowing
  memorization doesn't help by itself; and hiding ~40% of songs makes training
  windows unlike evaluation windows, which may cost some of what it gains.
- `p2_causal` (2026-10-06, `2796b29`): instead of one target per 10-song
  window, training playlists are cut into chunks of up to 11 songs
  (`recsys.data.make_chunks`) and the next song is predicted at every position,
  with causal attention (each position sees only earlier songs). Each
  transition is trained once per epoch, as before. Evaluation is unchanged
  (same held-out windows, scored from the last position). Clear gain: every
  seed improves on held-out NDCG@10 (+0.0021, +0.0075, +0.0046) and validation
  NDCG@10 (+0.0025, +0.0062, +0.0063); Hits@1 moves for the first time in
  Phase 2 (3.4% -> 3.8%, all three seeds 3.8%); the seed spread is the
  narrowest yet (0.0008); 10.9x most-popular. Also faster: ~15 s per epoch,
  5.2 min per seed. It still peaks at epoch 4-5 and overfits after, but more
  slowly (validation NDCG@10 0.055-0.057 at epoch 15 vs 0.042-0.046 at epoch 11).
  Two things changed together, so the gain can't be split between them:
  (1) the training method, and (2) the training data: chunks include
  transitions the window rule skipped (short playlists, songs next to
  out-of-vocab songs), 172,048 targets vs 74,115, but only 10,102 of them have
  a full 10-song context, vs all 74,115 windows. A control that trains chunks
  only on the window targets would separate them.
- `p2_causal_augment` (2026-10-06, `e738e7a`): `p2_causal` + augmentation of
  the training chunks: mask 20% and reorder 50% as in `p2_augment`, crop off.
  With chunks the targets sit inside the sequence, so targets inside a
  shuffled run (whose context could hold a later song, even the answer) and
  targets with nothing visible are left out of the loss: 19% of targets per
  epoch. Crop was turned off because causal training already sees every
  context length, and on short chunks (half have 5 songs or fewer) it hid
  whole chunks, leaving out 48% of targets. Result: worse than `p2_causal`.
  Held-out NDCG@10 -0.0032, -0.0039, -0.0004 per seed; validation NDCG@10 down
  on every seed (-0.0032, -0.0022, -0.0029); Hits@1 down on every seed (3.8%
  -> 3.3%), Hits@10 slightly up (12.1% -> 12.3%). Hiding and shuffling the
  most recent songs blurs the signal that picks the exact next song. It peaks
  later (epoch 7-8) and declines more slowly, as `p2_augment` did. The
  training losses logged for this run are ~19% too low (divided by all
  targets, not the ones used; fixed in `ddc449d`); validation and held-out
  numbers are unaffected.

**Overfitting experiments so far:** only every-position (causal) training
raised the peak. Augmentation, on windows or on chunks, slows memorization
but doesn't raise the peak, and on chunks it lowers it. `p2_causal` (0.0739
NDCG@10, 12.1% Hits@10, 3.8% Hits@1, 10.9x most-popular) is the best setup.

# Phase 3: scaling up

**Data.** `recsys/mpd_store.py` reads the 1,000 MPD JSON slices (31 GB) once
and writes a compact store (`data/mpd_store/`, 502 MB): 1,000,000 playlists,
66,346,428 track entries, 2,262,292 unique tracks, 295,860 artists, 734,684
albums; no track URI has conflicting metadata. Build: 3.6 min, 3.3 GB peak
memory. Datasets built from the store are identical, field for field, to those
built from the JSON (`v1_baseline`, `p2_causal`, `p2_genre` checked).

**Sampled softmax** (`recsys/sampled.py`). A million-song catalog can't be
scored in full for every training prediction, so each batch scores its
predictions against its own target slots plus random songs, with a logQ
correction for how often each song lands in that set. Validation and held-out
still rank the full catalog. Tested at 5,000 playlists first, against
`p2_causal`:

| Config | Seeds | NDCG@10 mean (min–max) | Change vs `p2_causal` | Hits@10 | Hits@1 | Best epochs | Min / seed |
|---|---|---|---|---|---|---|---|
| `p2_causal` (full softmax) | 3 | 0.0739 (0.0736–0.0744) | — | 12.1% (12.0%–12.2%) | 3.8% (3.8%–3.8%) | 5, 5, 4 | 5.2 |
| `p3_sampled` (1,024 random negatives) | 3 | 0.0623 (0.0608–0.0639) | -0.0116 | 10.4% (10.2%–10.5%) | 3.1% (2.9%–3.3%) | 5, 3, 3 | 3.6 |
| `p3_neg8k` (8,192 uniform negatives) | 3 | 0.0706 (0.0703–0.0707) | -0.0033 | 11.6% (11.4%–11.8%) | 3.6% (3.5%–3.8%) | 5, 4, 4 | 6.2 |
| `p3_popneg` (1,024 popularity-weighted negatives) | 3 | 0.0661 (0.0641–0.0675) | -0.0078 | 10.9% (10.9%–11.1%) | 3.3% (3.2%–3.4%) | 3, 5, 4 | 3.4 |
| `p3_neg8k_pop` (8,192 popularity-weighted negatives) | 3 | 0.0728 (0.0702–0.0747) | -0.0011 | 12.0% (11.6%–12.2%) | 3.7% (3.7%–3.8%) | 4, 5, 4 | 5.9 |

- `p3_sampled` (2026-10-07, `e2e0597`): each batch's 320 predictions are
  scored against 1,344 candidates (its target slots + 1,024 uniformly random
  songs) instead of 30,587. 16% worse on NDCG@10, on every seed (-0.0114,
  -0.0105, -0.0129); ~1.5x faster per epoch (10-11 s vs 15-16 s). The logQ
  correction isn't the problem: both models' top-10 picks have the same
  popularity profile (median 28 vs 30 training targets; 27% vs 28% from the 1%
  most popular songs). Validation loss is nearly unchanged (best 7.53 vs 7.49)
  while top-of-list ranking drops: most uniformly random negatives are songs
  the model would never rank highly anyway, so it rarely trains against the
  hard look-alikes that compete for the top 10. Training losses logged for
  this run are over the candidate set, not comparable to full-softmax runs.
- A first version de-duplicated the candidates (`torch.unique`), whose output
  size depends on the data; the GPU then had to report back to the CPU every
  step, and it ran slower than the full softmax (46 vs 28 ms per step). The
  fixed-size version runs at 11 ms per step.
- `p3_neg8k` (2026-10-07, `81ea651`): 8,192 uniform random negatives instead
  of 1,024. Better than `p3_sampled` on every seed: held-out NDCG@10 +0.0082,
  +0.0067, +0.0099; validation +0.0101, +0.0077, +0.0087. Closes 72% of the
  gap to the full softmax (-0.0033 left). ~18 ms per step vs 11; at this
  catalog size about the cost of the full softmax, but it stays constant as
  the catalog grows.
- `p3_popneg` (2026-10-07, `81ea651`): 1,024 negatives drawn in proportion to
  (how often the song is a training target)^0.75 instead of uniformly, with
  logQ computed from those probabilities. Better than `p3_sampled` on every
  seed: held-out +0.0020, +0.0035, +0.0060; validation +0.0038, +0.0060,
  +0.0038. Smaller gain than 8x the negatives, at no extra cost.
- `p3_neg8k_pop` (2026-10-07, `83717df`): both fixes, 8,192 negatives drawn by
  (target frequency)^0.75. Closes 90% of the gap to the full softmax (0.0728
  vs 0.0739). Against `p3_neg8k` alone the extra gain is uncertain: held-out
  NDCG@10 -0.0001, +0.0028, +0.0040, validation -0.0003, +0.0023, -0.0004.
  At no extra cost over `p3_neg8k`, it's the sampling setup to scale with.

**Scaling, stage 1: 50,000 playlists** (`p3_neg8k_pop` setup, read from the
store; genres from the MusicBrainz dump for all MPD artists). 40,000 / 5,000 /
5,000 playlists: 166,627 songs, 30,204 artists, 2,288,549 training targets,
validation sample of 20,000 windows (of 156,681), 159,081 held-out windows.
Not directly comparable with the 5,000-playlist rows: the catalog is 5.4x
bigger (more songs to rank against) and the held-out windows differ.

| Config | Seeds | NDCG@10 mean (min–max) | Hits@10 | Hits@1 | Most-popular NDCG@10 / Hits@10 | Best checks (epochs) | Min / seed |
|---|---|---|---|---|---|---|---|
| `p3_neg8k_pop` (5,000 playlists, 30,587 songs) | 3 | 0.0728 (0.0702–0.0747) | 12.0% | 3.7% | 0.0068 / 1.5% | 4, 5, 4 (5-6) | 5.9 |
| `p3_50k` (50,000 playlists, 166,627 songs) | 2 | **0.1060** (0.1059–0.1062) | **17.4%** | **5.4%** | 0.0034 / 0.7% | 14, 14 (3.75) | 40.6 |

- `p3_50k` (2026-10-07, `35cb591`): 10x the playlists. NDCG@10 +46% despite a
  5.4x bigger catalog; 31x the most-popular baseline (10.7x at 5,000). The two
  seeds agree to 0.0003.
- Overfitting is much milder. Validation NDCG@10 climbs for ~2.25 epochs,
  then plateaus around 0.105-0.110 until the best check at 3.75 epochs; it
  stopped at 5 epochs. Validation loss bottoms around 7.24 and then drifts up
  only slightly (to ~7.3-7.45), where at 5,000 playlists it climbed past 10.
  It ticks up in the first quarter of each new epoch (checks 8, 12, 16), when
  the model starts seeing the same playlists again.
- Pacing: ~114 s of training + ~5.5 s of validation per check (validation
  ~4.6% of run time); ~40 min per seed. The held-out evaluation scores all
  159,081 windows against all 166,627 songs, chunked on the GPU.
- Genres (dump, cutoff 5): 613 genres; 12,304 of 30,204 artists (40.7%) have
  genres, covering 90.1% of training targets; 25.7% of songs have none.

**Retrieval recall@K (step 4)**: how often the true next song is in the
retriever's top K, the most a second-stage ranker reranking those K could
find. Full held-out windows, exact search over the whole catalog.

| Config | Songs | top 10 | top 100 | top 500 | top 1,000 | top 2,000 | top 5,000 |
|---|---|---|---|---|---|---|---|
| `p3_neg8k_pop` seed 1 (5,000 playlists) | 30,587 | 11.6% | 34.2% | 61.3% | 72.5% | 82.3% | 91.8% |
| `p3_50k` seeds 1 / 2 (50,000 playlists) | 166,627 | 17.4% | 42.9% | 66.5% / 66.3% | 75.4% / 75.2% | 82.7% / 82.6% | 89.8% / 89.7% |
| most-popular (50,000 playlists) | 166,627 | 0.7% | 5.2% | 17.1% | 27.2% | 40.1% | 59.2% |

- Top-500 recall rises with more data (61% -> 66%) despite a 5.4x bigger
  catalog. Building top-500 shortlists for all 159,081 held-out windows takes
  91 s (0.57 ms per window, exact scores + top-k on the GPU).
- Misses aren't mainly rare songs. By how often the true song is a training
  target: never (0.0% of windows), 1-5 times (5.9%, 27% in the top 500),
  6-50 (25.6%, 50%), 51-500 (53.7%, 73%), over 500 (14.7%, 87%). Only 13% of
  the windows the top 500 misses have a true song seen 5 times or fewer.

**Two-stage ranking (step 5).** The part-A retriever (`p3_50k_partA`: `p3_50k`
trained on 80% of the training playlists, 32,000) builds each window's top-500
shortlist; the ranker (`r_50k`, `recsys/ranker.py`) rescores the 500 by reading
each candidate together with the 10 context songs (the retriever's layers and
weights, plus a candidate marker and position; context songs see only earlier
context songs, each candidate sees the context and itself). Its score is the
retriever's score plus a learned correction that starts at zero, so before
training it reproduces the retriever exactly (check 0 = the retriever's
validation score). It's trained on windows from the other 20% of training
playlists (8,000 playlists, 302,495 windows), which the retriever never saw:
their top-500 recall is 59.0%, close to held-out's 62.2%, so its training
shortlists look like the ones it meets on new playlists. Each training example:
the true song + 31 songs sampled from its shortlist; lr 1e-4 (fine-tuning).

| Held-out (159,081 windows, 166,627 songs) | Seeds | NDCG@10 mean (min–max) | Hits@10 | Hits@1 | Top 500 | Min / seed |
|---|---|---|---|---|---|---|
| `p3_50k` retriever (40,000 playlists) | 2 | 0.1060 (0.1059–0.1062) | 17.4% | 5.4% | 66.4% | 40.6 |
| `p3_50k_partA` retriever alone (32,000 playlists) | 2 | 0.1020 (0.1012–0.1027) | 16.4% | 5.4% | 62.2% | 36.6 |
| **`p3_50k_partA` + `r_50k` ranker (top 500)** | 2 | **0.1146 (0.1142–0.1149)** | **18.2%** | **6.1%** | 62.2% | 36.6 + 7.5 |
| most-popular | — | 0.0034 | 0.7% | 0.1% | 17.2% | — |

- `r_50k` (2026-10-07, `01c1322`): reranking adds +0.0126 NDCG@10 (+12.3%) over
  the retriever it reranks, on both seeds (+0.0130, +0.0122), with the same
  shortlist; Hits@10 +1.8 points, Hits@1 +0.7. It also beats `p3_50k`, which
  trained on 25% more playlists, by +8.1%. The ranker peaks after one pass over
  its 302,495 windows (check 5 of 11, both seeds; validation NDCG@10 0.1061 ->
  0.1202 and 0.1080 -> 0.1226) and then declines: it overfits its 8,000
  playlists after one epoch.
- Cost: shortlists for part B ~170 s, training ~7.5 min per seed, reranking
  all 159,081 held-out windows x 500 candidates ~100 s.
- This was the last 2-seed run; new runs use 1 seed.

**Training speed-up, lazy AdamW** (`recsys/lazy_adam.py`). Profiling a
`p3_50k` training step (`scripts/profile_training_step.py`): 45.7 ms per step,
58% of it the AdamW update, almost all on the per-ID tables (35.25M of 35.43M
parameters) while a batch uses ~282 of 166,627 songs as context and targets
(~8,500 with its sampled negatives). Lazy AdamW applies the AdamW update
(momentum, step, weight decay) only to the table rows a batch used; the shared
layers keep regular AdamW. Rows are worked out on the CPU, so the GPU never
reports back. With every row used it equals AdamW exactly (tested).

| Config (50,000 playlists, seed 1) | Held-out NDCG@10 | Hits@10 | Hits@1 | Top 500 | Top 2,000 | Best at | Training time |
|---|---|---|---|---|---|---|---|
| `p3_50k` (AdamW) | 0.1062 | 17.4% | 5.4% | 66.5% | 82.7% | 3.75 epochs | 39.5 min |
| `p3_50k_lazy` (lazy AdamW) | **0.1130** | 17.6% | **6.4%** | 65.6% | 82.1% | 6.0 epochs | **23.7 min** |
| `p3_50k_lazy_b128` (lazy AdamW, 128 chunks per batch) | **0.1199** | **18.3%** | **7.0%** | 65.9% | 82.3% | 6.0 epochs | **15.7 min** |
| `p3_50k_lazy_b128_lr2` (128 chunks, learning rate 2e-3) | **0.1213** | **18.6%** | **7.0%** | 66.3% | 82.4% | 6.0 epochs | 16.0 min |

- `p3_50k_lazy` (2026-10-07, `d1009b0`): +6.4% NDCG@10 and +1.0 point Hits@1,
  in 60% of the time (2.6x faster per epoch: ~43 s vs ~114 s per quarter
  epoch; it needs 6 epochs to peak instead of 3.75). One seed, but `p3_50k`'s
  two seeds differed by only 0.0003.
- Why it learns differently: with AdamW, a song row keeps moving on its
  momentum for many steps after each time it appears (each later push 0.9x the
  previous, adding up to ~10x the first step), and every row shrinks with
  weight decay every step. With lazy AdamW a row moves only when its song is in
  the batch. For rarely seen rows that acts like a much lower learning rate:
  slower learning early (validation NDCG@10 0.0509 vs 0.0693 after 1 epoch),
  a later and higher peak, slower decline after it.
- Trade-off for two-stage ranking: the deep end of the ranking is slightly
  worse (top-500 recall 65.6% vs 66.5%, top-2,000 82.1% vs 82.7%) while the
  top is better, so the ranker's ceiling would drop ~1 point.
- `p3_50k_lazy_b128` (2026-10-07, `cacb3bd`): 128 chunks per batch instead
  of 32, same learning rate. +6.1% NDCG@10 over `p3_50k_lazy` and +0.6 points
  Hits@1, in two-thirds of the time (~27 s vs ~43 s per quarter epoch); the
  peak still comes at 6 epochs. The likely reason quality rose: each step's
  predictions also compete against the batch's other ~1,280 real next songs
  (in-batch negatives), harder wrong answers than random songs. One seed.
- `p3_50k_lazy_b128_lr2` (2026-10-07, `cacb3bd`): 128 chunks and twice the
  learning rate (2e-3; x sqrt(4) for 4x the batch). A first run was stopped at
  check 12 by choice; this is a fresh rerun. Held-out NDCG@10 +0.0014 (+1.2%)
  over `p3_50k_lazy_b128`, Hits@10 +0.3 points, top-500 recall +0.4. It climbs
  faster early (validation NDCG@10 0.0610 vs 0.0438 after 1 epoch) but peaks at
  the same check (23, 6.0 epochs) and takes the same time (16.0 min), so the
  higher rate doesn't shorten training. One seed; a small gain.
- In both 128-chunk runs the best checks fall at epoch ends (checks 23 and 27
  are the top two in each): validation NDCG@10 dips at the start of each epoch,
  when the model starts revisiting the same playlists, and recovers by its end.
