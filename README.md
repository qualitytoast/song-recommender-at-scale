# Song Recommender at Scale

A next-song recommender in PyTorch: given the last 10 songs of a playlist, it
predicts the next one. It is trained on the Spotify Million Playlist Dataset
(MPD), uses song and playlist features, and works in two stages: a fast
retriever narrows hundreds of thousands of songs down to 500, then a slower,
more careful ranker orders those 500.

## Where it comes from

This is v2 of
[transformer-song-recommender](https://github.com/qualitytoast/transformer-song-recommender),
a next-song Transformer I built from scratch in NumPy, including its own
autograd engine, attention, LayerNorm, optimizer and metrics, trained on 5,000
MPD playlists and served as a REST API.

v2 starts by rebuilding that model layer for layer in PyTorch and proving the
two match: v1's trained weights, loaded into the v2 model, give the same
outputs (max difference 5e-6) and the same scores
(`scripts/check_against_v1.py`). Every change after that is a separate run,
measured against the one before it.

| | v1 (NumPy) | v2 (this repo) |
|---|---|---|
| Framework | Hand-written autograd in NumPy, CPU | PyTorch, Apple GPU (mps) |
| Data | 5,000 playlists, 33,770 songs | 200,000 playlists, 412,404 songs so far; 1M next |
| What a song is | Its ID only | ID + artist, album, length, genres (MusicBrainz); plus the playlist's name |
| Training | One target per 10-song window, full softmax, SGD | Next song predicted at every position (causal attention), sampled softmax, lazy AdamW |
| Finding recommendations | Score every song | Retriever + FAISS IVF search for the top 500, then a ranker that rescores them |
| Held-out NDCG@10 | 0.0330 (6.7x most-popular) | 0.1257 (39x most-popular; retriever alone, 200k, a 12x bigger catalog) |

The v1 model's architecture is still the core: a small Transformer (2 layers,
64-dimensional, single-head attention), the same metrics (NDCG@10, Hits@k) and
the same most-popular baseline.

## How it works

```
last 10 songs + playlist name
        │
        ▼
 ┌──────────────┐   playlist vector h
 │  Retriever   │───────────────────────┐
 └──────────────┘                       ▼
                         ┌───────────────────────────────┐
                         │  Search (FAISS IVF, separate  │  top 500 of 412,404 songs
                         │  process): highest h · song   │  (98.6% of the exact top 500)
                         └───────────────────────────────┘
                                         │
                                         ▼
                                 ┌──────────────┐
                                 │    Ranker    │  rescores the 500, reading each
                                 └──────────────┘  candidate next to the 10 songs
                                         │
                                         ▼
                                  top 10 next songs
```

**Retriever** (`recsys/model.py`, `recsys/train.py`). A 2-layer Transformer
over the 10 songs. Each song's input is the sum of learned vectors for the song,
its artist, album, length bucket and genres; the playlist name's words are
averaged and added at every position. The last position's output is the
playlist vector `h`, and a song's score is `h · (its output vector) + bias`, so
scoring is one dot product per song. Training:

- *Every position:* playlists are cut into chunks and the next song is predicted
  at every position, with causal attention (each position sees only earlier
  songs). This was the biggest fix for overfitting.
- *Sampled softmax* (`recsys/sampled.py`): instead of scoring every song for
  every prediction, each batch scores its own true next songs plus 8,192 random
  songs drawn by popularity, with a correction for how often each song is drawn.
- *Lazy AdamW* (`recsys/lazy_adam.py`): a batch uses a few thousand of the
  412k song rows, so only those rows are updated; this made training ~1.7x
  faster and better.

**Search** (`recsys/search.py`, `recsys/ann.py`, `recsys/search_worker.py`).
Finding the 500 highest dot products. Exact search scores every song; FAISS IVF
groups songs into 2,048 clusters and searches only the 128 nearest the query.
FAISS runs in its own process because it and PyTorch each bundle a copy of the
OpenMP library and can't share one. `scripts/search_benchmark.py` compares exact
search, FAISS Flat, IVF and HNSW, with and without sharding.

**Ranker** (`recsys/ranker.py`). A copy of the trained retriever that reads the
10 context songs and the 500 candidates together, so attention can relate each
candidate to each song in the playlist. Context songs see only earlier context
songs; each candidate sees the context and itself, never another candidate, so
candidates are scored independently. Its score is the retriever's score plus a
learned correction that starts at zero, so before training it reproduces the
retriever exactly. It trains on 20% of the training playlists that the
retriever never saw, so its shortlists look like those of new playlists.

## Results

Held-out NDCG@10 (1 if the true next song is ranked first, less the lower it
is, 0 outside the top 10). Rows use different catalogs and held-out sets, so
compare each with its own most-popular baseline. Full tables and notes are in
[RESULTS.md](RESULTS.md).

| Stage | Playlists | Songs | NDCG@10 | × most-popular |
|---|---|---|---|---|
| v1, NumPy | 5,000 | 33,770 | 0.0330 | 6.7x |
| PyTorch rebuild, AdamW + fixes to v1's data split and quirks | 5,000 | 30,587 | 0.0539 | 7.9x |
| + song and playlist features, every-position training | 5,000 | 30,587 | 0.0739 | 10.9x |
| Sampled softmax, lazy AdamW, bigger batches | 50,000 | 166,627 | 0.1213 | 36x |
| Two-stage at 50k: retriever (80% of training playlists) alone → + ranker | 50,000 | 166,627 | 0.1020 → 0.1146 | 34x |
| Retriever at 50k, current settings (80% of training playlists) | 50,000 | 166,627 | 0.1172 | 34x |
| Two-stage at 50k: that retriever (IVF shortlist) alone → + improved ranker | 50,000 | 166,627 | 0.1171 → 0.1351 | 40x |
| + input songs left out of the shortlist, small-network correction | 50,000 | 166,627 | 0.1255 → 0.1484 | 44x |
| Retriever at 200k (80% of training playlists) | 200,000 | 412,404 | 0.1257 | 39x |
| Two-stage at 200k: retriever alone (IVF shortlist) → + ranker | 200,000 | 412,404 | 0.1255 → 0.1255 (no gain yet) | 39x |

Some things that didn't work are recorded too: window augmentation (masking,
cropping, shuffling songs) slowed overfitting but never raised the best score;
uniform random negatives were clearly worse than popularity-weighted ones.

## Status

- [x] Phase 1: rebuild v1 in PyTorch and match it; AdamW; fix v1's data quirks
- [x] Phase 2: song and playlist features, one at a time; overfitting experiments
- [x] Phase 3 so far: compact data store, sampled softmax, 50k and 200k
      retrievers, two-stage ranking at 50k, faster training, search benchmark
- [x] Ranker at 200k (with FAISS IVF shortlists): no gain over the retriever, and none
      at 5k or 50k with the same retriever settings (its 50k gain was on an older retriever)
- [x] Ranker improvements at 50k: +15.4% over the retriever (in-shortlist training windows,
      top-100 reranking, overlap and co-occurrence inputs); then input songs left out and a
      small-network correction: 0.1484 held-out NDCG@10; 200k confirmation next
- [ ] Serving: a search service holding the song catalog, a model that calls it
- [ ] All 1M playlists
- [ ] Personalization to one listener's history (add distillation?)

## Running it

Python 3.13; `pip install -r requirements.txt` (PyTorch, NumPy, FAISS).

**Data.** Download the MPD from
[AIcrowd](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)
into `data/mpd/` (1,000 JSON slices, 31 GB), then build the compact store once
(~4 min, 502 MB):

```bash
python -m recsys.mpd_store --raw data/mpd --out data/mpd_store
# genres for every MPD artist, from the MusicBrainz JSON dump (~20 min, streamed)
python scripts/genres_from_dump.py --store data/mpd_store --out data/genres/musicbrainz_all_mpd_artists.jsonl
```

**Train and evaluate.** Every run is a config in `configs/` (no hidden
defaults: a missing setting is an error); results go to
`runs/<group>/<config>/seed<k>/` (e.g.
`runs/retriever_ranker_runs/retriever_200k/seed1/`).

```bash
python -m recsys.train    --config configs/retriever_200k.toml   # retriever (~1 h on an M-series Mac)
python -m recsys.evaluate --config configs/retriever_200k.toml   # held-out NDCG@10, Hits@k, recall@K
python -m recsys.ranker   --config configs/ranker_200k.toml          # ranker: train, then evaluate
python scripts/search_benchmark.py real --config configs/retriever_200k.toml
python -m pytest                                                # tests
```

Config names follow the phases: `v1_*` (Phase 1), `p2_*` (features and
overfitting), `p3_*` (scaling), `retriever_<size>` and `ranker_<size>` (two-stage
pairs: `ranker_200k` is trained on `retriever_200k`; earlier rankers are
`retriever_ranker_*`). Runs go in matching
folders: `runs/v1_runs/`, `runs/p2_runs/`, `runs/p3_runs/`, `runs/retriever_ranker_runs/`.

## Layout

| Path | What's there |
|---|---|
| `recsys/data.py`, `recsys/mpd_store.py` | Playlists → vocab, splits, windows, chunks, features; the compact MPD store |
| `recsys/model.py` | The retriever (`SongRecommender`) |
| `recsys/train.py`, `recsys/evaluate.py` | Training loop with early stopping; held-out evaluation |
| `recsys/sampled.py`, `recsys/lazy_adam.py`, `recsys/augment.py` | Sampled softmax, lazy AdamW, window augmentation |
| `recsys/ranker.py` | The second-stage ranker: shortlists, training, evaluation |
| `recsys/search.py`, `recsys/ann.py`, `recsys/search_worker.py` | Exact search; FAISS indexes in their own process |
| `recsys/metrics.py`, `recsys/baselines.py` | NDCG@k, Hits@k; most-popular baseline |
| `scripts/` | v1 equivalence check, genre fetching, profiling, search benchmark |
| `configs/`, `tests/` | One config per run; tests |
| `RESULTS.md` | Every run, with what changed and what was learned |

Genre data: MusicBrainz (CC BY-NC-SA).
