# Song Recommender at Scale

[![Tests](https://github.com/qualitytoast/song-recommender-at-scale/actions/workflows/tests.yml/badge.svg)](https://github.com/qualitytoast/song-recommender-at-scale/actions/workflows/tests.yml)

**Live: [songrec.duckdns.org](https://songrec.duckdns.org)**: pick a few songs and it suggests what comes next.

A next-song recommender in PyTorch. Given up to 10 songs, a fast **retriever** narrows a catalog of over a
million songs down to 100, and a slower, more careful **ranker** orders those 100. It was built and scaled on
the Spotify Million Playlist Dataset (MPD); the live site serves the same design trained on 66 million real
listens from ListenBrainz.

It is served as a small distributed system: the catalog split across four gRPC shard servers, behind
load-balanced front desks with a shared Redis cache, request batching, overload protection and automatic
recovery from crashed or frozen shards, all in Docker Compose on Oracle Cloud's free tier, with typo-tolerant
song search and a private per-listener taste toggle. What's next: [ROADMAP.md](ROADMAP.md). Every run, in
detail: [RESULTS.md](RESULTS.md).

## Where it comes from

This is v2 of [transformer-song-recommender](https://github.com/qualitytoast/transformer-song-recommender), a
next-song Transformer built from scratch in NumPy (its own autograd, attention, LayerNorm, optimizer and
metrics), trained on 5,000 MPD playlists. v2 starts by rebuilding that model layer for layer in PyTorch and
proving the two match: v1's trained weights in the v2 model give the same outputs (max difference 5e-6) and
scores (`scripts/check_against_v1.py`). Every change after that is a separate run, measured against the one
before it.

| | v1 (NumPy) | v2 (this repo) |
|---|---|---|
| Data | 5,000 playlists, 33,770 songs | 1M playlists (1.05M songs); then 66M ListenBrainz listens (1.18M songs) |
| What a song is | Its ID only | ID + artist, album, length, genres (MusicBrainz) |
| Training | One target per 10-song window, full softmax, SGD | Next song at every position (causal attention), sampled softmax, lazy AdamW, `torch.compile` |
| Finding recommendations | Score every song | Retriever + FAISS IVF top 100, then a ranker that rescores them |
| Serving | One process, REST API | 4 gRPC shards, front desks behind nginx, Redis, batching, fault tolerance, zero-downtime updates, Docker, a webpage, public hosting |
| Held-out NDCG@10 | 0.0330 (6.7x most-popular) | 0.1731 at 1M (59x most-popular, a 31x bigger catalog) |

The core is still v1's architecture: a small Transformer (2 layers, 64-dimensional, single-head attention),
the same metrics (NDCG@10, Hits@k) and the same most-popular baseline.

## How it works

```
 Up to 10 songs
       │
       ▼
 ┌─────────────────────┐
 │ Retriever           │  2-layer Transformer
 └─────────────────────┘
       │  "what comes next" vector h
       ▼
 ┌─────────────────────┐
 │ Search (FAISS IVF)  │  100 songs with highest h · song
 └─────────────────────┘
       │  100 candidates
       ▼
 ┌─────────────────────┐
 │ Ranker              │  rescores next to the input songs,
 └─────────────────────┘  plus overlap and co-occurrence counts
       │
       ▼
 Top 10 next songs
```

**Retriever** (`recsys/model.py`, `recsys/train.py`). A 2-layer Transformer. Each song's input is the sum of
learned vectors for the song, its artist, album, length bucket and genres (the MPD models also add the
playlist name's words); a song's score is
`h · (its output vector) + bias`, one dot product per song. Training predicts the next song at every
position with causal attention (the biggest fix for overfitting), scores each batch's true next songs
against 4,096 songs drawn by popularity (sampled softmax, `recsys/sampled.py`), and updates only the song
rows a batch uses (lazy AdamW, `recsys/lazy_adam.py`: ~1.7x faster and better). `torch.compile` and fewer
sampled songs halved a 1M epoch (16 → 7.7 min), with NDCG@10 within 0.001 at 50k.

**Search** (`recsys/search.py`, `recsys/ann.py`). FAISS IVF groups the songs into 4,096 clusters and
searches the 128 nearest the query, finding 99.5% of the exact top 100 on the MPD. It runs in its own
process: FAISS and PyTorch each bundle OpenMP and can't share it.

**Ranker** (`recsys/ranker.py`, `recsys/rank_features.py`). A copy of the trained retriever that reads the
input songs and the 100 candidates together (each candidate sees the input and itself, never another
candidate). Its score is the retriever's plus a learned correction from a small network that also reads
counts the retriever can't see: input songs sharing the candidate's artist, album or genres, and how often it
came right after them in training. The correction starts at zero, so an untrained ranker reproduces the
retriever exactly. It trains on data the retriever never saw (20% of training users or playlists).

## Results

Held-out NDCG@10 (1 if the true next song is ranked first, less the lower it is, 0 outside the top 10).
Rows use different catalogs and held-out sets, so compare each with its own most-popular baseline.

| Stage | Data | Songs | NDCG@10 | × most-popular |
|---|---|---|---|---|
| v1, NumPy | 5k playlists | 33,770 | 0.0330 | 6.7x |
| PyTorch rebuild, AdamW, v1's data quirks fixed | 5k playlists | 30,587 | 0.0539 | 7.9x |
| + song features, every-position training | 5k playlists | 30,587 | 0.0739 | 10.9x |
| Sampled softmax, lazy AdamW, bigger batches | 50k playlists | 166,627 | 0.1213 | 36x |
| Two-stage: retriever alone → + ranker | 50k playlists | 166,627 | 0.1255 → 0.1484 | 44x |
| Two-stage | 200k playlists | 412,404 | 0.1344 → 0.1567 | 49x |
| **Two-stage, ranker trained on inputs of any length** | **1M playlists** | **1,053,328** | **0.1479 → 0.1731** | **59x** |
| ListenBrainz, any-length inputs, next-10 ranker | 4.9M sessions | 1,182,261 | 0.2246 → 0.2477 | 17.7x |

- **Inputs of any length.** Requests hold 1 to 10 songs; over all lengths the 1M pair scores 0.1692 (shorter
  inputs are harder: the retriever alone 0.1082 with 1 song, 0.1397 with 10).
- **Scaling.** The same settings at 200k give 0.1576; 5x the data adds +0.015 to retriever and pair alike.
  The ranker adds ~+0.024 at both sizes.
- **Songs that belong in the playlist** (50k). A ranker trained on the next 10 songs (weighted) puts 3.4% more
  of its top 10 later in the playlist and cuts the last artist's share from 41% to 30%, for -7.4% on the very
  next song; kept. Multi-token heads (from DeepSeek-V3) added nothing.
- **Real listening.** Trained from scratch on ListenBrainz (66.4M listens from 61,752 users active in the
  past year, cut into sessions; users split 80/10/10), then asked the same questions as the MPD pair on
  held-out listeners' sessions whose next song both catalogs know:

| 4,614 held-out ListenBrainz cases | Next song | Later in the session | By the last song's artist |
|---|---|---|---|
| MPD pair | 0.1996 | 9.3% | 63% |
| **ListenBrainz pair** | **0.2433** | **11.8%** | **51%** |

  BTS holds 3.5% of the training listens, yet fills 0.08% of the top 10 for sessions without BTS, the same as
  its share of their real next songs: no per-artist cap needed.
- **Personalization** (the owner's own later listening): a skip penalty from their Spotify history more than
  doubled the new songs they then listened to (1.96 → 4.40 per 100 recommendations), for 29% fewer exact
  next songs. See [Your taste](#your-taste).
- **Didn't work:** window augmentation (masking, cropping, shuffling) slowed overfitting but never raised the
  best score; uniform random negatives were clearly worse than popularity-weighted ones.

## Serving

```
 Browser / API client
          │
          ▼
 ┌──────────────────┐
 │ Caddy            │  HTTPS
 └──────────────────┘
          │
          ▼
 ┌──────────────────┐
 │ nginx            │  per-visitor rate limits, spreads load,
 └──────────────────┘  503 when every front desk is full
          │
          ▼
 ┌──────────────────┐      ┌─────────────────┐
 │ Front desks      │◄────►│ Redis (cache)   │
 │ (FastAPI)        │◄────►│ Meilisearch     │
 └──────────────────┘      └─────────────────┘
          │  gRPC, 3 rounds, all shards at once
          ▼
 ┌────────┬────────┬────────┬────────┐
 │shard 0 │shard 1 │shard 2 │shard 3 │   song j → shard j mod 4
 └────────┴────────┴────────┴────────┘
```

**From a model to a server.** `recsys/export.py` writes everything serving needs into one versioned bundle:
the models' small shared layers, one precomputed input vector per song, the search index, per-song tables,
co-occurrence counts and the serving defaults. One process (`recsys/serve.py`, FastAPI) answers a request in
~1.5 ms on the CPU, with the same top 10 as the offline evaluation for 2,000 of 2,000 held-out requests.

**The catalog split across processes** (`recsys/shard.py`, `recsys/frontdesk.py`). Each shard holds a quarter
of every per-song table (~0.5 GB) and an IVF index built on the full index's cluster centres, so merged top
lists equal the full index's. A request takes three rounds: the input songs' rows, the search on every shard
(merged by score, then song ID), the 100 candidates' rows. Messages are protobuf (arrays as raw bytes, never
pickle) and carry the bundle version, so a front desk can never mix two models. At 1, 2 and 4 shards, natively
and in Docker, the sharded top 10 equals the single-process one for every request; ~10 ms in Docker.

**When a shard fails.** 250 ms deadlines and retries; shard copies used in turn, skipping unhealthy ones; a
partial answer (flagged) from the other shards; a circuit breaker that skips a frozen shard for 1 s; Docker
restarts crashed containers and an autoheal container restarts frozen ones. Under `scripts/chaos.py`:

| Event (requests back to back) | Requests | Partial | Failed | Complete again after |
|---|---|---|---|---|
| Shard program killed | 3,043 | 13 | 0 | 1.1 s |
| Shard frozen | 2,922 | 652 | 23 (one-song inputs on that shard) | 11.2 s |

**More traffic and overload.** Front-desk copies behind nginx (least connections): 58 → 182 recommendations/s
from 1 to 3 copies. Dynamic batching (requests within 2 ms answered together, `recsys/batching.py`) and the
shared Redis cache (keyed with the model version; skipped for a second if slow or down, `recsys/cache.py`)
took one front desk from 107 to 241 recommendations/s (p99 470 → 210 ms). With 1,000 simulated browsing users
on 3 front desks: p50 15 ms, p99 182 ms, nothing failed. Past capacity, each front desk admits 64 requests and
drops any that waited 1 s, and nginx refuses the rest at once with `503 Retry-After: 1` (`recsys/admission.py`):
a separate visitor still got answers in p50 21 ms, and refusals in 1.5 ms.

**Updates without downtime.** Blue/green (`scripts/bluegreen.py`): a second stack starts, an entrance nginx
switches to it with a graceful reload, the old one drains. Under load: 6,126 requests, 0 errors, p99 14 ms.

**Webpage** (`web/`, plain HTML, CSS and JavaScript): type-ahead song search, the playlist kept in the address
bar (a link reopens the same recommendations), Spotify's embedded player (songs without a Spotify link, 17% of
the catalog, say so instead), light and dark mode, phone layouts.

### What gets recommended

Trained on the very next song, the models follow the input's *last* song, often all the way into 9 songs by
its artist. Two serving settings, each a request field whose default comes with the model (`/health`):
`max_per_artist` (at most that many of the 10 by one artist) and `mix` (blend in the retriever's guesses after
earlier songs, each weighing `mix` times the next). Tuned for the ListenBrainz pair on 5,000 held-out listening
cases (`scripts/lb_compare.py tune`), where mixing never helped and listeners really do stay on an artist:

| ListenBrainz pair | Next song NDCG@10 | Later in the session | Different artists | By the last song's artist |
|---|---|---|---|---|
| No limit, no mixing | 0.2407 | 13.5% | 4.6 | 48% |
| 2 per artist, mix 0.5 (the MPD pair's default) | 0.2058 | 7.4% | 7.7 | 13% |
| **3 per artist, no mixing (the default)** | **0.2321** | **8.9%** | **7.0** | **21%** |

### Song search

The search box uses Meilisearch, an open-source search engine in its own container (`recsys/song_search.py`),
with one index per model version. Typed words match in any order across title, artist and album, the last as
a prefix, with a typo in words of 4+ letters (two in 8+). Results are ranked by words matched, then fewer
typos, then title or artist before album-only matches, then most listens. "one more tpme", "radiohed creep",
"daft punk one more" and "discovery daft punk" (an album) found nothing before and now find the right songs.
Per keystroke: p50 9 ms, p99 26 ms on the server. If Meilisearch errors or takes over 250 ms, the box answers
from the front desk's own name index (names starting with the text) and skips Meilisearch for a second.

### Your taste

A request can name a taste profile (`"profile": <key>`) built from one listener's Spotify Extended Streaming
History (`recsys/personal.py`, `recsys/taste.py`); without one, or with an unknown key, the answer is the
usual one, and personalized answers have their own cache entries. The webpage keeps a key given once as
`?me=<key>` in the browser (never in shared links) and shows a "Use my taste" toggle, off by default. A
profile can pull the search toward the listener's average taste, boost artists they play, and push down
songs they usually skip. Tuned on the owner's later listening (6 weeks to choose, 6 to report; 70% weight on
new songs they went on to listen to, 30% on their next song), only the skip penalty helped: 43% of the top 10
new to them instead of 17%, and 4.40 instead of 1.96 new songs per 100 that they then listened to (95% range
of the gain +1.8 to +3.1).

### Hosting

Live at **https://songrec.duckdns.org** on one Oracle Cloud Arm machine (2 cores, 8 GB, Always Free; quota
rules make Oracle refuse anything beyond the free allowance, and a budget emails on any spend), running the
whole Compose system (~4.3 GB). `docker-compose.oracle.yml` adds Caddy for HTTPS (Let's Encrypt, renewed by
itself). Each visitor may ask for 5 recommendations and 20 searches a second. On the server, 2,000 of 2,000
held-out requests gave the offline top 10 in order (p50 29 ms, p99 74 ms); a model switch is a restart on the
new bundle, ~10-30 s of "busy".

```bash
ssh ubuntu@<server> 'cd song-recommender-at-scale && git pull && docker compose -f docker-compose.yml -f docker-compose.oracle.yml up -d --build'
# a new model: copy its bundle, index its songs (BUNDLE=<its folder> docker compose ... run --rm search-index),
# then point .env's BUNDLE at it
```

## Running it

Python 3.13; `pip install -r requirements.txt`. Docker for the containerized system.

**MPD data.** Download the MPD from
[AIcrowd](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge) into `data/mpd/`
(1,000 JSON slices, 31 GB), then build the compact store (~4 min, 502 MB) and genres:

```bash
python -m recsys.mpd_store --raw data/mpd --out data/mpd_store
python scripts/genres_from_dump.py --store data/mpd_store --out data/genres/musicbrainz_all_mpd_artists.jsonl
```

**ListenBrainz data.** A full listens dump (~258 GB, one `.tar.zst`) from
[ListenBrainz's dumps](https://data.metabrainz.org/pub/musicbrainz/listenbrainz/fullexport/), then a 12-month
extract (~1 h, 14 GB), MusicBrainz genres for its artists, and the training store (~5 min):

```bash
python scripts/parallel_download.py <dump url> --out ~/Downloads/<file> --connections 16   # resumable
python -m recsys.lb_extract --source ~/Downloads/<file> --start 2025-10-15 --end 2026-10-15 --out data/lb_extract
python scripts/genres_from_dump.py --file artist.tar.xz --listenbrainz data/lb_extract --out data/genres/musicbrainz_lb_artists.jsonl
python -m recsys.lb_store --extract data/lb_extract --out data/lb_store
```

**Train and evaluate.** Every run is a config in `configs/` (no hidden defaults: a missing setting is an
error), with results in `runs/<group>/<config>/seed<k>/`. Names follow the phases: `v1_*`, `p2_*` (features,
overfitting), `p3_*` (scaling), then `retriever_<size>[_variant]` and `ranker_<size>[_variant]` (e.g.
`retriever_1m_lb`, `ranker_1m_lb_next10w`). A ranker starts from its retriever's checkpoint.

```bash
python -m recsys.train    --config configs/retriever_1m_lb.toml       # retriever (~2 h on an M-series Mac)
python -m recsys.evaluate --config configs/retriever_1m_lb.toml       # held-out NDCG@10, Hits@k, recall
python -m recsys.ranker   --config configs/ranker_1m_lb_next10w.toml  # ranker: train, then evaluate (~2 h)
scripts/run_queue.sh configs/a.toml configs/b.toml                    # runs one after another
python -m pytest
```

**Serve.**

```bash
python -m recsys.export --ranker configs/ranker_1m_lb_next10w.toml --max-per-artist 3 --mix 0
python -m recsys.shard_slices --bundle bundles/<bundle> --shards 4
docker compose up -d --build                      # http://localhost:8000 (MEILI_MASTER_KEY in .env)
docker compose run --rm search-index              # the song search's index, once per bundle
python scripts/serving_check.py cases --ranker configs/ranker_1m_lb_next10w.toml --out runs/serving/lb_cases
python scripts/serving_check.py serve --cases runs/serving/lb_cases --url http://127.0.0.1:8000 --label docker
python -m recsys.serve --bundle bundles/<bundle>  # or: one process, no Docker
```

Also: `scripts/chaos.py`, `scripts/load_test.py`, `scripts/bluegreen.py` (failure, load and update tests),
`scripts/lb_compare.py` (MPD vs ListenBrainz pairs, serving settings), `scripts/search_check.py`
(song search). In `docker-compose.yml`, `BUNDLE` picks the bundle; empty `REDIS_URL=` or `SEARCH_URL=` turn
the cache or Meilisearch off, `BATCH=0` batching.

**Personalize.** From a Spotify Extended Streaming History zip:

```bash
python -m recsys.personal plays --export <zip> --bundle bundles/<bundle>   # match plays, cut sessions
python scripts/personal_eval.py --bundle bundles/<bundle>                  # tune and measure (held-out weeks)
python -m recsys.personal profile --bundle bundles/<bundle> --beta 7.02    # the served profile, secret key
```

## Layout

| Path | What's there |
|---|---|
| `recsys/data.py`, `recsys/mpd_store.py` | Playlists or sessions → vocab, splits, windows, features; the compact MPD store |
| `recsys/listenbrainz.py`, `recsys/lb_extract.py`, `recsys/lb_sessions.py`, `recsys/lb_store.py` | ListenBrainz: reading the dump, the 12-month extract, session rules, the training store |
| `recsys/model.py`, `recsys/train.py`, `recsys/evaluate.py` | The retriever, training with early stopping, held-out evaluation |
| `recsys/sampled.py`, `recsys/lazy_adam.py`, `recsys/augment.py` | Sampled softmax, lazy AdamW, window augmentation |
| `recsys/ranker.py`, `recsys/rank_features.py` | The ranker: shortlists, extra inputs, training, evaluation |
| `recsys/search.py`, `recsys/ann.py`, `recsys/search_worker.py` | Exact search; FAISS indexes in their own process |
| `recsys/export.py`, `recsys/recommender.py`, `recsys/serve.py` | The versioned bundle, answering requests, the FastAPI server |
| `recsys/shard.py`, `recsys/shard_slices.py`, `recsys/frontdesk.py`, `recsys/wire.py`, `recsys/proto/` | Shards and their slices, the front desk, arrays in messages, the gRPC service |
| `recsys/batching.py`, `recsys/cache.py`, `recsys/admission.py`, `recsys/healthcheck.py`, `recsys/launch.py` | Batching, the Redis cache, overload limits, health checks, the one-command launcher |
| `recsys/song_search.py`, `recsys/personal.py`, `recsys/taste.py` | Song search (Meilisearch); personalization |
| `recsys/metrics.py`, `recsys/baselines.py` | NDCG@k, Hits@k; most-popular baseline |
| `docker/`, `docker-compose.yml`, `docker-compose.oracle.yml`, `web/` | The image, nginx, blue/green stacks, the server's Caddy; the webpage |
| `scripts/`, `configs/`, `tests/`, `.github/workflows/` | Tools and checks; one config per run; tests; CI on every pull request |
| `RESULTS.md`, `ROADMAP.md` | Every run and what was learned; done and planned work |

Data: ListenBrainz listens (CC0); genres from MusicBrainz (CC BY-NC-SA); the MPD isn't redistributed.
