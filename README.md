# Song Recommender at Scale

[![Tests](https://github.com/qualitytoast/song-recommender-at-scale/actions/workflows/tests.yml/badge.svg)](https://github.com/qualitytoast/song-recommender-at-scale/actions/workflows/tests.yml)

A next-song recommender in PyTorch: given the last 10 songs of a playlist, it
predicts the next one. It is trained on the Spotify Million Playlist Dataset
(MPD), uses song and playlist features, and works in two stages: a fast
retriever narrows a million songs down to 100, then a slower, more careful
ranker orders those 100.

It is served as a small distributed system: the million-song catalog is split
across four gRPC shard servers, behind load-balanced front desks with a shared
Redis cache, request batching, overload protection and automatic recovery from
crashed or frozen shards, all in Docker Compose, with a webpage on top.

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
| Data | 5,000 playlists, 33,770 songs | All 1,000,000 playlists, 1,053,328 songs |
| What a song is | Its ID only | ID + artist, album, length, genres (MusicBrainz); plus the playlist's name |
| Training | One target per 10-song window, full softmax, SGD | Next song predicted at every position (causal attention), sampled softmax, lazy AdamW |
| Finding recommendations | Score every song | Retriever + FAISS IVF search for the top 100 (songs already in the input left out), then a ranker that rescores them |
| Serving | One process behind a REST API | The catalog split across 4 gRPC shard servers; front-desk copies behind nginx; Redis cache, batching, fault tolerance, zero-downtime updates; Docker Compose; a webpage |
| Held-out NDCG@10 | 0.0330 (6.7x most-popular) | 0.1731 on 10-song windows like v1's; 0.1692 on inputs of any length, 1 to 10 songs (59x most-popular; retriever + ranker, 1M, a 31x bigger catalog) |

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
                         │  Search (FAISS IVF, separate  │  top 100 of 1,053,328 songs,
                         │  process): highest h · song   │  songs in the input left out
                         └───────────────────────────────┘
                                         │
                                         ▼
                                 ┌──────────────┐
                                 │    Ranker    │  rescores the 100, reading each
                                 └──────────────┘  candidate next to the 10 songs,
                                                   plus overlap and co-occurrence counts
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
  every prediction, each batch scores its own true next songs plus 4,096 random
  songs drawn by popularity, with a correction for how often each song is drawn.
- *Lazy AdamW* (`recsys/lazy_adam.py`): a batch uses a few thousand of the
  million song rows, so only those rows are updated; this made training ~1.7x
  faster and better.
- *Speed-ups for 1M:* `torch.compile` on the training step (it fuses many small
  GPU operations into fewer) and 4,096 instead of 8,192 random songs cut a 1M
  epoch from 16 to 7.7 min, with held-out NDCG@10 within 0.001 at 50k.

**Search** (`recsys/search.py`, `recsys/ann.py`, `recsys/search_worker.py`).
Finding the 100 highest dot products. Exact search scores every song; FAISS IVF
groups the 1M songs into 4,096 clusters (2,048 at 50k and 200k) and searches only
the 128 nearest the query, which finds 99.5% of the exact top 100.
FAISS runs in its own process because it and PyTorch each bundle a copy of the
OpenMP library and can't share one. `scripts/search_benchmark.py` compares exact
search, FAISS Flat, IVF and HNSW, with and without sharding.

**Ranker** (`recsys/ranker.py`, `recsys/rank_features.py`). A copy of the trained
retriever that reads the 10 context songs and the 100 candidates together, so
attention can relate each candidate to each song in the playlist. Context songs
see only earlier context songs; each candidate sees the context and itself, never
another candidate, so candidates are scored independently. Its score is the
retriever's score plus a learned correction: a small network that reads the
transformer's output for the candidate and a few counts the retriever can't see,
such as how many of the context songs share its artist, album or genres, and how
often it came right after them in the retriever's training playlists. The
correction starts at zero, so before training the ranker reproduces the retriever
exactly. It trains on the 20% of training playlists that the retriever never saw,
on the windows whose next song is in the shortlist, each against 31 wrong answers
drawn from that shortlist.

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
| Two-stage at 200k with the improved ranker: retriever alone (input songs left out) → + ranker | 200,000 | 412,404 | 0.1344 → 0.1567 | 49x |
| Two-stage at 200k with the 1M settings: retriever alone (input songs left out) → + ranker | 200,000 | 412,404 | 0.1337 → 0.1576 | 49x |
| Retriever at 1M (80% of training playlists; faster training settings) | 1,000,000 | 1,053,328 | 0.1397 | 49x |
| Two-stage at 1M: retriever alone (input songs left out) → + ranker | 1,000,000 | 1,053,328 | 0.1479 → 0.1725 | 59x |
| **Two-stage at 1M, ranker trained on inputs of any length** | **1,000,000** | **1,053,328** | **0.1479 → 0.1731** | **59x** |

The 1M numbers score all 4,938,520 held-out windows. Day to day, runs at 200k
and up score a fixed sample of 200,000 of them, which read within 0.001 of the
full set.

**Inputs of any length.** Every number in the table is on 10-song windows. A
request can hold 1 to 10 songs, so held-out cases now also include every song
with fewer songs before it (6,242,133 cases at 1M; 79% still have 10 songs). On
all of them the 1M retriever scores 0.1363 and the final pair **0.1692**:
shorter inputs are harder (the retriever alone: 0.1082 with 1 song, 0.1397 with
10). The ranker retrained on inputs of every length beats the one trained on
10-song windows by +0.0079 with 1 song, +0.0011 overall, and isn't worse at 10.

**What scaling to 1M adds.** The same settings trained on 200k playlists give a
retriever of 0.1245 and a pair of 0.1576 (all 836,433 of its held-out windows).
At 1M the retriever gains +0.0152 (+12%) and the pair +0.0149 (+9.5%), with 2.6x
more songs to choose from. The ranker adds about the same at both sizes (+0.024),
so the gain comes from the better retriever, carried through.

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
      small-network correction: 0.1484 held-out NDCG@10
- [x] 200k confirmation of the improved ranker: 0.1567 held-out NDCG@10 (49x most-popular),
      +16.6% over the retriever on the same shortlists
- [x] Training 2x faster before 1M (torch.compile, fewer random negatives), NDCG@10 within 0.001 at 50k
- [x] All 1M playlists: retriever 0.1397, retriever + ranker 0.1725 held-out NDCG@10
      (59x most-popular), the ranker +16.6% over the retriever on the same shortlists
- [x] Scaling check: the same settings at 200k give 0.1576, so 5x more data adds +0.015
- [x] Inputs of any length (1 to 10 songs) in validation and testing, and a ranker retrained on
      them: 0.1692 held-out NDCG@10 over all lengths
- [x] Serving, phase A: the 1M pair behind a FastAPI REST API: the same top 10 as the offline
      evaluation for 2,000 of 2,000 requests, ~1.5 ms per request on the CPU
- [x] Serving, phase B: the song catalog split across 4 gRPC shard servers (same answers as one
      process for 2,000 of 2,000 requests), Docker Compose, fault tolerance, front-desk copies behind
      nginx, zero-downtime updates, a shared Redis cache, request batching, overload protection
- [x] A webpage: type-ahead song search, shareable playlist links, Spotify's embedded player
- [x] More varied recommendations: at most 2 songs per artist, and guesses from the whole playlist
      mixed in, not only from its last song
- [ ] Public hosting on Oracle Cloud's free tier (waiting for a free Arm machine)
- [ ] Retrain for "songs that belong in this playlist" (any of the next several songs, not only the
      very next one)
- [ ] Train on ListenBrainz user streaming history
- [ ] Personalization to one listener's history (add distillation?)

## Serving

### One process (phase A)

```bash
python -m recsys.export --ranker configs/ranker_1m_any_length.toml   # once: writes bundles/<name>-<version>/ (2.2 GB)
python -m recsys.serve --device cpu                                   # the newest bundle, on http://127.0.0.1:8000
curl -s -X POST localhost:8000/recommend -H 'content-type: application/json' \
  -d '{"songs": ["Nightcall - Kavinsky", "spotify:track:6HZILIRieu8S0iqY8kIKhj"], "name": "night drive"}'
curl -s 'localhost:8000/songs?q=nightc'   # type-ahead: songs starting with "nightc", most-played first
```

A request is a playlist's songs, as Spotify track URIs or `"Track - Artist"`
text (any number; the last 10 known ones are read), and optionally its name.
`GET /songs` finds songs as the user types: track or artist names starting with
the text, case and accents ignored, in the most playlists first, so typing
"nightcall" lists Kavinsky's original before the covers and remixes (~1 ms). The answer is the top 10 songs with
track, artist and score, plus the songs it left out (not in the catalog), the
model version, and the time each stage took. The export (`recsys/export.py`)
writes everything the server needs into one versioned folder: each model's
small shared layers, one precomputed input vector per song (its song, artist,
album, length and genre vectors added up once), the saved search index, the
ranker's per-song tables and the appeared-together counts. The server
(`recsys/serve.py`, FastAPI + uvicorn; the work is in `recsys/recommender.py`)
loads it in 1-2 s. The webpage is at `/`, and FastAPI's page at `/docs` lets
you try requests in a browser.

Checked on 2,000 held-out requests of every length: the served top 10 equals
the offline evaluation's in every case, and a request takes ~1.5 ms end to end
on the CPU (~7x faster than on the Mac GPU, which only pays off with big
batches).

### What gets recommended

Trained to predict the very next song, the models mostly follow a playlist's
*last* song, and since people often add several songs by one artist in a row,
the 10 individually most likely next songs are often all by that artist (after
a varied 10-song playlist ending in Daft Punk: 9 Daft Punk songs). Two serving
settings fix that without retraining, each a request field:

- `max_per_artist` (default 2): going down the ranker's order of all 100
  candidates, a song is skipped once its artist has 2 songs in the top 10.
- `mix` (default 0.5): the retriever makes a "what comes next" guess after
  every song of the playlist, not just the last (that's how it was trained).
  The playlist vector becomes a weighted average of those guesses: the last
  song's weighs 1, the one before it 0.5, then 0.25, and so on. Scores are
  dot products, so this averages each song's scores the same way. Mixing the
  ranker's guesses too scored worse.

Judged on 5,000 held-out cases (`scripts/playlist_eval.py`) by the usual
question (is the true next song in the top 10?) and a new one, *later in the
playlist*: how many of the 10 the playlist actually goes on to contain.

| Setting | Next song in the top 10 | Later in the playlist | Different artists | By the last song's artist |
|---|---|---|---|---|
| Neither (the models as trained) | 24.1% | 16.6% | 5.7 | 44.3% |
| Artist limit only | 19.8% | 15.0% | 8.3 | 15.3% |
| Mixing only | 22.4% | 17.7% | 6.7 | 29.7% |
| **Both (the default)** | **18.5%** | **15.8%** | **8.5** | **11.6%** |

On varied playlists (3+ songs, every one by a different artist) the default
raises "later in the playlist" from 13.0% to 14.8% and cuts the last artist's
share from 24.5% to 6.6%. The artist limit trades some measured accuracy for
variety: many real playlists do repeat artists. `"max_per_artist": 0, "mix": 0`
gives the models' own answers, which the checks against the offline evaluation use.

### The catalog split across processes (phase B)

```
                    browser (webpage) / API clients
                                 │ HTTP
                                 ▼
                        ┌─────────────────┐  least-connections load balancing; answers 503 itself
                        │      nginx      │  when every front desk is full
                        └─────────────────┘
                          │             │
                          ▼             ▼
                 ┌──────────────┐ ┌──────────────┐        ┌───────┐
                 │ front desk 1 │ │ front desk 2 │  ...   │ Redis │  one answer cache for all of them
                 └──────────────┘ └──────────────┘        └───────┘
                     FastAPI: the models' small shared layers, song names, batching, admission limits
                          │ gRPC: 3 rounds per request, each sent to the shards involved at once
        ┌──────────────┬──┴───────────┬──────────────┐
        ▼              ▼              ▼              ▼
   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐   shard i: the songs with ID % 4 == i, and
   │ shard 0 │    │ shard 1 │    │ shard 2 │    │ shard 3 │   their search index, input vectors, artist,
   └─────────┘    └─────────┘    └─────────┘    └─────────┘   album, genres and appeared-together counts
```

**Shards** (`recsys/shard.py`, `recsys/shard_slices.py`). Song *j* lives on
shard *j* % 4. Each shard is a gRPC server holding everything per-song for its
quarter of the catalog (~0.5 GB): an IVF index of its songs' search vectors,
built on the full index's cluster centres so that the shards' top lists merged
are exactly the full index's top list, both models' input vectors, the songs'
artist, album and genres, and the appeared-together counts whose later song it
holds.

**Front desk** (`recsys/frontdesk.py`). Runs both stages, the retriever's
transformer and the ranker, but holds only the models' small shared layers and
the song names: the per-song tables and the search live on the shards, so the
two-stage system is the front desk and the shards together. A request takes three rounds, each
sent to every shard involved at the same time: (1) the input songs' rows,
(2) the playlist vector, searched on every shard, the top lists merged (by
score, then song ID), (3) the 100 candidates' rows and appeared-together counts.
The models then run on a table of just this request's ~110 songs, the same
code and numbers as with the whole catalog. Messages are protobuf
(`recsys/proto/shard.proto`; arrays as raw bytes + type + shape, never pickle),
and every one carries the bundle version: a shard refuses another version, so
a front desk can never mix two models.

gRPC (HTTP/2 + protobuf) was chosen for its built-in deadlines, status codes,
retries and health checks. For 2,000 held-out requests at 1, 2 and 4 shards,
natively and in Docker, the sharded top 10 equals the single-process one in
every case. A request takes ~10 ms in Docker, against ~1.5 ms in one process:
the price of three network rounds.

### Containers

One Docker image (`docker/Dockerfile`: CPU PyTorch, model files mounted rather
than baked in) runs every service in `docker-compose.yml`: 4 shards, the front
desk (scaled with `--scale frontdesk=N`), nginx, Redis, and an autoheal
container. Each shard and front desk has a health check every 2 s; a container
whose program dies is restarted by Docker, and one that stops answering its
health check (frozen, not dead) is restarted by autoheal. One PyTorch thread
per front desk (`OMP_NUM_THREADS=1`) halved a single request's work in a
container (17 → 8 ms). `python -m recsys.launch --shards 4` starts the same
system without Docker.

### When a shard fails

- **Deadlines and retries:** every call to a shard has a 250 ms deadline; a
  call to a shard that's down is retried up to twice.
- **Copies:** a shard can run as several copies, used in turn, skipping any
  whose health check fails.
- **Partial answers:** if every copy of a shard fails, the answer is built from
  the other shards and flagged (`"partial": true`, `"missing_shards"`). A request
  fails (503) only if no shard holding its input songs answered.
- **Circuit breaker:** a shard that let a call run into its deadline is skipped
  for 1 s, so a frozen shard costs one slow request instead of slowing every one.
- **Automatic restarts:** Docker restarts a crashed shard; autoheal restarts a
  frozen one.

`scripts/chaos.py` breaks shards while requests flow, one after another (cache off):

| Event | Requests | Partial answers | Failed | Complete answers again after |
|---|---|---|---|---|
| Shard 1's program killed (Docker restarts it) | 3,043 | 13 | 0 | 1.1 s |
| Shard 2 frozen (autoheal restarts it) | 2,922 | 652 | 23 | 11.2 s |

The 23 failures were one-song playlists whose only song lives on the frozen
shard. Through the freeze, requests took p50 13 ms; before the circuit breaker,
every request during a freeze waited ~520 ms.

### More traffic

- **Front-desk copies behind nginx** (least connections): load tests
  (`scripts/load_test.py`, locust) with 64 users sending requests back to back:
  1 front desk tops out at 58 recommendations/s, 3 copies reach 182/s (p50
  220 ms, p99 310 ms), with no failures.
- **Dynamic batching** (`recsys/batching.py`): requests arriving within 2 ms of
  each other (up to 16) are answered together, each shard asked once per round
  for the whole batch.
- **A shared Redis cache** (`recsys/cache.py`): one cache for every front desk,
  keyed with the model version (a new version never serves old answers);
  partial answers are never stored. If Redis is slow or down, a request waits at
  most 100 ms and Redis is skipped for the next second.

| One front desk, 64 users | Recommendations/s | p50 | p99 |
|---|---|---|---|
| Neither | 107 | 370 ms | 470 ms |
| Batching | 180 | 180 ms | 280 ms |
| Cache | 226 | 10 ms | 2.2 s |
| Both | 241 | 15 ms | 210 ms |

(The cache hit 98% of requests, because the test users cycle through 2,000
playlists; real traffic would repeat less. The two load-test sets were measured
in separate sessions, so compare within each.)

With 1,000 simulated users on 3 front desks (batching and cache on; locust
running on the same Mac): users browsing with 2-8 s between actions got
recommendations in p50 15 ms, p99 182 ms, and the type-ahead menu in p50 2 ms,
p99 57 ms; users clicking non-stop got p50 455 ms, p99 1.15 s at 205
recommendations/s. No request failed in either. (nginx's defaults had dropped
half the requests at 1,000 users: now a worker per core and 8,192 connections
each.)

### Overload

When more requests arrive than the front desks can answer, waiting in line only
makes every answer late. So each front desk admits at most 64 recommendation
requests at once and drops any that waited 1 s without starting
(`recsys/admission.py`), and nginx admits at most 96 per front desk: the rest
get `503` with `Retry-After: 1` at once, and the webpage retries once. One front
desk under 1,000 non-stop users: a separate visitor (`scripts/probe.py`) got
answers in p50 21 ms (p99 119 ms) and refusals in 1.5 ms; before nginx did the
refusing, a refusal took ~800 ms. The container health check (`/live`) never
waits behind requests, so a busy front desk isn't mistaken for a frozen one.

### Updating without downtime

Blue/green (`scripts/bluegreen.py`, `docker/bluegreen/`): the new version
starts as a complete second stack, an entrance nginx switches to it with a
graceful reload (requests already in progress finish on the old stack), and the
old stack is stopped once drained. During an update under load: 6,126
requests, 0 errors, 0 partial answers, p99 14 ms; the last old-version answer
and the first new-version answer each came within 0.1 s of the switch.

### Webpage

`web/` (plain HTML, CSS and JavaScript, served at `/`): type-ahead song search,
the playlist kept in the address bar (a link reopens the same recommendations),
Play buttons with Spotify's embedded player, light and dark mode, and phone
layouts.

### Hosting (in progress)

The plan is Oracle Cloud's Always Free tier: one Arm machine (2 cores, 8 GB)
runs the whole Docker Compose system (~2.9 GB), with Caddy for HTTPS. Free Arm
machines are often out of capacity, so a GitHub Actions workflow
(`.github/workflows/oracle-launch.yml`, running `scripts/oracle_launch.py`)
asks for one every 5 minutes and turns itself off once it has it.

## Ideas to revisit after serving (before adding more data)

Model and serving ideas to test once serving is done, before the ListenBrainz
data comes in. Each would be a separate run, measured like every other change.

- **Multi-token prediction** (from DeepSeek-V3/V4): train the model to predict
  the next song *and* the one after it (and so on) from the same playlist
  vector. More training signal from the same data, and a better fit for
  recommending a short run of songs. One extra output and loss; test at 50k.
- **Low-precision song vectors for serving:** store the search index's song
  vectors in 8 bits instead of 32 (FAISS scalar quantization), about 4x less
  memory (282 MB → ~70 MB at 1M). Measure what it costs in search recall.
- **Mixture of experts, only if a bigger model helps:** first test a wider or
  deeper transformer at 50k (it is still v1's 2 layers, 64 wide). If size
  helps, a mixture-of-experts layer (several small feed-forward "experts" and
  a router that picks 1–2 per position) adds that capacity cheaply. Once
  listening history brings several goals (next song, skips, play length),
  multi-gate MoE (MMoE) shares experts across them.

## Notes for later: songs the model doesn't know

A request can include songs outside the catalog. For now they are left out of
the input and listed as ignored; a song the model knows nothing about (not the
song, its artist, album or genres) can't be recommended either. Two things
make "knows nothing" rare in practice:

- **Spotify can always tell us something.** Any real Spotify song ID has a
  name, artist and album available through Spotify's API. Once those lookups
  are in place (planned for the ListenBrainz stage), "nothing known" really
  means "the artist is also new to us." A song with a known artist or album can
  get a features-only vector: its artist, album and genre vectors without a
  song vector of its own.
- **An "unknown song" placeholder** is a known trick: train the model with one
  shared vector that stands in for every rare song, so it learns what "some
  song I don't know" usually means. That's a model change for later, not a fix
  for now.

## Running it

Python 3.13; `pip install -r requirements.txt` (PyTorch, NumPy, FAISS, FastAPI,
gRPC, the Redis client, locust). Docker for the containerized system.

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
`runs/retriever_ranker_runs/1m/retriever_1m/seed1/`).

```bash
python -m recsys.train    --config configs/retriever_1m.toml   # retriever (~75 min on an M-series Mac)
python -m recsys.evaluate --config configs/retriever_1m.toml   # held-out NDCG@10, Hits@k, recall@K
python -m recsys.ranker   --config configs/ranker_1m.toml      # ranker: train, then evaluate (~45 min)
# final numbers on every held-out window instead of the 200,000-window sample (eval_all.json)
python -m recsys.evaluate --config configs/retriever_1m.toml --all-held-out
python -m recsys.ranker   --config configs/ranker_1m.toml --all-held-out
python scripts/search_benchmark.py real --config configs/retriever_1m.toml --k 100
python -m pytest                                              # tests
```

**Serve.** After the export (see Serving):

```bash
python -m recsys.launch --shards 4                    # sharded, without Docker (cuts the slices if needed)
python -m recsys.shard_slices --bundle bundles/<bundle> --shards 4   # slices for Docker, once
docker compose up -d --build --scale frontdesk=3      # shards, front desks, nginx, Redis, autoheal: http://localhost:8000
python scripts/serving_check.py cases --ranker configs/ranker_1m_any_length.toml --out runs/serving/cases
python scripts/serving_check.py serve --cases runs/serving/cases --url http://127.0.0.1:8000 --label docker
REDIS_URL= docker compose up -d --scale frontdesk=1   # cache off, then break shards under load:
python scripts/chaos.py --cases runs/serving/cases --out runs/serving/chaos.json
python scripts/load_test.py --copies 1 3 --users 1 16 64 --seconds 30 --out runs/serving/load.json
python scripts/bluegreen.py --bundle bundles/<bundle> --cases runs/serving/cases --out runs/serving/bluegreen.json
python scripts/playlist_eval.py cases --config configs/retriever_1m.toml --bundle bundles/<bundle> --out runs/serving/playlist_eval
python scripts/playlist_eval.py score --cases runs/serving/playlist_eval --bundle bundles/<bundle>
```

`docker-compose.yml` reads the bundle folder from `BUNDLE`; `BATCH=0` turns
batching off and an empty `REDIS_URL=` turns the cache off.

Config names follow the phases: `v1_*` (Phase 1), `p2_*` (features and
overfitting), `p3_*` (scaling), `retriever_<size>` and `ranker_<size>` (two-stage
pairs: `ranker_1m` is trained on `retriever_1m`; earlier rankers are
`retriever_ranker_*`). Runs go in matching
folders: `runs/v1_runs/`, `runs/p2_runs/`, `runs/p3_runs/`, and `runs/retriever_ranker_runs/<size>/`
(`5k/`, `50k/`, `200k/`, `1m/`).

## Layout

| Path | What's there |
|---|---|
| `recsys/data.py`, `recsys/mpd_store.py` | Playlists → vocab, splits, windows, chunks, features; the compact MPD store |
| `recsys/model.py` | The retriever (`SongRecommender`) |
| `recsys/train.py`, `recsys/evaluate.py` | Training loop with early stopping; held-out evaluation |
| `recsys/sampled.py`, `recsys/lazy_adam.py`, `recsys/augment.py` | Sampled softmax, lazy AdamW, window augmentation |
| `recsys/ranker.py` | The second-stage ranker: shortlists, training, evaluation |
| `recsys/search.py`, `recsys/ann.py`, `recsys/search_worker.py` | Exact search; FAISS indexes in their own process |
| `recsys/export.py`, `recsys/recommender.py`, `recsys/serve.py` | Serving: the versioned bundle, answering requests from it, the FastAPI server |
| `recsys/shard.py`, `recsys/shard_slices.py`, `recsys/frontdesk.py`, `recsys/wire.py`, `recsys/proto/` | Phase B: shard servers and their slices, the front desk, arrays in messages, the gRPC service |
| `recsys/batching.py`, `recsys/cache.py`, `recsys/admission.py`, `recsys/healthcheck.py`, `recsys/launch.py` | Batching, the Redis cache, overload limits, container health checks, the one-command launcher |
| `docker/`, `docker-compose.yml` | The image, nginx, the blue/green stacks |
| `web/` | The webpage |
| `.github/workflows/`, `deploy/` | The Oracle Cloud launch workflow; the server's SSH public key |
| `recsys/metrics.py`, `recsys/baselines.py` | NDCG@k, Hits@k; most-popular baseline |
| `scripts/` | v1 equivalence check, genre fetching, profiling, search benchmark, short-input test, serving check, chaos test, load tests (locust), probe, blue/green update, playlist evaluation, Oracle launch |
| `configs/`, `tests/` | One config per run; tests |
| `RESULTS.md` | Every run, with what changed and what was learned |

Genre data: MusicBrainz (CC BY-NC-SA).
