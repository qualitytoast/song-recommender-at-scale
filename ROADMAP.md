# Roadmap

What's done, what's being worked on, and what's planned. Every change is a separate run, measured against
the one before it ([RESULTS.md](RESULTS.md)).

## Done

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
- [x] Training for the songs that belong in a playlist, tested at 50k: a ranker trained on the next 10
      songs puts 3.4% more of its top 10 later in the playlist and cuts the last artist's share from 41%
      to 30%, for -7.4% on the very next song; multi-token heads added nothing
- [x] Public hosting on Oracle Cloud's free tier: https://songrec.duckdns.org (HTTPS, per-visitor rate limits)
- [x] Train on ListenBrainz listening history: 66M listens from 61,752 users, 1.18M songs; the pair
      predicts held-out listeners' next song 22% better than today's MPD pair (0.2433 vs 0.1996)
- [x] The live site serves the ListenBrainz pair (its own tuned settings: 3 per artist, no mixing),
      swapped in with ~12 s of downtime
- [x] Song search that fixes typos and matches title, artist and album words in any order (Meilisearch)
- [x] Personalization A: a taste profile from the owner's Spotify history, behind a private "Use my taste"
      toggle: more than twice the new songs they then listened to (1.96 -> 4.40 per 100 recommendations)

## Now: growing the song catalog

The catalog is every song at least N = 5 listeners in the training sample played (1.18M songs). The sample
is 66M listens: each of 61,752 people's most recent 2,759 from the past year. The whole year is already on
disk (`data/lb_extract`: 308M usable listens), so the first two options need no new download.

| Rule (from the step 2 report's counts) | Songs | vs today |
|---|---|---|
| N = 5 (today) | 1.18M | 1x |
| N = 3 | ~2.0M | ~1.7x |
| N = 2 | ~3.2M | ~2.7x |

Ways to grow it:

- [ ] **Lower N to 3 or 2, on the same 66M listens.** The cheapest option. The added songs are ones only 2-4
      people played, so the model sees each a handful of times and learns them mostly through their artist,
      album and genre: they'd score like "a typical song by this artist" and rank below that artist's
      well-known songs. They'd mainly help as search results and inputs: more of a listener's unmatched
      songs found (9% of the owner's), fewer "not in the catalog" messages.
- [ ] **Sample more listens, keeping N = 5.** Raise the per-person cap (to ~4,000-5,000) to use more of the
      listens already on disk. More people's listening pushes more songs past 5 listeners, and the new songs
      are better learned, but the catalog grows slower than the data and training time grows with it
      (roughly 2x the listens for 2x the time). N = 3 plus ~1.5x the listens is about 2x the songs.
- [ ] **Older years.** Re-download the full 258 GB dump (~14 h at ~5 MB/s), keep more than 12 months, and
      re-extract (~1 h): back-catalog songs, at a high cost in time and disk.
- [ ] **Songs nobody in the sample played.** Teach the model to place a song from its artist, album and genre
      alone (training with the song's own vector sometimes hidden), then add songs from elsewhere: MPD songs
      ListenBrainz lacks, a listener's unmatched songs, discographies. Limited to what's known about each
      song: one by a known artist with an album or genre gets a usable vector; one with only a known artist
      looks like all that artist's other unseen songs; one by an unknown artist can't be placed. A model
      change, and these are the weakest recommendations.

Serving a bigger catalog:

- [ ] **Resize the server to 12 GB.** Oracle's Always Free Ampere allowance is 1,500 OCPU hours and 9,000 GB
      hours a month (2 OCPUs and 12 GB), and the account's quota allows 12 GB (8 used), so it stays $0; a stop
      and start of a few minutes. At ~2x the songs, the shards, front desk and search index would need ~8 GB
      of today's 7.7.
- [ ] **Store the song vectors in 8 bits instead of 32** (search vectors and song inputs; FAISS scalar
      quantization): about 4x less memory for them. First measure the cost on held-out cases through the
      serving code (top-10 overlap, NDCG@10, shortlist recall), with 16 bits as a middle option.

A 2x catalog: store rebuild (after trimming its memory peak: ~27 GB on a 25.8 GB Mac last time), a new
retriever and baseline ranker (~4.5-6 h of training; 1.5x the listens: ~6-7 h), export and shards (~20 min),
the server's search index (~6 min) and the owner's taste profile (seconds): about a day, mostly training.

## Later

- [ ] **Personalization B-D:** B, fine-tune the pair on one listener's sessions (with general "replay" data
      against forgetting; a private model); C, a learned vector per listener (fold-in for new ones);
      D, listening history as an input (an average of the last ~100 plays), trained on ListenBrainz
      listeners' histories, so any visitor's own on-site history personalizes their requests with no
      per-person training. Distillation (a retriever trained to agree with the ranker) once a finished
      personalized model exists.
- [ ] **The playlist-fit ranker at 1M** (MPD): the next-10 ranker (`ranker_50k_1m_settings_next10w` at 50k)
      on the 1M MPD retriever, re-tuned serving settings; only the ranker retrained.
- [ ] **Faster retriever training** (on hold): at 50k, validation is 44% of a retriever run; at 1M, the
      last 5 of ~9 epochs added only +0.005. Candidates: a learning rate that decays over a fixed number of
      epochs, validating once an epoch at 50k, starting retriever variants from the trained retriever.
- [ ] **Mixture of experts, only if a bigger model helps:** first a wider or deeper transformer at 50k (it is
      still v1's 2 layers, 64 wide); if size helps, a mixture-of-experts layer adds capacity cheaply, and
      multi-gate MoE (MMoE) once there are several goals (next song, skips, play length).
- [ ] **Songs the model doesn't know:** today they are left out of a request and listed as ignored. Spotify's
      API can name any Spotify song's artist and album, which with the features-only vectors above would
      cover most of them; a shared "unknown song" vector trained on rare songs is the other known trick.
