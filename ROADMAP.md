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
- [x] Server resized to 12 GB, the Always Free maximum, for a bigger catalog (8.5 GB free; ~3 min down)
- [x] The site stores song vectors at 16 bits: same NDCG@10 (0.2322), ~430 MB less on the shards, ~2 ms slower p50
- [x] Store builder trimmed for bigger catalogs: 18.0 -> 13.1 GB in use, the same store byte for byte
- [x] Tested N = 2 on the same listens: 2.99M songs, +5.2% next-song NDCG@10 over today's pair on shared cases
- [x] Tested 1.5x the listens at N = 5: 1.51M songs, +7.0% on the same cases (+4.7% on its own): the better one
- [x] Tested N = 3 with 1.5x the listens: 2.46M songs, +7.8% on the same cases, ahead of N = 2 on all three case sets
- [x] The site serves the N = 3 + 1.5x pair: 2.46M songs, the offline top 10 for 2,000 of 2,000, 51 s of downtime
- [x] Co-occurrence counts at 5.26 bytes a pair, not 12: 1.82 -> 0.80 GB on the shards, ~0.9 GB freed, every count exact

## TODO Now

Today: 2.46M songs, those played by at least 3 listeners (N) in a 99.5M-listen sample; the full year is on disk.

- [ ] Older years from a fresh 258 GB dump (back-catalog songs; ~14 h download + ~1 h extract)
- [ ] Daily updates from ListenBrainz's incremental dumps: pair counts, popularity and new songs, no full rebuild
- [ ] Daily version switches: all 4 shards change together, no request mixes versions, rollback after a bad update
- [ ] Personalization B: fine-tune the pair on one listener's sessions (a private model)
- [ ] Personalization C: a learned vector per listener
- [ ] Personalization D: listening history as an input, so each visitor's own history personalizes their requests
- [ ] Personalization in the browser: ONNX retriever, song rows fetched in blocks; the server gets only a 64-number vector
- [ ] Ranker for browser personalization: pair lookups reveal the input songs, so measure what each alternative gives up
- [ ] Online logging of impressions, plays and skips (Spotify's iFrame API): next-song NDCG and new finds disagree
- [ ] Interleaving: merge two models' lists and count whose songs get played; needs far fewer users than an A/B test
- [ ] Off-policy evaluation: add a little randomness to serving, log the probabilities, score new models on old logs
- [ ] Distillation: a retriever trained to agree with the ranker, once a personalized model is finished
- [ ] The playlist-fit (next-10) ranker on the 1M MPD retriever
- [ ] Faster retriever training (on hold): a decaying learning rate, fewer validation checks
- [ ] A bigger transformer if size helps, then mixture of experts
- [ ] Songs the model doesn't know: their artist and album from Spotify's API, or a shared "unknown song" vector
- [ ] MusicBrainz recording IDs for the catalog, AcousticBrainz's key (the store keeps none; 7% of listens carry one)
- [ ] Vectors for songs outside the catalog: map AcousticBrainz audio features to song vectors, scored on known songs
- [ ] C++ shard hot path with SIMD: 16-bit dot products with no 32-bit decode, IVF scan, top-k merge (FAISS's job today)
- [ ] Hedged requests across the 4-shard fan-out (84 ms p99 on 2 cores), using a second copy of each shard
