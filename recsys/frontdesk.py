"""The front desk of the sharded system: the two models' small shared layers and the song lookups,
with every per-song table held by the shards (recsys/shard.py), reached over gRPC.

    rec = ShardedRecommender("bundles/<bundle>", ["127.0.0.1:50051", "127.0.0.1:50052"], device="cpu")
    rec.recommend(["Nightcall - Kavinsky"], name="drive")   # the same answer as recommender.Recommender

Song j lives on shard j % N. A request takes three rounds, each sent to the shards involved at the same
time (gRPC futures), each with a deadline:
  1. the input songs' rows (both models' input vectors, artist, album, genres) from their shards
  2. the retriever's playlist vector h, searched on every shard; their top k merged (score, then song
     ID) into the overall top k, input songs dropped, the best `shortlist` kept
  3. the candidates' rows from their shards, and the appeared-together counts (input song -> candidate)
     from the candidates' shards
The ~110 songs of a request get local IDs (inputs first, then candidates), and the models and the
ranker's extra inputs run on tables of just those songs: the same code and numbers as with the whole
catalog, so the answers equal the single-process Recommender's. Startup checks that the shards serve
this bundle's version, as shards 0..N-1 of N over the same catalog (and every request carries the
version, which a shard refuses if it differs).

Faults (gRPC's own machinery, set in SERVICE_CONFIG): every call has a deadline; a call that finds its
shard unavailable is retried up to twice; a shard given as several copies ("a:1|b:2", or a DNS name with
several addresses, as `docker compose --scale` makes) gets each call on the next healthy copy (round
robin, skipping copies whose gRPC health service doesn't say SERVING). If every copy of a shard fails
a call, the answer is made from the other shards and flagged: "partial": true and "missing_shards"
(input songs and candidates it holds are left out). Only if no input song or no search result is left
does the request fail (ShardsUnavailable: HTTP 503). A circuit breaker keeps one stuck shard from
slowing every request: a shard that let a call run past its deadline is skipped, as if it had failed,
for `cooldown` seconds; then the next call tries it again. (An unreachable shard fails at once, so it
costs no waiting and isn't skipped: a copy that died mustn't take its shard's live copies out.)
"""
import json
import time

import grpc
import numpy as np
import torch
from torch import nn

from recsys.proto import shard_pb2, shard_pb2_grpc
from recsys.rank_features import CandidateFeatures
from recsys.ranker import drop_input_songs
from recsys.recommender import TOP, Catalog, Stopwatch, load_shared, serving_ranker, serving_retriever
from recsys.search import query_vectors
from recsys.wire import pack, unpack

ROWS = ("retriever_inputs", "ranker_inputs", "artist", "album", "genres")


# Per call to a shard: up to two retries if it was unavailable (after 10-50 ms), and copies used in turn,
# only those whose health service says SERVING.
SERVICE_CONFIG = json.dumps({
    "loadBalancingConfig": [{"round_robin": {}}],
    "healthCheckConfig": {"serviceName": ""},
    "methodConfig": [{"name": [{"service": "shard.Shard"}],
                      "retryPolicy": {"maxAttempts": 3, "initialBackoff": "0.01s", "maxBackoff": "0.05s",
                                      "backoffMultiplier": 2, "retryableStatusCodes": ["UNAVAILABLE"]}}]})


class ShardMismatch(RuntimeError):
    """The shards don't serve this bundle as shards 0..N-1 of N."""


class ShardsUnavailable(RuntimeError):
    """Too many shards failed to answer a request at all."""


def shard_target(spec):
    """The gRPC target for a shard given as "host:port" (a DNS name: every address it resolves to is a
    copy) or as copies "ip:port|ip:port"."""
    copies = spec.split("|")
    return f"ipv4:{','.join(copies)}" if len(copies) > 1 else f"dns:///{spec}"


class RequestRows(nn.Module):
    """Stands in for a serving model's song table: the input vectors of one request's songs (rows, set
    before each use), addressed by local ID. The front desk holds no per-song data."""

    def __init__(self):
        super().__init__()
        self.rows = None

    def forward(self, ids):
        return self.rows[ids]


def merge_top(id_parts, score_parts, k):
    """The overall top k of the shards' top-k lists (each (1, k)), best first; equal scores by song ID,
    and empty slots (ID -1) last."""
    ids, scores = np.concatenate(id_parts, axis=1)[0], np.concatenate(score_parts, axis=1)[0]
    scores = np.where(ids >= 0, scores, -np.inf)
    order = np.lexsort((ids, -scores))[:k]
    return ids[order][None, :], scores[order][None, :].astype(np.float32)


class ShardedRecommender(Catalog):
    def __init__(self, bundle, shards, device="cpu", deadline=2.0, startup_wait=120.0, cooldown=1.0):
        """shards: the shard servers' addresses ("host:port"), shard i's at position i; deadline: seconds
        each call to a shard may take; cooldown: seconds a shard that missed a call is skipped."""
        super().__init__(bundle, device)
        s = self.settings
        blank = np.zeros((1, s["embed_dim"]), dtype=np.float32)
        self.retriever = serving_retriever(s, blank)
        self.retriever.song_embedding = RequestRows()
        self.retriever.to(self.device)
        load_shared(self.retriever, torch.load(self.path / "retriever_shared.pt", weights_only=True))
        self.ranker = serving_ranker(s, blank)
        self.ranker.retriever.song_embedding = RequestRows()
        self.ranker.to(self.device)
        load_shared(self.ranker, torch.load(self.path / "ranker_shared.pt", weights_only=True))
        self.deadline, self.cooldown = deadline, cooldown
        self.addresses = list(shards)
        self.skip_until = [0.0] * len(self.addresses)  # the circuit breaker: shard i skipped until then
        options = [("grpc.max_receive_message_length", 64 << 20), ("grpc.max_send_message_length", 64 << 20),
                   ("grpc.enable_retries", 1), ("grpc.service_config", SERVICE_CONFIG),
                   # reconnect to a shard (copy) that was down within 0.1-1 s, not gRPC's default of up to
                   # 2 minutes, so a restarted shard is used again right away
                   ("grpc.initial_reconnect_backoff_ms", 100), ("grpc.min_reconnect_backoff_ms", 100),
                   ("grpc.max_reconnect_backoff_ms", 1000)]
        self.channels = [grpc.insecure_channel(shard_target(a), options=options) for a in self.addresses]
        self.stubs = [shard_pb2_grpc.ShardStub(c) for c in self.channels]
        self.check_shards(startup_wait)

    def check_shards(self, wait):
        """Wait up to `wait` seconds for every shard, then make sure each is the right one."""
        n = len(self.stubs)
        for i, stub in enumerate(self.stubs):
            info = stub.Info(shard_pb2.InfoRequest(), timeout=wait, wait_for_ready=True)
            problems = [f"version {info.version}, not {self.version}" if info.version != self.version else "",
                        f"shard {info.index} of {info.shards}, not {i} of {n}" if (info.index, info.shards) != (i, n) else "",
                        f"a catalog of {info.catalog:,} songs, not {self.settings['songs']:,}"
                        if info.catalog != self.settings["songs"] else ""]
            if any(problems):
                raise ShardMismatch(f"{self.addresses[i]} serves " + "; ".join(p for p in problems if p))

    def close(self):
        for channel in self.channels:
            channel.close()

    def shard_states(self):
        """Each shard's answer to Info within the deadline: [{index, address, up}]."""
        pending = [stub.Info.future(shard_pb2.InfoRequest(), timeout=self.deadline) for stub in self.stubs]
        states, now = [], time.monotonic()
        for i, p in enumerate(pending):
            try:
                p.result()
                up = True
            except grpc.RpcError:
                up = False
            states.append({"index": i, "address": self.addresses[i], "up": up,
                           "skipped_for_s": round(max(0.0, self.skip_until[i] - now), 2)})
        return states

    def call_all(self, calls):
        """Send every (shard, method, request) at once; the replies, with None for each call that failed
        (after gRPC's retries) or went to a shard the circuit breaker skips, and the shards that failed one.
        A shard that let a call run past its deadline is skipped for `cooldown` seconds."""
        now = time.monotonic()
        pending = [(shard, None if now < self.skip_until[shard] else method.future(request, timeout=self.deadline))
                   for shard, method, request in calls]
        replies, failed = [], set()
        for shard, p in pending:
            try:
                if p is None:
                    raise LookupError("skipped")
                replies.append(p.result())
            except (grpc.RpcError, LookupError) as e:
                replies.append(None)
                failed.add(shard)
                if isinstance(e, grpc.RpcError) and e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    self.skip_until[shard] = time.monotonic() + self.cooldown
        return replies, failed

    def by_shard(self, ids):
        """{shard: positions in ids of the songs it holds}."""
        owner = ids % len(self.stubs)
        return {int(o): np.flatnonzero(owner == o) for o in np.unique(owner)}

    def lookup_calls(self, ids):
        parts = self.by_shard(ids)
        calls = [(o, self.stubs[o].Lookup, shard_pb2.LookupRequest(version=self.version, ids=pack(ids[at])))
                 for o, at in parts.items()]
        return parts, calls

    @staticmethod
    def gather_rows(n, parts, replies):
        """The Lookup replies' arrays, back in the order of the n songs asked for (rows of songs whose
        shard didn't answer stay unset; the caller leaves those songs out)."""
        out = {}
        for at, reply in zip(parts.values(), replies):
            if reply is None:
                continue
            for name in ROWS:
                a = unpack(getattr(reply, name))
                out.setdefault(name, np.empty((n,) + a.shape[1:], dtype=a.dtype))[at] = a
        return out

    @torch.no_grad()
    def recommend_ids(self, song_ids, name_ids):
        clock = Stopwatch(self.device)
        s, device, songs = self.settings, self.device, self.settings["songs"]
        X = np.asarray(song_ids[-s["input_length"]:], dtype=np.int64)
        N = np.asarray(name_ids, dtype=np.int64).reshape(1, -1)
        missing = set()
        # 1. the input songs' rows (songs on a shard that didn't answer are left out)
        parts, calls = self.lookup_calls(X)
        replies, failed = self.call_all(calls)
        missing |= failed
        inputs = self.gather_rows(len(X), parts, replies)
        keep = ~np.isin(X % len(self.stubs), list(failed))
        if not keep.any():
            raise ShardsUnavailable(f"no shard holding the input songs answered (shards {sorted(failed)})")
        X, inputs = X[keep], {name: a[keep] for name, a in inputs.items()}
        L = len(X)
        clock.mark("shards: input songs")
        self.retriever.song_embedding.rows = torch.from_numpy(inputs["retriever_inputs"]).to(device)
        context = np.arange(L)[None, :]  # local IDs: the inputs are 0..L-1
        h = query_vectors(self.retriever, context, N, device)
        clock.mark("retriever")
        # 2. search every shard, merge what came back
        k = s["shortlist"] + (L if s["exclude_input"] else 0)
        replies, failed = self.call_all([(i, stub.Search, shard_pb2.SearchRequest(version=self.version,
                                                                                  queries=pack(h), k=k))
                                         for i, stub in enumerate(self.stubs)])
        missing |= failed
        replies = [r for r in replies if r is not None]
        if not replies:
            raise ShardsUnavailable("no shard answered the search")
        ids, scores = merge_top([unpack(r.ids) for r in replies], [unpack(r.scores) for r in replies], k)
        if s["exclude_input"]:
            ids, scores = drop_input_songs(ids, scores, X[None, :], None, s["shortlist"])
        clock.mark("shards: search")
        # 3. the candidates' rows and the counts input song -> candidate, from the candidates' shards
        cands = ids[0]
        keys = X[None, :] * songs + cands[:, None]  # (C, L): pair (input l, candidate c)
        parts, calls = self.lookup_calls(cands)
        count_parts = self.by_shard(cands)
        calls += [(o, self.stubs[o].Counts, shard_pb2.CountsRequest(version=self.version,
                                                                    keys=pack(keys[at].ravel())))
                  for o, at in count_parts.items()]
        replies, failed = self.call_all(calls)
        missing |= failed
        rows = self.gather_rows(len(cands), parts, replies[:len(parts)])
        counts = np.zeros((len(cands), L), dtype=np.int64)
        for at, reply in zip(count_parts.values(), replies[len(parts):]):
            if reply is not None:
                counts[at] = unpack(reply.counts).reshape(len(at), L)
        if failed:  # candidates on a shard that didn't answer can't be ranked: leave them out
            keep = ~np.isin(cands % len(self.stubs), list(failed))
            cands, scores, counts = cands[keep], scores[:, keep], counts[keep]
            rows = {name: a[keep] for name, a in rows.items()}
        C = len(cands)
        clock.mark("shards: candidates")
        # the ranker's extra inputs on tables of this request's songs: local IDs 0..L-1 inputs, L.. candidates
        n_local = L + C
        position, cand = np.nonzero(counts.T)  # (input position, candidate) pairs ever seen
        local_keys = position * n_local + (L + cand)  # pair (local input ID, local candidate ID)
        order = np.argsort(local_keys)
        features = CandidateFeatures(s["features"], n_local, None, (local_keys[order], counts.T[position, cand][order]),
                                     device, song_artist=np.concatenate([inputs["artist"], rows["artist"]]),
                                     song_album=np.concatenate([inputs["album"], rows["album"]]),
                                     song_genres=np.concatenate([inputs["genres"], rows["genres"]]))
        context_t = torch.from_numpy(context).to(device)
        cands_t = torch.arange(L, n_local, device=device)[None, :]
        base = torch.from_numpy(np.ascontiguousarray(scores, dtype=np.float32)).to(device)
        ranks = torch.arange(1, C + 1, device=device)[None, :]
        f = features(context_t, cands_t, base, ranks, base[:, 0], playlist_len=torch.tensor([L], device=device))
        clock.mark("ranker inputs")
        self.ranker.retriever.song_embedding.rows = torch.from_numpy(
            np.concatenate([inputs["ranker_inputs"], rows["ranker_inputs"]])).to(device)
        rerank = self.ranker(context_t, torch.from_numpy(N).to(device), cands_t, base, f).cpu().numpy()[0]
        clock.mark("ranker")
        best = np.argsort(-rerank, kind="stable")[:TOP]
        notes = {"partial": bool(missing)}
        if missing:
            notes["missing_shards"] = sorted(int(i) for i in missing)
        return cands[best], rerank[best], clock.times, notes
