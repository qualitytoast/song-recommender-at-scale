"""One shard of the song catalog as a gRPC server (recsys/proto/shard.proto): it holds the songs with
song ID % shards == index and answers searches, per-song lookups and appeared-together counts for them.

    python -m recsys.shard --slice bundles/<bundle>/shards-4/shard-0 --port 50051

The slice comes from recsys.shard_slices. Every request carries the bundle version; a request for
another version is refused (FAILED_PRECONDITION), so a front desk can never mix model versions. The
standard gRPC health service reports SERVING once the slice is loaded. Imports numpy, FAISS and gRPC
only, never PyTorch (they can't share a process, see recsys/ann.py).
"""
import argparse
import json
import time
from concurrent import futures
from pathlib import Path

import faiss
import grpc
import numpy as np
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from recsys.pair_table import PairTable
from recsys.proto import shard_pb2, shard_pb2_grpc
from recsys.wire import pack, unpack

PER_SONG = ("retriever_song_inputs", "ranker_song_inputs", "song_artist", "song_album", "song_genres")


class Shard(shard_pb2_grpc.ShardServicer):
    def __init__(self, folder):
        folder = Path(folder)
        self.info = json.loads((folder / "info.json").read_text())
        self.ids = np.load(folder / "ids.npy")
        self.rows = {name: np.load(folder / f"{name}.npy") for name in PER_SONG}
        self.pairs = PairTable(folder, self.info["catalog"])
        self.index = faiss.read_index(str(folder / "search_index.faiss"))

    def check(self, request, context):
        if request.version != self.info["version"]:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                          f"shard {self.info['index']} serves version {self.info['version']}, not {request.version}")

    def rows_of(self, ids, context):
        """Each global song ID's row here; refuses songs another shard holds."""
        at = np.searchsorted(self.ids, ids)
        if len(ids) and (at.max() >= len(self.ids) or not np.array_equal(self.ids[at], ids)):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"shard {self.info['index']} doesn't hold some of {ids}")
        return at

    def Info(self, request, context):
        i = self.info
        return shard_pb2.InfoReply(version=i["version"], index=i["index"], shards=i["shards"], songs=i["songs"],
                                   catalog=i["catalog"])

    def Search(self, request, context):
        self.check(request, context)
        scores, ids = self.index.search(np.ascontiguousarray(unpack(request.queries), dtype=np.float32), request.k)
        return shard_pb2.SearchReply(ids=pack(ids), scores=pack(scores))

    def Lookup(self, request, context):
        self.check(request, context)
        at = self.rows_of(unpack(request.ids), context)
        r = {name: rows[at] for name, rows in self.rows.items()}
        # the input tables may be kept at 16 bits here (export --vector-bits 16); the front desk gets float32
        return shard_pb2.LookupReply(retriever_inputs=pack(r["retriever_song_inputs"].astype(np.float32)),
                                     ranker_inputs=pack(r["ranker_song_inputs"].astype(np.float32)),
                                     artist=pack(r["song_artist"]),
                                     album=pack(r["song_album"]), genres=pack(r["song_genres"]))

    def Counts(self, request, context):
        self.check(request, context)
        return shard_pb2.CountsReply(counts=pack(self.pairs.lookup(unpack(request.keys))))


def serve(folder, port, host="0.0.0.0", workers=8):
    """Start a shard server on host:port (returns it, already serving)."""
    start = time.perf_counter()
    shard = Shard(folder)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers),
                         options=[("grpc.max_receive_message_length", 64 << 20),
                                  ("grpc.max_send_message_length", 64 << 20)])
    shard_pb2_grpc.add_ShardServicer_to_server(shard, server)
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    i = shard.info
    print(f"shard {i['index']}/{i['shards']} of version {i['version']}: {i['songs']:,} songs, "
          f"{len(shard.pairs):,} pairs, on {host}:{port} after {time.perf_counter() - start:.1f}s", flush=True)
    return server


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Serve one shard of the song catalog over gRPC.")
    ap.add_argument("--slice", required=True, help="a shard folder from recsys.shard_slices")
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    serve(args.slice, args.port, args.host).wait_for_termination()
