"""One shard of the song catalog as a gRPC server (recsys/proto/shard.proto): it holds the songs with
song ID % shards == index and answers searches, per-song lookups and appeared-together counts for them.

    python -m recsys.shard --slice bundles/<bundle>/shards-4/shard-0 --port 50051
    python -m recsys.shard --slice ... --switch bundles/<bundle>/switch.json   # follows the switch file

The slice comes from recsys.shard_slices. Every request carries the bundle version; a request for a
version the shard doesn't hold is refused (FAILED_PRECONDITION), so a front desk can never mix model
versions. With --switch the shard holds the versions the switch file names (recsys/switch.py): the live
one and the standby, each the same slice (shards-N/shard-i) of its folder (the bundle, or an update of it,
recsys/updates.py). A newly named one is loaded beside the others while the shard keeps answering, by one shard
at a time (take_turn), and one the file no longer names is dropped (the live one only once its replacement is
loaded). The standard gRPC
health service reports SERVING once the live version is loaded. Imports numpy, FAISS and gRPC only, never
PyTorch (they can't share a process, see recsys/ann.py).
"""
import argparse
import fcntl
import json
import os
import time
from concurrent import futures
from pathlib import Path

import faiss
import grpc
import numpy as np
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from recsys import switch
from recsys.pair_table import PairTable
from recsys.proto import shard_pb2, shard_pb2_grpc
from recsys.wire import pack, unpack

PER_SONG = ("retriever_song_inputs", "ranker_song_inputs", "song_artist", "song_album", "song_genres")
RETRY_S = 60.0  # a version that failed to load is tried again this much later
# A version the switch file stops naming is still held this long: front desks read the file once a second, so one
# can still be sending it for a moment after a flip and a finish written right after it.
KEEP_S = 5.0


class Slice:
    """One version's slice of the catalog, loaded."""

    def __init__(self, folder):
        folder = Path(folder)
        self.info = json.loads((folder / "info.json").read_text())
        self.ids = np.load(folder / "ids.npy")
        self.rows = {name: np.load(folder / f"{name}.npy") for name in PER_SONG}
        self.pairs = PairTable(folder, self.info["catalog"])
        self.index = faiss.read_index(str(folder / "search_index.faiss"))


class Shard(shard_pb2_grpc.ShardServicer):
    def __init__(self, folder, switch_file=None, log=switch.say):
        """folder: the base version's slice; switch_file: the switch file to follow (None: that slice alone)."""
        self.slice, self.log = Path(folder), log
        self.switch = Path(switch_file) if switch_file else None
        self.relative = self.slice.resolve().relative_to(self.switch.parent.resolve()) if self.switch else None
        # (live folder, {folder: Slice}, {version: Slice}): replaced whole, never changed in place, so a request
        # reads one consistent state without a lock
        self.state = (None, {}, {})
        self.failed = {}  # folder -> when to try loading it again
        self.unnamed = {}  # folder -> since when the switch file hasn't named it (dropped KEEP_S later)
        self.waiting = {}  # folder -> since when another shard's turn has kept it from loading
        live = switch.read(self.switch)["live"] if self.switch else switch.BASE
        self.sync({"live": live, "standby": None}, strict=True)  # the standby (if any) loads after startup

    def slice_of(self, folder):
        return self.slice if self.switch is None else self.switch.parent / folder / self.relative

    def take_turn(self):
        """The loading turn, or None if another shard has it: the bundle's folder, opened and locked (flock).
        Every shard following the bundle's switch file loads a version while serving only during its turn, so
        they read the disk one at a time: all 4 at once held the server's disk (~50 MB/s) for 38 s and stalled
        answers up to 6.3 s (2026-11-03). The kernel lets go of a dead shard's turn. Closing the folder ends it."""
        turn = os.open(self.switch.parent, os.O_RDONLY)
        try:
            fcntl.flock(turn, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(turn)
            return None
        return turn

    def sync(self, state, strict=False):
        """Hold what the switch state names: load the folders not held, then drop the ones it doesn't name. At
        startup (strict) a load starts at once and a failure raises; later, a load waits for this shard's turn,
        and a failure is logged and tried again RETRY_S later."""
        live, held, _ = self.state
        for folder in switch.wanted(state):
            if folder in held or time.monotonic() < self.failed.get(folder, 0.0):
                continue
            turn = None
            if not strict:  # serving already (starting up, a shard serves nothing: it loads at once)
                turn = self.take_turn()
                if turn is None:
                    self.waiting.setdefault(folder, time.monotonic())
                    continue  # tried again next second
            waited = time.monotonic() - self.waiting.pop(folder, time.monotonic())
            start = time.perf_counter()
            try:
                loaded = Slice(self.slice_of(folder))
                first = next(iter(held.values()), None)
                if first and (loaded.info["index"], loaded.info["shards"]) != (first.info["index"], first.info["shards"]):
                    raise ValueError(f"{folder} is shard {loaded.info['index']} of {loaded.info['shards']}, not "
                                     f"{first.info['index']} of {first.info['shards']}")
            except Exception as e:
                if strict:
                    raise
                self.failed[folder] = time.monotonic() + RETRY_S
                self.log(f"shard: couldn't load {folder}: {type(e).__name__}: {e}")
                continue
            finally:
                if turn is not None:
                    os.close(turn)  # the next shard's turn
            held = {**held, folder: loaded}
            i = loaded.info
            self.log(f"shard {i['index']}/{i['shards']}: loaded version {i['version']} ({folder}): {i['songs']:,} "
                     f"songs, {len(loaded.pairs):,} pairs, in {time.perf_counter() - start:.1f}s"
                     + (f", after {waited:.0f}s waiting for its turn" if waited >= 0.5 else ""))
        if state["live"] in held:
            live = state["live"]
        now, named = time.monotonic(), set(switch.wanted(state))
        self.unnamed = {f: self.unnamed.get(f, now) for f in held if f not in named}
        self.waiting = {f: since for f, since in self.waiting.items() if f in named}
        # the live one stays until its replacement is loaded, any other KEEP_S after the file stops naming it
        keep = named | {live} | {f for f, since in self.unnamed.items() if now - since < KEEP_S}
        for folder in [f for f in held if f not in keep]:
            self.log(f"shard: dropped version {held[folder].info['version']} ({folder})")
        held = {f: s for f, s in held.items() if f in keep}
        order = [live] + [f for f in held if f != live]
        self.state = (live, held, {held[f].info["version"]: held[f] for f in order})

    def versions(self):
        return list(self.state[2])  # live first

    def slice_for(self, version, context):
        """The slice of the version a request names; refuses one this shard doesn't hold."""
        by_version = self.state[2]
        if version not in by_version:
            held = list(by_version)
            index = next(iter(by_version.values())).info["index"]
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                          f"shard {index} serves version{'s' if len(held) > 1 else ''} {', '.join(held)}, not {version}")
        return by_version[version]

    def rows_of(self, s, ids, context):
        """Each global song ID's row in slice s; refuses songs another shard holds."""
        at = np.searchsorted(s.ids, ids)
        if len(ids) and (at.max() >= len(s.ids) or not np.array_equal(s.ids[at], ids)):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"shard {s.info['index']} doesn't hold some of {ids}")
        return at

    def Info(self, request, context):
        live, held, by_version = self.state
        i = (self.slice_for(request.version, context) if request.version else held[live]).info
        return shard_pb2.InfoReply(version=i["version"], index=i["index"], shards=i["shards"], songs=i["songs"],
                                   catalog=i["catalog"], versions=list(by_version))

    def Search(self, request, context):
        s = self.slice_for(request.version, context)
        scores, ids = s.index.search(np.ascontiguousarray(unpack(request.queries), dtype=np.float32), request.k)
        return shard_pb2.SearchReply(ids=pack(ids), scores=pack(scores))

    def Lookup(self, request, context):
        s = self.slice_for(request.version, context)
        at = self.rows_of(s, unpack(request.ids), context)
        r = {name: rows[at] for name, rows in s.rows.items()}
        # the input tables may be kept at 16 bits here (export --vector-bits 16); the front desk gets float32
        return shard_pb2.LookupReply(retriever_inputs=pack(r["retriever_song_inputs"].astype(np.float32)),
                                     ranker_inputs=pack(r["ranker_song_inputs"].astype(np.float32)),
                                     artist=pack(r["song_artist"]),
                                     album=pack(r["song_album"]), genres=pack(r["song_genres"]))

    def Counts(self, request, context):
        s = self.slice_for(request.version, context)
        return shard_pb2.CountsReply(counts=pack(s.pairs.lookup(unpack(request.keys))))


def serve(folder, port, host="0.0.0.0", workers=8, switch_file=None):
    """Start a shard server on host:port (returns it, already serving); with switch_file, it follows that file."""
    start = time.perf_counter()
    shard = Shard(folder, switch_file)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers),
                         options=[("grpc.max_receive_message_length", 64 << 20),
                                  ("grpc.max_send_message_length", 64 << 20)])
    shard_pb2_grpc.add_ShardServicer_to_server(shard, server)
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    if switch_file:
        switch.follow(switch_file, shard.sync)
    print(f"shard serving on {host}:{port} after {time.perf_counter() - start:.1f}s", flush=True)
    return server


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Serve one shard of the song catalog over gRPC.")
    ap.add_argument("--slice", required=True, help="a shard folder from recsys.shard_slices")
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--switch", help="the bundle's switch.json (recsys/switch.py) to follow")
    args = ap.parse_args()
    serve(args.slice, args.port, args.host, switch_file=args.switch).wait_for_termination()
