"""The search process: holds FAISS indexes and answers requests over a pipe.

Started by recsys.search.SearchWorker as `python -m recsys.search_worker`. It
imports FAISS and numpy only, never PyTorch (see recsys/ann.py for why).

Protocol: each message is an 8-byte length followed by a pickled dict, in both
directions (stdin in, stdout out). Requests:
  {"cmd": "build", "name": str, "vectors": array, "kind": str, "shards": int, **params}
      -> {"build_seconds": float, "memory_bytes": int}
  {"cmd": "search", "name": str, "queries": array, "k": int}
      -> {"ids": array, "scores": array, "seconds": float}
  {"cmd": "latency", "name": str, "queries": array, "k": int}  -> {"ms": array}
  {"cmd": "drop", "name": str}  -> {}
  {"cmd": "quit"}
Errors come back as {"error": message}. pickle is fine here: both ends are this project's code.
"""
import os
import pickle
import struct
import sys
import time
import traceback

from recsys.ann import Index


def read_message(stream):
    header = stream.read(8)
    if len(header) < 8:
        return None
    (length,) = struct.unpack("<Q", header)
    return pickle.loads(stream.read(length))


def write_message(stream, message):
    data = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack("<Q", len(data)) + data)
    stream.flush()


def main():
    # Messages go out on a private copy of stdout; anything else printed to stdout
    # (by FAISS or any library, even from C code) goes to stderr, so it can't
    # corrupt the message stream.
    stdout = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    stdin = sys.stdin.buffer
    indexes = {}
    while (request := read_message(stdin)) is not None:
        cmd = request.pop("cmd")
        if cmd == "quit":
            break
        try:
            if cmd == "build":
                name = request.pop("name")
                indexes[name] = Index(request.pop("vectors"), request.pop("kind"), **request)
                reply = {"build_seconds": indexes[name].build_seconds,
                         "memory_bytes": indexes[name].memory_bytes()}
            elif cmd == "search":
                start = time.perf_counter()
                ids, scores = indexes[request["name"]].search(request["queries"], request["k"])
                reply = {"ids": ids, "scores": scores, "seconds": time.perf_counter() - start}
            elif cmd == "latency":
                reply = {"ms": indexes[request["name"]].latency_ms(request["queries"], request["k"])}
            elif cmd == "drop":
                indexes.pop(request["name"], None)
                reply = {}
            else:
                reply = {"error": f"unknown command {cmd!r}"}
        except Exception:
            reply = {"error": traceback.format_exc()}
        write_message(stdout, reply)


if __name__ == "__main__":
    main()
