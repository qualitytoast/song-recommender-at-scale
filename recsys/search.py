"""Finding a playlist's top-k songs: exact search here, FAISS in a separate process.

The retriever scores song j as h . v_j + b_j, where h is the playlist vector and
v_j, b_j the song's output vector and bias (its own output row plus its artist,
album, duration and genre output vectors). Appending the bias to each song vector
and a 1 to h makes that a plain dot product:
    [h, 1] . [v_j, b_j] = h . v_j + b_j
so any maximum-dot-product search over the song vectors finds the retriever's
top-k. Every searcher returns (ids, scores), each (n_queries, k), best first.

This module uses PyTorch (vectors, exact search on the GPU) and never imports
FAISS: the two can't share a process (see recsys/ann.py). SearchWorker starts
the FAISS process (recsys/search_worker.py) and talks to it over a pipe:

    PyTorch process                      search process (FAISS)
    retriever: songs -> h    --- h -->   index.search(h, k)
    ranker <- shortlist      <-- ids, scores ---
"""
import pickle
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def song_vectors(model):
    """(vocab_size, embed_dim + 1) float32: each song's output vector with its bias
    appended, so [h, 1] . vector = the retriever's score for that song."""
    vectors = model.output.weight.detach().clone()
    for name in model.feature_names:
        vectors += model.output_features[name](model.song_feature_ids(name))
    if model.output_genres is not None:
        vectors += model.mean_vector(model.output_genres, model.song_genres)
    return torch.cat([vectors, model.output.bias[:, None]], dim=1).float().cpu().numpy()


@torch.no_grad()
def query_vectors(model, X, N, device, batch=4096):
    """(windows, embed_dim + 1) float32: each window's playlist vector h with a 1 appended."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        h = model.hidden_states(torch.from_numpy(X[i:i + batch]).to(device),
                                torch.from_numpy(N[i:i + batch]).to(device))[:, -1, :]
        out.append(torch.cat([h, torch.ones(len(h), 1, device=device)], dim=1).float().cpu())
    return torch.cat(out).numpy()


class ExactSearch:
    """Score every song, keep the top k: one matrix multiply per chunk of queries."""

    def __init__(self, vectors, device, max_scores=2**26):
        self.vectors = torch.from_numpy(vectors).to(device)
        self.device = device
        self.chunk = max(1, max_scores // len(vectors))

    def sync(self):
        if self.device.type == "mps":
            torch.mps.synchronize()

    @torch.no_grad()
    def search(self, queries, k):
        ids, scores = [], []
        for i in range(0, len(queries), self.chunk):
            top = torch.topk(torch.from_numpy(queries[i:i + self.chunk]).to(self.device) @ self.vectors.T, k, dim=1)
            ids.append(top.indices.cpu())
            scores.append(top.values.cpu())
        return torch.cat(ids).numpy(), torch.cat(scores).numpy()

    def memory_bytes(self):
        return self.vectors.numel() * 4

    def latency_ms(self, queries, k):
        times = np.empty(len(queries))
        for i, q in enumerate(queries):
            start = time.perf_counter()
            self.search(q[None, :], k)  # .cpu() at the end waits for the GPU
            times[i] = (time.perf_counter() - start) * 1000
        return times


class SearchWorker:
    """A FAISS search process. Build named indexes in it, then search them.

        with SearchWorker() as worker:
            worker.build("ivf", vectors, kind="ivf", nlist=2048, nprobe=32)
            ids, scores = worker.search("ivf", queries, 500)
    """

    def __init__(self):
        root = Path(__file__).resolve().parents[1]  # so `-m recsys.search_worker` resolves from anywhere
        self.process = subprocess.Popen([sys.executable, "-m", "recsys.search_worker"], cwd=root,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def _request(self, **message):
        data = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        self.process.stdin.write(struct.pack("<Q", len(data)) + data)
        self.process.stdin.flush()
        header = self.process.stdout.read(8)
        if len(header) < 8:
            raise RuntimeError(f"search worker exited (code {self.process.poll()})")
        reply = pickle.loads(self.process.stdout.read(struct.unpack("<Q", header)[0]))
        if "error" in reply:
            raise RuntimeError(f"search worker: {reply['error']}")
        return reply

    def build(self, name, vectors, kind, shards=1, **params):
        """Build an index in the worker; returns {"build_seconds", "memory_bytes"}."""
        return self._request(cmd="build", name=name, vectors=np.ascontiguousarray(vectors, dtype=np.float32),
                             kind=kind, shards=shards, **params)

    def search(self, name, queries, k):
        reply = self._request(cmd="search", name=name, queries=np.ascontiguousarray(queries, dtype=np.float32), k=k)
        return reply["ids"], reply["scores"]

    def latency_ms(self, name, queries, k):
        """Per-query search time measured inside the worker (no pipe overhead)."""
        return self._request(cmd="latency", name=name, queries=np.ascontiguousarray(queries, dtype=np.float32),
                             k=k)["ms"]

    def end_to_end_ms(self, name, queries, k):
        """Per-query time as seen from this process: sending h, searching, receiving the result."""
        times = np.empty(len(queries))
        for i, q in enumerate(np.ascontiguousarray(queries, dtype=np.float32)):
            start = time.perf_counter()
            self.search(name, q[None, :], k)
            times[i] = (time.perf_counter() - start) * 1000
        return times

    def drop(self, name):
        self._request(cmd="drop", name=name)

    def close(self):
        if self.process.poll() is None:
            data = pickle.dumps({"cmd": "quit"})
            self.process.stdin.write(struct.pack("<Q", len(data)) + data)
            self.process.stdin.close()
            self.process.wait(timeout=30)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def search_recall(approx_ids, exact_ids):
    """Per query: the share of the exact top-k that the approximate top-k also found."""
    k = exact_ids.shape[1]
    return np.array([len(np.intersect1d(a, e)) / k for a, e in zip(approx_ids, exact_ids)])
