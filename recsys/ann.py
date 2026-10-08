"""Approximate nearest-neighbour search with FAISS. Never imports PyTorch.

FAISS and PyTorch each ship their own copy of the OpenMP library (which spreads
work over CPU cores); loading both in one Python process makes OpenMP abort it.
So FAISS runs in its own process (recsys/search_worker.py), which imports only
this module and numpy. The PyTorch side makes the vectors and talks to that
process (recsys.search.SearchWorker).

Songs are searched by maximum dot product (METRIC_INNER_PRODUCT) of
[h, 1] . [v_j, b_j], which equals the retriever's score (see recsys/search.py).

  flat: exact search on the CPU, every song scored
  ivf:  songs grouped into nlist clusters (k-means on the song vectors); a query
        searches only the nprobe clusters whose centres best match it
  hnsw: a graph linking each song to similar songs (hnsw_m links each); a query
        walks it toward higher scores, keeping ef_search candidates (must be >= k)
  shards > 1: the catalog split into that many contiguous parts, one index each,
        searched in parallel threads; their top-k merged into one top-k
"""
import time

import faiss
import numpy as np


def faiss_index(vectors, kind, nlist=None, nprobe=None, hnsw_m=32, ef_search=None, ef_construction=200):
    d = vectors.shape[1]
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if kind == "flat":
        index = faiss.IndexFlatIP(d)
    elif kind == "ivf":
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)  # k-means: the nlist cluster centres
        index.nprobe = nprobe
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
    else:
        raise ValueError(f"unknown index kind {kind!r}")
    index.add(vectors)
    return index


class Index:
    """One FAISS index over the song vectors, or several shards of the catalog."""

    def __init__(self, vectors, kind, shards=1, **params):
        start = time.perf_counter()
        if shards == 1:
            self.parts = [faiss_index(vectors, kind, **params)]
            self.index = self.parts[0]
        else:
            # Shard i gets the i-th contiguous slice of songs; ids continue across shards,
            # so results use the same song ids as the unsplit catalog.
            self.parts = [faiss_index(part, kind, **params) for part in np.array_split(vectors, shards)]
            self.index = faiss.IndexShards(vectors.shape[1], True, True)  # threaded, successive ids
            for part in self.parts:
                self.index.add_shard(part)
        self.build_seconds = time.perf_counter() - start

    def search(self, queries, k):
        """(ids, scores), each (queries, k), best first."""
        scores, ids = self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        return ids, scores

    def memory_bytes(self):
        return sum(len(faiss.serialize_index(part)) for part in self.parts)

    def latency_ms(self, queries, k):
        """Time to answer each query on its own (batch of 1), inside this process."""
        times = np.empty(len(queries))
        for i, q in enumerate(np.ascontiguousarray(queries, dtype=np.float32)):
            start = time.perf_counter()
            self.index.search(q[None, :], k)
            times[i] = (time.perf_counter() - start) * 1000
        return times
