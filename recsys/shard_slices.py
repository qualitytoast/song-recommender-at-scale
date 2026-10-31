"""Cut a serving bundle (recsys/export.py) into per-shard slices: shard i of N holds the songs with
song ID % N == i, and everything per-song for them.

    python -m recsys.shard_slices --bundle bundles/ranker_1m_any_length-<version> --shards 4

Writes <bundle>/shards-N/shard-<i>/:
  info.json                  the bundle version, i, N, how many songs it holds, the catalog size
  ids.npy                    its songs' global IDs (sorted)
  search_index.faiss         an IVF index of its songs' search vectors, under their global IDs, using
                             the bundle index's cluster centres: every shard searches the same clusters
                             for a query, so the shards' top k merged is exactly the whole index's top k
                             (at 16 bits if the bundle's vectors are: settings.json vector_bits)
  retriever_song_inputs.npy, ranker_song_inputs.npy, song_artist.npy, song_album.npy, song_genres.npy
                             the bundle's per-song rows for its songs, in ids.npy's order
  pair_*.npy                 the appeared-together pairs (a, b) whose later song b it holds (recsys/pair_table.py)
Imports numpy and FAISS only (never PyTorch: they can't share a process, see recsys/ann.py).
"""
import argparse
import json
import shutil
from pathlib import Path

import faiss
import numpy as np

from recsys.ann import ivf_index
from recsys.pair_table import save_pairs

PER_SONG = ("retriever_song_inputs", "ranker_song_inputs", "song_artist", "song_album", "song_genres")


def cluster_centres(index_path):
    """The (nlist, d) cluster centres of a saved IVF index."""
    index = faiss.read_index(str(index_path))  # keep it referenced while its parts are read
    ivf = faiss.extract_index_ivf(index)
    return np.array(faiss.downcast_index(ivf.quantizer).reconstruct_n(0, ivf.nlist))


def shard_index(centres, vectors, ids, nprobe, encoding=None):
    """An IVF index over vectors (stored under ids) with the given, already computed, cluster centres; encoding
    "fp16" keeps the vectors at 16 bits, like the bundle's index (each vector's code depends only on it and its
    centre, so a shard's codes equal the whole index's)."""
    quantizer = faiss.IndexFlatIP(centres.shape[1])
    quantizer.add(np.ascontiguousarray(centres, dtype=np.float32))
    index = ivf_index(quantizer, centres.shape[1], len(centres), encoding)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if not index.is_trained:  # the 16-bit encoder: nothing to learn (the centres are given, not clustered again)
        index.train(vectors[:max(len(centres), 1)])
    assert index.is_trained  # the centres are the training: nothing is clustered again
    index.add_with_ids(vectors, np.asarray(ids, dtype=np.int64))
    index.nprobe = nprobe  # (FAISS's Python wrapper keeps the quantizer alive as long as the index)
    return index


def write_slices(bundle, shards):
    """Write the N slices of bundle (see the module docstring); returns the shards-N folder."""
    bundle = Path(bundle)
    settings = json.loads((bundle / "settings.json").read_text())
    version = json.loads((bundle / "manifest.json").read_text())["version"]
    songs = settings["songs"]
    out = bundle / f"shards-{shards}"
    shutil.rmtree(out, ignore_errors=True)
    centres = cluster_centres(bundle / "search_index.faiss")
    vectors = np.load(bundle / "search_vectors.npy", mmap_mode="r")
    rows = {name: np.load(bundle / f"{name}.npy", mmap_mode="r") for name in PER_SONG}
    keys, counts = np.load(bundle / "pair_keys.npy"), np.load(bundle / "pair_counts.npy")
    later = keys % songs
    for i in range(shards):
        folder = out / f"shard-{i}"
        folder.mkdir(parents=True)
        ids = np.arange(i, songs, shards, dtype=np.int64)
        np.save(folder / "ids.npy", ids)
        for name, array in rows.items():
            np.save(folder / f"{name}.npy", np.asarray(array[ids]))
        mine = later % shards == i  # still sorted: a subset of sorted keys
        save_pairs(folder, keys[mine], counts[mine], songs)
        faiss.write_index(shard_index(centres, vectors[ids], ids, settings["ivf_nprobe"],
                                      "fp16" if settings.get("vector_bits") == 16 else None),
                          str(folder / "search_index.faiss"))
        (folder / "info.json").write_text(json.dumps({"version": version, "index": i, "shards": shards,
                                                      "songs": len(ids), "catalog": songs}))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cut a serving bundle into per-shard slices.")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--shards", type=int, required=True)
    args = ap.parse_args()
    print(write_slices(args.bundle, args.shards))
