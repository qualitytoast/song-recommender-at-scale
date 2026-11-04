"""A tiny serving bundle for the serving and sharding tests, and shard server processes for it."""
import signal
import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from recsys.export import write_bundle
from recsys.model import SongRecommender
from recsys.rank_features import pair_counts
from recsys.ranker import CandidateRanker

ROOT = Path(__file__).resolve().parents[1]
V, CPU = 40, torch.device("cpu")
SETTINGS = {"shortlist": 15, "exclude_input": True, "ivf_nlist": 2, "ivf_nprobe": 2,  # every cluster: exact
            "features": ["overlap", "cooccurrence"], "correction_hidden": 4, "windows": "any_length", "sources": {}}
WORDS = ["chill", "gym", "road", "trip", "mix"]
ARTISTS = ["Kavinsky", "London Grammar", "Chromatics", "Beyoncé", "Artist 4", "Artist 5"]
# A few named songs (the rest "Song i"): song i's artist is ARTIST_OF[i]; PLAYLISTS[i] its playlist count.
NAMED = {0: "Nightcall", 1: "Nightcall", 2: "Nightcall - Breakbot Remix", 3: "Night Drive", 4: "Halo", 5: "Odd Look"}
ARTIST_OF = [0, 1, 0, 2, 3, 0]
PLAYLISTS = np.array([50, 5, 3, 20, 30, 10] + [1 + i % 4 for i in range(V - 6)])
TRACKS = [NAMED.get(i, f"Song {i}") for i in range(V)]


def public(notes):
    """A sharded answer's notes without its shard calls' records (frontdesk.ShardedRecommender.call_all)."""
    return {k: v for k, v in notes.items() if k != "shard_calls"}


def tiny_pair(names=True):
    """A retriever with artist, album, length and genre features and name words (unless names=False, as for
    listening sessions), and a ranker on it, every weight random (so each table matters), the ranker's song
    tables fine-tuned apart."""
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    genres = rng.randint(0, 5, (V, 2))
    genres[:4] = 0  # some songs with no genres
    artist = rng.randint(0, 6, V)
    artist[:6] = ARTIST_OF
    features = {"artist": artist, "album": rng.randint(0, 12, V), "duration": rng.randint(0, 4, V)}
    retriever = SongRecommender(V, 8, 3, 2, 0.0, True, "pytorch", features,
                                name_word_count=len(WORDS) if names else None, song_genres=genres, causal=True).eval()
    with torch.no_grad():
        for table in [*retriever.input_features.values(), *retriever.output_features.values(), retriever.input_genres,
                      retriever.output_genres, *([retriever.name_words] if names else [])]:
            table.weight[table.padding_idx is not None:].normal_()
    ranker = CandidateRanker(retriever, n_features=7, hidden=4).eval()
    with torch.no_grad():
        for p in [ranker.candidate_marker, ranker.candidate_position, *ranker.correction.parameters(),
                  ranker.retriever.song_embedding.weight, ranker.retriever.input_features["artist"].weight]:
            p.add_(torch.randn_like(p))
        ranker.feature_mean.normal_()
        ranker.feature_std.uniform_(0.5, 2.0)
    pairs = pair_counts(rng.randint(0, V, 300), np.arange(0, 301, 10), V)
    return retriever, ranker, pairs


def write_tiny_bundle(root, names=True, settings=None, albums=None):
    """(bundle path, retriever, ranker, pairs) for the tiny pair, written under root (settings: added to
    SETTINGS; albums: per song)."""
    retriever, ranker, pairs = tiny_pair(names)
    path = write_bundle(root, "tiny", retriever, ranker, {**SETTINGS, **(settings or {})},
                        [f"spotify:track:{i}" for i in range(V)], TRACKS,
                        ARTISTS, WORDS if names else [], pairs, PLAYLISTS, CPU, albums)
    return path, retriever, ranker, pairs


def free_ports(n):
    """n distinct free ports (all held open while choosing, so none is handed out twice)."""
    sockets = [socket.socket() for _ in range(n)]
    for s in sockets:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in sockets]
    for s in sockets:
        s.close()
    return ports


def slices(bundle, n):
    """Cut the bundle into n shard slices (in its own process: FAISS and PyTorch can't share one)."""
    subprocess.run([sys.executable, "-m", "recsys.shard_slices", "--bundle", str(bundle), "--shards", str(n)],
                   cwd=ROOT, check=True, capture_output=True)
    return [Path(bundle) / f"shards-{n}" / f"shard-{i}" for i in range(n)]


@contextmanager
def shard_servers(folders, with_processes=False, switch_file=None, log=subprocess.DEVNULL):
    """A shard server process per slice folder (following switch_file, if given); yields their addresses (and
    processes)."""
    procs, addresses = [], []
    for folder, port in zip(folders, free_ports(len(folders))):
        procs.append(subprocess.Popen([sys.executable, "-m", "recsys.shard", "--slice", str(folder), "--port", str(port),
                                       "--host", "127.0.0.1", *(["--switch", str(switch_file)] if switch_file else [])],
                                      cwd=ROOT, stdout=log, stderr=log))
        addresses.append(f"127.0.0.1:{port}")
    try:
        yield (addresses, procs) if with_processes else addresses
    finally:
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGCONT)
                p.terminate()
            p.wait(timeout=30)
