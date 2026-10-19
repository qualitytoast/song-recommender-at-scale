"""A tiny serving bundle for the serving and sharding tests."""
import numpy as np
import torch

from recsys.export import write_bundle
from recsys.model import SongRecommender
from recsys.rank_features import pair_counts
from recsys.ranker import CandidateRanker

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


def write_tiny_bundle(root, names=True, settings=None):
    """(bundle path, retriever, ranker, pairs) for the tiny pair, written under root (settings: added to
    SETTINGS)."""
    retriever, ranker, pairs = tiny_pair(names)
    path = write_bundle(root, "tiny", retriever, ranker, {**SETTINGS, **(settings or {})},
                        [f"spotify:track:{i}" for i in range(V)], TRACKS,
                        ARTISTS, WORDS if names else [], pairs, PLAYLISTS, CPU)
    return path, retriever, ranker, pairs
