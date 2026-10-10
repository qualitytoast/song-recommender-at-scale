"""Recommendations from a serving bundle (recsys/export.py): the two-stage pair without the training
data or the full checkpoints.

    rec = Recommender("bundles/ranker_1m_any_length-<version>", device="cpu")
    rec.recommend(["spotify:track:...", ...], name="road trip")   # top 10, as a dict
    rec.close()

A request's songs become song IDs (unknown ones are left out and listed); its last input_length known
songs and the playlist name go through the retriever (playlist vector h), the search process finds the
top shortlist songs (input songs left out), and the ranker reorders them; the top 10 are the answer.
Per-song tables are the bundle's precomputed input vectors (one row per song), so the models here hold
only their small shared layers plus those rows.
"""
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from recsys.data import encode_names
from recsys.model import SongRecommender
from recsys.rank_features import CandidateFeatures
from recsys.ranker import CandidateRanker, drop_input_songs
from recsys.search import SearchWorker, query_vectors

TOP = 10  # recommendations per request


class NoKnownSongs(ValueError):
    """None of a request's songs are in the catalog."""

    def __init__(self, ignored):
        super().__init__("none of the songs are in the catalog")
        self.ignored = ignored


def serving_retriever(settings, song_inputs):
    """A retriever whose input side is one precomputed vector per song (song_inputs (songs, d)): a
    featureless SongRecommender whose song table holds those vectors, with no output table (search
    vectors do the scoring). Its shared layers are loaded separately (load_shared)."""
    model = SongRecommender(vocab_size=1, embed_dim=settings["embed_dim"], context_length=settings["input_length"],
                            num_layers=settings["num_layers"], dropout=0.0,
                            scale_attention=settings["scale_attention"], init="pytorch",
                            name_word_count=settings["name_word_count"], causal=settings["causal"])
    model.song_embedding = nn.Embedding.from_pretrained(torch.as_tensor(song_inputs), freeze=True)
    return model.eval()


def serving_ranker(settings, song_inputs):
    """The ranker on a serving_retriever built from its own (fine-tuned) input vectors."""
    return CandidateRanker(serving_retriever(settings, song_inputs), settings["n_features"],
                           settings["correction_hidden"]).eval()


# What a serving model keeps of a trained one, besides the precomputed per-song input vectors: the
# transformer layers, position and name-word vectors, and the ranker's own new weights and input
# statistics. Every per-song table (song, artist, album, length, genre, output) is left behind.
SHARED = ("blocks.", "position_embedding.", "name_words.", "candidate_marker", "candidate_position",
          "correction.", "feature_mean", "feature_std")
# A serving model's own per-song tables: its song table holds the precomputed input vectors and its
# output table is a stand-in (search vectors do the scoring).
SERVING_PER_SONG = ("song_embedding.", "output.")
# Extra-input groups a request can supply (the others need playlist history or part-A playlists).
SERVED_GROUPS = ("retriever_score", "overlap", "cooccurrence")


def shared_state(model):
    """The weights of a trained retriever or ranker that its serving copy needs (SHARED)."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k.removeprefix("retriever.").startswith(SHARED)}


def load_shared(model, state):
    """Load shared_state into a serving model; only its own per-song tables may be left out."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.removeprefix("retriever.").startswith(SERVING_PER_SONG) for k in missing):
        raise ValueError(f"bundle weights don't fit the serving model: missing {missing}, unexpected {unexpected}")


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


class Recommender:
    """A loaded bundle: the two models on `device`, the ranker's extra-input tables, the song
    lookups, and a search process holding the saved IVF index. Requests are answered one at a
    time (a lock), since the models and the search process serve one request at a time."""

    def __init__(self, bundle, device="cpu"):
        self.path = Path(bundle)
        self.device = torch.device(device)
        load = lambda name: np.load(self.path / f"{name}.npy")
        self.settings = s = json.loads((self.path / "settings.json").read_text())
        self.version = json.loads((self.path / "manifest.json").read_text())["version"]
        self.retriever = serving_retriever(s, load("retriever_song_inputs")).to(self.device)
        load_shared(self.retriever, torch.load(self.path / "retriever_shared.pt", weights_only=True))
        self.ranker = serving_ranker(s, load("ranker_song_inputs")).to(self.device)
        load_shared(self.ranker, torch.load(self.path / "ranker_shared.pt", weights_only=True))
        self.features = CandidateFeatures(s["features"], s["songs"], None, (load("pair_keys"), load("pair_counts")),
                                          self.device, song_artist=load("song_artist"), song_album=load("song_album"),
                                          song_genres=load("song_genres"))
        songs = json.loads((self.path / "songs.json").read_text())
        self.uris, self.tracks = songs["uris"], songs["tracks"]
        self.artists = [songs["artist_names"][a] for a in load("song_artist")]
        self.song_id = {uri: i for i, uri in enumerate(self.uris)}
        words = json.loads((self.path / "name_words.json").read_text())
        self.word_id = {w: i + 1 for i, w in enumerate(words)}  # 0 is padding, as in training
        self.search = SearchWorker()
        self.search.load("songs", (self.path / "search_index.faiss").resolve(), s["ivf_nprobe"])
        self.lock = threading.Lock()

    def close(self):
        self.search.close()

    def encode_name(self, name):
        """(1, width) name-word IDs, as in training: unknown words left out, all padding if none known."""
        return encode_names([name or ""], self.word_id)

    @torch.no_grad()
    def recommend_ids(self, song_ids, name_ids):
        """The top 10 for a playlist's songs (song IDs, oldest first; the last input_length are read)
        and its name-word IDs (1, width): (song IDs, ranker scores, timings in ms by stage)."""
        times, clock = {}, [time.perf_counter()]

        def mark(stage):
            sync(self.device)
            now = time.perf_counter()
            times[stage] = (now - clock[0]) * 1000
            clock[0] = now
        s, device = self.settings, self.device
        X = np.asarray(song_ids[-s["input_length"]:], dtype=np.int64)[None, :]
        N = np.asarray(name_ids, dtype=np.int64).reshape(1, -1)
        h = query_vectors(self.retriever, X, N, device)  # (1, d + 1): the playlist vector with a 1 appended
        mark("retriever")
        extra = X.shape[1] if s["exclude_input"] else 0  # search deeper by up to that many input songs dropped
        ids, scores = self.search.search("songs", h, s["shortlist"] + extra)
        if s["exclude_input"]:
            ids, scores = drop_input_songs(ids, scores, X, None, s["shortlist"])
        mark("search")
        context, cands = torch.from_numpy(X).to(device), torch.from_numpy(ids).long().to(device)
        base = torch.from_numpy(np.ascontiguousarray(scores, dtype=np.float32)).to(device)
        ranks = torch.arange(1, cands.shape[1] + 1, device=device)[None, :]  # the shortlist's own order
        f = self.features(context, cands, base, ranks, base[:, 0], playlist_len=torch.tensor([X.shape[1]], device=device))
        mark("ranker inputs")
        rerank = self.ranker(context, torch.from_numpy(N).to(device), cands, base, f).cpu().numpy()[0]
        mark("ranker")
        order = np.argsort(-rerank, kind="stable")[:TOP]
        return ids[0, order], rerank[order], times

    def recommend(self, uris, name=None):
        """The response to a request: the top 10 songs {uri, track, artist, score}, the songs read
        and the ones left out (not in the catalog), the bundle version, and the time per stage."""
        start = time.perf_counter()
        known = [(u, self.song_id[u]) for u in uris if u in self.song_id]
        ignored = [u for u in uris if u not in self.song_id]
        if not known:
            raise NoKnownSongs(ignored)
        used = known[-self.settings["input_length"]:]
        name_ids = self.encode_name(name)
        lookup_ms = (time.perf_counter() - start) * 1000
        with self.lock:
            ids, scores, times = self.recommend_ids([i for _, i in used], name_ids)
        times = {"lookup": lookup_ms, **times}
        times["total"] = (time.perf_counter() - start) * 1000
        return {"recommendations": [{"uri": self.uris[j], "track": self.tracks[j], "artist": self.artists[j],
                                     "score": float(x)} for j, x in zip(ids.tolist(), scores)],
                "used": [u for u, _ in used], "ignored": ignored, "version": self.version,
                "timings_ms": {k: round(v, 3) for k, v in times.items()}}
