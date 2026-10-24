"""Recommendations from a serving bundle (recsys/export.py): the two-stage pair without the training
data or the full checkpoints.

    rec = Recommender("bundles/ranker_1m_any_length-<version>", device="cpu")
    rec.recommend(["spotify:track:...", "Nightcall - Kavinsky"], name="road trip")   # top 10, as a dict
    rec.find_songs("nightc")   # songs whose name or artist starts with it (or, with rec.song_search set, typo-
                               # tolerant matches), most played first
    rec.close()

A request's songs (Spotify URIs, or "Track - Artist" text) become song IDs (unknown ones are left out
and listed); its last input_length known
songs and the playlist name go through the retriever (playlist vector h: its guesses at the next song
after each of the songs, mixed, see MIX), the search process finds the top shortlist songs (input songs
left out), and the ranker reorders them; the top 10 are the answer, with at most max_per_artist songs by
any one artist (pick_top).
Per-song tables are the bundle's precomputed input vectors (one row per song), so the models here hold
only their small shared layers plus those rows.
"""
import bisect
import json
import threading
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
from torch import nn

from recsys.admission import Overloaded  # noqa: F401  (raised through compute)
from recsys.data import encode_names
from recsys.model import SongRecommender
from recsys.rank_features import CandidateFeatures
from recsys.ranker import CandidateRanker, drop_input_songs
from recsys.search import SearchWorker, mixed_query_vectors
from recsys.taste import adjust, blend, load_profile

TOP = 10  # recommendations per request
# MAX_PER_ARTIST and MIX: the defaults for a bundle without its own (settings.json's max_per_artist and mix,
# set at export and tuned per model; a request can still set either). Both were tuned on the MPD pair.
# At most this many of the 10 by one artist, unless a request says otherwise (0: no limit). The models
# learned to predict the very next song, which is often by the last song's artist (people add albums), so
# the 10 most likely next songs alone are mostly that artist's.
MAX_PER_ARTIST = 2
# The retriever's guess after the last song alone follows that song (the very next song mostly depends on
# the one before it); mixing in its guesses after the earlier songs makes the shortlist fit the whole
# playlist. Each earlier song's guess weighs MIX times the next one's (search.mixed_query_vectors); 0 is
# the last song's guess alone, as in training and the offline evaluation. 0.5 had the most recommendations
# that the held-out playlists went on to contain (scripts/playlist_eval.py).
MIX = 0.5
LAST = "\U0010ffff"  # sorts after every character: keys starting with p lie in [p, p + LAST)


def name_key(text):
    """How song and artist names are compared: accents removed, case folded, spaces collapsed
    ("Beyoncé  Knowles" -> "beyonce knowles")."""
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return " ".join(text.casefold().split())


def splits(key):
    """(track, artist) for each " - " in a name key: "a - b - c" -> ("a", "b - c"), ("a - b", "c")."""
    out, i = [], key.find(" - ")
    while i > 0:
        out.append((key[:i], key[i + 3:]))
        i = key.find(" - ", i + 1)
    return [(t, a) for t, a in out if t and a]


def pick_top(ids, scores, song_artist, max_per_artist, k=TOP):
    """The top k of a ranked list (ids, scores: best first), skipping a song once max_per_artist songs by
    its artist are in (0: no limit). Too few artists to fill k: the best skipped songs fill the rest."""
    if not max_per_artist:
        return ids[:k], scores[:k]
    artists = song_artist[ids]
    count = {}
    keep = np.zeros(len(ids), dtype=bool)
    for i, a in enumerate(artists.tolist()):
        if count.get(a, 0) < max_per_artist:
            count[a] = count.get(a, 0) + 1
            keep[i] = True
            if keep.sum() == k:
                break
    chosen = np.flatnonzero(keep)
    if len(chosen) < k:
        chosen = np.sort(np.concatenate([chosen, np.flatnonzero(~keep)[:k - len(chosen)]]))
    return ids[chosen], scores[chosen]


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


class Stopwatch:
    """Milliseconds per stage of a request (waiting for the GPU first, so each stage is its own)."""

    def __init__(self, device):
        self.device, self.times, self.last = device, {}, time.perf_counter()

    def mark(self, stage):
        sync(self.device)
        now = time.perf_counter()
        self.times[stage] = (now - self.last) * 1000
        self.last = now


class Catalog:
    """What every recommender front end needs besides the models: the bundle's settings and version,
    the song lookups (URI <-> ID, track and artist names), finding songs by name, the playlist-name
    words, and turning a request into a response (recommend). Subclasses answer recommend_ids."""

    def __init__(self, bundle, device="cpu"):
        self.path = Path(bundle)
        self.device = torch.device(device)
        self.settings = json.loads((self.path / "settings.json").read_text())
        self.version = json.loads((self.path / "manifest.json").read_text())["version"]
        # a request's artist limit and mix unless it sets them: the bundle's own, if it has them
        self.max_per_artist = self.settings.get("max_per_artist", MAX_PER_ARTIST)
        self.mix = self.settings.get("mix", MIX)
        songs = json.loads((self.path / "songs.json").read_text())
        self.uris, self.tracks = songs["uris"], songs["tracks"]
        self.song_artist = self.load("song_artist")
        self.artists = [songs["artist_names"][a] for a in self.song_artist]
        self.song_id = {uri: i for i, uri in enumerate(self.uris)}
        # Finding songs by name: song IDs sorted by track name key and by artist name key (each
        # most-played first among equal names), with the keys in that order for binary search.
        self.playlists = self.load("song_playlists")
        self.popularity = self.settings.get("popularity", "playlists")  # what self.playlists counts
        keys = json.loads((self.path / "name_keys.json").read_text())
        self.by_track, self.by_artist = self.load("songs_by_track"), self.load("songs_by_artist")
        self.track_keys = [keys["tracks"][i] for i in self.by_track]
        self.artist_keys = [keys["artists"][self.song_artist[i]] for i in self.by_artist]
        words = json.loads((self.path / "name_words.json").read_text())
        self.word_id = {w: i + 1 for i, w in enumerate(words)}  # 0 is padding, as in training
        self.lock = threading.Lock()
        self.cache = None  # a recsys.cache.Cache shared by every front desk, if set
        self.song_search = None  # a recsys.song_search.SongSearch (typo-tolerant), if set: the menu asks it first
        self.profiles = {}  # key -> recsys.taste.Profile: a request naming one is personalized (load_profiles)
        self.batcher = None  # a recsys.batching.Batcher, if requests are answered in batches
        self.answers = {"total": 0, "partial": 0, "cached": 0}  # counts for /health
        self.admission = None  # a recsys.admission.Admission: requests that waited too long are dropped

    def load(self, name):
        return np.load(self.path / f"{name}.npy")

    def close(self):
        pass

    def load_profiles(self, folder):
        """Load the taste profiles in folder (*.npz, recsys.taste) built for this bundle version; returns how many
        are loaded. One built for another version is skipped: its song and artist IDs don't fit this bundle."""
        for path in sorted(Path(folder).glob("*.npz")):
            profile = load_profile(path)
            if profile.bundle == self.version:
                self.profiles[profile.key] = profile
        return len(self.profiles)

    def _starting(self, keys, order, prefix):
        """Song IDs whose key (in sorted keys, songs in order) starts with prefix."""
        return order[bisect.bisect_left(keys, prefix):bisect.bisect_left(keys, prefix + LAST)]

    def _exactly(self, keys, order, key):
        return order[bisect.bisect_left(keys, key):bisect.bisect_right(keys, key)]

    def _most_played(self, ids, limit):
        """ids (no repeats) in the most playlists first (then lower ID), at most limit of them."""
        if len(ids) > limit:
            ids = ids[np.argpartition(-self.playlists[ids], limit - 1)[:limit]]
        return ids[np.lexsort((ids, -self.playlists[ids]))]

    def find_songs(self, query, limit=10):
        """Songs for a menu as the user types: [{uri, track, artist, and the count: "playlists" or "listens",
        as self.popularity says}]. With a search set, its answer (typed words in any order across track,
        artist and album names, typos allowed: recsys.song_search); without one, or when it can't answer,
        the songs whose track or artist name starts with query (or, for "track - artist" text, whose track
        starts with the part before " - " and artist with the part after), most played first. Cached, if a
        cache is set (a fallback answer isn't: it mustn't outlive the search's outage)."""
        q = name_key(query)
        if not q:
            return []
        key = f"songs:{self.version}:{limit}:{q}"
        if self.cache and (found := self.cache.get(key)) is not None:
            return found
        ids = self.song_search.search(query, limit) if self.song_search else None
        found = [self.describe(j) for j in ids] if ids is not None else self._find_songs(q, limit)
        if self.cache and (ids is not None or not self.song_search):
            self.cache.set(key, found)
        return found

    def _find_songs(self, q, limit):
        parts = [self._most_played(self._starting(self.track_keys, self.by_track, q), limit),
                 self._most_played(self._starting(self.artist_keys, self.by_artist, q), limit)]
        for t, a in splits(q):
            both = np.intersect1d(self._starting(self.track_keys, self.by_track, t),
                                  self._starting(self.artist_keys, self.by_artist, a))
            parts.append(self._most_played(both, limit))
        ids = self._most_played(np.unique(np.concatenate(parts)), limit)
        return [self.describe(j) for j in ids.tolist()]

    def describe(self, j):
        return {"uri": self.uris[j], "track": self.tracks[j], "artist": self.artists[j],
                self.popularity: int(self.playlists[j])}

    def resolve(self, song):
        """The song ID for a request's song: a Spotify URI in the catalog, or "Track - Artist" text (or
        just a track name) matching a song's names exactly (ignoring case and accents), the most-played
        one if several do; None if nothing matches."""
        if song in self.song_id:
            return self.song_id[song]
        q = name_key(song)
        if not q or song.startswith("spotify:"):
            return None
        matches = [np.intersect1d(self._exactly(self.track_keys, self.by_track, t),
                                  self._exactly(self.artist_keys, self.by_artist, a)) for t, a in splits(q)]
        found = np.concatenate(matches) if matches else np.zeros(0, dtype=np.int64)
        if len(found) == 0:  # no "track - artist" match: a track name on its own
            found = self._exactly(self.track_keys, self.by_track, q)
        return int(self._most_played(np.unique(found), 1)[0]) if len(found) else None

    def encode_name(self, name):
        """(1, width) name-word IDs, as in training: unknown words left out, all padding if none known."""
        return encode_names([name or ""], self.word_id)

    def recommend_ids(self, song_ids, name_ids, mix=MIX, taste=None, w=0.0):
        """The reranked shortlist for a playlist's songs (song IDs, oldest first; the last input_length are
        read) and its name-word IDs (1, width), best first, with the retriever's guesses mixed by mix (MIX) and,
        with a taste vector, the query pulled toward it by w (recsys.taste.blend): (song IDs, ranker scores,
        timings in ms by stage, notes for the response: {"partial": bool} and, if partial, "missing_shards")."""
        raise NotImplementedError

    def recommend(self, songs, name=None, arrived=None, max_per_artist=None, mix=None, profile=None):
        """The response to a request (songs: Spotify URIs or "Track - Artist" text, oldest first): the
        top 10 songs {uri, track, artist, score} with at most max_per_artist by one artist (0: no limit),
        the songs read {uri, track, artist} and the ones left out (no match in the catalog), the bundle
        version, and the time per stage. mix: how much the retriever's guesses after earlier songs count
        (MIX). None for either: the bundle's default. profile: a loaded profile's key (self.profiles) for an
        answer nudged toward that listener's taste (recsys.taste); an unknown key gets the usual answer, and
        "personalized" says which. arrived: when the request came in (time.monotonic()), for dropping it if it
        waits too long (Overloaded)."""
        max_per_artist = self.max_per_artist if max_per_artist is None else max_per_artist
        mix = self.mix if mix is None else mix
        profile = self.profiles.get(profile) if profile else None
        start = time.perf_counter()
        arrived = time.monotonic() if arrived is None else arrived
        ids = [self.resolve(s) for s in songs]
        known = [(s, j) for s, j in zip(songs, ids) if j is not None]
        ignored = [s for s, j in zip(songs, ids) if j is None]
        if not known:
            raise NoKnownSongs(ignored)
        used = known[-self.settings["input_length"]:]
        name_ids = self.encode_name(name)
        lookup_ms = (time.perf_counter() - start) * 1000
        answer = {"used": [{k: v for k, v in self.describe(j).items() if k != self.popularity} for _, j in used],
                  "ignored": ignored, "version": self.version, "personalized": profile is not None}
        key = (f"rec:{self.version}:{','.join(str(j) for _, j in used)}:{','.join(map(str, name_ids[0]))}"
               f":a{max_per_artist}:m{mix}" + (f":p{profile.cache_id()}" if profile else ""))
        self.answers["total"] += 1
        if self.cache and (cached := self.cache.get(key)) is not None:
            self.answers["cached"] += 1
            return {"recommendations": cached, **answer, "partial": False, "cached": True,
                    "timings_ms": {"lookup": round(lookup_ms, 3), "total": round((time.perf_counter() - start) * 1000, 3)}}
        times, recommendations, notes = self.compute(used, name_ids, arrived, max_per_artist, mix, profile)
        self.answers["partial"] += notes["partial"]
        if self.cache and not notes["partial"]:  # a partial answer mustn't outlive the shard's outage
            self.cache.set(key, recommendations)
        times = {"lookup": lookup_ms, **times}
        times["total"] = (time.perf_counter() - start) * 1000
        return {"recommendations": recommendations, **answer, **notes, "cached": False,
                "timings_ms": {k: round(v, 3) for k, v in times.items()}}

    def compute(self, used, name_ids, arrived=None, max_per_artist=MAX_PER_ARTIST, mix=MIX, profile=None):
        """(timings, the top 10 as dicts (pick_top), notes) for the resolved songs `used` and name-word IDs;
        in a batch with other requests if a batcher is set, else one request at a time. With an admission
        policy, a request that waited too long to start raises Overloaded instead. With a profile
        (recsys.taste), its taste vector pulls the query and its artist bonus and skip penalty adjust the
        ranker's scores before the top 10 are picked."""
        arrived = time.monotonic() if arrived is None else arrived
        taste, w = (profile.taste, profile.w) if profile else (None, 0.0)
        if self.batcher is not None:
            deadline = self.admission.deadline(arrived) if self.admission else None
            ids, scores, times, notes = self.batcher.submit(([i for _, i in used], name_ids, mix, taste, w), deadline)
        else:
            with self.lock:
                if self.admission:
                    self.admission.check_started(arrived)
                ids, scores, times, notes = self.recommend_ids([i for _, i in used], name_ids, mix, taste, w)
        if profile:
            ids, scores = adjust(profile, ids, scores, self.song_artist)
        ids, scores = pick_top(ids, scores, self.song_artist, max_per_artist)
        return times, [{"uri": self.uris[j], "track": self.tracks[j], "artist": self.artists[j], "score": float(x)}
                       for j, x in zip(ids.tolist(), scores)], notes


class Recommender(Catalog):
    """A loaded bundle in one process: the two models on `device`, the ranker's extra-input tables,
    and a search process holding the saved IVF index. Requests are answered one at a time (a lock),
    since the models and the search process serve one request at a time."""

    def __init__(self, bundle, device="cpu"):
        super().__init__(bundle, device)
        s = self.settings
        inputs = lambda name: self.load(name).astype(np.float32, copy=False)  # 16-bit bundles: float32 here
        self.retriever = serving_retriever(s, inputs("retriever_song_inputs")).to(self.device)
        load_shared(self.retriever, torch.load(self.path / "retriever_shared.pt", weights_only=True))
        self.ranker = serving_ranker(s, inputs("ranker_song_inputs")).to(self.device)
        load_shared(self.ranker, torch.load(self.path / "ranker_shared.pt", weights_only=True))
        self.features = CandidateFeatures(s["features"], s["songs"], None, (self.load("pair_keys"),
                                          self.load("pair_counts")), self.device, song_artist=self.load("song_artist"),
                                          song_album=self.load("song_album"), song_genres=self.load("song_genres"))
        self.search = SearchWorker()
        self.search.load("songs", (self.path / "search_index.faiss").resolve(), s["ivf_nprobe"])

    def close(self):
        self.search.close()

    @torch.no_grad()
    def recommend_ids(self, song_ids, name_ids, mix=MIX, taste=None, w=0.0):
        clock = Stopwatch(self.device)
        s, device = self.settings, self.device
        X = np.asarray(song_ids[-s["input_length"]:], dtype=np.int64)[None, :]
        N = np.asarray(name_ids, dtype=np.int64).reshape(1, -1)
        h = mixed_query_vectors(self.retriever, X, N, device, mix)  # (1, d + 1): playlist vector, a 1 appended
        if taste is not None:
            h = blend(h, taste, w)
        clock.mark("retriever")
        extra = X.shape[1] if s["exclude_input"] else 0  # search deeper by up to that many input songs dropped
        ids, scores = self.search.search("songs", h, s["shortlist"] + extra)
        if s["exclude_input"]:
            ids, scores = drop_input_songs(ids, scores, X, None, s["shortlist"])
        clock.mark("search")
        context, cands = torch.from_numpy(X).to(device), torch.from_numpy(ids).long().to(device)
        base = torch.from_numpy(np.ascontiguousarray(scores, dtype=np.float32)).to(device)
        ranks = torch.arange(1, cands.shape[1] + 1, device=device)[None, :]  # the shortlist's own order
        f = self.features(context, cands, base, ranks, base[:, 0], playlist_len=torch.tensor([X.shape[1]], device=device))
        clock.mark("ranker inputs")
        rerank = self.ranker(context, torch.from_numpy(N).to(device), cands, base, f).cpu().numpy()[0]
        clock.mark("ranker")
        order = np.argsort(-rerank, kind="stable")
        return ids[0, order], rerank[order], clock.times, {"partial": False}
