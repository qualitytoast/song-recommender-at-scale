"""A listener's taste profile (personalization A), the serving side: what a profile holds and the two ways a
request that names one is nudged toward it. Built by recsys.personal from the listener's history; a request
without a profile is answered exactly as before.

- Taste vector: the average of the retriever's "what comes next" vectors over the listener's past sessions
  (the same kind of vector a request's songs give, so it blends in directly). The request's query becomes
  (1 - w) * its own + w * taste, which pulls the shortlist toward the listener's taste.
- After the ranker: each candidate's score + alpha * the listener's affinity for its artist (0-1: how much
  they play the artist) - beta * how often they skip the song (0-1, for songs played 3+ times), re-sorted.
"""
import hashlib
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Profile:
    key: str                      # what a request names it by (on the site, a long random secret)
    bundle: str                   # the bundle version it was built for: song and artist IDs are that bundle's
    taste: np.ndarray             # (embed_dim + 1,), a 1 last, like a query vector
    artist_affinity: np.ndarray   # (artists,) 0-1
    skip_rate: np.ndarray         # (songs,) 0-1
    w: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    digest: str = ""              # a hash of the arrays: in cache keys, so a rebuilt profile isn't served stale

    def with_settings(self, w, alpha, beta):
        return replace(self, w=float(w), alpha=float(alpha), beta=float(beta))

    def cache_id(self):
        return f"{self.key}:{self.digest}:{self.w}:{self.alpha}:{self.beta}"


def digest_of(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:12]


def make_profile(key, bundle, taste, artist_affinity, skip_rate, w=0.0, alpha=0.0, beta=0.0):
    taste, affinity, skip = (np.asarray(a, dtype=np.float32) for a in (taste, artist_affinity, skip_rate))
    return Profile(key, bundle, taste, affinity, skip, float(w), float(alpha), float(beta),
                   digest_of(taste, affinity, skip))


def save_profile(path, profile):
    np.savez(path, key=profile.key, bundle=profile.bundle, taste=profile.taste,
             artist_affinity=profile.artist_affinity, skip_rate=profile.skip_rate,
             settings=np.array([profile.w, profile.alpha, profile.beta]))


def load_profile(path):
    with np.load(path) as f:
        w, alpha, beta = f["settings"]
        return make_profile(str(f["key"]), str(f["bundle"]), f["taste"], f["artist_affinity"], f["skip_rate"],
                            w, alpha, beta)


def blend(h, taste, w):
    """Query vectors h (n, d + 1) pulled toward taste vectors: (1 - w) h + w taste, per row. taste: (d + 1,) or
    (n, d + 1); w: one value or one per row (0 leaves a row exactly as it was). The 1 in the last column stays 1."""
    w = np.broadcast_to(np.asarray(w, dtype=np.float64), (len(h),))[:, None]
    if not w.any():
        return h
    return ((1 - w) * h + w * np.broadcast_to(taste, h.shape)).astype(h.dtype)


def adjust(profile, ids, scores, song_artist):
    """A ranked list (ids, scores: best first) with the profile's artist bonus and skip penalty, re-sorted
    (ties keep their order)."""
    if not (profile.alpha or profile.beta):
        return ids, scores
    scores = (scores + profile.alpha * profile.artist_affinity[song_artist[ids]]
              - profile.beta * profile.skip_rate[ids]).astype(scores.dtype)
    order = np.argsort(-scores, kind="stable")
    return ids[order], scores[order]
