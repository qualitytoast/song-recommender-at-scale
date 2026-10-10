"""Second stage: rerank the retriever's shortlist with a candidate-aware transformer.

Stage 1, the retriever (SongRecommender), scores every song with one dot product
between the playlist vector h and each song's output vector, and keeps the top K
(the shortlist). The top K are found by exact search (every song scored) or by
an IVF index in a FAISS search process (only the songs in the nprobe clusters
nearest the query are scored; recsys/search.py), set by the config's `search`.
Training, validation and held-out shortlists all come from the same search, so
the ranker trains on the kind of shortlist it is tested on.

Stage 2, the ranker, scores each shortlisted song by reading it *together with*
the 10 context songs, so attention can relate the candidate to each song in the
playlist ("same artist as the last song", ...).

The ranker is a copy of the retriever (same layers, starting from its weights) fed
10 context songs + K candidates in one pass, where
  - context songs see only earlier context songs, exactly as in the retriever
    (so they never see a candidate), and
  - each candidate sees the 10 context songs and itself, never another candidate,
so every candidate is scored independently and the context is computed once.
A candidate token is its song + feature vectors, a learned "candidate" vector and
a learned position vector for the next slot.

Attention computes only those allowed scores (ranker_block): per layer, 10 x 10
for the context and 11 per candidate (the 10 context keys, computed once and
shared by every candidate, plus its own key). With 500 candidates that's ~5,600
scores instead of 510 x 510 = 260,100, all but ~5,600 of which a mask would give
zero weight: the same result, without computing what is thrown away.

Score = the retriever's own score for the song (frozen, from building the
shortlist) + a learned correction from the transformer's output at the
candidate's position. The correction layer starts at zero, so before training
the ranker reproduces the retriever's ranking exactly; training learns changes
to it.

Training data: windows from the playlists the retriever never trained on
(data.ranker_split), so the shortlists look like those for new playlists. Each
example is the true next song + `negatives` songs sampled from its shortlist; the
loss picks the true one out of them.

Options, each a config setting, for making the ranker beat a strong retriever:
  train_windows = "in_shortlist": train only on windows whose true song is in the
      shortlist (the ranker can only ever rank songs the retriever passes it)
  shortlist / negatives / negatives_from: how many songs are reranked, how many of them
      each training example's true song is compared against (all: shortlist - 1), and
      from how many of the shortlist's first songs those are drawn
  freeze_tables: keep the copied per-ID tables at the retriever's values
  features: extra per-candidate inputs (recsys/rank_features.py) for the correction
      layer, standardized with statistics from training examples
  correction_hidden: the correction as a small network (this many hidden units) that can
      combine its inputs, instead of a weighted sum of them (0)
  exclude_input: songs in the retriever's input never enter a shortlist
  val_set = "separate": validate on validation windows outside the retriever's validation
      sample, so the windows that picked the retriever's checkpoint don't also judge the ranker

    python -m recsys.ranker --config configs/retriever_ranker_50k.toml            # train every seed, then evaluate
    python -m recsys.ranker --config configs/retriever_ranker_50k.toml --evaluate # evaluate only
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import shutil
import time
import tomllib
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from recsys.baselines import popularity_ranks, popularity_scores
from recsys.config import load_config
from recsys.data import build_dataset
from recsys.evaluate import eval_file, score, with_all_held_out
from recsys.lazy_adam import LazyAdamW, is_table, lr_groups, used_rows
from recsys.model import build_model, rank_and_loss
from recsys.rank_features import (GROUPS, CandidateFeatures, history_matrix, membership_keys, neighbour_lists,
                                  pair_counts, playlist_vectors)
from recsys.search import ExactSearch, SearchWorker, query_vectors, song_vectors
from recsys.train import EarlyStopping, git_state, pick_device, run_dir_for

MISSED = 10**9  # rank given to a true song that isn't in the shortlist: never in any top k


@dataclass(frozen=True)
class RankerConfig:
    retriever: str         # retriever config; seed s uses run_dir_for(retriever, s)/best.pt
    train_seeds: list
    shortlist: int         # K: how many retriever songs the ranker reranks
    search: str            # how the shortlist is found: "exact" or "ivf" (FAISS IVF index)
    ivf_nlist: int         # ivf: clusters the songs are grouped into (0 for exact)
    ivf_nprobe: int        # ivf: clusters searched per query (0 for exact)
    negatives: int         # shortlist songs sampled per training example, besides the true one
    negatives_from: int    # ... drawn from the first this many shortlist songs (= shortlist: all of it)
    optimizer: str         # "adamw" or "lazy_adamw" (only the table rows a batch uses are updated)
    lr: float              # the weights copied from the retriever
    new_lr: float          # the ranker's own new weights (candidate marker and position, correction)
    weight_decay: float
    batch_size: int        # windows per step
    epochs: int            # maximum passes over the ranker's windows
    eval_every_examples: int  # validation check every this many training windows
    min_checks: int
    patience: int
    val_windows: int       # validate on the first this many validation windows
    train_windows: str     # "all", or "in_shortlist": only windows whose true song is in the shortlist
    freeze_tables: bool    # true: the copied per-ID tables keep the retriever's values
    features: list         # extra per-candidate inputs: groups from recsys/rank_features.py ([] = none)
    correction_hidden: int # 0: the correction is a weighted sum of its inputs; else a network with this
                           # many hidden units, which can combine them
    exclude_input: bool    # true: songs in the retriever's input are left out of every shortlist
    val_set: str           # "retriever": the retriever's validation sample; "separate": other validation
                           # windows (data.ranker_validation_keep)

    def __post_init__(self):
        if self.optimizer not in ("adamw", "lazy_adamw"):
            raise ValueError(f"optimizer must be adamw or lazy_adamw, got {self.optimizer!r}")
        if self.search not in ("exact", "ivf"):
            raise ValueError(f"search must be exact or ivf, got {self.search!r}")
        if self.train_windows not in ("all", "in_shortlist"):
            raise ValueError(f"train_windows must be all or in_shortlist, got {self.train_windows!r}")
        if not 1 <= self.negatives < self.negatives_from <= self.shortlist:
            raise ValueError(f"need 1 <= negatives < negatives_from <= shortlist, got {self.negatives}, "
                             f"{self.negatives_from}, {self.shortlist}")
        if self.val_set not in ("retriever", "separate"):
            raise ValueError(f"val_set must be retriever or separate, got {self.val_set!r}")
        if self.correction_hidden < 0:
            raise ValueError(f"correction_hidden must be 0 or more, got {self.correction_hidden}")
        unknown = sorted(set(self.features) - set(GROUPS))
        if unknown:
            raise ValueError(f"unknown features {unknown}; known: {list(GROUPS)}")

    def search_name(self):
        return "exact" if self.search == "exact" else f"ivf nlist={self.ivf_nlist} nprobe={self.ivf_nprobe}"


def load_ranker_config(path):
    with open(path, "rb") as f:
        return RankerConfig(**tomllib.load(f)["ranker"])


class CandidateRanker(nn.Module):
    def __init__(self, retriever, n_features=0, hidden=0):
        super().__init__()
        self.retriever = copy.deepcopy(retriever)  # fine-tuned; starts as the retriever
        dim = self.retriever.output.in_features
        # New parameters start at zero: no random draws, and no effect before training.
        self.candidate_marker = nn.Parameter(torch.zeros(dim))
        self.candidate_position = nn.Parameter(torch.zeros(dim))
        # The correction reads the transformer's output for the candidate and, if used, its
        # n_features extra inputs, standardized with statistics from training examples
        # (set_feature_scale): a weighted sum of them (hidden = 0), or a small network with
        # `hidden` units that can combine them. Its last layer starts at zero, so it still
        # changes nothing before training.
        if hidden:
            self.correction = nn.Sequential(nn.Linear(dim + n_features, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            last = self.correction[-1]
        else:
            self.correction = last = nn.Linear(dim + n_features, 1)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        if n_features:
            self.register_buffer("feature_mean", torch.zeros(n_features))
            self.register_buffer("feature_std", torch.ones(n_features))

    def forward(self, ids, names, candidates, base_scores, features=None):
        """ids (b, L) context songs; names (b, w) name word IDs; candidates (b, C) song
        IDs; base_scores (b, C) the retriever's scores for them; features (b, C, n_features)
        their extra inputs, if the ranker uses them. Returns (b, C) scores."""
        r = self.retriever
        L = ids.shape[1]
        context = r.embed_songs(ids) + r.position_embedding(torch.arange(L, device=ids.device))
        cand = r.embed_songs(candidates) + self.candidate_position + self.candidate_marker
        if r.name_words is not None:
            name = r.mean_vector(r.name_words, names)[:, None, :]  # name at every position, as in the retriever
            context, cand = context + name, cand + name
        context, cand = r.dropout(context), r.dropout(cand)
        later = torch.triu(torch.ones(L, L, dtype=torch.bool, device=ids.device), diagonal=1)
        for block in r.blocks:
            context, cand = ranker_block(block, context, cand, later)
        if features is not None:
            cand = torch.cat([cand, (features - self.feature_mean) / self.feature_std], dim=-1)
        return base_scores + self.correction(cand).squeeze(-1)


def ranker_block(block, context, cand, later):
    """One TransformerBlock over context (b, L, d) and candidates (b, C, d), computing
    only the allowed attention scores: context position i attends to context <= i
    (`later` (L, L) marks the rest); each candidate to the L context songs and itself.
    Returns the new (context, cand)."""
    a = block.attention
    q, k, v = a.query(context), a.key(context), a.value(context)
    cq, ck, cv = a.query(cand), a.key(cand), a.value(cand)
    scale = math.sqrt(q.shape[-1]) if a.scale else 1.0
    weights = ((q @ k.transpose(-2, -1)) / scale).masked_fill(later, float("-inf")).softmax(dim=-1)
    context_out = weights @ v
    # Each candidate: scores against the context keys (b, C, L) and its own key (b, C, 1).
    scores = torch.cat([cq @ k.transpose(-2, -1), (cq * ck).sum(dim=-1, keepdim=True)], dim=-1) / scale
    weights = scores.softmax(dim=-1)
    cand_out = weights[..., :-1] @ v + weights[..., -1:] * cv
    # The rest of the block works on each position alone: the same as the retriever's.
    x = torch.cat([context, cand], dim=1)
    x = block.norm1(x + block.dropout(torch.cat([context_out, cand_out], dim=1)))
    x = block.norm2(x + block.dropout(block.ffn(x)))
    return x[:, :context.shape[1]], x[:, context.shape[1]:]


@contextmanager
def song_search(rc, vectors, device):
    """A function (queries, k) -> (ids, scores), each query's top k songs best first,
    searching the song vectors (recsys.search.song_vectors) the way rc.search says:
    exact on the device, or IVF in a FAISS search process that closes afterwards."""
    if rc.search == "exact":
        yield ExactSearch(vectors, device).search
        return
    with SearchWorker() as worker:
        worker.build("songs", vectors, kind="ivf", nlist=rc.ivf_nlist, nprobe=rc.ivf_nprobe)
        yield lambda queries, k: worker.search("songs", queries, k)


def build_shortlists(model, X, N, Y, device, k, search, vectors, chunk=20000, lengths=None, exclude_input=False):
    """Each window's top-k songs found by `search` (song_search) over `vectors`: (ids
    (n, k) int32, scores (n, k) float32, best first), plus each true song's retriever
    score (n,). X: the retriever's inputs, with lengths if padded (data.window_inputs).
    exclude_input: leave the input's songs out (drop_input_songs). A chunk of windows at a
    time, so only one chunk's results are in flight, written into arrays made up front (joining
    per-chunk pieces at the end would briefly hold two copies: ~5 GB extra at 1M)."""
    ids, scores = np.empty((len(X), k), dtype=np.int32), np.empty((len(X), k), dtype=np.float32)
    true = np.empty(len(X), dtype=np.float32)
    extra = X.shape[1] if exclude_input else 0  # search deeper by up to that many songs dropped
    for i in range(0, len(X), chunk):
        n = None if lengths is None else lengths[i:i + chunk]
        queries = query_vectors(model, X[i:i + chunk], N[i:i + chunk], device, lengths=n)
        top_ids, top_scores = search(queries, k + extra)
        if (top_ids < 0).any():  # FAISS pads with -1 when the searched clusters hold fewer than k songs
            raise RuntimeError(f"search found fewer than {k + extra} songs for some windows; search more clusters")
        if exclude_input:
            top_ids, top_scores = drop_input_songs(top_ids, top_scores, X[i:i + chunk], n, k)
        ids[i:i + chunk], scores[i:i + chunk] = top_ids, top_scores
        true[i:i + chunk] = np.einsum("nd,nd->n", queries, vectors[Y[i:i + chunk]])
    return ids, scores, true


SHORTLIST_CACHE = Path("runs/shortlist_cache")


def file_fingerprint(path, block=1 << 24):
    """SHA-256 of a file's bytes, read a block at a time."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def shortlist_key(rc, retriever_print, X, N, Y, lengths):
    """What decides a set of shortlists: the retriever's weights (its checkpoint's fingerprint),
    how the search runs, how many songs are kept and whether input songs are dropped, and the
    windows themselves. Any change gives a different key."""
    digest = hashlib.sha256(f"shortlists-v1|{retriever_print}|{rc.search_name()}|{rc.shortlist}|"
                            f"{rc.exclude_input}".encode())
    for a in (X, N, Y, lengths):
        digest.update(b"|" if a is None else np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()[:32]


def cached_shortlists(rc, retriever_path, retriever, vectors, device, sets, cache_dir=SHORTLIST_CACHE):
    """build_shortlists for each window set (X, N, Y, lengths) in sets, as a list of (ids, scores,
    true), loaded from cache_dir when an identical set was built before (shortlist_key). The
    search (song_search, which builds the FAISS index) runs only if some set isn't cached.
    Returns (shortlists, how many were loaded from the cache)."""
    retriever_print = file_fingerprint(retriever_path)
    paths = [cache_dir / f"{shortlist_key(rc, retriever_print, *s)}.npz" for s in sets]
    out = [None] * len(sets)
    for i, path in enumerate(paths):
        if path.exists():
            with np.load(path) as f:
                out[i] = (f["ids"], f["scores"], f["true"])
    missing = [i for i, o in enumerate(out) if o is None]
    if missing:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with song_search(rc, vectors, device) as search:
            for i in missing:
                X, N, Y, lengths = sets[i]
                out[i] = build_shortlists(retriever, X, N, Y, device, rc.shortlist, search, vectors, lengths=lengths,
                                          exclude_input=rc.exclude_input)
                np.savez(paths[i], ids=out[i][0], scores=out[i][1], true=out[i][2])
    return out, len(sets) - len(missing)


def drop_input_songs(ids, scores, inputs, lengths, k):
    """The best k of each row's songs ids (n, m) with scores, best first, leaving out songs in
    the row's input (inputs (n, I), its first lengths[r] real, or all of them without lengths)."""
    width = inputs.shape[1]
    real = np.arange(width) < (np.full(len(inputs), width) if lengths is None else lengths)[:, None]
    in_input = ((ids[:, :, None] == inputs[:, None, :]) & real[:, None, :]).any(-1)
    keep = np.argsort(in_input, axis=1, kind="stable")[:, :k]  # songs not in the input first, order kept
    return np.take_along_axis(ids, keep, 1), np.take_along_axis(scores, keep, 1)


def sample_negatives(short_ids, y, n, generator, top=None):
    """For each row, n distinct shortlist positions whose song isn't the true song y,
    chosen uniformly (CPU tensors), from the first `top` positions only if given.
    Returns (rows, n) positions."""
    keys = torch.rand(short_ids.shape, generator=generator)
    keys[short_ids == y[:, None]] = 2.0  # the true song is never a negative
    if top is not None:
        keys[:, top:] = 2.0  # nor is a song past the first `top`
    return torch.topk(keys, n, dim=1, largest=False).indices


def with_negatives(short_ids, short_scores, true_scores, y, n, generator, top=None):
    """One training example per row: (candidates, base scores), each (rows, 1 + n) CPU
    tensors, the true song y first with its retriever score, then n songs sampled from
    its shortlist's first `top` (sample_negatives) with theirs. The loss picks column 0."""
    pos = sample_negatives(short_ids, y, n, generator, top)
    return (torch.cat([y[:, None], short_ids.gather(1, pos).long()], dim=1),
            torch.cat([true_scores[:, None], short_scores.gather(1, pos)], dim=1))


def final_ranks(rerank_scores, short_ids, y):
    """Rank of each true song after reranking its shortlist: 1 + shortlist songs scored
    strictly higher (ties in the true song's favour, as everywhere else); MISSED if the
    retriever didn't shortlist it. numpy in, numpy out."""
    hit = short_ids == y[:, None]
    in_list = hit.any(axis=1)
    true_score = np.where(in_list, (rerank_scores * hit).sum(axis=1), np.inf)
    ranks = (rerank_scores > true_score[:, None]).sum(axis=1) + 1
    return np.where(in_list, ranks, MISSED)


@torch.no_grad()
def rerank(ranker, X, N, cands, base, device, batch=64, features=None):
    """The ranker's scores for each window's candidates cands (n, C), whose retriever scores
    are base (n, C), as (n, C) numpy. features: feature_fn(...) for the same windows, or None."""
    ranker.eval()
    out = []
    for s in range(0, len(X), batch):
        i = np.arange(s, min(s + batch, len(X)))
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a[i])).to(device)
        context, c, b = t(X), t(cands).long(), t(base)
        f = None if features is None else features(i, context, c, b)
        out.append(ranker(context, t(N), c, b, f).cpu())
    return torch.cat(out).numpy()


def shortlisted(short_ids, y):
    """(n,) bool: whether each window's true song y is in its shortlist short_ids (numpy)."""
    return (short_ids == y[:, None]).any(axis=1)


def shortlist_ranks(short_ids, cands):
    """(b, C): each candidate's place in its window's shortlist short_ids (b, K), 1 = first,
    K + 1 if it isn't in it (tensors)."""
    match = (short_ids[:, None, :] == cands[:, :, None]).float()
    return torch.where(match.amax(-1) > 0, match.argmax(-1) + 1, short_ids.shape[1] + 1)


def make_featurizer(rc, ds, retriever, device):
    """The extra-input computer for rc.features (recsys/rank_features.py), or None. Popularity
    and pair counts come from the retriever's part-A playlists only."""
    if not rc.features:
        return None
    popularity = np.bincount(ds.Yc_train[ds.Yc_train != -100], minlength=len(ds.vocab))
    no_pairs = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    pairs = (pair_counts(ds.fit_songs, ds.fit_offsets, len(ds.vocab))
             if {"cooccurrence", "history"} & set(rc.features) else no_pairs)
    members = (membership_keys(ds.fit_songs, ds.fit_offsets, len(ds.vocab))
               if "neighbours" in rc.features else None)
    return CandidateFeatures(rc.features, retriever, popularity, pairs, device, ds.song_duration_ms, members)


def window_neighbours(featurizer, ds, retriever, XI, N, LI, device):
    """(windows, NEIGHBOURS) part-A playlists most like each window (rank_features.neighbour_lists),
    from the retriever's inputs XI (lengths LI); None unless the featurizer uses "neighbours"."""
    if featurizer is None or "neighbours" not in featurizer.groups:
        return None
    if featurizer.playlists is None:  # each part-A playlist's average song vector, made once
        featurizer.playlists = playlist_vectors(ds.fit_songs, ds.fit_offsets, song_vectors(retriever))
    return neighbour_lists(query_vectors(retriever, XI, N, device, lengths=LI), featurizer.playlists, device)


def feature_fn(featurizer, short_ids, short_scores, histories, rows, device, neighbours=None):
    """None without a featurizer. Otherwise a function (i, context, cands, base) -> the extra
    inputs for windows i (numpy indices) of a set whose shortlists are short_ids / short_scores
    (numpy, best first), whose playlists so far are entries rows of histories, and whose most
    alike part-A playlists are neighbours (window_neighbours, aligned with i)."""
    if featurizer is None:
        return None
    songs, begin, end = histories

    def features(i, context, cands, base):
        ranks = shortlist_ranks(torch.from_numpy(short_ids[i]).to(device), cands)
        top = torch.from_numpy(short_scores[i, 0]).to(device)
        history = (torch.from_numpy(history_matrix(histories, rows[i])).to(device)
                   if "history" in featurizer.groups else None)
        nbrs = torch.from_numpy(neighbours[i]).to(device) if neighbours is not None else None
        playlist_len = torch.from_numpy(end[rows[i]] - begin[rows[i]]).to(device)
        return featurizer(context, cands, base, ranks, top, history, nbrs, playlist_len)
    return features


def ranker_validation(rc, ds):
    """The ranker's validation windows, the first rc.val_windows of the retriever's validation
    sample (val_set "retriever") or of the windows outside it (val_set "separate"):
    (X, N, Y, XI, LI, histories)."""
    if rc.val_set == "retriever":
        arrays = (ds.X_val, ds.N_val, ds.Y_val, ds.XI_val, ds.LI_val, ds.H_val)
    elif ds.Y_rval is None or len(ds.Y_rval) == 0:
        raise ValueError("val_set = \"separate\" needs validation windows outside the retriever's sample "
                         "(val_max_windows below the number of validation windows)")
    else:
        arrays = (ds.X_rval, ds.N_rval, ds.Y_rval, ds.XI_rval, ds.LI_rval, ds.H_rval)
    v = slice(0, rc.val_windows)
    X, N, Y, XI, LI, (songs, begin, end) = arrays
    return X[v], N[v], Y[v], XI[v], LI[v], (songs, begin[v], end[v])


@torch.no_grad()
def set_feature_scale(ranker, features, i, X, cands, base, device, batch=256):
    """Store each extra input's mean and standard deviation over the candidates cands (with
    retriever scores base) of windows i of X in the ranker, which standardizes inputs with them.
    A constant input keeps a standard deviation of 1."""
    cols = []
    for s in range(0, len(i), batch):
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a[s:s + batch])).to(device)
        context = torch.from_numpy(X[i[s:s + batch]]).to(device)
        cols.append(features(i[s:s + batch], context, t(cands).long(), t(base)).flatten(0, 1).cpu())
    f = torch.cat(cols)
    std = f.std(0)
    ranker.feature_mean.copy_(f.mean(0))
    ranker.feature_std.copy_(torch.where(std < 1e-4, torch.ones_like(std), std))


def weight_lr(rc, name):
    """The learning rate for a ranker parameter: rc.lr for the weights copied from the
    retriever (named "retriever. ..."), rc.new_lr for the ranker's own new ones."""
    return rc.lr if name.startswith("retriever.") else rc.new_lr


def freeze_tables(ranker):
    """Stop training the ranker's copied per-ID tables (lazy_adam.TABLES: songs, artists,
    albums, genres, name words, ...): they keep the retriever's values."""
    for name, p in ranker.retriever.named_parameters():
        if is_table(name):
            p.requires_grad_(False)


def ranker_rows(context, candidates, names, song_features, song_genres, device):
    """The table rows a ranker batch uses (numpy in), for LazyAdamW. The ranker reads
    context and candidate songs alike through the input-side tables; it never uses
    the output-side ones (the retriever's score comes in frozen), so those get none."""
    songs = np.concatenate([context, candidates], axis=1)
    return used_rows(songs, np.empty(0, dtype=np.int64), names, song_features, song_genres, device)


def check_vocab(checkpoint, vocab, path):
    """Song IDs come from the rebuilt dataset's vocab, the model's rows from the one it
    was trained on. If they differ, every score would be meaningless, so refuse.
    (Ranker checkpoints saved before this check have no vocab and are let through.)"""
    if "vocab" in checkpoint and checkpoint["vocab"] != vocab:
        raise ValueError(f"{path} was trained on a different vocab than its config builds now.")


def load_retriever(rc, seed, device, all_held_out=False):
    cfg = with_all_held_out(load_config(rc.retriever), all_held_out)
    ds = build_dataset(cfg)
    path = run_dir_for(rc.retriever, seed) / "best.pt"
    checkpoint = torch.load(path, weights_only=True)
    check_vocab(checkpoint, ds.vocab, path)
    model = build_model(cfg, ds).to(device)
    model.load_state_dict(checkpoint["model"])
    return cfg, ds, model.eval()


def train_ranker(config_path, seed):
    rc = load_ranker_config(config_path)
    run_dir = run_dir_for(config_path, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, run_dir / "config.toml")
    commit, dirty = git_state()
    torch.manual_seed(seed)
    device = pick_device()
    cfg, ds, retriever = load_retriever(rc, seed, device)
    start = time.perf_counter()
    X_val, N_val, Y_val, XI_val, LI_val, H_val = ranker_validation(rc, ds)
    vectors = song_vectors(retriever)
    ((train_ids, train_scores, train_true), (val_ids, val_scores, val_true)), from_cache = cached_shortlists(
        rc, run_dir_for(rc.retriever, seed) / "best.pt", retriever, vectors, device,
        [(ds.XI_rank, ds.N_rank, ds.Y_rank, ds.LI_rank), (XI_val, N_val, Y_val, LI_val)])
    in_list = shortlisted(train_ids, ds.Y_rank)
    rows = np.flatnonzero(in_list) if rc.train_windows == "in_shortlist" else np.arange(len(ds.Y_rank))
    X_tr, N_tr, Y_tr = ds.X_rank[rows], ds.N_rank[rows], ds.Y_rank[rows]
    train_ids, train_scores, train_true = train_ids[rows], train_scores[rows], train_true[rows]
    print(f"{run_dir} | {len(ds.Y_rank):,} ranker windows ({len(rows):,} trained on), {len(Y_val):,} val windows | "
          f"shortlists ({rc.search_name()}{', cached' if from_cache == 2 else ''}) in "
          f"{time.perf_counter() - start:.0f}s | train recall@{rc.shortlist} {in_list.mean():.1%}", flush=True)

    featurizer = make_featurizer(rc, ds, retriever, device)
    ranker = CandidateRanker(retriever, len(featurizer.names) if featurizer else 0, rc.correction_hidden).to(device)
    if rc.freeze_tables:
        freeze_tables(ranker)
    num_params = sum(p.numel() for p in ranker.parameters())
    new_params = num_params - sum(p.numel() for p in retriever.parameters())
    trained = sum(p.numel() for p in ranker.parameters() if p.requires_grad)
    print(f"{num_params:,} params: the retriever's + {new_params:,} new (candidate marker, position, correction); "
          f"{trained:,} trained", flush=True)
    lazy = rc.optimizer == "lazy_adamw"
    if lazy:  # AdamW on only the table rows each batch uses (recsys/lazy_adam.py); frozen tables get none
        optimizer = LazyAdamW(ranker, rc.lr, rc.weight_decay, prefix="retriever.",
                              dense_lr=lambda name: weight_lr(rc, name))
    else:
        trainable = [(n, p) for n, p in ranker.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(lr_groups(trainable, rc.lr, lambda name: weight_lr(rc, name)), lr=rc.lr,
                                      weight_decay=rc.weight_decay)
    song_features = {name: ds.song_features[name][0] for name in retriever.feature_names}
    song_genres = ds.song_genres if retriever.input_genres is not None else None
    stopper = EarlyStopping(rc.min_checks, rc.patience)
    order_gen, negative_gen = torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed + 3_000_000)
    X, N, Y = (torch.from_numpy(a) for a in (X_tr, N_tr, Y_tr))
    short_ids, short_scores, true_scores = (torch.from_numpy(a) for a in (train_ids, train_scores, train_true))
    train_nbrs = window_neighbours(featurizer, ds, retriever, ds.XI_rank[rows], N_tr, ds.LI_rank[rows], device)
    val_nbrs = window_neighbours(featurizer, ds, retriever, XI_val, N_val, LI_val, device)
    train_features = feature_fn(featurizer, train_ids, train_scores, ds.H_rank, rows, device, train_nbrs)
    val_features = feature_fn(featurizer, val_ids, val_scores, H_val, np.arange(len(Y_val)), device, val_nbrs)
    # Validation loss: the training loss on validation windows of the kind trained on (all, or
    # those with the true song shortlisted), their negatives drawn once (own generator, so
    # training's draws don't change) and reused at every check.
    loss_rows = np.arange(len(Y_val)) if rc.train_windows == "all" else np.flatnonzero(shortlisted(val_ids, Y_val))
    val_cands, val_base = (t.numpy() for t in with_negatives(
        *(torch.from_numpy(a[loss_rows]) for a in (val_ids, val_scores, val_true, Y_val)), rc.negatives,
        torch.Generator().manual_seed(seed + 4_000_000), rc.negatives_from))
    loss_features = feature_fn(featurizer, val_ids[loss_rows], val_scores[loss_rows], H_val, loss_rows, device,
                               None if val_nbrs is None else val_nbrs[loss_rows])
    if featurizer is not None:  # standardize the extra inputs with a fixed sample of training examples
        sample = np.sort(np.random.RandomState(seed).choice(len(Y_tr), min(20000, len(Y_tr)), replace=False))
        cands, base = with_negatives(
            *(torch.from_numpy(a[sample]) for a in (train_ids, train_scores, train_true, Y_tr)), rc.negatives,
            torch.Generator().manual_seed(seed + 5_000_000), rc.negatives_from)
        set_feature_scale(ranker, train_features, sample, X_tr, cands.numpy(), base.numpy(), device)
        print("extra inputs (mean / std): " + ", ".join(
            f"{n} {m:.3g}/{s:.3g}" for n, m, s in zip(featurizer.names, ranker.feature_mean.tolist(),
                                                    ranker.feature_std.tolist())), flush=True)
    log, best_check, seen, next_check = [], None, 0, rc.eval_every_examples
    loss_sum, steps, interval_start = torch.zeros((), device=device), 0, time.perf_counter()

    def validate():
        nonlocal loss_sum, steps, interval_start, best_check
        check = len(log)
        train_loss = loss_sum.item() / max(steps, 1)  # mean batch loss since the last check
        if not math.isfinite(train_loss):
            raise FloatingPointError(f"Ranker training loss became {train_loss} by check {check}.")
        train_seconds, val_start = time.perf_counter() - interval_start, time.perf_counter()
        ranks = final_ranks(rerank(ranker, X_val, N_val, val_ids, val_scores, device, features=val_features),
                            val_ids, Y_val)
        result = score(ranks)
        sampled = torch.from_numpy(rerank(ranker, X_val[loss_rows], N_val[loss_rows], val_cands, val_base, device,
                                          features=loss_features))
        val_loss = nn.functional.cross_entropy(sampled, torch.zeros(len(loss_rows), dtype=torch.long)).item()
        val_seconds = time.perf_counter() - val_start
        improved, stop = stopper.update(check, result["ndcg@10"])
        if improved:
            best_check = check
            torch.save({"model": ranker.state_dict(), "vocab": ds.vocab, "config": asdict(rc), "check": check,
                        "val_ndcg": result["ndcg@10"]}, run_dir / "best.pt")
        log.append({"check": check, "examples": seen, "train_loss": train_loss, "val_loss": val_loss,
                    "val_ndcg": result["ndcg@10"],
                    "val_hits10": result["hits@10"], "val_hits1": result["hits@1"],
                    "train_seconds": round(train_seconds, 2), "val_seconds": round(val_seconds, 2)})
        note = "  *best" if improved else (
            f"  no improvement {stopper.bad_checks}/{stopper.patience}" if check >= stopper.min_checks else "")
        print(f"check {check:3d} | {seen:,} windows | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
              f"val NDCG@10 {result['ndcg@10']:.4f} Hits@10 {result['hits@10']:.3f} Hits@1 {result['hits@1']:.3f} | "
              f"train {train_seconds:.0f}s + val {val_seconds:.0f}s{note}", flush=True)
        loss_sum.zero_()
        steps = 0
        ranker.train()
        interval_start = time.perf_counter()
        return stop

    validate()  # check 0, before any training: the ranker equals the retriever here
    stop = False
    for epoch in range(rc.epochs):
        ranker.train()
        order = torch.randperm(len(Y), generator=order_gen)
        for i in range(0, len(Y), rc.batch_size):
            idx = order[i:i + rc.batch_size]
            candidates, base = with_negatives(short_ids[idx], short_scores[idx], true_scores[idx], Y[idx],
                                              rc.negatives, negative_gen, rc.negatives_from)
            context, cands, base = X[idx].to(device), candidates.to(device), base.to(device)
            f = None if train_features is None else train_features(idx.numpy(), context, cands, base)
            scores = ranker(context, N[idx].to(device), cands, base, f)
            loss = nn.functional.cross_entropy(scores, torch.zeros(len(idx), dtype=torch.long, device=device))
            optimizer.zero_grad()
            loss.backward()
            if lazy:
                optimizer.step(ranker_rows(X[idx].numpy(), candidates.numpy(), N[idx].numpy(), song_features,
                                           song_genres, device))
            else:
                optimizer.step()
            loss_sum += loss.detach()
            steps += 1
            seen += len(idx)
            if seen >= next_check:
                next_check = (seen // rc.eval_every_examples + 1) * rc.eval_every_examples
                stop = validate()
                if stop:
                    break
        if stop:
            print(f"early stop at check {len(log) - 1}")
            break
    if not stop and log[-1]["examples"] < seen:
        validate()  # training ended between checks: validate what it learned since the last one

    with open(run_dir / "log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log[0].keys())
        writer.writeheader()
        writer.writerows(log)
    summary = {"config": str(config_path), "train_seed": seed, "retriever": rc.retriever, "search": rc.search_name(),
               "num_params": num_params, "new_params": new_params, "trained_params": trained,
               "train_windows_used": len(rows), "features": featurizer.names if featurizer else [],
               "best_check": best_check, "best_val_ndcg": stopper.best, "retriever_val_ndcg": log[0]["val_ndcg"],
               "checks_run": len(log), "train_seconds": round(time.perf_counter() - start, 1), "git_commit": commit,
               "git_dirty": dirty}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"best val NDCG@10 {stopper.best:.4f} at check {best_check} (retriever alone {log[0]['val_ndcg']:.4f}) | "
          f"{summary['train_seconds'] / 60:.1f} min")


def retriever_exact_score(rc, seed, ds, retriever, device, scored, all_held_out=False):
    """The retriever alone ranking every song for the held-out windows: read from its own held-out
    evaluation (recsys.evaluate's eval.json, or eval_all.json for every held-out window) when that is
    newer than its checkpoint, scored the same number of windows and has recall at this shortlist
    size; otherwise computed here."""
    run_dir = run_dir_for(rc.retriever, seed)
    path = run_dir / eval_file(all_held_out)
    if path.exists() and path.stat().st_mtime >= (run_dir / "best.pt").stat().st_mtime:
        saved = json.loads(path.read_text()).get("full held-out")
        if saved and saved["n"] == len(ds.Y_test) and f"recall@{rc.shortlist}" in saved:
            return {**saved, "from": str(path)}
    return scored(rank_and_loss(retriever, ds.XI_test, ds.N_test, ds.Y_test, device, lengths=ds.LI_test)[0])


def evaluate_ranker(config_path, seed, all_held_out=False):
    """Held-out: the retriever's top-k reranked by the best ranker, vs the retriever alone. all_held_out:
    every held-out window instead of the retriever config's sample, saved to eval_all.json."""
    rc = load_ranker_config(config_path)
    run_dir = run_dir_for(config_path, seed)
    device = pick_device()
    cfg, ds, retriever = load_retriever(rc, seed, device, all_held_out)
    start = time.perf_counter()
    vectors = song_vectors(retriever)
    held_out = [(ds.XI_test, ds.N_test, ds.Y_test, ds.LI_test)]
    ((ids, scores, _),), from_cache = cached_shortlists(rc, run_dir_for(rc.retriever, seed) / "best.pt", retriever,
                                                       vectors, device, held_out)
    search_seconds = time.perf_counter() - start
    featurizer = make_featurizer(rc, ds, retriever, device)
    ranker = CandidateRanker(retriever, len(featurizer.names) if featurizer else 0, rc.correction_hidden).to(device)
    checkpoint = torch.load(run_dir / "best.pt", weights_only=True)
    check_vocab(checkpoint, ds.vocab, run_dir / "best.pt")
    ranker.load_state_dict(checkpoint["model"])
    rerank_start = time.perf_counter()
    nbrs = window_neighbours(featurizer, ds, retriever, ds.XI_test, ds.N_test, ds.LI_test, device)
    features = feature_fn(featurizer, ids, scores, ds.H_test, np.arange(len(ds.Y_test)), device, nbrs)
    ranks = final_ranks(rerank(ranker, ds.X_test, ds.N_test, ids, scores, device, features=features), ids, ds.Y_test)
    rerank_seconds = time.perf_counter() - rerank_start
    retriever_ranks = final_ranks(scores, ids, ds.Y_test)  # its own order within its shortlist = its full ranks
    pop, _ = popularity_scores(ds.Y_train, len(ds.vocab))
    scored = lambda r: score(r, (rc.shortlist,))  # with recall at the shortlist size, whatever it is
    results = {"best_check": checkpoint["check"], "search": rc.search_name(),
               f"retriever + ranker (top {rc.shortlist})": scored(ranks),
               "retriever alone": scored(retriever_ranks)}
    if rc.search != "exact":  # what approximate search costs: the retriever's own ranking of every song
        results["retriever alone, exact search"] = retriever_exact_score(rc, seed, ds, retriever, device, scored,
                                                                         all_held_out)
    results.update({"most-popular": scored(popularity_ranks(pop, ds.Y_test)),
                    "search_seconds": round(search_seconds, 1), "rerank_seconds": round(rerank_seconds, 1),
                    "total_seconds": round(time.perf_counter() - start, 1)})
    print(f"\n{run_dir}/best.pt (check {checkpoint['check']}) | held-out {len(ds.Y_test):,} windows | "
          f"shortlists ({rc.search_name()}{', cached' if from_cache else ''}) {search_seconds:.0f}s | "
          f"reranking {rerank_seconds:.0f}s")
    print(f"{'':34}{'NDCG@10':>10}{'hits@1':>9}{'hits@5':>9}{'hits@10':>9}{f'top {rc.shortlist}':>10}")
    for name, r in results.items():
        if isinstance(r, dict):
            print(f"{name:34}{r['ndcg@10']:>10.4f}{r['hits@1']:>9.3f}{r['hits@5']:>9.3f}{r['hits@10']:>9.3f}"
                  f"{r[f'recall@{rc.shortlist}']:>10.3f}")
    (run_dir / eval_file(all_held_out)).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train and evaluate the second-stage ranker.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, help="only this seed (default: every seed in train_seeds)")
    ap.add_argument("--evaluate", action="store_true", help="evaluate only, no training")
    ap.add_argument("--all-held-out", action="store_true", help="evaluate only, on every held-out window instead "
                    "of the retriever config's sample; saves eval_all.json (run the retriever's first)")
    args = ap.parse_args()
    for seed in [args.seed] if args.seed is not None else load_ranker_config(args.config).train_seeds:
        if not (args.evaluate or args.all_held_out):
            train_ranker(args.config, seed)
        evaluate_ranker(args.config, seed, args.all_held_out)
