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

    python -m recsys.ranker --config configs/retriever_ranker_50k.toml            # train every seed, then evaluate
    python -m recsys.ranker --config configs/retriever_ranker_50k.toml --evaluate # evaluate only
"""
import argparse
import copy
import csv
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
from recsys.evaluate import score
from recsys.lazy_adam import LazyAdamW, used_rows
from recsys.model import build_model, rank_and_loss
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
    optimizer: str         # "adamw" or "lazy_adamw" (only the table rows a batch uses are updated)
    lr: float
    weight_decay: float
    batch_size: int        # windows per step
    epochs: int            # maximum passes over the ranker's windows
    eval_every_examples: int  # validation check every this many training windows
    min_checks: int
    patience: int
    val_windows: int       # validate on the first this many validation windows

    def __post_init__(self):
        if self.optimizer not in ("adamw", "lazy_adamw"):
            raise ValueError(f"optimizer must be adamw or lazy_adamw, got {self.optimizer!r}")
        if self.search not in ("exact", "ivf"):
            raise ValueError(f"search must be exact or ivf, got {self.search!r}")

    def search_name(self):
        return "exact" if self.search == "exact" else f"ivf nlist={self.ivf_nlist} nprobe={self.ivf_nprobe}"


def load_ranker_config(path):
    with open(path, "rb") as f:
        return RankerConfig(**tomllib.load(f)["ranker"])


class CandidateRanker(nn.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retriever = copy.deepcopy(retriever)  # fine-tuned; starts as the retriever
        dim = self.retriever.output.in_features
        # New parameters start at zero: no random draws, and no effect before training.
        self.candidate_marker = nn.Parameter(torch.zeros(dim))
        self.candidate_position = nn.Parameter(torch.zeros(dim))
        self.correction = nn.Linear(dim, 1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, ids, names, candidates, base_scores):
        """ids (b, L) context songs; names (b, w) name word IDs; candidates (b, C) song
        IDs; base_scores (b, C) the retriever's scores for them. Returns (b, C) scores."""
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


def build_shortlists(model, X, N, Y, device, k, search, vectors, chunk=20000):
    """Each window's top-k songs found by `search` (song_search) over `vectors`: (ids
    (n, k) int32, scores (n, k) float32, best first), plus each true song's retriever
    score (n,). A chunk of windows at a time, so only one chunk's results are in flight."""
    ids, scores, true = [], [], []
    for i in range(0, len(X), chunk):
        queries = query_vectors(model, X[i:i + chunk], N[i:i + chunk], device)
        top_ids, top_scores = search(queries, k)
        if (top_ids < 0).any():  # FAISS pads with -1 when the searched clusters hold fewer than k songs
            raise RuntimeError(f"search found fewer than {k} songs for some windows; search more clusters")
        ids.append(top_ids.astype(np.int32))
        scores.append(top_scores.astype(np.float32))
        true.append(np.einsum("nd,nd->n", queries, vectors[Y[i:i + chunk]]))
    return np.concatenate(ids), np.concatenate(scores), np.concatenate(true)


def sample_negatives(short_ids, y, n, generator):
    """For each row, n distinct shortlist positions whose song isn't the true song y,
    chosen uniformly (CPU tensors). Returns (rows, n) positions."""
    keys = torch.rand(short_ids.shape, generator=generator)
    keys[short_ids == y[:, None]] = 2.0  # the true song is never a negative
    return torch.topk(keys, n, dim=1, largest=False).indices


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
def rerank(ranker, X, N, short_ids, short_scores, device, batch=64):
    """The ranker's scores for every shortlisted song, (n, k) numpy."""
    ranker.eval()
    out = []
    for i in range(0, len(X), batch):
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a[i:i + batch])).to(device)
        out.append(ranker(t(X), t(N), t(short_ids).long(), t(short_scores)).cpu())
    return torch.cat(out).numpy()


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


def load_retriever(rc, seed, device):
    cfg = load_config(rc.retriever)
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
    v = slice(0, rc.val_windows)
    X_val, N_val, Y_val = ds.X_val[v], ds.N_val[v], ds.Y_val[v]
    vectors = song_vectors(retriever)
    with song_search(rc, vectors, device) as search:
        train_ids, train_scores, train_true = build_shortlists(retriever, ds.X_rank, ds.N_rank, ds.Y_rank, device,
                                                               rc.shortlist, search, vectors)
        val_ids, val_scores, _ = build_shortlists(retriever, X_val, N_val, Y_val, device, rc.shortlist, search,
                                                  vectors)
    print(f"{run_dir} | {len(ds.Y_rank):,} ranker windows, {len(Y_val):,} val windows | shortlists "
          f"({rc.search_name()}) in {time.perf_counter() - start:.0f}s | train recall@{rc.shortlist} "
          f"{(train_ids == ds.Y_rank[:, None]).any(1).mean():.1%}", flush=True)

    ranker = CandidateRanker(retriever).to(device)
    num_params = sum(p.numel() for p in ranker.parameters())
    new_params = num_params - sum(p.numel() for p in retriever.parameters())
    print(f"{num_params:,} params: the retriever's + {new_params:,} new (candidate marker, position, correction)",
          flush=True)
    lazy = rc.optimizer == "lazy_adamw"
    if lazy:  # AdamW on only the table rows each batch uses (recsys/lazy_adam.py)
        optimizer = LazyAdamW(ranker, rc.lr, rc.weight_decay, prefix="retriever.")
    else:
        optimizer = torch.optim.AdamW(ranker.parameters(), lr=rc.lr, weight_decay=rc.weight_decay)
    song_features = {name: ds.song_features[name][0] for name in retriever.feature_names}
    song_genres = ds.song_genres if retriever.input_genres is not None else None
    stopper = EarlyStopping(rc.min_checks, rc.patience)
    order_gen, negative_gen = torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed + 3_000_000)
    X, N, Y = (torch.from_numpy(a) for a in (ds.X_rank, ds.N_rank, ds.Y_rank))
    short_ids, short_scores, true_scores = (torch.from_numpy(a) for a in (train_ids, train_scores, train_true))
    log, best_check, seen, next_check = [], None, 0, rc.eval_every_examples
    loss_sum, steps, interval_start = torch.zeros((), device=device), 0, time.perf_counter()

    def validate():
        nonlocal loss_sum, steps, interval_start, best_check
        check = len(log)
        train_loss = loss_sum.item() / max(steps, 1)  # mean batch loss since the last check
        if not math.isfinite(train_loss):
            raise FloatingPointError(f"Ranker training loss became {train_loss} by check {check}.")
        train_seconds, val_start = time.perf_counter() - interval_start, time.perf_counter()
        ranks = final_ranks(rerank(ranker, X_val, N_val, val_ids, val_scores, device), val_ids, Y_val)
        result = score(ranks)
        val_seconds = time.perf_counter() - val_start
        improved, stop = stopper.update(check, result["ndcg@10"])
        if improved:
            best_check = check
            torch.save({"model": ranker.state_dict(), "vocab": ds.vocab, "config": asdict(rc), "check": check,
                        "val_ndcg": result["ndcg@10"]}, run_dir / "best.pt")
        log.append({"check": check, "examples": seen, "train_loss": train_loss, "val_ndcg": result["ndcg@10"],
                    "val_hits10": result["hits@10"], "val_hits1": result["hits@1"],
                    "train_seconds": round(train_seconds, 2), "val_seconds": round(val_seconds, 2)})
        note = "  *best" if improved else (
            f"  no improvement {stopper.bad_checks}/{stopper.patience}" if check >= stopper.min_checks else "")
        print(f"check {check:3d} | {seen:,} windows | train loss {train_loss:.4f} | val NDCG@10 {result['ndcg@10']:.4f} "
              f"Hits@10 {result['hits@10']:.3f} Hits@1 {result['hits@1']:.3f} | train {train_seconds:.0f}s + "
              f"val {val_seconds:.0f}s{note}", flush=True)
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
            pos = sample_negatives(short_ids[idx], Y[idx], rc.negatives, negative_gen)
            candidates = torch.cat([Y[idx][:, None], short_ids[idx].gather(1, pos).long()], dim=1)
            base = torch.cat([true_scores[idx][:, None], short_scores[idx].gather(1, pos)], dim=1)
            scores = ranker(X[idx].to(device), N[idx].to(device), candidates.to(device), base.to(device))
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

    with open(run_dir / "log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log[0].keys())
        writer.writeheader()
        writer.writerows(log)
    summary = {"config": str(config_path), "train_seed": seed, "retriever": rc.retriever, "search": rc.search_name(),
               "num_params": num_params, "new_params": new_params, "best_check": best_check,
               "best_val_ndcg": stopper.best, "retriever_val_ndcg": log[0]["val_ndcg"], "checks_run": len(log),
               "train_seconds": round(time.perf_counter() - start, 1), "git_commit": commit, "git_dirty": dirty}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"best val NDCG@10 {stopper.best:.4f} at check {best_check} (retriever alone {log[0]['val_ndcg']:.4f}) | "
          f"{summary['train_seconds'] / 60:.1f} min")


def evaluate_ranker(config_path, seed):
    """Held-out: the retriever's top-k reranked by the best ranker, vs the retriever alone."""
    rc = load_ranker_config(config_path)
    run_dir = run_dir_for(config_path, seed)
    device = pick_device()
    cfg, ds, retriever = load_retriever(rc, seed, device)
    start = time.perf_counter()
    vectors = song_vectors(retriever)
    with song_search(rc, vectors, device) as search:
        ids, scores, _ = build_shortlists(retriever, ds.X_test, ds.N_test, ds.Y_test, device, rc.shortlist, search,
                                          vectors)
    search_seconds = time.perf_counter() - start
    ranker = CandidateRanker(retriever).to(device)
    checkpoint = torch.load(run_dir / "best.pt", weights_only=True)
    check_vocab(checkpoint, ds.vocab, run_dir / "best.pt")
    ranker.load_state_dict(checkpoint["model"])
    rerank_start = time.perf_counter()
    ranks = final_ranks(rerank(ranker, ds.X_test, ds.N_test, ids, scores, device), ids, ds.Y_test)
    rerank_seconds = time.perf_counter() - rerank_start
    retriever_ranks = final_ranks(scores, ids, ds.Y_test)  # its own order within its shortlist = its full ranks
    pop, _ = popularity_scores(ds.Y_train, len(ds.vocab))
    results = {"best_check": checkpoint["check"], "search": rc.search_name(),
               f"retriever + ranker (top {rc.shortlist})": score(ranks),
               "retriever alone": score(retriever_ranks)}
    if rc.search != "exact":  # what approximate search costs: the retriever's own ranking of every song
        results["retriever alone, exact search"] = score(rank_and_loss(retriever, ds.X_test, ds.N_test, ds.Y_test,
                                                                       device)[0])
    results.update({"most-popular": score(popularity_ranks(pop, ds.Y_test)),
                    "search_seconds": round(search_seconds, 1), "rerank_seconds": round(rerank_seconds, 1),
                    "total_seconds": round(time.perf_counter() - start, 1)})
    print(f"\n{run_dir}/best.pt (check {checkpoint['check']}) | held-out {len(ds.Y_test):,} windows | "
          f"shortlists ({rc.search_name()}) {search_seconds:.0f}s | reranking {rerank_seconds:.0f}s")
    print(f"{'':34}{'NDCG@10':>10}{'hits@1':>9}{'hits@5':>9}{'hits@10':>9}{f'top {rc.shortlist}':>10}")
    for name, r in results.items():
        if isinstance(r, dict):
            print(f"{name:34}{r['ndcg@10']:>10.4f}{r['hits@1']:>9.3f}{r['hits@5']:>9.3f}{r['hits@10']:>9.3f}"
                  f"{r[f'recall@{rc.shortlist}']:>10.3f}")
    (run_dir / "eval.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train and evaluate the second-stage ranker.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, help="only this seed (default: every seed in train_seeds)")
    ap.add_argument("--evaluate", action="store_true", help="evaluate only, no training")
    args = ap.parse_args()
    for seed in [args.seed] if args.seed is not None else load_ranker_config(args.config).train_seeds:
        if not args.evaluate:
            train_ranker(args.config, seed)
        evaluate_ranker(args.config, seed)
