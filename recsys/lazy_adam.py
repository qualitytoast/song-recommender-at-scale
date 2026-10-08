"""AdamW that updates only the rows of the per-ID tables a batch used ("lazy" AdamW).

The model's big tables have one row per song, artist, album, genre or name word
(99.5% of its parameters at 50,000 playlists), but a batch uses a tiny share of
them (~282 of 166,627 songs as context and targets, ~8,500 counting sampled
negatives). Regular AdamW still updates every row every step: rows with zero
gradient keep moving with their momentum and shrinking with weight decay, and
the arithmetic covers all of them. That was 58% of a training step.

LazyAdamW applies the same AdamW update only to the rows the batch used:
  - the per-ID tables (TABLES below): Adam's running averages, the step and
    weight decay are applied to used rows only. An unused row doesn't change at
    all: no decay, no drifting on old momentum. (Decay therefore happens only
    when a row is used, so rarely seen songs decay less often than with AdamW.)
  - everything else (attention, feed-forward, LayerNorm, positions): regular AdamW.
Bias correction uses the global step count, as in AdamW.

The second-stage ranker (recsys/ranker.py) holds a copy of the retriever under
the name "retriever", so its tables are "retriever.song_embedding.weight", ...:
LazyAdamW(ranker, ..., prefix="retriever.") finds them, and rows keep the
retriever's names. The ranker's own new parameters get regular AdamW.

The rows a batch uses are worked out on the CPU from the batch's song IDs
(used_rows), so the GPU never has to report back which rows it touched: that
would make the CPU wait for the GPU every step.
"""
import math

import numpy as np
import torch

# Per-ID tables of a SongRecommender, by parameter name prefix. Feature tables are
# "input_features.<name>.weight" / "output_features.<name>.weight".
TABLES = ("song_embedding.weight", "output.weight", "output.bias", "input_features.", "output_features.",
          "input_genres.weight", "output_genres.weight", "name_words.weight")


def is_table(name):
    return any(name == t or (t.endswith(".") and name.startswith(t)) for t in TABLES)


class LazyAdamW:
    def __init__(self, model, lr, weight_decay, betas=(0.9, 0.999), eps=1e-8, prefix=""):
        named = dict(model.named_parameters())
        table = lambda n: n.startswith(prefix) and is_table(n[len(prefix):])
        self.tables = {n[len(prefix):]: p for n, p in named.items() if table(n)}
        self.dense = torch.optim.AdamW([p for n, p in named.items() if not table(n)],
                                       lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        self.lr, self.weight_decay, (self.beta1, self.beta2), self.eps = lr, weight_decay, betas, eps
        self.exp_avg = {n: torch.zeros_like(p) for n, p in self.tables.items()}
        self.exp_avg_sq = {n: torch.zeros_like(p) for n, p in self.tables.items()}
        self.steps = 0

    def zero_grad(self):
        self.dense.zero_grad()
        for p in self.tables.values():
            p.grad = None

    @torch.no_grad()
    def step(self, rows):
        """rows: {table parameter name: 1-D LongTensor of unique row indices, on its device}."""
        self.dense.step()
        self.steps += 1
        bias1, bias2 = 1 - self.beta1 ** self.steps, 1 - self.beta2 ** self.steps
        for name, p in self.tables.items():
            r = rows[name]
            if p.grad is None or len(r) == 0:
                continue
            g = p.grad[r]
            # The same operations, in the same order, as torch.optim.AdamW, on the used rows.
            w = p[r].mul_(1 - self.lr * self.weight_decay)
            m = self.exp_avg[name][r].lerp_(g, 1 - self.beta1)
            v = self.exp_avg_sq[name][r].mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
            denom = (v.sqrt() / math.sqrt(bias2)).add_(self.eps)
            w.addcdiv_(m, denom, value=-self.lr / bias1)
            p[r], self.exp_avg[name][r], self.exp_avg_sq[name][r] = w, m, v


def used_rows(context, candidates, names, song_features, song_genres, device):
    """The rows of each per-ID table a batch uses, worked out on the CPU (numpy in).

    context: song IDs fed to the model (input side); candidates: songs scored on
    the output side; names: name word IDs; song_features: {name: per-song feature
    IDs}; song_genres: (songs, k) genre IDs or None. Returns {parameter name: LongTensor}."""
    unique = lambda a: torch.from_numpy(np.unique(np.asarray(a).ravel())).to(device, non_blocking=True)
    rows = {"song_embedding.weight": unique(context), "output.weight": unique(candidates),
            "name_words.weight": unique(names)}
    rows["output.bias"] = rows["output.weight"]
    for name, ids in song_features.items():
        rows[f"input_features.{name}.weight"] = unique(ids[context])
        rows[f"output_features.{name}.weight"] = unique(ids[candidates])
    if song_genres is not None:
        rows["input_genres.weight"] = unique(song_genres[context])
        rows["output_genres.weight"] = unique(song_genres[candidates])
    return rows
