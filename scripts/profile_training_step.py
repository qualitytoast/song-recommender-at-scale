"""Where does a training step's time go? Measures the phases of the retriever's
training step for a config, to decide which speed-ups are worth making.

    python scripts/profile_training_step.py --config configs/p3_50k.toml

Two measurements:
  1. Real speed: steps run exactly as in training (no waiting between phases), averaged.
  2. Breakdown: the same step with the GPU forced to finish after each phase
     (torch.mps.synchronize), so each phase can be timed on its own. Forcing
     those waits stops the CPU and GPU from overlapping work, so the phases add
     up to a bit more than the real step; the shares are what matter.
Also splits the optimizer step between the big per-ID tables (song, artist,
album, genre, name-word vectors and output rows) and the small shared layers.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys.config import load_config  # noqa: E402
from recsys.data import build_dataset  # noqa: E402
from recsys.lazy_adam import LazyAdamW, used_rows  # noqa: E402
from recsys.model import build_model  # noqa: E402
from recsys.sampled import (candidate_set, draw_random_songs, log_q, random_probs,  # noqa: E402
                            sampled_softmax_loss)
from recsys.train import make_optimizer, pick_device  # noqa: E402


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def per_id_table(name, param, vocab_size):
    """True for the big tables with one row per song/artist/album/genre/word."""
    return param.dim() >= 1 and param.shape[0] > 1000 and ("embedding" in name or "features" in name
                                                           or "genres" in name or "name_words" in name
                                                           or name.startswith("output"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    cfg = load_config(args.config)
    t = cfg.train
    assert t.objective == "every_position" and t.softmax == "sampled", "profiles the p3 training setup"
    ds = build_dataset(cfg)
    device = pick_device()
    torch.manual_seed(1)
    model = build_model(cfg, ds).to(device)
    lazy = t.optimizer == "lazy_adamw"
    optimizer = LazyAdamW(model, t.lr, t.weight_decay) if lazy else make_optimizer(model.parameters(), t)
    song_features = {name: ds.song_features[name][0] for name in model.feature_names}
    X, Y, N = (torch.from_numpy(a).to(device) for a in (ds.Xc_train, ds.Yc_train, ds.Nc_train))
    V, B = len(ds.vocab), t.batch_size
    counts = torch.bincount(Y[Y != -100], minlength=V).float()
    target_freq = counts / counts.sum()
    probs = random_probs(target_freq, t.negative_power)
    draw = None if t.negative_power == 0 else probs.cpu()
    gen = torch.Generator().manual_seed(0)
    order_cpu = torch.randperm(len(X), generator=torch.Generator().manual_seed(1))
    order = order_cpu.to(device)

    table_params = [p for n, p in model.named_parameters() if per_id_table(n, p, V)]
    shared_params = [p for n, p in model.named_parameters() if not per_id_table(n, p, V)]
    n_table, n_shared = sum(p.numel() for p in table_params), sum(p.numel() for p in shared_params)
    print(f"{args.config}: {V:,} songs | {sum(p.numel() for p in model.parameters()):,} params: "
          f"{n_table:,} in per-ID tables, {n_shared:,} in shared layers | batch {B} chunks")

    def step(i, timings=None):
        mark = lambda name: None
        if timings is not None:
            clock = [time.perf_counter()]

            def mark(name):
                sync(device)
                now = time.perf_counter()
                timings[name] = timings.get(name, 0.0) + now - clock[0]
                clock[0] = now
        model.train()
        start_row = (i * B) % len(X)
        idx = order[start_row:start_row + B]
        xb, yb, nb = X[idx], Y[idx], N[idx]
        random_songs = draw_random_songs(t.sampled_negatives, V, gen, draw)
        candidates, real = candidate_set(yb, t.sampled_negatives, V, gen, draw, random_songs)
        correction = log_q(candidates, (yb != -100).sum(), target_freq, t.sampled_negatives, probs)
        mark("batch + candidate sampling")
        logits = model(xb, nb, all_positions=True, candidates=candidates)
        mark("forward")
        loss = sampled_softmax_loss(logits, yb, candidates, real, correction)
        mark("loss")
        optimizer.zero_grad()
        loss.backward()
        mark("backward")
        if lazy:
            idx_cpu = order_cpu[start_row:start_row + B].numpy()
            cand_cpu = np.concatenate([np.clip(ds.Yc_train[idx_cpu].ravel(), 0, None), random_songs.numpy()])
            optimizer.step(used_rows(ds.Xc_train[idx_cpu], cand_cpu, ds.Nc_train[idx_cpu], song_features,
                                     ds.song_genres if model.input_genres is not None else None, device))
        else:
            optimizer.step()
        mark(f"optimizer step ({t.optimizer})")
        return loss

    for i in range(20):  # warm up
        step(i)
    sync(device)
    start = time.perf_counter()
    for i in range(args.steps):
        loss = step(i)
    loss.item()
    real_ms = (time.perf_counter() - start) / args.steps * 1000
    steps_per_epoch = -(-len(X) // B)
    print(f"\n1. real speed: {real_ms:.1f} ms per step x {steps_per_epoch:,} steps = "
          f"{real_ms * steps_per_epoch / 60000:.1f} min per epoch")

    timings = {}
    for i in range(args.steps):
        step(i, timings)
    total = sum(timings.values())
    print(f"\n2. breakdown (GPU forced to finish after each phase; {total / args.steps * 1000:.1f} ms per step):")
    for name, seconds in timings.items():
        print(f"   {name:28}{seconds / args.steps * 1000:7.1f} ms  {seconds / total:6.1%}")

    # Regular AdamW's cost split by parameter group, for reference (timed separately).
    for label, params in [("per-ID tables", table_params), ("shared layers", shared_params)]:
        opt = torch.optim.AdamW(params, lr=t.lr, weight_decay=t.weight_decay)
        for p in params:
            p.grad = torch.zeros_like(p)
        for _ in range(10):
            opt.step()
        sync(device)
        s = time.perf_counter()
        for _ in range(args.steps):
            opt.step()
        sync(device)
        print(f"   AdamW on {label:14} {(time.perf_counter() - s) / args.steps * 1000:7.1f} ms  "
              f"({sum(p.numel() for p in params):,} params)")
    rows = [len(torch.unique(torch.cat([Y[order[i * B:(i + 1) * B]].flatten(), X[order[i * B:(i + 1) * B]].flatten()])))
            for i in range(50)]
    print(f"\nsongs a batch actually uses (context + targets, before random negatives): median {int(np.median(rows)):,} "
          f"of {V:,} ({np.median(rows) / V:.2%}); with {t.sampled_negatives:,} random negatives at most "
          f"~{int(np.median(rows)) + t.sampled_negatives:,} ({(np.median(rows) + t.sampled_negatives) / V:.1%})")


if __name__ == "__main__":
    main()
