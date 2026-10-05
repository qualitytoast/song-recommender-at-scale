"""Train SongRecommender with v1's loop: mini-batch SGD, early stopping on
validation NDCG@10, keep the best epoch.

    python -m recsys.train --config configs/v1_baseline.toml

Writes runs/<config name>/:
    best.pt       weights of the best epoch, plus the vocab and config
    log.csv       one row per epoch
    summary.json  best epoch, training time, git commit, ...
    config.toml   copy of the config used
"""
import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from recsys.config import load_config
from recsys.data import build_dataset
from recsys.metrics import ndcg_at_k
from recsys.model import build_model, predict


def pick_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def make_optimizer(params, t):
    if t.optimizer == "sgd":
        # Same update as v1's SGD: w -= lr * (grad + weight_decay * w)
        return torch.optim.SGD(params, lr=t.lr, weight_decay=t.weight_decay)
    if t.optimizer == "adam":
        # weight_decay is added to the gradient (L2), the same rule as SGD above, not AdamW
        return torch.optim.Adam(params, lr=t.lr, weight_decay=t.weight_decay)
    if t.optimizer == "adamw":
        # weight_decay applied separately: every weight shrinks by lr * weight_decay per step
        return torch.optim.AdamW(params, lr=t.lr, weight_decay=t.weight_decay)
    raise ValueError(f"unknown optimizer {t.optimizer!r}")


class EarlyStopping:
    """v1's rule. A new best score resets the counter. An epoch without one only
    counts once epoch >= min_epochs (NDCG is noisy early). Stop at `patience` counts.
    """

    def __init__(self, min_epochs, patience):
        self.min_epochs = min_epochs
        self.patience = patience
        self.best = -1.0
        self.bad_epochs = 0

    def update(self, epoch, score):
        """Record one epoch's score. Returns (improved, should_stop)."""
        if score > self.best:
            self.best = score
            self.bad_epochs = 0
            return True, False
        if epoch >= self.min_epochs:
            self.bad_epochs += 1
        return False, self.bad_epochs >= self.patience


def git_state():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True).stdout.strip())
    return commit, dirty


def train(config_path):
    cfg = load_config(config_path)
    run_dir = Path("runs") / Path(config_path).stem
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, run_dir / "config.toml")

    torch.manual_seed(cfg.seed)  # weight init, dropout masks
    device = pick_device()
    ds = build_dataset(cfg)
    model = build_model(cfg, len(ds.vocab)).to(device)
    optimizer = make_optimizer(model.parameters(), cfg.train)
    stopper = EarlyStopping(cfg.train.min_epochs, cfg.train.patience)

    X_train = torch.from_numpy(ds.X_train).to(device)
    Y_train = torch.from_numpy(ds.Y_train).to(device)
    shuffle_gen = torch.Generator().manual_seed(cfg.seed)  # batch order
    n, batch_size = len(X_train), cfg.train.batch_size
    print(f"device {device} | {len(ds.vocab):,} songs | {n:,} train | {len(ds.X_val):,} val | "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    log, best_epoch, start = [], None, time.perf_counter()
    for epoch in range(cfg.train.epochs):
        epoch_start = time.perf_counter()
        model.train()  # dropout on
        order = torch.randperm(n, generator=shuffle_gen).to(device)
        # Summed on the device: calling .item() every batch would make the CPU
        # wait for the GPU each step.
        loss_sum = torch.zeros((), device=device)
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]  # last batch may be smaller, like v1
            loss = nn.functional.cross_entropy(model(X_train[idx]), Y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(idx)
        train_loss = loss_sum.item() / n
        # v1 checked every batch; once per epoch avoids the per-step wait above.
        if not math.isfinite(train_loss):
            raise FloatingPointError(f"Training loss became {train_loss} in epoch {epoch}. Training diverged.")

        val_logits = predict(model, ds.X_val, device)
        val_loss = nn.functional.cross_entropy(torch.from_numpy(val_logits),
                                               torch.from_numpy(ds.Y_val)).item()
        val_ndcg = ndcg_at_k(val_logits, ds.Y_val, k=10)
        improved, stop = stopper.update(epoch, val_ndcg)
        if improved:
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "vocab": ds.vocab, "config": asdict(cfg),
                        "epoch": epoch, "val_ndcg": val_ndcg}, run_dir / "best.pt")

        seconds = time.perf_counter() - epoch_start
        log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                    "val_ndcg": val_ndcg, "seconds": seconds})
        note = "  *best" if improved else (
            f"  no improvement {stopper.bad_epochs}/{stopper.patience}" if epoch >= stopper.min_epochs else "")
        print(f"epoch {epoch:2d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
              f"val NDCG@10 {val_ndcg:.4f} | {seconds:.1f}s{note}", flush=True)
        if stop:
            print(f"early stop at epoch {epoch}")
            break

    train_seconds = time.perf_counter() - start
    with open(run_dir / "log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log[0].keys())
        writer.writeheader()
        writer.writerows(log)
    commit, dirty = git_state()
    summary = {"config": str(config_path), "device": str(device), "best_epoch": best_epoch,
               "best_val_ndcg": stopper.best, "epochs_run": len(log),
               "train_seconds": round(train_seconds, 1), "git_commit": commit, "git_dirty": dirty}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"best val NDCG@10 {stopper.best:.4f} at epoch {best_epoch} | "
          f"{train_seconds / 60:.1f} min | saved {run_dir}/best.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train SongRecommender.")
    ap.add_argument("--config", required=True)
    train(ap.parse_args().config)
