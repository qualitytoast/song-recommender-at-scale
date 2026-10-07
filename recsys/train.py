"""Train SongRecommender with v1's loop: mini-batch SGD, early stopping on
validation NDCG@10, keep the best epoch.

    python -m recsys.train --config configs/p2_base.toml            # every seed in train_seeds
    python -m recsys.train --config configs/p2_base.toml --seed 1   # one seed

Writes runs/<config name>/seed<train seed>/:
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
from recsys.augment import augment_plan, ignored_targets
from recsys.metrics import gains_from_ranks
from recsys.sampled import candidate_set, log_q, random_probs, sampled_softmax_loss
from recsys.model import build_model, rank_and_loss


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
    """v1's rule, counted in validation checks. A new best score resets the counter.
    A check without one only counts once check >= min_checks (NDCG is noisy early).
    Stop at `patience` counts.
    """

    def __init__(self, min_checks, patience):
        self.min_checks = min_checks
        self.patience = patience
        self.best = -1.0
        self.bad_checks = 0

    def update(self, check, score):
        """Record one validation check's score. Returns (improved, should_stop)."""
        if score > self.best:
            self.best = score
            self.bad_checks = 0
            return True, False
        if check >= self.min_checks:
            self.bad_checks += 1
        return False, self.bad_checks >= self.patience


class EvalSchedule:
    """When to validate: each time the count of training targets passes another
    multiple of `every`. Set every to one epoch's targets to validate at each
    epoch's end. If one batch passes several multiples, that's one check."""

    def __init__(self, every):
        self.every = every
        self.seen = 0
        self.next = every

    def add(self, n_targets):
        """Count a batch's training targets; True if a validation check is due now."""
        self.seen += n_targets
        if self.seen < self.next:
            return False
        self.next = (self.seen // self.every + 1) * self.every
        return True


def git_state():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True).stdout.strip())
    return commit, dirty


def run_dir_for(config_path, train_seed):
    return Path("runs") / Path(config_path).stem / f"seed{train_seed}"


def train(config_path, train_seed):
    cfg = load_config(config_path)
    run_dir = run_dir_for(config_path, train_seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, run_dir / "config.toml")

    commit, dirty = git_state()  # at the start: code may be committed while this runs
    torch.manual_seed(train_seed)  # weight init, dropout masks
    device = pick_device()
    ds = build_dataset(cfg)
    model = build_model(cfg, ds).to(device)
    optimizer = make_optimizer(model.parameters(), cfg.train)
    stopper = EarlyStopping(cfg.train.min_checks, cfg.train.patience)

    t = cfg.train
    every_position = t.objective == "every_position"
    # Training examples: windows (one target each) or chunks (a target at every position).
    X_train, Y_train, N_train = ((ds.Xc_train, ds.Yc_train, ds.Nc_train) if every_position
                                 else (ds.X_train, ds.Y_train, ds.N_train))
    X_train, Y_train, N_train = (torch.from_numpy(a).to(device) for a in (X_train, Y_train, N_train))
    # Real targets per training example, on the CPU: counting a batch's targets for the
    # validation schedule then needs no GPU -> CPU wait. Counted before augmentation,
    # so one epoch is the same number of targets with or without it.
    example_targets = (Y_train != -100).reshape(len(Y_train), -1).sum(dim=1).cpu()
    n_targets = int(example_targets.sum())
    shuffle_gen = torch.Generator().manual_seed(train_seed)  # batch order
    augment_gen = torch.Generator().manual_seed(train_seed + 1_000_000)  # augmentation draws
    n, batch_size = len(X_train), cfg.train.batch_size
    vocab_size, sampled = len(ds.vocab), t.softmax == "sampled"
    if sampled:  # how often each song is a training target, for the logQ correction
        counts = torch.bincount(Y_train[Y_train != -100], minlength=vocab_size).float()
        target_freq = counts / counts.sum()
        negative_gen = torch.Generator().manual_seed(train_seed + 2_000_000)  # random candidates
        probs = random_probs(target_freq, t.negative_power)
        draw_probs = None if t.negative_power == 0 else probs.cpu()  # None: uniform, same draws as before
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{run_dir} | device {device} | {len(ds.vocab):,} songs | {n:,} train "
          f"{'chunks' if every_position else 'windows'} ({n_targets:,} targets) | "
          f"{len(ds.X_val):,} val | {num_params:,} params")

    schedule = EvalSchedule(t.eval_every_targets)
    log, best_check, start = [], None, time.perf_counter()
    # Training loss since the last check, summed on the device: calling .item()
    # every batch would make the CPU wait for the GPU each step.
    loss_sum = torch.zeros((), device=device)
    targets_used = torch.zeros((), device=device)  # augmentation can leave chunk targets out
    interval_start = time.perf_counter()

    def validate():
        """One validation check: log it, save the model if it's the best so far.
        Returns True if early stopping says to stop."""
        nonlocal loss_sum, targets_used, interval_start, best_check
        check, epochs = len(log), schedule.seen / n_targets
        train_loss = loss_sum.item() / max(targets_used.item(), 1)
        # v1 checked every batch; once per check avoids a GPU wait every step.
        if not math.isfinite(train_loss):
            raise FloatingPointError(f"Training loss became {train_loss} by check {check}. Training diverged.")
        train_seconds = time.perf_counter() - interval_start
        val_start = time.perf_counter()
        val_ranks, val_losses = rank_and_loss(model, ds.X_val, ds.N_val, ds.Y_val, device)
        val_loss = float(val_losses.mean())
        val_ndcg = float(gains_from_ranks(val_ranks, k=10).mean())
        val_seconds = time.perf_counter() - val_start
        improved, stop = stopper.update(check, val_ndcg)
        if improved:
            best_check = check
            torch.save({"model": model.state_dict(), "vocab": ds.vocab, "config": asdict(cfg),
                        "check": check, "epochs": epochs, "val_ndcg": val_ndcg}, run_dir / "best.pt")
        log.append({"check": check, "epochs": round(epochs, 4), "targets": schedule.seen,
                    "train_loss": train_loss, "val_loss": val_loss, "val_ndcg": val_ndcg,
                    "train_seconds": round(train_seconds, 2), "val_seconds": round(val_seconds, 2)})
        note = "  *best" if improved else (
            f"  no improvement {stopper.bad_checks}/{stopper.patience}" if check >= stopper.min_checks else "")
        print(f"check {check:3d} | epoch {epochs:5.2f} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
              f"val NDCG@10 {val_ndcg:.4f} | train {train_seconds:.1f}s + val {val_seconds:.1f}s{note}", flush=True)
        loss_sum.zero_()
        targets_used.zero_()
        model.train()  # back to training mode (predict switched dropout off)
        interval_start = time.perf_counter()
        return stop

    stop = False
    for epoch in range(cfg.train.epochs):
        model.train()  # dropout on
        order_cpu = torch.randperm(n, generator=shuffle_gen)
        order = order_cpu.to(device)
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]  # last batch may be smaller, like v1
            X, Y, hidden = X_train[idx], Y_train[idx], None
            if t.augmenting:
                lengths = (Y != -100).sum(dim=1) if every_position else None  # real songs per chunk
                reorder, hidden, mid_shuffle = augment_plan(len(idx), X.shape[1], augment_gen, t.augment_mask,
                                                            t.augment_crop, t.augment_reorder, lengths)
                X, hidden = X.gather(1, reorder.to(device)), hidden.to(device)
                if every_position:
                    Y = Y.masked_fill(ignored_targets(hidden, mid_shuffle.to(device)), -100)
            if sampled:  # score only this batch's candidates (recsys/sampled.py)
                candidates, real = candidate_set(Y, t.sampled_negatives, vocab_size, negative_gen, draw_probs)
                logits = model(X, N_train[idx], hidden, all_positions=every_position, candidates=candidates)
                correction = log_q(candidates, (Y != -100).sum(), target_freq, t.sampled_negatives, probs)
                loss = sampled_softmax_loss(logits, Y, candidates, real, correction)
            else:
                logits = model(X, N_train[idx], hidden, all_positions=every_position)
                # Mean over this batch's real targets; -100 marks chunk padding.
                loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), Y.reshape(-1),
                                                   ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * (Y != -100).sum()
            targets_used += (Y != -100).sum()
            if schedule.add(int(example_targets[order_cpu[i:i + batch_size]].sum())):
                stop = validate()
                if stop:
                    break
        if stop:
            print(f"early stop at check {len(log) - 1}")
            break
    if not stop and (not log or log[-1]["targets"] < schedule.seen):
        validate()  # training ended between checks: validate what it learned since the last one

    train_seconds = time.perf_counter() - start
    with open(run_dir / "log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log[0].keys())
        writer.writeheader()
        writer.writerows(log)
    val_seconds = [row["val_seconds"] for row in log]
    summary = {"config": str(config_path), "train_seed": train_seed, "device": str(device),
               "num_params": num_params, "eval_every_targets": t.eval_every_targets,
               "best_check": best_check, "best_epochs": log[best_check]["epochs"],
               "best_val_ndcg": stopper.best, "checks_run": len(log), "epochs_run": round(log[-1]["epochs"], 2),
               "train_seconds": round(train_seconds, 1), "val_seconds_total": round(sum(val_seconds), 1),
               "val_seconds_per_check": round(sum(val_seconds) / len(log), 2),
               "git_commit": commit, "git_dirty": dirty}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"best val NDCG@10 {stopper.best:.4f} at check {best_check} ({log[best_check]['epochs']:.2f} epochs) | "
          f"{train_seconds / 60:.1f} min, of which validation {sum(val_seconds) / 60:.1f} min | saved {run_dir}/best.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train SongRecommender.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, help="train only this seed (default: every seed in train_seeds)")
    args = ap.parse_args()
    for seed in [args.seed] if args.seed is not None else load_config(args.config).train_seeds:
        train(args.config, seed)
