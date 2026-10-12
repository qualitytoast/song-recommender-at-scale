"""Train SongRecommender: mini-batches, a validation check every
eval_every_targets training targets, early stopping on validation NDCG@10 (over
the songs it trains to find: the next one, or the next target_songs weighted by
target_decay), keep the best check. Each check also logs plain next-song NDCG@10,
how much of the top 10 the playlist goes on to contain, and artist variety.

    python -m recsys.train --config configs/p2_base.toml            # every seed in train_seeds
    python -m recsys.train --config configs/p2_base.toml --seed 1   # one seed

Writes runs/<group>/[<size>/]<config name>/seed<train seed>/ (see run_dir_for):
    best.pt       weights at the best check, plus the vocab and config
    log.csv       one row per validation check
    summary.json  best check, training time, git commit, ...
    config.toml   copy of the config used
"""
import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from recsys.config import load_config
from recsys.data import (build_dataset, case_playlists, case_targets, chunk_targets, drop_input_targets,
                         last_positions)
from recsys.augment import augment_plan, ignored_targets
from recsys.metrics import artist_variety, gains_from_ranks, later_precision, multi_target_gains, target_weights
from recsys.lazy_adam import LazyAdamW, used_rows
from recsys.sampled import (candidate_set, draw_random_songs, log_q, multi_target_loss, random_probs,
                            sampled_softmax_loss)
from recsys.model import build_model, rank_targets


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


# NDCG@10 over the next 5 and next 10 songs, equal and nearer-weighing-more (0.8 per step): reported for every
# model whatever it trains on, so models trained on different targets can be compared on the same numbers.
FIXED_TARGETS = (("next5", 5, 1.0), ("next5w", 5, 0.8), ("next10", 10, 1.0), ("next10w", 10, 0.8))
# A validation check computes "later@10" and artist variety (which need each case's top 10: a full-catalog top-k
# search, the slow part) on its first this many cases; evaluations use every case.
CHECK_SAMPLE = 2000


class ScoredCases:
    """Cases a model is scored on (validation checks, held-out evaluation). Each case's target songs (data.
    case_targets: its next song, then the songs after it, up to `columns` of them; later ones that are input
    songs dropped, data.drop_input_targets), the first target_songs of them weighted as in training, each
    playlist's last positions (data.last_positions; for "later in the playlist"), each song's artist and each
    case's last input song. summary() gives:
      "ndcg@10"         NDCG@10 over the songs the model trains on (early stopping; one song: plain NDCG@10)
      "next_ndcg@10"    the next song only
      "ndcg@10_next5" ... "ndcg@10_next10w"   FIXED_TARGETS, when columns >= 10
      "later@10"        share of the top 10 (input songs left out, as served) the playlist goes on to contain
      "artists@10", "last_artist@10"          different artists in the top 10, share by the last input's artist
    """

    def __init__(self, ds, X, N, Y, lengths, histories, offsets, target_songs, decay, device, columns=1):
        self.X, self.N, self.lengths, self.device = X, N, lengths, device
        self.n, self.weights = target_songs, target_weights(target_songs, decay)
        columns = max(columns, target_songs)
        T = case_targets(Y, histories, offsets, columns)
        self.T = drop_input_targets(T, X, lengths) if columns > 1 else T
        self.n_songs = len(ds.vocab)
        self.keys = None
        if offsets is not None:
            self.playlist, self.end = case_playlists(histories, offsets), histories[2]
            self.keys, self.last = last_positions(histories[0], offsets, self.n_songs)
        self.artist = ds.song_features["artist"][0] if "artist" in ds.song_features else None
        self.last_input = X[np.arange(len(X)), lengths - 1]

    def rank(self, model, rows=None, k=10):
        """rank_targets for the cases (or rows of them): (target ranks, next-song losses, top k or None)."""
        rows = np.arange(len(self.T)) if rows is None else rows
        return rank_targets(model, self.X[rows], self.N[rows], self.T[rows], self.device, lengths=self.lengths[rows],
                            k=k)

    def scores(self, model, sample=CHECK_SAMPLE):
        """A validation check: the loss and NDCG@10s over every case, later@10 and artist variety over the first
        `sample` cases."""
        ranks, losses, _ = self.rank(model, k=0)
        rows = np.arange(min(sample, len(self.T)))
        top = self.rank(model, rows)[2]
        return {"loss": float(losses.mean()), **self.summary(ranks), **self.top_summary(top, rows)}

    def summary(self, ranks, top=None):
        """The NDCG@10s from target ranks (cases, columns) (0: no target; a missed shortlist's MISSED counts as not
        found), and with top (cases, 10) the top-10 scores too."""
        out = {"ndcg@10": float(multi_target_gains(ranks[:, :self.n], self.weights).mean()),
               "next_ndcg@10": float(gains_from_ranks(ranks[:, 0], k=10).mean())}
        if ranks.shape[1] >= 10:
            for label, n, decay in FIXED_TARGETS:
                out[f"ndcg@10_{label}"] = float(multi_target_gains(ranks[:, :n], target_weights(n, decay)).mean())
        return out if top is None else {**out, **self.top_summary(top)}

    def top_summary(self, top, rows=None):
        """later@10 and artist variety of the top 10 songs of the cases (or rows of them)."""
        rows = np.arange(len(top)) if rows is None else rows
        out = {}
        if self.keys is not None:
            out["later@10"] = float(later_precision(top, self.playlist[rows], self.end[rows], self.keys, self.last,
                                                    self.n_songs).mean())
        if self.artist is not None:
            distinct, last = artist_variety(top, self.artist, self.last_input[rows])
            out["artists@10"], out["last_artist@10"] = float(distinct.mean()), float(last.mean())
        return out


def git_state():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    # Changes to tracked files only: runs/ is in the repository, so a run's own new output files mustn't count.
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True).stdout.strip())
    return commit, dirty


def run_dir_for(config_path, train_seed):
    """runs/<group>/<config name>/seed<k>. Two-stage runs, named by what they test
    (retriever_*, ranker_*, retriever_ranker_*), go in runs/retriever_ranker_runs/<size>,
    the size being the name's 5k / 50k / 200k / 1m part:
    configs/ranker_50k_v2_mlp.toml -> runs/retriever_ranker_runs/50k/ranker_50k_v2_mlp/seed1.
    The rest go by the name's prefix: configs/v1_baseline.toml -> runs/v1_runs/v1_baseline/seed1."""
    name = Path(config_path).stem
    if name.startswith(("retriever_", "ranker_")):
        size = next((part for part in name.split("_") if re.fullmatch(r"\d+[km]", part)), None)
        if size is None:
            raise ValueError(f"{name}: retriever and ranker config names need a size (e.g. retriever_50k_...)")
        return Path("runs") / "retriever_ranker_runs" / size / name / f"seed{train_seed}"
    return Path("runs") / f"{name.split('_')[0]}_runs" / name / f"seed{train_seed}"


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
    lazy = cfg.train.optimizer == "lazy_adamw"
    if lazy:  # AdamW on only the table rows each batch uses (recsys/lazy_adam.py)
        assert cfg.train.objective == "every_position" and cfg.train.softmax == "sampled" \
            and not cfg.train.augmenting, "lazy_adamw is implemented for the p3 training setup"
        optimizer = LazyAdamW(model, cfg.train.lr, cfg.train.weight_decay)
    else:
        optimizer = make_optimizer(model.parameters(), cfg.train)
    stopper = EarlyStopping(cfg.train.min_checks, cfg.train.patience)

    t = cfg.train
    every_position = t.objective == "every_position"
    # Training examples: windows (one target each) or chunks (a target at every position).
    X_train, Y_train, N_train = ((ds.Xc_train, ds.Yc_train, ds.Nc_train) if every_position
                                 else (ds.X_train, ds.Y_train, ds.N_train))
    X_cpu, Y_cpu, N_cpu = X_train, Y_train, N_train  # numpy copies: the CPU works out which rows a batch uses
    song_features = {name: ds.song_features[name][0] for name in model.feature_names}
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

    def forward_loss(X, N, hidden, Y, candidates, real, correction):
        """Sampled-softmax forward pass and loss: the part torch.compile fuses (t.compile)."""
        logits = model(X, N, hidden, all_positions=every_position, candidates=candidates)
        return sampled_softmax_loss(logits, Y, candidates, real, correction)

    # Training on each position's next target_songs songs, and/or with multi-token prediction heads for the songs
    # 2.. predict_steps steps ahead: the songs after every chunk are looked up per batch (data.chunk_targets)
    # instead of stored, so memory doesn't grow with either.
    several = t.target_songs > 1 or t.predict_steps > 1
    if several:
        # a chunk's own next songs and the songs just after it, as far as either goal looks ahead
        width = X_train.shape[1] + max(t.target_songs, t.predict_steps) - 1
        weights = torch.from_numpy(target_weights(t.target_songs, t.target_decay)).float().to(device)
        step_weights = target_weights(t.predict_steps, t.predict_decay).tolist()  # head h: predict_decay^(h-1)
        one = torch.ones(1, device=device)

    def forward_several(X, N, positions, T, candidates, real, correction):
        """The usual head's loss (its target_songs goal) plus each multi-token head's: the exact song h steps
        ahead for head h, weighted predict_decay^(h-1)."""
        logits = model.step_logits(X, N, candidates)
        loss = multi_target_loss(logits[0], positions, T, weights, candidates, real, correction)
        for h in range(1, len(logits)):
            loss = loss + step_weights[h] * multi_target_loss(logits[h], positions, T, one, candidates, real,
                                                              correction, offset=h)
        return loss
    if t.compile:  # same model and weights; only how the GPU runs the step changes
        forward_loss = torch.compile(forward_loss)
        forward_several = torch.compile(forward_several)
    val_cases = ScoredCases(ds, ds.XI_val, ds.N_val, ds.Y_val, ds.LI_val, ds.H_val, ds.O_val, t.target_songs,
                                t.target_decay, device)

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
        val = val_cases.scores(model)
        val_loss, val_ndcg = val["loss"], val["ndcg@10"]
        val_seconds = time.perf_counter() - val_start
        improved, stop = stopper.update(check, val_ndcg)
        if improved:
            best_check = check
            torch.save({"model": model.state_dict(), "vocab": ds.vocab, "config": asdict(cfg),
                        "check": check, "epochs": epochs, "val_ndcg": val_ndcg}, run_dir / "best.pt")
        log.append({"check": check, "epochs": round(epochs, 4), "targets": schedule.seen,
                    "train_loss": train_loss, "val_loss": val_loss, "val_ndcg": val_ndcg,
                    **{f"val_{k}": v for k, v in val.items() if k not in ("loss", "ndcg@10")},
                    "train_seconds": round(train_seconds, 2), "val_seconds": round(val_seconds, 2)})
        note = "  *best" if improved else (
            f"  no improvement {stopper.bad_checks}/{stopper.patience}" if check >= stopper.min_checks else "")
        trained_for = f" (next {t.target_songs}, next song {val['next_ndcg@10']:.4f})" if several else ""
        later = f" | later {val['later@10']:.3f} (first {CHECK_SAMPLE:,})" if "later@10" in val else ""
        print(f"check {check:3d} | epoch {epochs:5.2f} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
              f"val NDCG@10 {val_ndcg:.4f}{trained_for}{later} | train {train_seconds:.1f}s + val {val_seconds:.1f}s"
              f"{note}", flush=True)
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
            if several:  # the next target_songs songs (and, per extra head, the song further ahead)
                T_cpu = chunk_targets(ds.fit_songs, ds.fit_offsets, ds.Gc_train[order_cpu[i:i + batch_size].numpy()],
                                      width)
                T = torch.from_numpy(T_cpu).to(device)
                random_songs = draw_random_songs(t.sampled_negatives, vocab_size, negative_gen, draw_probs)
                candidates, real = candidate_set(T, t.sampled_negatives, vocab_size, negative_gen, draw_probs,
                                                 random_songs)
                correction = log_q(candidates, (T != -100).sum(), target_freq, t.sampled_negatives, probs)
                loss = forward_several(X, N_train[idx], Y != -100, T, candidates, real, correction)
            elif sampled:  # score only this batch's candidates (recsys/sampled.py)
                random_songs = draw_random_songs(t.sampled_negatives, vocab_size, negative_gen, draw_probs)
                candidates, real = candidate_set(Y, t.sampled_negatives, vocab_size, negative_gen, draw_probs,
                                                 random_songs)
                correction = log_q(candidates, (Y != -100).sum(), target_freq, t.sampled_negatives, probs)
                loss = forward_loss(X, N_train[idx], hidden, Y, candidates, real, correction)
            else:
                logits = model(X, N_train[idx], hidden, all_positions=every_position)
                # Mean over this batch's real targets; -100 marks chunk padding.
                loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), Y.reshape(-1),
                                                   ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            if lazy:
                idx_cpu = order_cpu[i:i + batch_size].numpy()
                targets_cpu = T_cpu if several else Y_cpu[idx_cpu]
                candidates_cpu = np.concatenate([np.clip(targets_cpu.ravel(), 0, None), random_songs.numpy()])
                optimizer.step(used_rows(X_cpu[idx_cpu], candidates_cpu, N_cpu[idx_cpu], song_features,
                                         ds.song_genres if model.input_genres is not None else None, device))
            else:
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
               "target_songs": t.target_songs, "target_decay": t.target_decay, "predict_steps": t.predict_steps,
               "predict_decay": t.predict_decay, "best_val_ndcg": stopper.best,
               "best_val_next_ndcg": log[best_check]["val_next_ndcg@10"], "checks_run": len(log),
               "epochs_run": round(log[-1]["epochs"], 2),
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
