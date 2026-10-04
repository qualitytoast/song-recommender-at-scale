"""Score a trained run on the held-out set, next to the most-popular baseline.

    python -m recsys.evaluate --config configs/v1_baseline.toml

Rebuilds the dataset from the config (same seed, same split), loads
runs/<config name>/best.pt, prints the results and saves them to
runs/<config name>/eval.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from recsys.baselines import popularity_scores
from recsys.config import load_config
from recsys.data import build_dataset
from recsys.metrics import gains_at_k, hits_at_k
from recsys.model import SongRecommender, predict
from recsys.train import pick_device

KS = (1, 5, 10)
CHUNK = 512  # full held-out logits would be 14,844 x 33,770 floats; score in chunks


def score(logits_fn, X, Y):
    """NDCG@10 and Hits@k for k in KS. logits_fn maps a chunk of X to (chunk, vocab) scores,
    so the model and the baseline are measured by identical code."""
    gains, hits = [], {k: [] for k in KS}
    for i in range(0, len(X), CHUNK):
        logits, y = logits_fn(X[i:i + CHUNK]), Y[i:i + CHUNK]
        gains.append(gains_at_k(logits, y, k=10))
        for k in KS:
            hits[k].append(hits_at_k(logits, y, k))
    result = {"n": len(X), "ndcg@10": float(np.mean(np.concatenate(gains)))}
    result.update({f"hits@{k}": float(np.mean(np.concatenate(hits[k]))) for k in KS})
    return result


def evaluate(config_path):
    cfg = load_config(config_path)
    run_dir = Path("runs") / Path(config_path).stem
    ds = build_dataset(cfg)

    checkpoint = torch.load(run_dir / "best.pt", weights_only=True)
    # Held-out IDs come from the rebuilt vocab and the model's rows from the saved one.
    # If they differ, every score would be meaningless, so refuse.
    if checkpoint["vocab"] != ds.vocab:
        raise ValueError(f"{config_path} builds a different vocab than {run_dir}/best.pt was trained on.")
    device = pick_device()
    model = SongRecommender(len(ds.vocab), cfg.model.embed_dim, cfg.data.context_length,
                            cfg.model.num_layers, cfg.model.dropout).to(device)
    model.load_state_dict(checkpoint["model"])
    model_fn = lambda xb: predict(model, xb, device)

    pop, counts = popularity_scores(ds.Y_train, len(ds.vocab))  # training targets only
    pop_fn = lambda xb: np.broadcast_to(pop, (len(xb), len(pop)))

    n_val = len(ds.X_val)  # the validation subset is the first n_val held-out rows
    results = {
        "best_epoch": checkpoint["epoch"],
        "validation subset": score(model_fn, ds.X_val, ds.Y_val),
        "held-out minus validation": score(model_fn, ds.X_test[n_val:], ds.Y_test[n_val:]),
        "full held-out": score(model_fn, ds.X_test, ds.Y_test),
        "most-popular (full held-out)": score(pop_fn, ds.X_test, ds.Y_test),
    }

    print(f"\n{run_dir}/best.pt (epoch {checkpoint['epoch']})")
    print(f"{'':30}{'n':>8}{'NDCG@10':>10}{'hits@1':>9}{'hits@5':>9}{'hits@10':>9}")
    for name, r in results.items():
        if isinstance(r, dict):
            print(f"{name:30}{r['n']:>8,}{r['ndcg@10']:>10.4f}"
                  + "".join(f"{r[f'hits@{k}']:>9.3f}" for k in KS))
    top = np.argsort(-pop)[:3]
    print("\nmost common next songs in training: "
          + ", ".join(f"{ds.vocab[i]} ({counts[i]}x)" for i in top))
    model_ndcg = results["full held-out"]["ndcg@10"]
    pop_ndcg = results["most-popular (full held-out)"]["ndcg@10"]
    print(f"model / most-popular NDCG@10: {model_ndcg / pop_ndcg:.1f}x")

    (run_dir / "eval.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate a trained run on the held-out set.")
    ap.add_argument("--config", required=True)
    evaluate(ap.parse_args().config)
