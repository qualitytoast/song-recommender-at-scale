"""Score a trained run on the held-out set, next to the most-popular baseline.

    python -m recsys.evaluate --config configs/p2_base.toml            # every seed in train_seeds
    python -m recsys.evaluate --config configs/p2_base.toml --seed 1   # one seed

Rebuilds the dataset from the config (same data seed, same split), loads
runs/<prefix>_runs/<config name>/seed<train seed>/best.pt, prints the results and saves
them to eval.json in the same folder.
"""
import argparse
import json

import numpy as np
import torch

from recsys.baselines import popularity_ranks, popularity_scores
from recsys.config import load_config
from recsys.data import build_dataset
from recsys.metrics import gains_from_ranks
from recsys.model import build_model, rank_and_loss
from recsys.train import pick_device, run_dir_for

KS = (1, 5, 10)
# Shortlist sizes for a second-stage ranker: recall@K = how often the true next song
# is in the retriever's top K, the most a ranker reranking those K songs could find.
RECALL_KS = (100, 500, 1000, 2000, 5000)


def score(ranks):
    """NDCG@10 and Hits@k for k in KS from target ranks, so the model and the baseline
    are measured by identical code."""
    result = {"n": len(ranks), "ndcg@10": float(np.mean(gains_from_ranks(ranks, k=10)))}
    result.update({f"hits@{k}": float(np.mean(ranks <= k)) for k in KS})
    result.update({f"recall@{k}": float(np.mean(ranks <= k)) for k in RECALL_KS})
    return result


def evaluate(config_path, train_seed):
    cfg = load_config(config_path)
    run_dir = run_dir_for(config_path, train_seed)
    ds = build_dataset(cfg)

    checkpoint = torch.load(run_dir / "best.pt", weights_only=True)
    # Held-out IDs come from the rebuilt vocab and the model's rows from the saved one.
    # If they differ, every score would be meaningless, so refuse.
    if checkpoint["vocab"] != ds.vocab:
        raise ValueError(f"{config_path} builds a different vocab than {run_dir}/best.pt was trained on.")
    device = pick_device()
    model = build_model(cfg, ds).to(device)
    model.load_state_dict(checkpoint["model"])
    model_score = lambda X, Y, N: score(rank_and_loss(model, X, N, Y, device)[0])

    pop, counts = popularity_scores(ds.Y_train, len(ds.vocab))  # training targets only


    # Older checkpoints (validated once per epoch) store "epoch" instead of "check".
    best_check = checkpoint.get("check", checkpoint.get("epoch"))
    results = {"best_check": best_check}
    if cfg.data.validation == "held_out_prefix":
        # v1 style: validation is the first n_val held-out windows, so also report
        # the held-out windows early stopping never saw
        n_val = len(ds.X_val)
        results["validation subset"] = model_score(ds.X_val, ds.Y_val, ds.N_val)
        results["held-out minus validation"] = model_score(ds.X_test[n_val:], ds.Y_test[n_val:], ds.N_test[n_val:])
    else:
        results["validation"] = model_score(ds.X_val, ds.Y_val, ds.N_val)  # separate playlists
    results["full held-out"] = model_score(ds.X_test, ds.Y_test, ds.N_test)
    results["most-popular (full held-out)"] = score(popularity_ranks(pop, ds.Y_test))

    print(f"\n{run_dir}/best.pt (check {best_check})")
    print(f"{'':30}{'n':>8}{'NDCG@10':>10}{'hits@1':>9}{'hits@5':>9}{'hits@10':>9}"
          + "".join(f"{f'top {k}':>9}" for k in RECALL_KS))
    for name, r in results.items():
        if isinstance(r, dict):
            print(f"{name:30}{r['n']:>8,}{r['ndcg@10']:>10.4f}"
                  + "".join(f"{r[f'hits@{k}']:>9.3f}" for k in KS)
                  + "".join(f"{r[f'recall@{k}']:>9.3f}" for k in RECALL_KS))
    top = np.argsort(-pop)[:3]
    print("\nmost common next songs in training: "
          + ", ".join(f"{ds.names[i]} ({counts[i]}x)" for i in top))
    model_ndcg = results["full held-out"]["ndcg@10"]
    pop_ndcg = results["most-popular (full held-out)"]["ndcg@10"]
    print(f"model / most-popular NDCG@10: {model_ndcg / pop_ndcg:.1f}x")

    (run_dir / "eval.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate a trained run on the held-out set.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, help="evaluate only this seed (default: every seed in train_seeds)")
    args = ap.parse_args()
    for seed in [args.seed] if args.seed is not None else load_config(args.config).train_seeds:
        evaluate(args.config, seed)
