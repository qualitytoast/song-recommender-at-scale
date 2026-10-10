"""Score a trained run on the held-out set, next to the most-popular baseline.

    python -m recsys.evaluate --config configs/p2_base.toml            # every seed in train_seeds
    python -m recsys.evaluate --config configs/p2_base.toml --seed 1   # one seed

Rebuilds the dataset from the config (same data seed, same split), loads
runs/<group>/<config name>/seed<train seed>/best.pt, prints the results and saves
them to eval.json in the same folder.

    python -m recsys.evaluate --config configs/retriever_1m.toml --all-held-out

scores every held-out window instead of the config's fixed sample (test_max_windows), for
a final model's headline number, and saves to eval_all.json, leaving eval.json as it was.

    python -m recsys.evaluate --config configs/retriever_1m.toml --windows any_length

scores test cases of every length (data.windows) whatever the config says, to re-score a
model trained before any-length validation; saved to eval_any_length.json (or
eval_any_length_all.json). With any-length cases the results also give the cases with the
full input_length songs (the old windows) and a breakdown by length.
"""
import argparse
import json
from dataclasses import replace

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


def score(ranks, recall_ks=()):
    """NDCG@10 and Hits@k for k in KS from target ranks, so the model and the baseline
    are measured by identical code; recall@K for RECALL_KS and any other recall_ks."""
    result = {"n": len(ranks), "ndcg@10": float(np.mean(gains_from_ranks(ranks, k=10)))}
    result.update({f"hits@{k}": float(np.mean(ranks <= k)) for k in KS})
    result.update({f"recall@{k}": float(np.mean(ranks <= k)) for k in sorted({*RECALL_KS, *recall_ks})})
    return result


def eval_file(all_held_out, windows="full"):
    """Where an evaluation is saved: eval.json for the config's held-out windows (day to day, a
    fixed sample at 200k+), eval_all.json for every held-out window (final numbers); with
    any-length test cases, eval_any_length.json and eval_any_length_all.json."""
    return f"eval{'_any_length' if windows == 'any_length' else ''}{'_all' if all_held_out else ''}.json"


def evaluation_config(cfg, all_held_out=False, windows=None):
    """cfg, or a copy for this evaluation only: every held-out window (all_held_out:
    test_max_windows = 0) and/or other test cases (windows: "full" or "any_length")."""
    data = cfg.data
    if all_held_out:
        data = replace(data, test_max_windows=0)
    if windows is not None:
        data = replace(data, windows=windows)
    return cfg if data is cfg.data else replace(cfg, data=data)


def length_groups(max_length):
    """(label, shortest, longest) for a breakdown by input length: each length up to 10, then
    ranges (11-25, 26-50, 51-100, ...), so long inputs don't need one line per length."""
    groups = [(str(k), k, k) for k in range(1, min(max_length, 10) + 1)]
    low = 11
    for high in (25, 50, 100, 250, 500, 1000):
        if low > max_length:
            break
        groups.append((f"{low}-{min(high, max_length)}", low, min(high, max_length)))
        low = high + 1
    if low <= max_length:
        groups.append((f"{low}-{max_length}", low, max_length))
    return groups


def by_length(ranks, lengths, max_length, scorer=score):
    """scorer(ranks) for the cases of each length group (length_groups) that has any."""
    out = {}
    for label, low, high in length_groups(max_length):
        keep = (lengths >= low) & (lengths <= high)
        if keep.any():
            out[label] = scorer(ranks[keep])
    return out


def print_by_length(groups, indent=2):
    """One line per length group: NDCG@10 and Hits@10, with how many cases."""
    for label, r in groups.items():
        print(f"{' ' * indent}{label + (' song' if label == '1' else ' songs'):<{28 - indent}}{r['n']:>10,}"
              f"{r['ndcg@10']:>10.4f}{r['hits@10']:>9.3f}")


def evaluate(config_path, train_seed, all_held_out=False, windows=None):
    cfg = evaluation_config(load_config(config_path), all_held_out, windows)
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
    # The retriever reads each window's input (data.window_inputs: X itself unless input_length > context_length).
    model_ranks = lambda X, Y, N, L: rank_and_loss(model, X, N, Y, device, lengths=L)[0]
    model_score = lambda X, Y, N, L: score(model_ranks(X, Y, N, L))

    pop, counts = popularity_scores(ds.Y_train, len(ds.vocab))  # training targets only


    # Older checkpoints (validated once per epoch) store "epoch" instead of "check".
    best_check = checkpoint.get("check", checkpoint.get("epoch"))
    results = {"best_check": best_check}
    if cfg.data.validation == "held_out_prefix":
        # v1 style: validation is the first n_val held-out windows, so also report
        # the held-out windows early stopping never saw
        n_val = len(ds.X_val)
        results["validation subset"] = model_score(ds.XI_val, ds.Y_val, ds.N_val, ds.LI_val)
        results["held-out minus validation"] = model_score(ds.XI_test[n_val:], ds.Y_test[n_val:], ds.N_test[n_val:],
                                                           ds.LI_test[n_val:])
    else:
        results["validation"] = model_score(ds.XI_val, ds.Y_val, ds.N_val, ds.LI_val)  # separate playlists
    held_out_ranks = model_ranks(ds.XI_test, ds.Y_test, ds.N_test, ds.LI_test)
    results["full held-out"] = score(held_out_ranks)  # with any-length cases, every length: the headline
    any_length, longest = cfg.data.windows == "any_length", cfg.data.input_length
    if any_length:  # the cases with every song an input can hold: the old full windows
        results[f"held-out, {longest} songs"] = score(held_out_ranks[ds.LI_test == longest])
    results["most-popular (full held-out)"] = score(popularity_ranks(pop, ds.Y_test))
    if any_length:
        results["held-out by length"] = by_length(held_out_ranks, ds.LI_test, longest)

    print(f"\n{run_dir}/best.pt (check {best_check})")
    print(f"{'':30}{'n':>8}{'NDCG@10':>10}{'hits@1':>9}{'hits@5':>9}{'hits@10':>9}"
          + "".join(f"{f'top {k}':>9}" for k in RECALL_KS))
    for name, r in results.items():
        if isinstance(r, dict) and "n" in r:
            print(f"{name:30}{r['n']:>8,}{r['ndcg@10']:>10.4f}"
                  + "".join(f"{r[f'hits@{k}']:>9.3f}" for k in KS)
                  + "".join(f"{r[f'recall@{k}']:>9.3f}" for k in RECALL_KS))
    if any_length:
        print(f"\nheld-out by input length{'':4}{'n':>10}{'NDCG@10':>10}{'hits@10':>9}")
        print_by_length(results["held-out by length"])
    top = np.argsort(-pop)[:3]
    print("\nmost common next songs in training: "
          + ", ".join(f"{ds.names[i]} ({counts[i]}x)" for i in top))
    model_ndcg = results["full held-out"]["ndcg@10"]
    pop_ndcg = results["most-popular (full held-out)"]["ndcg@10"]
    print(f"model / most-popular NDCG@10: {model_ndcg / pop_ndcg:.1f}x")

    (run_dir / eval_file(all_held_out, cfg.data.windows)).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate a trained run on the held-out set.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, help="evaluate only this seed (default: every seed in train_seeds)")
    ap.add_argument("--all-held-out", action="store_true",
                    help="score every held-out window, not the config's sample; saves eval_all.json")
    ap.add_argument("--windows", choices=["full", "any_length"],
                    help="test cases for this evaluation instead of the config's data.windows")
    args = ap.parse_args()
    for seed in [args.seed] if args.seed is not None else load_config(args.config).train_seeds:
        evaluate(args.config, seed, args.all_held_out, args.windows)
