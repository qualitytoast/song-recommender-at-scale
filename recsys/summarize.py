"""Collect multi-seed results into a markdown table, one row per config.

    python -m recsys.summarize configs/p2_base.toml configs/p2_artist.toml ...

Reads eval.json and summary.json from runs/<config>/seed<k>/ for every train
seed in each config. Each row shows the mean over seeds with the min-max range,
and the change in mean NDCG@10 from the row above, so a feature's gain can be
compared against how much the seeds alone move the score.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from recsys.config import load_config
from recsys.train import run_dir_for


def load_seed_results(config_path):
    """One dict per train seed, plus the most-popular baseline's held-out scores."""
    results = []
    for seed in load_config(config_path).train_seeds:
        run_dir = run_dir_for(config_path, seed)
        held_out = json.loads((run_dir / "eval.json").read_text())
        summary = json.loads((run_dir / "summary.json").read_text())
        results.append({"ndcg": held_out["full held-out"]["ndcg@10"],
                        "hits10": held_out["full held-out"]["hits@10"],
                        "hits1": held_out["full held-out"]["hits@1"],
                        "best_epoch": summary["best_epoch"],
                        "minutes": summary["train_seconds"] / 60,
                        "params": summary.get("num_params")})
    return results, held_out["most-popular (full held-out)"]


def mean_range(values, fmt):
    v = np.asarray(values, dtype=float)
    return f"{fmt(v.mean())} ({fmt(v.min())}–{fmt(v.max())})"


def table(rows):
    """rows: [(name, [per-seed result dicts]), ...] -> markdown table lines."""
    four_dp = lambda x: f"{x:.4f}"
    pct = lambda x: f"{100 * x:.1f}%"
    lines = ["| Config | Seeds | NDCG@10 mean (min–max) | Change vs previous | Hits@10 | Hits@1 "
             "| Best epochs | Params | Min / seed |",
             "|---|---|---|---|---|---|---|---|---|"]
    previous = None
    for name, results in rows:
        col = lambda key: [r[key] for r in results]
        ndcg = float(np.mean(col("ndcg")))
        change = "—" if previous is None else f"{ndcg - previous:+.4f}"
        params = "—" if results[0]["params"] is None else f"{results[0]['params']:,}"
        lines.append(f"| `{name}` | {len(results)} | {mean_range(col('ndcg'), four_dp)} | {change} "
                     f"| {mean_range(col('hits10'), pct)} | {mean_range(col('hits1'), pct)} "
                     f"| {', '.join(str(e) for e in col('best_epoch'))} | {params} "
                     f"| {np.mean(col('minutes')):.1f} |")
        previous = ndcg
    return lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Markdown table of multi-seed results.")
    ap.add_argument("configs", nargs="+")
    args = ap.parse_args()
    rows, popular = [], None
    for path in args.configs:
        results, popular = load_seed_results(path)
        rows.append((Path(path).stem, results))
    print("\n".join(table(rows)))
    print(f"\nMost-popular baseline on the same held-out windows: NDCG@10 {popular['ndcg@10']:.4f}, "
          f"Hits@10 {100 * popular['hits@10']:.1f}%")
