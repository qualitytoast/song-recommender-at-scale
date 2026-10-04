"""Check that the PyTorch port computes exactly what v1 computed.

NOTE: needs the v1 repo (transformer-song-recommender) checked out next to this
one, with its trained bundle in artifacts/, and the MPD data in data/mpd/.
It imports v1's own code (engine.py, checkpoint.py), so it is a one-off
verification, not part of training or the test suite.

    python scripts/check_against_v1.py     # run from the repo root

It loads v1's trained NumPy weights into the PyTorch SongRecommender, then:
  1. compares the two models' logits on the same held-out windows
  2. scores v1's weights with this repo's evaluation code, which should
     reproduce v1's reported numbers (validation NDCG@10 0.0292, held-out
     NDCG@10 0.0330, Hits@10 5.9%)
If both hold, v2's model, data split and metrics match v1, and any difference
between v1's and v2's results comes from training, not from the port.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT.parent / "transformer-song-recommender"
sys.path.insert(0, str(ROOT))

from recsys.config import load_config  # noqa: E402
from recsys.data import build_dataset  # noqa: E402
from recsys.evaluate import score  # noqa: E402
from recsys.model import SongRecommender, predict  # noqa: E402

CPU = torch.device("cpu")


def v1_state_dict(weights):
    """Rename v1's arrays to v2's parameter names and convert their layout.

    v1 stored linear weights as (inputs, outputs) and biases as (1, outputs);
    nn.Linear wants (outputs, inputs) and (outputs,).
    """
    def linear(v1_name, v2_name):
        return {f"{v2_name}.weight": torch.from_numpy(weights[v1_name + ".W"].T.copy()),
                f"{v2_name}.bias": torch.from_numpy(weights[v1_name + ".B"].reshape(-1).copy())}

    state = {"song_embedding.weight": torch.from_numpy(weights["embedding.weight"]),
             "position_embedding.weight": torch.from_numpy(weights["position_embedding.weight"])}
    for b in range(2):
        for v1_name, v2_name in [("attention.W_query", "attention.query"), ("attention.W_key", "attention.key"),
                                 ("attention.W_value", "attention.value"), ("ffn_expand", "ffn.0"),
                                 ("ffn_compress", "ffn.2")]:
            state.update(linear(f"blocks.{b}.{v1_name}", f"blocks.{b}.{v2_name}"))
        for n in (1, 2):
            state[f"blocks.{b}.norm{n}.weight"] = torch.from_numpy(weights[f"blocks.{b}.norm{n}.gamma"])
            state[f"blocks.{b}.norm{n}.bias"] = torch.from_numpy(weights[f"blocks.{b}.norm{n}.beta"])
    state.update(linear("matchmaker", "output"))
    return state


def main():
    cfg = load_config(ROOT / "configs" / "v1_baseline.toml")
    ds = build_dataset(cfg)
    if json.loads((V1 / "artifacts" / "vocab.json").read_text()) != ds.vocab:
        sys.exit("FAIL: v1's saved vocab differs from the one this repo builds")

    model = SongRecommender(len(ds.vocab), cfg.model.embed_dim, cfg.data.context_length,
                            cfg.model.num_layers, cfg.model.dropout)
    # strict loading: fails if any parameter name or shape doesn't line up
    model.load_state_dict(v1_state_dict(np.load(V1 / "artifacts" / "weights.npz")))

    sys.path.insert(1, str(V1))
    import checkpoint  # v1's
    import engine      # v1's
    engine.TRAINING = False  # dropout off
    v1_model, _, _ = checkpoint.load_bundle(str(V1 / "artifacts"))

    X = ds.X_test[:256]
    diff = float(np.abs(v1_model(X).data - predict(model, X, CPU)).max())
    print(f"max |v1 logits - v2 logits| on {len(X)} held-out windows: {diff:.2e}")

    for name, X, Y in [("validation subset", ds.X_val, ds.Y_val), ("full held-out", ds.X_test, ds.Y_test)]:
        r = score(lambda x: predict(model, x, CPU), X, Y)
        print(f"v1 weights scored by v2 code, {name:18}: "
              f"NDCG@10 {r['ndcg@10']:.4f}  Hits@10 {r['hits@10']:.3f}")

    if diff > 1e-4:  # float32 rounding is ~1e-6; a real mismatch is far larger
        sys.exit("FAIL: logits differ, the PyTorch model does not match v1")
    print("PASS: PyTorch model reproduces v1's outputs")


if __name__ == "__main__":
    main()
