"""Short requests: the retriever alone vs the retriever + ranker when a playlist has only 1 to 9 songs
so far. The retriever learned to predict after any number of songs up to 10 (every-position training);
the ranker only ever saw 10-song windows, so this checks whether it still helps on shorter ones.

    python scripts/short_inputs.py --config configs/ranker_1m.toml

Test cases: held-out playlists whose first 11 songs are all in the catalog. For k = 1..10, a
playlist's first k songs (and its name) predict song k + 1, so every length uses the same playlists
and k = 10 is the length the ranker trained on. Both orders come from the same shortlist the ranker
config uses (IVF top 100, input songs left out), as in serving. Prints NDCG@10 and Hits@10 per
length, with the ranker's gain and its standard error, and saves them to short_inputs.json in the
ranker's run folder.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys.evaluate import score  # noqa: E402
from recsys.metrics import gains_from_ranks  # noqa: E402
from recsys.ranker import (CandidateRanker, build_shortlists, check_vocab, feature_fn, final_ranks,  # noqa: E402
                           load_ranker_config, load_retriever, make_featurizer, rerank, song_search)
from recsys.search import song_vectors  # noqa: E402
from recsys.train import pick_device, run_dir_for  # noqa: E402


def playlist_starts(histories, context_length):
    """Indices of the windows (data.window_histories) whose context starts at their playlist's
    first song: the playlists whose first context_length + 1 songs are all in the vocab."""
    _, begin, end = histories
    return np.flatnonzero(end - context_length == begin)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="ranker config")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rc = load_ranker_config(args.config)
    device = pick_device()
    cfg, ds, retriever = load_retriever(rc, args.seed, device, all_held_out=True)
    length = cfg.data.context_length
    songs, begin, _ = ds.H_test
    rows = playlist_starts(ds.H_test, length)
    first, names = begin[rows], ds.N_test[rows]  # each playlist's first song in songs; its name's words

    run_dir = run_dir_for(args.config, args.seed)
    featurizer = make_featurizer(rc, ds, retriever, device)
    ranker = CandidateRanker(retriever, len(featurizer.names) if featurizer else 0, rc.correction_hidden).to(device)
    checkpoint = torch.load(run_dir / "best.pt", weights_only=True)
    check_vocab(checkpoint, ds.vocab, run_dir / "best.pt")
    ranker.load_state_dict(checkpoint["model"])
    vectors = song_vectors(retriever)
    print(f"{run_dir}/best.pt | {len(rows):,} held-out playlists whose first {length + 1} songs are known | "
          f"shortlists: {rc.search_name()}, top {rc.shortlist}")
    print(f"{'songs':>5}{'retriever':>11}{'+ ranker':>10}{'gain':>9}{'(± s.e.)':>10}"
          f"{'hits@10':>9}{'hits@10':>9}{f'top {rc.shortlist}':>9}")

    results = {}
    with song_search(rc, vectors, device) as search:
        for k in range(1, length + 1):
            X, Y = songs[first[:, None] + np.arange(k)], songs[first + k]
            ids, scores, _ = build_shortlists(retriever, X, names, Y, device, rc.shortlist, search, vectors,
                                              exclude_input=rc.exclude_input)
            features = feature_fn(featurizer, ids, scores, (songs, first, first + k), np.arange(len(Y)), device)
            ranker_ranks = final_ranks(rerank(ranker, X, names, ids, scores, device, features=features), ids, Y)
            retriever_ranks = final_ranks(scores, ids, Y)  # its own order within its shortlist
            gain = gains_from_ranks(ranker_ranks, k=10) - gains_from_ranks(retriever_ranks, k=10)
            r, rr = score(retriever_ranks, (rc.shortlist,)), score(ranker_ranks, (rc.shortlist,))
            results[k] = {"retriever": r, "retriever + ranker": rr, "gain": float(gain.mean()),
                          "gain_standard_error": float(gain.std(ddof=1) / np.sqrt(len(gain)))}
            print(f"{k:>5}{r['ndcg@10']:>11.4f}{rr['ndcg@10']:>10.4f}{results[k]['gain']:>+9.4f}"
                  f"{results[k]['gain_standard_error']:>10.4f}{r['hits@10']:>9.3f}{rr['hits@10']:>9.3f}"
                  f"{r[f'recall@{rc.shortlist}']:>9.3f}", flush=True)
    (run_dir / "short_inputs.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
