"""Multi-token prediction (predict_steps): extra heads that predict the song 2, 3, ... steps ahead in training."""
import csv
import json
import os

import numpy as np
import pytest
import torch

from recsys.model import SongRecommender
from recsys.ranker import CandidateRanker
from recsys.recommender import shared_state
from recsys.sampled import candidate_set, multi_target_loss
from test_next_targets import write_config, write_ranker_config


def model(steps):
    torch.manual_seed(0)
    return SongRecommender(vocab_size=20, embed_dim=8, context_length=4, num_layers=1, dropout=0.0,
                           scale_attention=True, init="pytorch", causal=True, predict_steps=steps).eval()


def test_extra_heads_start_as_copies_of_the_usual_head_and_add_nothing_else():
    plain, multi = model(1), model(3)
    ids, cand = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]]), torch.arange(20)
    heads = multi.step_logits(ids, None, cand)
    assert len(heads) == 3 and len(plain.step_logits(ids, None, cand)) == 1
    for h in heads:  # the heads' last layers start at zero: each starts as the usual head
        torch.testing.assert_close(h, multi(ids, all_positions=True, candidates=cand))
    torch.testing.assert_close(heads[0], plain(ids, all_positions=True))  # everything else is unchanged
    assert {n.split(".")[0] for n in set(dict(multi.named_parameters())) - set(dict(plain.named_parameters()))} \
        == {"step_heads"}
    assert not any(k.startswith("step_heads") for k in shared_state(multi))  # never exported for serving


def test_a_step_heads_loss_targets_the_song_that_many_steps_ahead():
    torch.manual_seed(1)
    T = torch.tensor([[3, 5, 7, 9, 2], [4, 6, -100, 8, 1]])  # chunks of 3 positions, the 5 songs after their first
    positions = torch.tensor([[True, True, True], [True, True, False]])
    cand, real = candidate_set(T, 4, 12, torch.Generator().manual_seed(2))
    logits, correction = torch.randn(2, 3, len(cand)), torch.randn(len(cand))
    one = torch.ones(1)
    got = multi_target_loss(logits, positions, T, one, cand, real, correction, offset=2)
    # By hand: position k of chunk c has one right song, T[c, k + 2] (2 songs further along), in column
    # c * 5 + k + 2, scored against every real candidate holding another song.
    P = 6
    scores = logits.reshape(P, -1) - correction
    losses = []
    for c in range(2):
        for k in range(3):
            song = int(T[c, k + 2])
            if not positions[c, k] or song == -100:
                continue
            col = c * 5 + k + 2
            wrong = scores[c * 3 + k].masked_fill((cand == song) | ~real, float("-inf"))
            losses.append(torch.logaddexp(scores[c * 3 + k, col], torch.logsumexp(wrong, 0)) - scores[c * 3 + k, col])
    torch.testing.assert_close(got, torch.stack(losses).mean())
    with pytest.raises(AssertionError):  # T must reach the furthest song a head looks at
        multi_target_loss(logits, positions, T[:, :4], one, cand[:8 + 4], real[:8 + 4], correction[:12], offset=2)


def test_a_shifted_next_x_head_scores_the_next_songs_from_further_ahead():
    torch.manual_seed(3)
    T = torch.tensor([[3, 5, 7, 9, 2, 6], [4, 6, -100, 8, 1, 3]])  # 3 positions, the 6 songs after the first
    positions = torch.tensor([[True, True, True], [True, True, False]])
    cand, real = candidate_set(T, 4, 12, torch.Generator().manual_seed(2))
    logits, correction = torch.randn(2, 3, len(cand)), torch.randn(len(cand))
    w = torch.tensor([1.0, 0.8, 0.64])
    got = multi_target_loss(logits, positions, T, w, cand, real, correction, offset=1)
    scores = logits.reshape(6, -1) - correction
    per_position = []
    for c in range(2):
        for k in range(3):
            if not positions[c, k]:
                continue
            right = [(c * 6 + k + 1 + d, int(T[c, k + 1 + d]), float(w[d])) for d in range(3) if T[c, k + 1 + d] != -100]
            songs = {song for _, song, _ in right}
            wrong = scores[c * 3 + k].masked_fill(torch.tensor([int(x) in songs for x in cand]) | ~real, float("-inf"))
            lse = torch.logsumexp(wrong, 0)
            losses = [(torch.logaddexp(scores[c * 3 + k, col], lse) - scores[c * 3 + k, col]) * wt for col, _, wt in right]
            per_position.append(sum(losses) / sum(wt for _, _, wt in right))
    torch.testing.assert_close(got, torch.stack(per_position).mean())


def test_how_far_ahead_training_looks():
    from recsys.config import TrainConfig

    base = dict(optimizer="lazy_adamw", lr=0.01, weight_decay=0.0, batch_size=8, epochs=1, eval_every_targets=10,
                min_checks=1, patience=1, augment_mask=0.0, augment_crop=0, augment_reorder=0.0,
                objective="every_position", softmax="sampled", sampled_negatives=4, negative_power=0.75,
                compile=False, target_decay=1.0, predict_decay=0.5)
    for n, steps, kind, ahead in [(1, 1, "exact", 1), (5, 1, "exact", 5), (1, 3, "exact", 3), (5, 3, "exact", 5),
                                  (5, 3, "next", 7), (10, 3, "next", 12)]:
        t = TrainConfig(**base, target_songs=n, predict_steps=steps, predict_targets=kind)
        assert t.songs_ahead == ahead


def test_ranker_step_corrections_start_at_zero():
    r = CandidateRanker(model(1), n_features=0, hidden=4, predict_steps=3).eval()
    ids, names = torch.tensor([[1, 2, 3, 4]]), torch.zeros(1, 1, dtype=torch.long)
    cands, base = torch.tensor([[5, 6, 7]]), torch.tensor([[0.5, -1.0, 2.0]])
    heads = r(ids, names, cands, base, steps=True)
    assert len(heads) == 3
    for h in heads:
        torch.testing.assert_close(h, base)
    torch.testing.assert_close(r(ids, names, cands, base), base)


RUNS = (("retriever_5k_mtp", 1, "exact"), ("retriever_5k_next3_mtp", 3, "exact"),
        ("retriever_5k_next3_mtpnext", 3, "next"))


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Retrievers with 3 heads on a tiny store: on the next song, and on the next 3 songs."""
    from recsys.mpd_store import build_store
    from recsys.train import train
    from test_mpd_store import random_mpd

    tmp = tmp_path_factory.mktemp("multi_token")
    build_store(random_mpd(tmp), tmp / "store")
    old = os.getcwd()
    os.chdir(tmp)
    try:
        for name, n, kind in RUNS:
            write_config(tmp / f"{name}.toml", tmp / "store", tmp / "genres.jsonl", n, steps=3, kind=kind)
            train(tmp / f"{name}.toml", 1)
        yield tmp
    finally:
        os.chdir(old)


def test_multi_token_training_end_to_end(trained, monkeypatch):
    from recsys.evaluate import evaluate
    from recsys.ranker import evaluate_ranker, train_ranker

    monkeypatch.chdir(trained)
    for name, n, kind in RUNS:
        run = trained / "runs/retriever_ranker_runs/5k" / name / "seed1"
        summary = json.loads((run / "summary.json").read_text())
        assert summary["predict_steps"] == 3 and summary["target_songs"] == n and summary["predict_targets"] == kind
        assert any(k.startswith("step_heads.1.") for k in torch.load(run / "best.pt", weights_only=True)["model"])
        evaluate(trained / f"{name}.toml", 1)
        assert "later@10" in json.loads((run / "eval_any_length.json").read_text())["full held-out"]
        ranker = trained / f"ranker_5k_{name.removeprefix('retriever_5k_')}.toml"
        write_ranker_config(ranker, trained / f"{name}.toml", n, steps=3, kind=kind)
        train_ranker(ranker, 1)
        evaluate_ranker(ranker, 1)
        rrun = trained / "runs/retriever_ranker_runs/5k" / ranker.stem / "seed1"
        rows = list(csv.DictReader(open(rrun / "log.csv")))
        assert len(rows) >= 2 and np.isfinite([float(r["train_loss"]) for r in rows]).all()
        assert any(k.startswith("step_corrections.1.") for k in torch.load(rrun / "best.pt", weights_only=True)["model"])
        assert "later@10" in json.loads((rrun / "eval_any_length.json").read_text())["retriever + ranker (top 15)"]
