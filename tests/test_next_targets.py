"""Training and judging on the next several songs of a playlist (target_songs, target_decay)."""
import numpy as np
import torch
from torch import nn

from recsys.data import (any_length_cases, case_targets, chunk_targets, chunks_from_arrays, later_keys,
                         next_targets, playlist_stops)
from recsys.metrics import artist_variety, gains_from_ranks, later_precision, multi_target_gains, target_weights
from recsys.model import SongRecommender, rank_and_loss, rank_targets
from recsys.sampled import candidate_set, log_q, multi_target_loss, sampled_softmax_loss

# Two playlists: 1 2 3 2 4 -1 5 (song 2 repeats, one song outside the vocab) and 6 7 8.
SONGS = np.array([1, 2, 3, 2, 4, -1, 5, 6, 7, 8])
OFFSETS = np.array([0, 7, 10])


def test_chunk_starts_point_at_each_chunks_songs_and_their_next_songs():
    rng = np.random.RandomState(0)
    songs = np.where(rng.rand(400) < 0.05, -1, rng.randint(0, 30, 400))
    offsets = np.concatenate([[0], np.sort(rng.choice(np.arange(1, 400), 25, replace=False)), [400]])
    inputs, targets, _, starts = chunks_from_arrays(songs, offsets, 4, with_starts=True)
    np.testing.assert_array_equal(chunks_from_arrays(songs, offsets, 4)[1], targets)  # same chunks as before
    real = targets != -100
    idx = starts[:, None] + np.arange(4)
    np.testing.assert_array_equal(inputs[real], songs[idx][real])
    T = chunk_targets(songs, offsets, starts, 4 + 5)
    np.testing.assert_array_equal(T[:, :4][real], targets[real])  # the first L columns: each position's next song
    stop = offsets[np.searchsorted(offsets, starts, side="right")]
    for c in range(len(starts)):  # the rest: the songs after the chunk, in the same playlist and the vocab
        for j in range(9):
            g = starts[c] + 1 + j
            assert T[c, j] == (songs[g] if g < stop[c] and songs[g] >= 0 else -100)


def test_next_targets_stop_at_the_playlists_end_and_count_a_repeat_once():
    _, Y, _, histories = any_length_cases(SONGS, OFFSETS, input_length=3)
    np.testing.assert_array_equal(Y, [2, 3, 2, 4, 7, 8])  # every known song with a known song before it
    np.testing.assert_array_equal(playlist_stops(histories, OFFSETS), [7, 7, 7, 7, 10, 10])
    T = next_targets(histories, OFFSETS, 4)
    np.testing.assert_array_equal(T[:, 0], Y)
    np.testing.assert_array_equal(T, [[2, 3, -1, 4],    # from the first 2: 2 3 (2 again: counted once) 4
                                      [3, 2, 4, -1],    # from 3: 3 2 4, then the unknown song
                                      [2, 4, -1, 5],    # from the second 2: 2 4 (unknown) 5
                                      [4, -1, 5, -1],   # from 4: 4 (unknown) 5, then the playlist ends
                                      [7, 8, -1, -1],   # from 7: 7 8, then the playlist ends
                                      [8, -1, -1, -1]])
    np.testing.assert_array_equal(case_targets(Y, histories, OFFSETS, 1), Y[:, None])
    np.testing.assert_array_equal(case_targets(Y, histories, OFFSETS, 4), T)


def test_later_keys_are_every_vocab_song_from_the_target_on():
    _, _, _, histories = any_length_cases(SONGS, OFFSETS, input_length=3)
    keys = later_keys(histories, OFFSETS, 100)
    expected = {(0, s) for s in [2, 3, 4, 5]} | {(1, s) for s in [2, 3, 4, 5]} | {(2, s) for s in [2, 4, 5]} \
        | {(3, s) for s in [4, 5]} | {(4, s) for s in [7, 8]} | {(5, 8)}
    assert set(zip(*np.divmod(keys, 100))) == expected


def test_one_target_is_plain_ndcg_and_weights_count_like_relevance():
    ranks = np.array([1, 3, 10, 11, 500])
    np.testing.assert_allclose(multi_target_gains(ranks[:, None], [1.0]), gains_from_ranks(ranks))
    w = target_weights(3, 0.8)
    np.testing.assert_allclose(w, [1, 0.8, 0.64])
    got = multi_target_gains(np.array([[1, 3, 0]]), w)  # the third song doesn't exist
    np.testing.assert_allclose(got, [(1 + 0.8 / 2) / (1 + 0.8 / np.log2(3))])
    np.testing.assert_allclose(multi_target_gains(np.array([[2, 1, 3]]), [1, 1, 1]), [1.0])  # any order of the top 3
    np.testing.assert_allclose(multi_target_gains(np.array([[0, 0, 0]]), w), [0.0])


def test_later_precision_and_artist_variety():
    top = np.array([[2, 3, 9], [8, 1, 4]])
    later = np.array([0 * 100 + 3, 0 * 100 + 9, 1 * 100 + 2])
    np.testing.assert_allclose(later_precision(top, later, 100), [2 / 3, 0.0])
    artist = np.array([0, 0, 1, 1, 2, 2, 2, 2, 3, 3])
    distinct, last = artist_variety(top, artist, np.array([2, 4]))
    np.testing.assert_array_equal(distinct, [2, 3])
    np.testing.assert_allclose(last, [2 / 3, 1 / 3])


def test_with_one_target_the_loss_is_the_sampled_softmax_loss():
    torch.manual_seed(0)
    Y = torch.tensor([[3, 5, -100], [2, 2, 7]])
    cand, real = candidate_set(Y, 6, 10, torch.Generator().manual_seed(1))
    logits = torch.randn(2, 3, len(cand))
    correction = torch.randn(len(cand))
    expected = sampled_softmax_loss(logits, Y, cand, real, correction)
    got = multi_target_loss(logits, Y != -100, Y, torch.ones(1), cand, real, correction)
    torch.testing.assert_close(got, expected)


def test_each_right_answer_competes_only_with_the_wrong_ones():
    # One chunk, one position; candidates: its two right songs (columns 0, 1), then wrong songs 7 and 8.
    T = torch.tensor([[4, 5]])
    cand, real = torch.tensor([4, 5, 7, 8]), torch.ones(4, dtype=torch.bool)
    logits = torch.tensor([[[2.0, 1.0, 0.5, -1.0]]])
    w = torch.tensor([1.0, 0.8])
    got = multi_target_loss(logits, torch.tensor([[True]]), T, w, cand, real, torch.zeros(4))
    wrong = torch.logsumexp(torch.tensor([0.5, -1.0]), 0)
    each = [torch.logaddexp(torch.tensor(s), wrong) - s for s in (2.0, 1.0)]
    torch.testing.assert_close(got, (each[0] * 1.0 + each[1] * 0.8) / 1.8)
    # raising the second right song's score can't hurt the first one: they don't compete
    higher = multi_target_loss(torch.tensor([[[2.0, 5.0, 0.5, -1.0]]]), torch.tensor([[True]]), T, w, cand, real,
                               torch.zeros(4))
    assert higher < got


def test_a_repeated_right_song_counts_once():
    T = torch.tensor([[4, 4, 5]])  # song 4 is both 1 and 2 songs ahead
    cand, real = torch.tensor([4, 4, 5, 9]), torch.ones(4, dtype=torch.bool)
    logits = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
    pos = torch.tensor([[True]])
    once = multi_target_loss(logits, pos, T, torch.tensor([1.0, 1.0, 1.0]), cand, real, torch.zeros(4))
    T2, cand2 = torch.tensor([[4, 5, 6]]), torch.tensor([4, 5, 6, 9])
    no_repeat = multi_target_loss(torch.tensor([[[1.0, 0.0, -50.0, 0.0]]]), pos, T2, torch.tensor([1.0, 1.0, 0.0]),
                                  cand2, real, torch.zeros(4))
    torch.testing.assert_close(once, no_repeat)  # the same as songs 4 and 5 once each


def test_rank_targets_matches_rank_and_loss_and_leaves_inputs_out_of_the_top():
    torch.manual_seed(0)
    m = SongRecommender(vocab_size=12, embed_dim=8, context_length=3, num_layers=1, dropout=0.0,
                        scale_attention=True, init="pytorch", causal=True).eval()
    X = np.array([[1, 2, 3], [4, 5, 0], [6, 0, 0]])
    lengths = np.array([3, 2, 1])
    N = np.zeros((3, 1), dtype=np.int64)
    Y = np.array([7, 8, 9])
    T = np.array([[7, 2, -1], [8, 1, 3], [9, -1, -1]])
    ranks, losses, top = rank_targets(m, X, N, T, torch.device("cpu"), lengths=lengths, k=5)
    expected_ranks, expected_losses = rank_and_loss(m, X, N, Y, torch.device("cpu"), lengths=lengths)
    np.testing.assert_array_equal(ranks[:, 0], expected_ranks)
    np.testing.assert_allclose(losses, expected_losses, rtol=1e-6)
    assert (ranks[T < 0] == 0).all() and (ranks[T >= 0] > 0).all()
    with torch.no_grad():  # the top 5 by hand: every song's score, the real input songs (not padding) left out
        vectors, bias = m.output_vectors()
        h = m.hidden_states(torch.from_numpy(X), torch.from_numpy(N))[torch.arange(3), torch.from_numpy(lengths) - 1]
        logits = (h @ vectors.T + bias).numpy()
    for row, n in zip(range(3), lengths):
        scores = logits[row].copy()
        scores[X[row, :n]] = -np.inf
        np.testing.assert_array_equal(top[row], np.argsort(-scores, kind="stable")[:5])


def write_config(path, store, genres, target_songs, decay=0.8):
    path.write_text(f"""
data_seed = 42
train_seeds = [1]

[data]
folder = "{store}"
max_playlists = 1000
min_playlist_len = 4
min_freq = 2
song_key = "track_uri"
vocab_from = "train"
context_length = 3
input_length = 3
test_split = 0.1
validation = "separate_playlists"
val_split = 0.1
val_size = 0
genres_file = "{genres}"
val_max_windows = 0
test_max_windows = 0
ranker_split = 0.2
windows = "any_length"

[model]
embed_dim = 8
num_layers = 1
dropout = 0.0
scale_attention = true
init = "pytorch"
features = ["artist", "album", "duration", "playlist_name", "genre"]

[train]
optimizer = "lazy_adamw"
lr = 0.01
weight_decay = 0.0
batch_size = 8
epochs = 2
eval_every_targets = 60
min_checks = 1
patience = 3
augment_mask = 0.0
augment_crop = 0
augment_reorder = 0.0
objective = "every_position"
softmax = "sampled"
sampled_negatives = 20
negative_power = 0.75
compile = false
target_songs = {target_songs}
target_decay = {decay}
""")


def test_training_on_the_next_songs_end_to_end(tmp_path, monkeypatch):
    import csv
    import json

    from recsys.mpd_store import build_store
    from recsys.train import train
    from test_mpd_store import random_mpd

    raw = random_mpd(tmp_path)
    build_store(raw, tmp_path / "store")
    monkeypatch.chdir(tmp_path)
    for n in (1, 3):
        config = tmp_path / f"retriever_5k_next{n}.toml"
        write_config(config, tmp_path / "store", tmp_path / "genres.jsonl", n)
        train(config, 1)
        run = tmp_path / "runs/retriever_ranker_runs/5k" / config.stem / "seed1"
        rows = list(csv.DictReader(open(run / "log.csv")))
        assert len(rows) >= 2
        for row in rows:
            assert 0 <= float(row["val_later@10"]) <= 1 and 1 <= float(row["val_artists@10"]) <= 10
            if n == 1:  # one target: early stopping's NDCG is plain next-song NDCG@10, as before
                assert row["val_ndcg"] == row["val_next_ndcg@10"]
        summary = json.loads((run / "summary.json").read_text())
        assert summary["target_songs"] == n and (run / "best.pt").exists()


def test_target_settings_are_checked(tmp_path):
    import pytest

    from recsys.config import load_config

    path = tmp_path / "retriever_5k_x.toml"
    write_config(path, tmp_path, tmp_path, 3)
    assert load_config(path).train.target_songs == 3
    for old, new in [('target_decay = 0.8', 'target_decay = 0.0'), ('target_decay = 0.8', 'target_decay = 1.5'),
                     ('target_songs = 3', 'target_songs = 0'),
                     ('softmax = "sampled"', 'softmax = "full"'),  # several targets need sampled softmax
                     ('objective = "every_position"', 'objective = "last_position"')]:
        bad = tmp_path / "bad.toml"
        bad.write_text(path.read_text().replace(old, new))
        with pytest.raises(ValueError):
            load_config(bad)
