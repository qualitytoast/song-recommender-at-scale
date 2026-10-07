import math

import torch
from torch import nn

from recsys.model import SongRecommender
from recsys.sampled import candidate_set, log_q, sampled_softmax_loss

ARTIST = [0, 0, 1, 2, 2, 2, 1]
GENRES = [[1, 2], [1, 0], [2, 0], [0, 0], [1, 2], [2, 0], [1, 0]]


def model():
    torch.manual_seed(0)
    m = SongRecommender(vocab_size=7, embed_dim=4, context_length=3, num_layers=2, dropout=0.0,
                        scale_attention=True, init="pytorch", song_features={"artist": ARTIST},
                        song_genres=GENRES, causal=True).eval()
    with torch.no_grad():  # nonzero feature vectors, so they're part of the test
        m.output_features["artist"].weight.normal_()
        m.output_genres.weight[1:].normal_()
    return m


def test_candidate_scores_equal_those_columns_of_the_full_scores():
    m, ids = model(), torch.tensor([[0, 3, 6], [2, 5, 1]])
    full = m(ids, all_positions=True)
    cand = torch.tensor([1, 4, 6, 4])  # duplicates are fine
    torch.testing.assert_close(m(ids, all_positions=True, candidates=cand), full[..., cand])


def test_when_the_candidates_are_every_song_once_it_is_the_full_loss():
    torch.manual_seed(1)
    full = torch.randn(7, 7)                  # 7 predictions over a 7-song catalog
    Y = torch.tensor([3, 0, 6, 2, 5, 1, 4])   # every song is some prediction's answer, once
    cand, real = candidate_set(Y, 0, 7, torch.Generator())
    got = sampled_softmax_loss(full[:, cand], Y, cand, real, torch.zeros(7))
    torch.testing.assert_close(got, nn.functional.cross_entropy(full, Y))


def test_candidates_are_target_slots_then_random_songs():
    Y = torch.tensor([[3, 3, -100], [5, 1, 3]])
    cand, real = candidate_set(Y, 4, 100, torch.Generator().manual_seed(0))
    assert cand[:6].tolist() == [3, 3, 0, 5, 1, 3] and len(cand) == 10
    assert real.tolist() == [True, True, False, True, True, True] + [True] * 4


def test_accidental_hits_and_padding_are_masked():
    # Predictions 0 and 1 both have answer 3, prediction 2 is padding. For prediction 0,
    # column 1 (also song 3) and column 2 (padding) are masked, so only columns 0 and 3 compete.
    Y = torch.tensor([3, 3, -100])
    cand, real = torch.tensor([3, 3, 0, 5]), torch.tensor([True, True, False, True])
    logits = torch.tensor([[2.0, 9.0, 9.0, 1.0], [9.0, 2.0, 9.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    expected = -torch.log_softmax(torch.tensor([2.0, 1.0]), 0)[0]  # same for both real predictions
    torch.testing.assert_close(sampled_softmax_loss(logits, Y, cand, real, torch.zeros(4)), expected)


def test_log_q_is_expected_count_in_the_candidate_set():
    target_freq = torch.tensor([0.5, 0.25, 0.25, 0.0])
    # 8 targets per batch, 2 random draws over 4 songs: song 0 expected 8*0.5 + 2/4 = 4.5 times
    q = log_q(torch.tensor([0, 3]), 8, target_freq, n_random=2, vocab_size=4)
    torch.testing.assert_close(q, torch.tensor([math.log(4.5), math.log(0.5)]))
