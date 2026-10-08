import math

import numpy as np
import pytest
import torch

from recsys.model import SongRecommender
from recsys.rank_features import GROUPS, CandidateFeatures, history_matrix, pair_counts
from recsys.ranker import CandidateRanker, RankerConfig, freeze_tables, shortlist_ranks

ARTIST = [0, 0, 1, 1, 2, 2]
ALBUM = [0, 1, 2, 2, 3, 4]
GENRES = [[1, 2], [1, 0], [2, 0], [2, 3], [0, 0], [3, 0]]  # 0 = padding


def retriever():
    torch.manual_seed(0)
    return SongRecommender(vocab_size=6, embed_dim=4, context_length=3, num_layers=1, dropout=0.0,
                           scale_attention=True, init="pytorch", song_features={"artist": ARTIST, "album": ALBUM},
                           name_word_count=2, song_genres=GENRES, causal=True).eval()


def test_pair_counts_by_hand():
    # Playlists [0, 1, 2] and [2, -1, 0]; pairs up to 2 songs apart, never across playlists or with -1.
    songs, offsets = np.array([0, 1, 2, 2, -1, 0]), np.array([0, 3, 6])
    keys, counts = pair_counts(songs, offsets, vocab_size=3, window=2)
    # (0,1) (1,2) one apart, (0,2) and (2,0) two apart: keys a * 3 + b
    assert keys.tolist() == [1, 2, 5, 6] and counts.tolist() == [1, 1, 1, 1]


def test_history_matrix_is_right_aligned_and_stops_at_the_playlist_start():
    songs = np.array([5, 6, -1, 8, 9, 1, 2, 3])        # playlists [5, 6, -1, 8, 9] and [1, 2, 3]
    histories = (songs, np.array([0, 5]), np.array([4, 8]))  # windows ending after song 8, and after song 3
    np.testing.assert_array_equal(history_matrix(histories, np.array([0, 1]), length=6),
                                  [[-1, -1, 5, 6, -1, 8], [-1, -1, -1, 1, 2, 3]])


def test_every_feature_group_by_hand():
    popularity = np.array([10, 4, 3, 7, 0, 1])
    pairs = pair_counts(np.array([2, 3, 1, 3, 4]), np.array([0, 2, 5]), vocab_size=6)  # [2, 3] and [1, 3, 4]
    f = CandidateFeatures(list(GROUPS), retriever(), popularity, pairs, torch.device("cpu"))
    context, cands = torch.tensor([[0, 1, 2]]), torch.tensor([[3, 1, 4]])
    base, ranks, top = torch.tensor([[2.0, 1.0, 0.5]]), torch.tensor([[1, 2, 5]]), torch.tensor([3.0])
    history = torch.tensor([[-1, 0, 1, 2]])  # the playlist so far: the 3 context songs
    got = f(context, cands, base, ranks, top, history)[0]
    l1, l2 = math.log1p(1), math.log1p(2)
    # Context artists [0, 0, 1], albums [0, 1, 2]; genre 1 in 2 of 3 context songs, genre 2 in 2 of 3, genre 3 in none.
    # Pairs: 2 -> 3, 1 -> 3, 1 -> 4, 3 -> 4.
    overlap = [[1 / 3, 1 / 3, (2 / 3 + 0) / 2],  # candidate 3: artist 1, album 2, genres {2, 3}
               [2 / 3, 1 / 3, 2 / 3],             # candidate 1: artist 0, album 1, genres {1}
               [0, 0, 0]]                         # candidate 4: artist 2, album 3, no genres
    after = [[l1, l2, 2 / 3], [0, 0, 0], [0, l1, 1 / 3]]  # after the last song, after any summed, share
    expected = [
        [-1.0, 0.0, math.log1p(7), *overlap[0], 1.0, *after[0], *overlap[0], l2, 2 / 3, math.log1p(3)],
        [-2.0, math.log(2), math.log1p(4), *overlap[1], 0.0, *after[1], *overlap[1], 0.0, 0.0, math.log1p(3)],
        [-2.5, math.log(5), 0.0, *overlap[2], 0.0, *after[2], *overlap[2], l1, 1 / 3, math.log1p(3)],
    ]
    assert f.names == [n for g in GROUPS.values() for n in g]
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_groups_come_in_a_fixed_order_and_unknown_groups_are_refused():
    f = CandidateFeatures(["popularity", "retriever_score"], retriever(), np.zeros(6), (np.zeros(0), np.zeros(0)),
                          torch.device("cpu"))
    assert f.names == ["score_gap", "log_rank", "log_popularity"]
    with pytest.raises(ValueError, match="unknown"):
        CandidateFeatures(["vibes"], retriever(), np.zeros(6), (np.zeros(0), np.zeros(0)), torch.device("cpu"))


def test_with_extra_inputs_the_ranker_still_starts_as_the_retriever_and_then_uses_them():
    ranker = CandidateRanker(retriever(), n_features=2).eval()
    context, names, cands, base = torch.tensor([[0, 1, 2]]), torch.tensor([[1, 0]]), torch.tensor([[3, 4]]), \
        torch.tensor([[1.0, 2.0]])
    features = torch.randn(1, 2, 2)
    assert torch.equal(ranker(context, names, cands, base, features), base)  # correction starts at zero
    with torch.no_grad():
        ranker.correction.weight[0, -2:] = torch.tensor([1.0, 0.0])  # read the first extra input
        ranker.feature_mean.fill_(0.5)
    np.testing.assert_allclose(ranker(context, names, cands, base, features).detach().numpy(),
                               (base + features[..., 0] - 0.5).numpy(), rtol=1e-6)


def test_shortlist_ranks():
    short = torch.tensor([[7, 3, 9], [1, 2, 0]])
    cands = torch.tensor([[9, 7, 4], [0, 1, 2]])
    assert shortlist_ranks(short, cands).tolist() == [[3, 1, 4], [3, 1, 2]]  # 4 isn't shortlisted: K + 1


def test_freezing_keeps_the_copied_tables_while_the_layers_train():
    ranker = CandidateRanker(retriever())
    freeze_tables(ranker)
    tables = {n: p.detach().clone() for n, p in ranker.retriever.named_parameters() if not p.requires_grad}
    assert "song_embedding.weight" in tables and "input_features.artist.weight" in tables
    layer = ranker.retriever.blocks[0].ffn[0].weight.detach().clone()
    opt = torch.optim.AdamW([p for p in ranker.parameters() if p.requires_grad], lr=0.1)
    with torch.no_grad():
        ranker.correction.weight.fill_(1.0)  # so the layers get a gradient
    ranker(torch.tensor([[0, 1, 2]]), torch.tensor([[1, 0]]), torch.tensor([[3, 4]]),
           torch.zeros(1, 2)).sum().backward()
    opt.step()
    for n, p in ranker.retriever.named_parameters():
        if n in tables:
            assert torch.equal(p, tables[n]), n
    assert not torch.equal(ranker.retriever.blocks[0].ffn[0].weight, layer)


def config(**changes):
    settings = dict(retriever="r.toml", train_seeds=[1], shortlist=100, search="exact", ivf_nlist=0, ivf_nprobe=0,
                    negatives=31, optimizer="lazy_adamw", lr=1e-4, weight_decay=0.05, batch_size=64, epochs=1,
                    eval_every_examples=10, min_checks=1, patience=1, val_windows=10, train_windows="all",
                    freeze_tables=False, features=[])
    return RankerConfig(**{**settings, **changes})


def test_ranker_config_checks_the_new_settings():
    config(negatives=99, train_windows="in_shortlist", features=["popularity", "history"])  # fine
    for bad in (dict(negatives=100), dict(train_windows="some"), dict(features=["vibes"])):
        with pytest.raises(ValueError):
            config(**bad)
