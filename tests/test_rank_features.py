import math

import numpy as np
import pytest
import torch

from recsys.model import SongRecommender
from types import SimpleNamespace

from recsys.rank_features import (GROUPS, CandidateFeatures, history_matrix, membership_keys, pair_counts,
                                  playlist_vectors)
from recsys.ranker import (CandidateRanker, RankerConfig, drop_input_songs, freeze_tables, ranker_validation,
                           shortlist_ranks)

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
    durations = np.array([180_000, 240_000, 200_000, 300_000, 0, 210_000])  # song 4's length unknown (0)
    # 12 part-A playlists: song 3 in playlists 0 and 11, song 1 in 1-3, song 5 in 4-10.
    members = membership_keys(np.array([3, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 3]), np.arange(13), vocab_size=6)
    f = CandidateFeatures(list(GROUPS), retriever(), popularity, pairs, torch.device("cpu"), durations, members)
    context, cands = torch.tensor([[0, 1, 2]]), torch.tensor([[3, 1, 4]])
    base, ranks, top = torch.tensor([[2.0, 1.0, 0.5]]), torch.tensor([[1, 2, 5]]), torch.tensor([3.0])
    history = torch.tensor([[-1, 0, 1, 2]])  # the playlist so far: the 3 context songs
    neighbours = torch.arange(12)[None, :]   # the 12 playlists, most alike first
    got = f(context, cands, base, ranks, top, history, neighbours, torch.tensor([25]))[0]
    typical = np.mean(np.log([180_000, 240_000, 200_000]))
    gap = [abs(math.log(300_000) - typical), abs(math.log(240_000) - typical), abs(math.log(1000) - typical)]
    nbrs = [[2 / 12, 1 / 10], [3 / 12, 3 / 10], [0, 0]]  # share of all 12, share of the first 10
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
    for row, n, g in zip(expected, nbrs, gap):
        row += [*n, g, math.log1p(25)]
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
                    negatives=31, negatives_from=100, optimizer="lazy_adamw", lr=1e-4, weight_decay=0.05,
                    batch_size=64, epochs=1, eval_every_examples=10, min_checks=1, patience=1, val_windows=10,
                    train_windows="all", freeze_tables=False, features=[], correction_hidden=0,
                    exclude_input=False, val_set="retriever")
    return RankerConfig(**{**settings, **changes})


def test_ranker_config_checks_the_new_settings():
    config(negatives=99, train_windows="in_shortlist", features=["popularity", "history"])  # fine
    config(shortlist=500, negatives_from=100)                                                # fine
    config(correction_hidden=64, exclude_input=True, val_set="separate")                     # fine
    for bad in (dict(negatives=100), dict(negatives_from=600), dict(negatives=31, negatives_from=31),
                dict(train_windows="some"), dict(features=["vibes"]), dict(val_set="other"),
                dict(correction_hidden=-1)):
        with pytest.raises(ValueError):
            config(**bad)


def test_playlist_vectors_average_their_known_songs():
    vectors = np.array([[1.0, 0.0], [0.0, 2.0], [4.0, 4.0]], dtype=np.float32)
    # playlists [0, 1, -1] and [2]; -1 is outside the vocab and left out
    got = playlist_vectors(np.array([0, 1, -1, 2]), np.array([0, 3, 4]), vectors)
    np.testing.assert_allclose(got, [[0.5, 1.0], [4.0, 4.0]])


def test_input_songs_are_dropped_and_the_next_best_fill_the_shortlist():
    ids = np.array([[7, 3, 9, 1, 5], [2, 4, 6, 8, 0]])
    scores = -np.arange(5, dtype=np.float32)[None, :].repeat(2, 0)
    inputs, lengths = np.array([[3, 1, 0], [0, 9, 9]]), np.array([2, 1])  # row 1's real input: song 0 only
    got_ids, got_scores = drop_input_songs(ids, scores, inputs, lengths, 3)
    assert got_ids.tolist() == [[7, 9, 5], [2, 4, 6]]  # row 0 loses 3 and 1; row 1 would lose 0, past the top 3
    assert got_scores.tolist() == [[0, -2, -4], [0, -1, -2]]


def test_a_small_network_correction_also_starts_as_the_retriever():
    ranker = CandidateRanker(retriever(), n_features=2, hidden=8).eval()
    assert isinstance(ranker.correction, torch.nn.Sequential)
    base = torch.tensor([[1.0, 2.0]])
    out = ranker(torch.tensor([[0, 1, 2]]), torch.tensor([[1, 0]]), torch.tensor([[3, 4]]), base,
                 torch.randn(1, 2, 2))
    assert torch.equal(out, base)


def test_the_ranker_can_validate_on_windows_outside_the_retrievers_sample():
    rows = lambda first: (np.arange(first, first + 5)[:, None], np.zeros((5, 1)), np.arange(first, first + 5),
                          np.arange(first, first + 5)[:, None], np.ones(5), (np.zeros(9), np.arange(5), np.arange(5)))
    (Xv, Nv, Yv, XIv, LIv, Hv), (Xr, Nr, Yr, XIr, LIr, Hr) = rows(0), rows(100)
    ds = SimpleNamespace(X_val=Xv, N_val=Nv, Y_val=Yv, XI_val=XIv, LI_val=LIv, H_val=Hv,
                         X_rval=Xr, N_rval=Nr, Y_rval=Yr, XI_rval=XIr, LI_rval=LIr, H_rval=Hr)
    assert ranker_validation(config(val_set="retriever", val_windows=3), ds)[2].tolist() == [0, 1, 2]
    assert ranker_validation(config(val_set="separate", val_windows=3), ds)[2].tolist() == [100, 101, 102]
    ds.Y_rval = np.zeros(0)
    with pytest.raises(ValueError, match="outside the retriever"):
        ranker_validation(config(val_set="separate"), ds)
