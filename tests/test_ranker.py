import numpy as np
import torch

from recsys.model import SongRecommender
from recsys.ranker import MISSED, CandidateRanker, final_ranks, sample_negatives

ARTIST = [0, 0, 1, 2, 2, 2, 1]
GENRES = [[1, 2], [1, 0], [2, 0], [0, 0], [1, 2], [2, 0], [1, 0]]


def retriever(causal=True, scale=True):
    torch.manual_seed(0)
    return SongRecommender(vocab_size=7, embed_dim=4, context_length=3, num_layers=2, dropout=0.0,
                           scale_attention=scale, init="pytorch", song_features={"artist": ARTIST},
                           name_word_count=3, song_genres=GENRES, causal=causal).eval()


def trained_ranker(causal=True, scale=True):
    """A ranker whose new parameters and feature tables aren't zero, so their effects are visible."""
    ranker = CandidateRanker(retriever(causal, scale)).eval()
    r = ranker.retriever
    with torch.no_grad():
        for p in (ranker.candidate_marker, ranker.candidate_position, ranker.correction.weight,
                  r.input_features["artist"].weight, r.input_genres.weight[1:], r.name_words.weight[1:]):
            p.normal_()
    return ranker


def ranker_mask(L, C):
    """(L + C, L + C) bool, True where attention is blocked: context position i sees
    context positions <= i; candidate k sees the context and itself only."""
    i, j = torch.arange(L + C)[:, None], torch.arange(L + C)[None, :]
    return ~torch.where(i < L, j <= i, (j < L) | (j == i))


def masked_reference(ranker, ids, names, candidates, base_scores):
    """The ranker computed the plain way: full (L + C) x (L + C) attention with the
    blocked scores masked out. CandidateRanker computes only the allowed scores and
    must give the same result."""
    r = ranker.retriever
    L = ids.shape[1]
    context = r.embed_songs(ids) + r.position_embedding(torch.arange(L))
    cand = r.embed_songs(candidates) + ranker.candidate_position + ranker.candidate_marker
    x = torch.cat([context, cand], dim=1) + r.mean_vector(r.name_words, names)[:, None, :]
    blocked = ranker_mask(L, candidates.shape[1])
    for block in r.blocks:
        x = block(x, blocked)
    return base_scores + ranker.correction(x[:, L:]).squeeze(-1)


CONTEXT, NAMES = torch.tensor([[0, 3, 6]]), torch.tensor([[1, 2]])


def test_mask_by_hand():
    # 2 context songs (c0, c1), 2 candidates (k0, k1); True = blocked
    assert ranker_mask(2, 2).int().tolist() == [
        [0, 1, 1, 1],   # c0 sees c0
        [0, 0, 1, 1],   # c1 sees c0, c1
        [0, 0, 0, 1],   # k0 sees the context and itself
        [0, 0, 1, 0]]   # k1 sees the context and itself, not k0


def test_computing_only_the_allowed_scores_matches_full_masked_attention():
    torch.manual_seed(1)
    ids, names = torch.randint(0, 7, (5, 3)), torch.randint(0, 4, (5, 2))
    candidates, base = torch.randint(0, 7, (5, 6)), torch.randn(5, 6)
    for causal in (True, False):      # the context is causal in the ranker either way
        for scale in (True, False):   # v1's unscaled attention too
            ranker = trained_ranker(causal, scale)
            torch.testing.assert_close(ranker(ids, names, candidates, base),
                                       masked_reference(ranker, ids, names, candidates, base))


def test_before_training_the_ranker_returns_the_retriever_scores():
    ranker = CandidateRanker(retriever()).eval()
    base = torch.tensor([[2.5, -1.0, 0.3]])
    torch.testing.assert_close(ranker(CONTEXT, NAMES, torch.tensor([[4, 1, 2]]), base), base)


def test_each_candidate_is_scored_independently_of_the_others():
    ranker = trained_ranker()
    zeros = torch.zeros(1, 4)
    alone = ranker(CONTEXT, NAMES, torch.tensor([[4]]), zeros[:, :1])
    with_others = ranker(CONTEXT, NAMES, torch.tensor([[4, 1, 2, 5]]), zeros)
    with_others_swapped = ranker(CONTEXT, NAMES, torch.tensor([[5, 2, 1, 4]]), zeros)
    torch.testing.assert_close(with_others[0, 0], alone[0, 0])
    torch.testing.assert_close(with_others_swapped[0, 3], alone[0, 0])  # order doesn't matter either


def test_candidate_scores_depend_on_the_context():
    ranker = trained_ranker()
    cand, zeros = torch.tensor([[4, 1]]), torch.zeros(1, 2)
    a = ranker(CONTEXT, NAMES, cand, zeros)
    b = ranker(torch.tensor([[2, 2, 5]]), NAMES, cand, zeros)
    assert not torch.allclose(a, b)


def test_negatives_are_distinct_shortlist_songs_other_than_the_true_one():
    short = torch.tensor([[5, 3, 9, 1, 7], [2, 4, 6, 8, 0]])
    y = torch.tensor([9, 1])  # row 0's true song is shortlisted, row 1's isn't
    for seed in range(20):
        pos = sample_negatives(short, y, 4, torch.Generator().manual_seed(seed))
        for row in range(2):
            songs = short[row, pos[row]].tolist()
            assert len(set(pos[row].tolist())) == 4 and y[row].item() not in songs


def test_final_ranks():
    short = np.array([[5, 3, 9], [2, 4, 6], [1, 2, 3]])
    scores = np.array([[0.1, 0.9, 0.5], [3.0, 1.0, 2.0], [1.0, 1.0, 0.0]])
    y = np.array([9, 7, 2])
    # row 0: song 9 scores 0.5, one song higher -> 2; row 1: song 7 not shortlisted;
    # row 2: song 2 ties with song 1 -> ties go to the true song -> 1
    np.testing.assert_array_equal(final_ranks(scores, short, y), [2, MISSED, 1])
