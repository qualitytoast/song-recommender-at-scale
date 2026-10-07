import math

import torch
from torch import nn

from recsys.model import SelfAttention, SongRecommender


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def expected_params(V, E, C, L):
    """Hand count, layer by layer.

    song embedding V*E, position embedding C*E, output layer E*V + V,
    and per block: Q/K/V 3*(E*E + E), feed-forward (E*4E + 4E) + (4E*E + E),
    two LayerNorms 2*(E + E).
    """
    block = 3 * (E * E + E) + (E * 4 * E + 4 * E) + (4 * E * E + E) + 4 * E
    return V * E + C * E + (E * V + V) + L * block


def test_param_count_matches_hand_count():
    model = SongRecommender(vocab_size=7, embed_dim=4, context_length=3, num_layers=2, dropout=0.0, scale_attention=False, init="v1")
    assert count_params(model) == expected_params(7, 4, 3, 2)


def test_param_count_matches_v1():
    model = SongRecommender(vocab_size=33770, embed_dim=64, context_length=10, num_layers=2, dropout=0.1, scale_attention=False, init="v1")
    assert count_params(model) == 4_448_618


def test_output_shape():
    model = SongRecommender(vocab_size=7, embed_dim=4, context_length=3, num_layers=2, dropout=0.0, scale_attention=False, init="v1")
    ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    assert model(ids).shape == (2, 7)


def test_attention_is_unscaled_like_v1():
    # With Q, K, V all set to the identity, attention on x = [[1, 0], [0, 1]] gives
    # weights = softmax(x @ x.T) = softmax([[1, 0], [0, 1]]), row 0 = [e, 1] / (e + 1).
    # Scaling by 1/sqrt(2) would give a different answer.
    attn = SelfAttention(embed_dim=2, scale=False)
    with torch.no_grad():
        for layer in (attn.query, attn.key, attn.value):
            layer.weight.copy_(torch.eye(2))
            layer.bias.zero_()
    out = attn(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
    e = math.e
    torch.testing.assert_close(out[0, 0], torch.tensor([e / (e + 1), 1 / (e + 1)]))


def test_init_matches_v1():
    torch.manual_seed(0)
    model = SongRecommender(vocab_size=5000, embed_dim=64, context_length=10, num_layers=2, dropout=0.1, scale_attention=False, init="v1")
    assert abs(model.song_embedding.weight.std().item() - 0.1) < 0.005
    ffn_in = model.blocks[0].ffn[0]  # 64 -> 256
    assert abs(ffn_in.weight.std().item() - math.sqrt(2 / 64)) < 0.01
    for m in model.modules():
        if isinstance(m, nn.Linear):
            assert torch.all(m.bias == 0)


def test_dropout_only_in_train_mode():
    torch.manual_seed(0)
    model = SongRecommender(vocab_size=7, embed_dim=4, context_length=3, num_layers=2, dropout=0.5, scale_attention=False, init="v1")
    ids = torch.tensor([[0, 1, 2]])
    model.eval()
    torch.testing.assert_close(model(ids), model(ids))
    model.train()
    assert not torch.equal(model(ids), model(ids))


def test_scaled_attention_divides_by_sqrt_embed_dim():
    # Same setup as the unscaled test: scores [[1, 0], [0, 1]] become [[1, 0], [0, 1]] / sqrt(2),
    # so row 0 of the weights is [e^(1/sqrt 2), 1] / (e^(1/sqrt 2) + 1).
    attn = SelfAttention(embed_dim=2, scale=True)
    with torch.no_grad():
        for layer in (attn.query, attn.key, attn.value):
            layer.weight.copy_(torch.eye(2))
            layer.bias.zero_()
    out = attn(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
    a = math.exp(1 / math.sqrt(2))
    torch.testing.assert_close(out[0, 0], torch.tensor([a / (a + 1), 1 / (a + 1)]))


def test_pytorch_init_keeps_defaults():
    torch.manual_seed(0)
    model = SongRecommender(vocab_size=5000, embed_dim=64, context_length=10, num_layers=2,
                            dropout=0.1, scale_attention=True, init="pytorch")
    assert abs(model.song_embedding.weight.std().item() - 1.0) < 0.05  # N(0, 1)
    ffn_in = model.blocks[0].ffn[0]  # uniform in +-1/sqrt(64): std = 1/sqrt(64)/sqrt(3)
    assert abs(ffn_in.weight.std().item() - 1 / 8 / math.sqrt(3)) < 0.005
    assert not torch.all(ffn_in.bias == 0)


# --- song features (artist, ...) ---

ARTIST = [0, 0, 1, 2, 2, 2, 1]  # artist of each of 7 songs


def featured_model(seed=0, features=True):
    torch.manual_seed(seed)
    return SongRecommender(vocab_size=7, embed_dim=4, context_length=3, num_layers=2, dropout=0.1,
                           scale_attention=True, init="pytorch",
                           song_features={"artist": ARTIST} if features else None)


def test_zero_init_features_are_paired_with_no_feature_run():
    # Same seed: identical starting weights, and the zero artist vectors add nothing.
    ids = torch.tensor([[0, 3, 6], [2, 2, 5]])
    torch.testing.assert_close(featured_model().eval()(ids), featured_model(features=False).eval()(ids))
    # Same random draws afterwards too (dropout masks, ...)
    featured_model(); a = torch.rand(3)
    featured_model(features=False); b = torch.rand(3)
    torch.testing.assert_close(a, b)


def test_output_artist_vector_shifts_only_that_artists_songs_equally():
    model = featured_model().eval()
    ids = torch.tensor([[0, 3, 6]])
    before = model(ids)
    with torch.no_grad():
        model.output_features["artist"].weight[2] = torch.tensor([1.0, -2.0, 0.5, 3.0])
    shift = (model(ids) - before)[0]
    songs_of_artist_2 = [3, 4, 5]
    assert torch.all(shift[[0, 1, 2, 6]] == 0)
    torch.testing.assert_close(shift[songs_of_artist_2], shift[3].expand(3))  # h . v, same for each
    assert shift[3] != 0


def test_input_artist_vector_changes_predictions():
    model = featured_model().eval()
    ids = torch.tensor([[0, 3, 6]])
    before = model(ids)
    with torch.no_grad():
        model.input_features["artist"].weight[1] = torch.ones(4)  # artist of song 6
    assert not torch.allclose(model(ids), before)


def test_artist_adds_two_tables_of_parameters():
    extra = count_params(featured_model()) - count_params(featured_model(features=False))
    assert extra == 2 * 3 * 4  # input + output table, 3 artists x embed_dim 4


# --- playlist name ---

def named_model(seed=0, words=4):
    torch.manual_seed(seed)
    return SongRecommender(vocab_size=7, embed_dim=4, context_length=3, num_layers=2, dropout=0.1,
                           scale_attention=True, init="pytorch", name_word_count=words)


def test_zero_init_name_words_are_paired_with_no_name_run():
    ids, names = torch.tensor([[0, 3, 6]]), torch.tensor([[2, 1, 0]])
    torch.testing.assert_close(named_model().eval()(ids, names),
                               named_model(words=None).eval()(ids))


def test_name_vector_is_mean_of_real_words_ignoring_padding():
    model = named_model().eval()
    with torch.no_grad():
        model.name_words.weight[1:] = torch.randn(4, 4)
    ids = torch.tensor([[0, 3, 6]] * 3)
    out = model(ids, torch.tensor([[3, 0, 0], [3, 3, 0], [0, 0, 0]]))
    torch.testing.assert_close(out[0], out[1])  # [3] and [3, 3] average to word 3's vector
    torch.testing.assert_close(out[2], named_model(words=None).eval()(ids[:1])[0])  # no words: no change
    assert not torch.allclose(out[0], out[2])


# --- genre ---

SONG_GENRES = [[1, 2], [1, 0], [2, 0], [0, 0], [1, 2], [2, 0], [1, 0]]  # 7 songs, genres 1-2, 0 = padding


def genre_model(seed=0, genres=True):
    torch.manual_seed(seed)
    return SongRecommender(vocab_size=7, embed_dim=4, context_length=3, num_layers=2, dropout=0.1,
                           scale_attention=True, init="pytorch", song_genres=SONG_GENRES if genres else None)


def test_zero_init_genres_are_paired_with_no_genre_run():
    ids = torch.tensor([[0, 3, 6]])
    torch.testing.assert_close(genre_model().eval()(ids), genre_model(genres=False).eval()(ids))
    genre_model(); a = torch.rand(3)
    genre_model(genres=False); b = torch.rand(3)
    torch.testing.assert_close(a, b)


def test_output_genre_vectors_are_averaged_per_song():
    model = genre_model().eval()
    ids = torch.tensor([[0, 3, 6]])
    before = model(ids)
    with torch.no_grad():
        model.output_genres.weight[1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        model.output_genres.weight[2] = torch.tensor([0.0, 2.0, 0.0, 0.0])
    shift = (model(ids) - before)[0]
    h_dot = {1: shift[1], 2: shift[2]}  # songs with only genre 1 / only genre 2
    torch.testing.assert_close(shift[0], (h_dot[1] + h_dot[2]) / 2)  # genres [1, 2]: the average
    torch.testing.assert_close(shift[6], h_dot[1])                   # songs sharing a genre move together
    assert shift[3] == 0                                             # no genres: unaffected


def test_input_genre_vector_changes_predictions():
    model = genre_model().eval()
    ids = torch.tensor([[0, 3, 6]])
    before = model(ids)
    with torch.no_grad():
        model.input_genres.weight[1] = torch.ones(4)
    assert not torch.allclose(model(ids), before)


# --- augmentation: hidden songs ---

def masked_model(seed=0):
    torch.manual_seed(seed)
    return SongRecommender(vocab_size=7, embed_dim=4, context_length=3, num_layers=2, dropout=0.0,
                           scale_attention=True, init="pytorch", song_features={"artist": ARTIST},
                           song_genres=SONG_GENRES, mask_token=True)


def test_hidden_song_and_its_features_have_no_effect():
    model = masked_model().eval()
    with torch.no_grad():  # give the features real values, so hiding them is a real test
        model.input_features["artist"].weight.normal_()
        model.input_genres.weight[1:].normal_()
    hidden = torch.tensor([[False, True, False]])
    a = model(torch.tensor([[0, 3, 6]]), hidden=hidden)
    b = model(torch.tensor([[0, 5, 6]]), hidden=hidden)  # different song (other artist, genre) in the hidden slot
    torch.testing.assert_close(a, b)
    assert not torch.allclose(model(torch.tensor([[0, 3, 6]])), model(torch.tensor([[0, 5, 6]])))  # visible: it matters


def test_nothing_hidden_equals_no_hidden_argument():
    model = masked_model().eval()
    ids = torch.tensor([[0, 3, 6]])
    torch.testing.assert_close(model(ids, hidden=torch.zeros(1, 3, dtype=torch.bool)), model(ids))


# --- causal attention / every-position output ---

def causal_model(seed=0):
    torch.manual_seed(seed)
    return SongRecommender(vocab_size=7, embed_dim=4, context_length=4, num_layers=2, dropout=0.0,
                           scale_attention=True, init="pytorch", song_features={"artist": ARTIST},
                           causal=True)


def test_causal_positions_cannot_see_later_songs():
    model = causal_model().eval()
    a = model(torch.tensor([[0, 3, 6, 1]]), all_positions=True)
    b = model(torch.tensor([[0, 3, 2, 5]]), all_positions=True)  # songs at positions 2 and 3 changed
    torch.testing.assert_close(a[0, :2], b[0, :2])               # positions 0 and 1 unaffected
    assert not torch.allclose(a[0, 2], b[0, 2])


def test_last_position_output_matches_all_positions():
    model = causal_model().eval()
    ids = torch.tensor([[0, 3, 6, 1], [2, 2, 5, 4]])
    torch.testing.assert_close(model(ids), model(ids, all_positions=True)[:, -1])


def test_non_causal_positions_do_see_later_songs():
    torch.manual_seed(0)
    model = SongRecommender(vocab_size=7, embed_dim=4, context_length=4, num_layers=2, dropout=0.0,
                            scale_attention=True, init="pytorch").eval()
    a = model(torch.tensor([[0, 3, 6, 1]]), all_positions=True)
    b = model(torch.tensor([[0, 3, 2, 5]]), all_positions=True)
    assert not torch.allclose(a[0, 0], b[0, 0])  # without the mask, position 0 sees later songs
