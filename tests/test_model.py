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
    model = SongRecommender(vocab_size=7, embed_dim=4, context_length=3, num_layers=2, dropout=0.0)
    assert count_params(model) == expected_params(7, 4, 3, 2)


def test_param_count_matches_v1():
    model = SongRecommender(vocab_size=33770, embed_dim=64, context_length=10, num_layers=2, dropout=0.1)
    assert count_params(model) == 4_448_618


def test_output_shape():
    model = SongRecommender(vocab_size=7, embed_dim=4, context_length=3, num_layers=2, dropout=0.0)
    ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    assert model(ids).shape == (2, 7)


def test_attention_is_unscaled_like_v1():
    # With Q, K, V all set to the identity, attention on x = [[1, 0], [0, 1]] gives
    # weights = softmax(x @ x.T) = softmax([[1, 0], [0, 1]]), row 0 = [e, 1] / (e + 1).
    # Scaling by 1/sqrt(2) would give a different answer.
    attn = SelfAttention(embed_dim=2)
    with torch.no_grad():
        for layer in (attn.query, attn.key, attn.value):
            layer.weight.copy_(torch.eye(2))
            layer.bias.zero_()
    out = attn(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
    e = math.e
    torch.testing.assert_close(out[0, 0], torch.tensor([e / (e + 1), 1 / (e + 1)]))


def test_init_matches_v1():
    torch.manual_seed(0)
    model = SongRecommender(vocab_size=5000, embed_dim=64, context_length=10, num_layers=2, dropout=0.1)
    assert abs(model.song_embedding.weight.std().item() - 0.1) < 0.005
    ffn_in = model.blocks[0].ffn[0]  # 64 -> 256
    assert abs(ffn_in.weight.std().item() - math.sqrt(2 / 64)) < 0.01
    for m in model.modules():
        if isinstance(m, nn.Linear):
            assert torch.all(m.bias == 0)


def test_dropout_only_in_train_mode():
    torch.manual_seed(0)
    model = SongRecommender(vocab_size=7, embed_dim=4, context_length=3, num_layers=2, dropout=0.5)
    ids = torch.tensor([[0, 1, 2]])
    model.eval()
    torch.testing.assert_close(model(ids), model(ids))
    model.train()
    assert not torch.equal(model(ids), model(ids))
