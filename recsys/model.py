"""SongRecommender: v1's Transformer, rebuilt in PyTorch.

Song IDs -> embeddings + position embeddings -> stacked Transformer blocks ->
last position's vector -> a score for every song in the catalog.

Matches v1's model.py layer for layer, including two things PyTorch would do
differently by default:
  - attention scores are NOT divided by sqrt(embed_dim) (v1 quirk, see SelfAttention)
  - weights start from v1's init, not PyTorch's (see _init_like_v1)
"""
import math

import torch
from torch import nn


class SelfAttention(nn.Module):
    """Single-head attention: each song pulls in a weighted blend of the others."""

    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):  # x: (batch, seq, embed_dim)
        q, k, v = self.query(x), self.key(x), self.value(x)
        # scores[b, i, j] = how relevant song j is to song i
        scores = q @ k.transpose(-2, -1)
        # v1 quirk: no division by sqrt(embed_dim). v1 computed the scaled scores
        # but passed the unscaled ones to softmax, so its attention was unscaled.
        weights = scores.softmax(dim=-1)
        return weights @ v


class TransformerBlock(nn.Module):
    """Attention -> add & norm -> feed-forward -> add & norm (post-LN, like v1)."""

    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.attention = SelfAttention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attention(x)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class SongRecommender(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_length, num_layers, dropout):
        super().__init__()
        self.song_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(context_length, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(TransformerBlock(embed_dim, dropout) for _ in range(num_layers))
        # v1's "matchmaker": scores the final vector against every song (own weights, with bias)
        self.output = nn.Linear(embed_dim, vocab_size)
        self._init_like_v1()

    def forward(self, ids):  # ids: (batch, seq) song IDs
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.dropout(self.song_embedding(ids) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        return self.output(x[:, -1, :])  # (batch, vocab_size) logits

    def _init_like_v1(self):
        """Replace PyTorch's default starting weights with v1's.

        PyTorch starts embeddings at std 1.0 and linear layers uniform with
        nonzero biases. v1 used std 0.1 embeddings and He-normal linear weights
        (std sqrt(2 / inputs)) with zero biases. LayerNorm defaults already match.
        """
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=math.sqrt(2.0 / m.in_features))
                nn.init.zeros_(m.bias)


@torch.no_grad()
def predict(model, X, device, batch_size=512):
    """Logits for every row of X (numpy song IDs) as a numpy array, dropout off.

    Runs in chunks: the full held-out logits would be 14,844 x 33,770 floats at once.
    """
    model.eval()
    chunks = [model(torch.from_numpy(X[i:i + batch_size]).to(device)).cpu()
              for i in range(0, len(X), batch_size)]
    return torch.cat(chunks).numpy()
