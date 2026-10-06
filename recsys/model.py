"""SongRecommender: v1's Transformer, rebuilt in PyTorch.

Song IDs -> embeddings + position embeddings -> stacked Transformer blocks ->
last position's vector -> a score for every song in the catalog.

Matches v1's model.py layer for layer. Two v1 behaviours are settings, so
v1-matching runs keep reproducing:
  - scale_attention=False: attention scores are NOT divided by sqrt(embed_dim)
    (v1 quirk, see SelfAttention)
  - init="v1": weights start from v1's init, not PyTorch's (see _init_like_v1)
"""
import math

import torch
from torch import nn


class SelfAttention(nn.Module):
    """Single-head attention: each song pulls in a weighted blend of the others."""

    def __init__(self, embed_dim, scale):
        super().__init__()
        self.scale = scale
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):  # x: (batch, seq, embed_dim)
        q, k, v = self.query(x), self.key(x), self.value(x)
        # scores[b, i, j] = how relevant song j is to song i
        scores = q @ k.transpose(-2, -1)
        # Scores are sums of embed_dim products, so they grow with embed_dim;
        # dividing by sqrt(embed_dim) keeps softmax from saturating. v1 computed
        # the scaled scores but passed the unscaled ones to softmax (scale=False).
        if self.scale:
            scores = scores / math.sqrt(q.shape[-1])
        weights = scores.softmax(dim=-1)
        return weights @ v


class TransformerBlock(nn.Module):
    """Attention -> add & norm -> feed-forward -> add & norm (post-LN, like v1)."""

    def __init__(self, embed_dim, dropout, scale_attention):
        super().__init__()
        self.attention = SelfAttention(embed_dim, scale_attention)
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
    """song_features: {name: per-song ID array}, e.g. {"artist": ids} where ids[i] is
    song i's artist. Each feature adds a learned vector per ID on two sides:
      input:  a song in the window = song vector + its artist vector + ...
      output: candidate song j's score = h . (output row j + its artist's output vector + ...)
    so songs sharing an artist share what is learned about that artist.

    name_word_count: if set, the playlist name is used too: a learned vector per
    name word, averaged over the name's words and added at every position.

    song_genres: if set, (vocab_size, k) genre IDs of each song's artist (0 = padding).
    Each genre gets a learned input and output vector; a song's genre vector is the
    average over its genres, used on both sides like the song features. A song
    whose artist has no genres gets a zero genre vector.
    """

    def __init__(self, vocab_size, embed_dim, context_length, num_layers, dropout,
                 scale_attention, init, song_features=None, name_word_count=None,
                 song_genres=None):
        super().__init__()
        self.song_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(context_length, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(TransformerBlock(embed_dim, dropout, scale_attention)
                                    for _ in range(num_layers))
        # v1's "matchmaker": scores the final vector against every song (own weights, with bias)
        self.output = nn.Linear(embed_dim, vocab_size)
        if init == "v1":
            self._init_like_v1()
        # init == "pytorch": keep PyTorch's defaults (embeddings std 1.0; linear
        # weights and biases uniform in +-1/sqrt(inputs))

        # Feature tables come last and start at zero without drawing random numbers:
        # at step 0 the model computes exactly what it would without features, and a
        # given seed gets the same starting weights, batch order and dropout masks
        # with or without them, so runs with and without a feature are paired.
        self.feature_names = sorted(song_features or {})
        self.input_features, self.output_features = nn.ModuleDict(), nn.ModuleDict()
        for name in self.feature_names:
            ids = torch.as_tensor(song_features[name], dtype=torch.long)
            # A buffer is saved and moved with the model but not learned.
            self.register_buffer(f"song_{name}", ids)
            count = int(ids.max()) + 1
            self.input_features[name] = nn.Embedding.from_pretrained(
                torch.zeros(count, embed_dim), freeze=False)
            self.output_features[name] = nn.Embedding.from_pretrained(
                torch.zeros(count, embed_dim), freeze=False)
        self.name_words = None
        if name_word_count is not None:
            # Row 0 is padding: it stays zero and is left out of the average.
            self.name_words = nn.Embedding.from_pretrained(
                torch.zeros(name_word_count + 1, embed_dim), freeze=False, padding_idx=0)
        self.input_genres = self.output_genres = None
        if song_genres is not None:
            genres = torch.as_tensor(song_genres, dtype=torch.long)
            self.register_buffer("song_genres", genres)
            count = int(genres.max()) + 1  # including padding row 0
            self.input_genres = nn.Embedding.from_pretrained(
                torch.zeros(count, embed_dim), freeze=False, padding_idx=0)
            self.output_genres = nn.Embedding.from_pretrained(
                torch.zeros(count, embed_dim), freeze=False, padding_idx=0)

    def song_feature_ids(self, name):
        return getattr(self, f"song_{name}")

    @staticmethod
    def mean_vector(table, ids):
        """Average of table's vectors for ids along the last axis, leaving out padding (ID 0).
        All padding gives a zero vector. ids (..., k) -> (..., embed_dim)."""
        real = (ids != 0).sum(dim=-1, keepdim=True).clamp(min=1)
        return table(ids).sum(dim=-2) / real

    def forward(self, ids, names=None):  # ids: (batch, seq) song IDs; names: (batch, width) word IDs
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.song_embedding(ids) + self.position_embedding(positions)
        for name in self.feature_names:
            x = x + self.input_features[name](self.song_feature_ids(name)[ids])
        if self.input_genres is not None:
            x = x + self.mean_vector(self.input_genres, self.song_genres[ids])  # (batch, seq, embed_dim)
        if self.name_words is not None:
            name_vector = self.mean_vector(self.name_words, names)  # (batch, embed_dim)
            x = x + name_vector[:, None, :]  # same name vector at every position
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        h = x[:, -1, :]
        logits = self.output(h)  # (batch, vocab_size): h . output row + bias, per song
        # Every candidate song's feature vectors summed: (vocab_size, embed_dim)
        candidates = [self.output_features[name](self.song_feature_ids(name)) for name in self.feature_names]
        if self.output_genres is not None:
            candidates.append(self.mean_vector(self.output_genres, self.song_genres))
        if candidates:
            logits = logits + h @ sum(candidates).T
        return logits

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


def build_model(cfg, ds):
    """SongRecommender with the shape and settings from a Config, for Dataset ds."""
    m = cfg.model
    song_features = {name: ds.song_features[name][0] for name in m.features
                     if name not in ("playlist_name", "genre")}
    name_word_count = len(ds.name_words) if "playlist_name" in m.features else None
    song_genres = ds.song_genres if "genre" in m.features else None
    return SongRecommender(len(ds.vocab), m.embed_dim, cfg.data.context_length, m.num_layers,
                           m.dropout, m.scale_attention, m.init, song_features, name_word_count,
                           song_genres)


@torch.no_grad()
def predict(model, X, N, device, batch_size=512):
    """Logits for every row of X (numpy song IDs, with name word IDs N) as numpy, dropout off.

    Runs in chunks: the full held-out logits would be 14,844 x 33,770 floats at once.
    """
    model.eval()
    chunks = [model(torch.from_numpy(X[i:i + batch_size]).to(device),
                    torch.from_numpy(N[i:i + batch_size]).to(device)).cpu()
              for i in range(0, len(X), batch_size)]
    return torch.cat(chunks).numpy()
