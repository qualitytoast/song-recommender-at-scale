"""Run settings, loaded from a TOML file in configs/.

Nothing here has a default: every setting comes from the TOML file, and a
missing or misspelled key raises an error instead of silently falling back.
"""
import tomllib
from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    folder: str            # one MPD dataset folder, e.g. "data/mpd"
    max_playlists: int
    min_playlist_len: int  # playlists with fewer tracks are skipped
    min_freq: int          # songs seen fewer times are dropped from the vocab
    context_length: int    # songs in each input window
    test_split: float      # fraction of playlists held out
    val_size: int          # first N held-out windows, used for early stopping


@dataclass(frozen=True)
class ModelConfig:
    embed_dim: int
    num_layers: int        # Transformer blocks
    dropout: float


@dataclass(frozen=True)
class TrainConfig:
    optimizer: str         # "sgd"
    lr: float
    weight_decay: float
    batch_size: int
    epochs: int            # maximum epochs
    min_epochs: int        # early-stopping patience only counts from this epoch on
    patience: int          # stop after this many counted epochs without a new best NDCG@10


@dataclass(frozen=True)
class Config:
    seed: int
    data: DataConfig
    model: ModelConfig
    train: TrainConfig


def load_config(path):
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    data = DataConfig(**raw.pop("data"))
    model = ModelConfig(**raw.pop("model"))
    train = TrainConfig(**raw.pop("train"))
    return Config(data=data, model=model, train=train, **raw)
