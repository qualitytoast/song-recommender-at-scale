"""Run settings, loaded from a TOML file in configs/.

Nothing here has a default: every setting comes from the TOML file, and a
missing or misspelled key raises an error instead of silently falling back.
Settings with a fixed set of options are checked when the config loads.
"""
import tomllib
from dataclasses import dataclass


def _check_choice(name, value, choices):
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")


@dataclass(frozen=True)
class DataConfig:
    folder: str            # one MPD dataset folder, e.g. "data/mpd"
    max_playlists: int
    min_playlist_len: int  # playlists with fewer tracks are skipped
    min_freq: int          # songs seen fewer times are dropped from the vocab
    song_key: str          # "track_name" (v1: same-titled songs merge) or "track_uri"
    vocab_from: str        # count songs over "all" playlists (v1) or "train" only
    context_length: int    # songs in each input window
    test_split: float      # fraction of playlists held out
    validation: str        # "held_out_prefix": first val_size held-out windows (v1)
                           # "separate_playlists": val_split of playlists, never held out
    val_size: int          # used by "held_out_prefix"
    val_split: float       # used by "separate_playlists"; must be 0 otherwise

    def __post_init__(self):  # runs right after the dataclass fills in its fields
        _check_choice("song_key", self.song_key, ("track_name", "track_uri"))
        _check_choice("vocab_from", self.vocab_from, ("all", "train"))
        _check_choice("validation", self.validation, ("held_out_prefix", "separate_playlists"))
        if (self.validation == "separate_playlists") != (self.val_split > 0):
            raise ValueError("val_split must be > 0 with separate_playlists validation, "
                             "and 0 with held_out_prefix")


@dataclass(frozen=True)
class ModelConfig:
    embed_dim: int
    num_layers: int        # Transformer blocks
    dropout: float
    scale_attention: bool  # divide attention scores by sqrt(embed_dim); v1 didn't
    init: str              # starting weights: "v1" or "pytorch" (PyTorch's defaults)

    def __post_init__(self):
        _check_choice("init", self.init, ("v1", "pytorch"))


@dataclass(frozen=True)
class TrainConfig:
    optimizer: str         # "sgd", "adam" or "adamw"
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
