"""Run settings, loaded from a TOML file in configs/.

Nothing here has a default: every setting comes from the TOML file, and a
missing or misspelled key raises an error instead of silently falling back.
Settings with a fixed set of options are checked when the config loads.
"""
import tomllib
from dataclasses import dataclass


FEATURES = ("artist", "album", "duration", "playlist_name", "genre")


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
    context_length: int    # songs in each window: windows predict the song after this many known songs
    input_length: int      # songs the retriever reads (>= context_length): a window's context songs and
                           # the songs before them, up to this many; training chunks are this long
    test_split: float      # fraction of playlists held out
    validation: str        # "held_out_prefix": first val_size held-out windows (v1)
                           # "separate_playlists": val_split of playlists, never held out
    val_size: int          # used by "held_out_prefix"
    val_split: float       # used by "separate_playlists"; must be 0 otherwise
    genres_file: str       # artist genres JSONL (scripts/fetch_genres.py or genres_from_dump.py)
    val_max_windows: int   # validate on at most this many windows (a fixed random sample); 0 = all
    test_max_windows: int  # score at most this many held-out windows: a fixed random sample, each still ranked
                           # against every song, so scores stay unbiased; 0 = all
    ranker_split: float    # fraction of training playlists kept out of retriever training, for
                           # training the second-stage ranker (recsys/ranker.py); 0 = none

    def __post_init__(self):  # runs right after the dataclass fills in its fields
        _check_choice("song_key", self.song_key, ("track_name", "track_uri"))
        _check_choice("vocab_from", self.vocab_from, ("all", "train"))
        _check_choice("validation", self.validation, ("held_out_prefix", "separate_playlists"))
        if (self.validation == "separate_playlists") != (self.val_split > 0):
            raise ValueError("val_split must be > 0 with separate_playlists validation, "
                             "and 0 with held_out_prefix")
        if self.validation == "held_out_prefix" and self.test_max_windows:
            raise ValueError("held_out_prefix validation uses the first held-out windows; test_max_windows must be 0")
        if self.input_length < self.context_length:
            raise ValueError(f"input_length ({self.input_length}) can't be below context_length "
                             f"({self.context_length})")


@dataclass(frozen=True)
class ModelConfig:
    embed_dim: int
    num_layers: int        # Transformer blocks
    dropout: float
    scale_attention: bool  # divide attention scores by sqrt(embed_dim); v1 didn't
    init: str              # starting weights: "v1" or "pytorch" (PyTorch's defaults)
    features: list         # song features added to the song ID, e.g. ["artist"]

    def __post_init__(self):
        _check_choice("init", self.init, ("v1", "pytorch"))
        for name in self.features:
            _check_choice("feature", name, FEATURES)


@dataclass(frozen=True)
class TrainConfig:
    optimizer: str         # "sgd", "adam", "adamw" or "lazy_adamw" (recsys/lazy_adam.py)
    lr: float
    weight_decay: float
    batch_size: int
    epochs: int            # maximum epochs
    eval_every_targets: int  # run a validation check every this many training targets
    min_checks: int        # early-stopping patience only counts from this validation check on
    patience: int          # stop after this many counted checks without a new best NDCG@10
    augment_mask: float    # training windows: chance each song is hidden (see recsys/augment.py)
    augment_crop: int      # training windows: hide the first 0..augment_crop songs
    augment_reorder: float # training windows: chance a run of 3-5 songs is shuffled
    objective: str         # "last_position": one window -> its next song;
                           # "every_position": chunks of songs, the next song predicted at
                           # every position, with causal attention (see recsys.data.make_chunks)
    softmax: str           # "full": training scores every song; "sampled": only the batch's
                           # true next songs plus sampled_negatives random ones (see recsys.sampled)
    sampled_negatives: int # random songs added to each batch's candidates (sampled softmax)
    negative_power: float  # random songs drawn by (target frequency)^power; 0 = uniform

    def __post_init__(self):
        _check_choice("objective", self.objective, ("last_position", "every_position"))
        _check_choice("softmax", self.softmax, ("full", "sampled"))

    @property
    def augmenting(self):
        return self.augment_mask > 0 or self.augment_crop > 0 or self.augment_reorder > 0


@dataclass(frozen=True)
class Config:
    data_seed: int         # shuffles playlists into train / validation / held-out
    train_seeds: list      # one training run per seed: starting weights, batch order, dropout
    data: DataConfig
    model: ModelConfig
    train: TrainConfig

    def __post_init__(self):
        # With track names as song keys, same-titled songs by different artists merge,
        # so "the song's artist" isn't well defined.
        if self.model.features and self.data.song_key != "track_uri":
            raise ValueError("song features need song_key = \"track_uri\"")
        # A last-position model reads exactly the window's songs; only every-position
        # training (chunks) teaches the retriever to read inputs of any length.
        if (self.train is not None and self.train.objective == "last_position"
                and self.data.input_length != self.data.context_length):
            raise ValueError("input_length must equal context_length with the last_position objective")


def load_config(path):
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    data = DataConfig(**raw.pop("data"))
    model = ModelConfig(**raw.pop("model"))
    train = TrainConfig(**raw.pop("train"))
    return Config(data=data, model=model, train=train, **raw)
