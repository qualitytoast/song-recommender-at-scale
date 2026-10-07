import pytest

from recsys.config import load_config

DATA = """
[data]
folder = "data/mpd"
max_playlists = 5000
min_playlist_len = 4
min_freq = 2
song_key = "track_name"
vocab_from = "all"
context_length = 10
test_split = 0.1
val_size = 3000
validation = "held_out_prefix"
val_split = 0.0

[model]
embed_dim = 64
num_layers = 2
dropout = 0.1
scale_attention = false
init = "v1"
features = []

[train]
optimizer = "sgd"
lr = 0.05
weight_decay = 1e-3
batch_size = 32
epochs = 40
min_epochs = 20
patience = 10
augment_mask = 0.0
augment_crop = 0
augment_reorder = 0.0
objective = "last_position"
"""


def test_loads_values(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("data_seed = 42\ntrain_seeds = [1, 2, 3]\n" + DATA)
    cfg = load_config(path)
    assert cfg.data_seed == 42 and cfg.train_seeds == [1, 2, 3] and cfg.data.context_length == 10


def test_misspelled_key_raises(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("data_seed = 42\ntrain_seeds = [1, 2, 3]\n" + DATA.replace("min_freq", "min_frq"))
    with pytest.raises(TypeError):
        load_config(path)


def test_missing_key_raises(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(DATA)  # no seeds
    with pytest.raises(TypeError):
        load_config(path)


def test_invalid_choice_raises(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("data_seed = 42\ntrain_seeds = [1, 2, 3]\n" + DATA.replace('"track_name"', '"track-uri"'))
    with pytest.raises(ValueError):
        load_config(path)


def test_val_split_must_match_validation_mode(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("data_seed = 42\ntrain_seeds = [1, 2, 3]\n" + DATA.replace("val_split = 0.0", "val_split = 0.1"))
    with pytest.raises(ValueError):  # held_out_prefix with a nonzero val_split
        load_config(path)


def test_unknown_feature_raises(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("data_seed = 42\ntrain_seeds = [1]\n" + DATA.replace("features = []", 'features = ["mood"]'))
    with pytest.raises(ValueError):
        load_config(path)


def test_features_need_track_uri(tmp_path):
    path = tmp_path / "c.toml"  # DATA uses song_key = "track_name"
    path.write_text("data_seed = 42\ntrain_seeds = [1]\n" + DATA.replace("features = []", 'features = ["artist"]'))
    with pytest.raises(ValueError):
        load_config(path)


def test_every_position_with_augmentation_loads(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("data_seed = 42\ntrain_seeds = [1]\n" + DATA.replace('"last_position"', '"every_position"')
                    .replace("augment_mask = 0.0", "augment_mask = 0.2"))
    cfg = load_config(path)
    assert cfg.train.objective == "every_position" and cfg.train.augmenting
