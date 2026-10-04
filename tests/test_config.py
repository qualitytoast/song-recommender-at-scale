import pytest

from recsys.config import load_config

DATA = """
[data]
folder = "data/mpd"
max_playlists = 5000
min_playlist_len = 4
min_freq = 2
context_length = 10
test_split = 0.1
val_size = 3000
"""


def test_loads_values(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("seed = 42\n" + DATA)
    cfg = load_config(path)
    assert cfg.seed == 42 and cfg.data.context_length == 10


def test_misspelled_key_raises(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("seed = 42\n" + DATA.replace("min_freq", "min_frq"))
    with pytest.raises(TypeError):
        load_config(path)


def test_missing_key_raises(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(DATA)  # no seed
    with pytest.raises(TypeError):
        load_config(path)
