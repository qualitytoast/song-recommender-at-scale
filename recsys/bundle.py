"""Serving-bundle helpers that need neither PyTorch nor FAISS, so a FAISS process can use them too (the two can't
share a process, see recsys/ann.py): file checksums, the manifest and version, and the find-songs name index.
recsys/export.py writes bundles with them; recsys/daily.py writes updates of a bundle.
"""
import hashlib
import json
import unicodedata

import numpy as np


def file_fingerprint(path, block=1 << 24):
    """SHA-256 of a file's bytes, read a block at a time."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(folder):
    """manifest.json for every other file in folder; returns the version."""
    files = {p.name: {"sha256": file_fingerprint(p), "bytes": p.stat().st_size}
             for p in sorted(folder.iterdir()) if p.name != "manifest.json"}
    version = hashlib.sha256("".join(f"{n} {f['sha256']}\n" for n, f in files.items()).encode()).hexdigest()[:12]
    (folder / "manifest.json").write_text(json.dumps({"version": version, "files": files}, indent=2))
    return version


def name_key(text):
    """How song and artist names are compared: accents removed, case folded, spaces collapsed
    ("Beyoncé  Knowles" -> "beyonce knowles")."""
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return " ".join(text.casefold().split())


def name_index(tracks, artist_names, song_artist, song_playlists):
    """(songs_by_track, songs_by_artist, name_keys): song IDs sorted by track name key and by artist
    name key, each in the most playlists first among equal names, and the keys themselves."""
    track_keys = [name_key(t) for t in tracks]
    artist_keys = [name_key(a) for a in artist_names]
    by_track = sorted(range(len(tracks)), key=lambda j: (track_keys[j], -song_playlists[j], j))
    by_artist = sorted(range(len(tracks)), key=lambda j: (artist_keys[song_artist[j]], -song_playlists[j], j))
    return (np.array(by_track, dtype=np.int32), np.array(by_artist, dtype=np.int32),
            {"tracks": track_keys, "artists": artist_keys})
