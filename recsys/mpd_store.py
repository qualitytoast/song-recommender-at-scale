"""A compact copy of the Million Playlist Dataset, built once from the raw JSON.

The raw MPD is 1,000 JSON slices (~31 GB). Reading it into Python objects every
run is slow and, for all 1M playlists, needs about as much memory as the
laptop has. This module reads the slices once and writes:

  tracks:    one row per unique track URI, numbered in order of first appearance
             track_uris.json, track_names.json, track_artist.npy (artist ID),
             track_album.npy (album ID), track_duration_ms.npy
  artists:   artist_uris.json, artist_names.json
  albums:    album_uris.json, album_names.json
  playlists: playlist_names.json, playlist_pids.npy,
             playlist_tracks.npy - every playlist's track IDs, concatenated (int32)
             playlist_offsets.npy - playlist i is playlist_tracks[offsets[i]:offsets[i + 1]]
  store.json: counts and how the store was built

Playlists are stored in the order the JSON loader reads them (slices sorted by
file name, playlists in file order), so the first N playlists are the same.
The two big arrays are opened memory-mapped: the OS reads parts of the file
only when they're used.

    python -m recsys.mpd_store --raw data/mpd --out data/mpd_store
"""
import array
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MARKER = "store.json"  # a folder with this file is a store


def is_store(folder):
    return (Path(folder) / MARKER).exists()


def build_store(raw_folder, out_folder):
    """Read every MPD slice in raw_folder once and write the store to out_folder."""
    files = sorted(f for f in os.listdir(raw_folder) if f.endswith(".json"))
    start = time.monotonic()
    track_index, track_uris, track_names = {}, [], []
    track_artist, track_album, track_duration = array.array("i"), array.array("i"), array.array("i")
    artist_index, artist_uris, artist_names = {}, [], []
    album_index, album_uris, album_names = {}, [], []
    names, pids, offsets, sequence = [], array.array("i"), array.array("q", [0]), array.array("i")
    conflicting = set()  # track URIs whose metadata differs between occurrences

    def intern(index, uris, labels, uri, label):
        i = index.get(uri)
        if i is None:
            i = index[uri] = len(uris)
            uris.append(uri)
            labels.append(label)
        return i

    for n, file_name in enumerate(files, 1):
        with open(os.path.join(raw_folder, file_name), encoding="utf-8") as f:
            for playlist in json.load(f)["playlists"]:
                for t in playlist["tracks"]:
                    artist = intern(artist_index, artist_uris, artist_names, t["artist_uri"], t["artist_name"])
                    album = intern(album_index, album_uris, album_names, t["album_uri"], t["album_name"])
                    i = track_index.get(t["track_uri"])
                    if i is None:
                        i = track_index[t["track_uri"]] = len(track_uris)
                        track_uris.append(t["track_uri"])
                        track_names.append(t["track_name"])
                        track_artist.append(artist)
                        track_album.append(album)
                        track_duration.append(t["duration_ms"])
                    elif (track_names[i], track_artist[i], track_album[i], track_duration[i]) != (
                            t["track_name"], artist, album, t["duration_ms"]):
                        conflicting.add(i)
                    sequence.append(i)
                offsets.append(len(sequence))
                names.append(playlist["name"])
                pids.append(playlist["pid"])
        if n % 50 == 0 or n == len(files):
            print(f"  {n:,}/{len(files):,} slices | {len(names):,} playlists | {len(sequence):,} track entries | "
                  f"{len(track_uris):,} tracks | {time.monotonic() - start:.0f}s", flush=True)

    tmp = Path(str(out_folder) + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for file_name, values in [("track_uris.json", track_uris), ("track_names.json", track_names),
                              ("artist_uris.json", artist_uris), ("artist_names.json", artist_names),
                              ("album_uris.json", album_uris), ("album_names.json", album_names),
                              ("playlist_names.json", names)]:
        (tmp / file_name).write_text(json.dumps(values, ensure_ascii=False), encoding="utf-8")
    for file_name, values, dtype in [("track_artist.npy", track_artist, np.int32),
                                     ("track_album.npy", track_album, np.int32),
                                     ("track_duration_ms.npy", track_duration, np.int32),
                                     ("playlist_pids.npy", pids, np.int32),
                                     ("playlist_offsets.npy", offsets, np.int64),
                                     ("playlist_tracks.npy", sequence, np.int32)]:
        np.save(tmp / file_name, np.frombuffer(values, dtype=dtype))
    meta = {"raw_folder": str(raw_folder), "slices": len(files), "playlists": len(names),
            "track_entries": len(sequence), "tracks": len(track_uris), "artists": len(artist_uris),
            "albums": len(album_uris), "tracks_with_conflicting_metadata": len(conflicting),
            "build_seconds": round(time.monotonic() - start), "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    (tmp / MARKER).write_text(json.dumps(meta, indent=2))
    if Path(out_folder).exists():
        shutil.rmtree(out_folder)
    tmp.rename(out_folder)  # only a finished store appears under its real name
    return meta


@dataclass
class Store:
    meta: dict
    track_uris: list
    track_names: list
    track_artist: np.ndarray       # (tracks,) artist ID of each track
    track_album: np.ndarray        # (tracks,) album ID
    track_duration_ms: np.ndarray  # (tracks,)
    artist_uris: list
    artist_names: list
    album_uris: list
    album_names: list
    playlist_names: list
    playlist_pids: np.ndarray
    playlist_offsets: np.ndarray   # (playlists + 1,)
    playlist_tracks: np.ndarray    # (track entries,), memory-mapped

    def playlist(self, i):
        """Track IDs of playlist i."""
        return self.playlist_tracks[self.playlist_offsets[i]:self.playlist_offsets[i + 1]]


def load_store(folder):
    folder = Path(folder)
    read_json = lambda name: json.loads((folder / name).read_text(encoding="utf-8"))
    read_npy = lambda name, mmap=None: np.load(folder / name, mmap_mode=mmap)
    return Store(meta=read_json(MARKER),
                 track_uris=read_json("track_uris.json"), track_names=read_json("track_names.json"),
                 track_artist=read_npy("track_artist.npy"), track_album=read_npy("track_album.npy"),
                 track_duration_ms=read_npy("track_duration_ms.npy"),
                 artist_uris=read_json("artist_uris.json"), artist_names=read_json("artist_names.json"),
                 album_uris=read_json("album_uris.json"), album_names=read_json("album_names.json"),
                 playlist_names=read_json("playlist_names.json"), playlist_pids=read_npy("playlist_pids.npy"),
                 playlist_offsets=read_npy("playlist_offsets.npy"),
                 playlist_tracks=read_npy("playlist_tracks.npy", mmap="r"))


def playlists_from_store(store, max_playlists, min_playlist_len, song_key):
    """Same output as recsys.data.load_playlists, read from a store:
    (playlists of song keys, {key: track record}, playlist names).

    One difference: load_playlists reads whole slices, so its track records also
    include playlists past max_playlists in the last slice it reads. That only
    matters if max_playlists isn't a multiple of the slice size (1,000)."""
    keys = store.track_uris if song_key == "track_uri" else store.track_names
    playlists, names, kept_ids = [], [], []
    for i in range(len(store.playlist_names)):
        if len(playlists) >= max_playlists:
            break
        ids = store.playlist(i)
        if len(ids) >= min_playlist_len:
            ids = ids.tolist()
            playlists.append([keys[j] for j in ids])
            names.append(store.playlist_names[i])
            kept_ids.extend(ids)
    tracks = {}
    for j in kept_ids:  # in reading order, so with track names the last record read wins, as in load_playlists
        a, b = int(store.track_artist[j]), int(store.track_album[j])
        tracks[keys[j]] = {"track_uri": store.track_uris[j], "track_name": store.track_names[j],
                           "artist_uri": store.artist_uris[a], "artist_name": store.artist_names[a],
                           "album_uri": store.album_uris[b], "album_name": store.album_names[b],
                           "duration_ms": int(store.track_duration_ms[j])}
    return playlists, tracks, names


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the compact MPD store from the raw JSON slices.")
    ap.add_argument("--raw", default="data/mpd")
    ap.add_argument("--out", default="data/mpd_store")
    args = ap.parse_args()
    print(json.dumps(build_store(args.raw, args.out), indent=2))
