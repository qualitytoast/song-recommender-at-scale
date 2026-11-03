"""An update of a serving bundle: new songs and new counts as a new version, without retraining (recsys/switch.py
switches to it). The base bundle never changes; an update is a folder in it:

  <bundle>/updates/<date>-<version>/
    shards-N/shard-i/      complete slices in recsys/shard_slices.py's format, for the grown catalog
    added_songs.json       every song added since the base, in ID order (after the base's songs): uris, tracks,
                           artist (artist IDs, only artists the base has), albums (album names)
    song_playlists.npy     every song's count (the menu's order and number)
    songs_by_track.npy, songs_by_artist.npy   the name search's order (bundle.name_index)
    settings.json          the base's, with the new song count
    update.json            the base and parent versions, the songs added, a note, and the slices' checksums
    manifest.json          the version: as for a bundle, over the files above (update.json carries the slices')

Song IDs only append: song j stays song j, and a new song gets the next ID, so song j is still on shard j % N and
everything naming old songs stays right. Everything else (the models' shared layers, names, the playlist-name
words, taste profiles) is the base's, which is why a front desk keeps the base's song search index and profiles.

write_update makes one from its parent (the base or an earlier update) by extending the parent's slices, which
is all the server has: each new song's rows appended, its search vector added to the index, the pair table
re-keyed for the bigger catalog with the update's counts added. Imports numpy and FAISS only (never PyTorch:
they can't share a process, see recsys/ann.py).

    python -m recsys.updates fake --bundle bundles/<bundle> --copies 500   # a test update: copies of songs
"""
import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from recsys.bundle import file_fingerprint, name_index, write_manifest
from recsys.pair_table import AB, read_pairs, save_pairs, with_pairs
from recsys.shard_slices import PER_SONG

FIELDS = ("uris", "tracks", "artist", "albums")  # added_songs.json: one list per field, a song per position


def version_of(folder):
    return json.loads((Path(folder) / "manifest.json").read_text())["version"]


def added_songs(folder, base):
    """The songs added since the base, as of folder (the base itself: none)."""
    if Path(folder).resolve() == Path(base).resolve():
        return {f: [] for f in FIELDS}
    return json.loads((Path(folder) / "added_songs.json").read_text())


def slice_count(folder):
    """N for the folder's shards-N slices (exactly one set)."""
    sets = sorted(Path(folder).glob("shards-*"))
    if len(sets) != 1:
        raise ValueError(f"{folder}: expected one shards-N folder, found {[s.name for s in sets]}")
    return int(sets[0].name.removeprefix("shards-"))


def write_update(base, parent, added, rows, vectors, listens, pairs=None, note=""):
    """Write an update of the bundle base on top of parent (base itself or one of its updates); returns its folder.
    added: {"uris", "tracks", "artist", "albums"} for the songs it adds (their IDs follow the parent's songs);
    rows: {name: (added, ...)} each of the slices' per-song tables (shard_slices.PER_SONG) for them;
    vectors: (added, d + 1) their search vectors, the bias last; listens: every song's count after the update;
    pairs: (keys a * AB + b, sorted and distinct; counts) appeared-together counts to add, or None."""
    base, parent = Path(base), Path(parent)
    settings = json.loads((parent / "settings.json").read_text())
    before, new = settings["songs"], len(added["uris"])
    songs, shards = before + new, slice_count(parent)
    if set(added) != set(FIELDS) or any(len(added[f]) != new for f in FIELDS):
        raise ValueError(f"added songs need {FIELDS}, {new} of each")
    if len(listens) != songs or len(vectors) != new or any(len(rows[name]) != new for name in PER_SONG):
        raise ValueError(f"{new} added songs: rows, vectors and {songs} listens expected")
    pairs = pairs or (np.zeros(0, np.int64), np.zeros(0, np.int64))
    root = base / "updates"
    work = root / f".writing-{os.getpid()}"
    shutil.rmtree(work, ignore_errors=True)
    (work / f"shards-{shards}").mkdir(parents=True)
    new_ids = np.arange(before, songs, dtype=np.int64)
    checksums, sizes = {}, []
    for i in range(shards):
        src, out = parent / f"shards-{shards}" / f"shard-{i}", work / f"shards-{shards}" / f"shard-{i}"
        out.mkdir()
        mine = new_ids[new_ids % shards == i]
        ids = np.concatenate([np.load(src / "ids.npy"), mine])
        np.save(out / "ids.npy", ids)
        for name in PER_SONG:
            old = np.load(src / f"{name}.npy")
            np.save(out / f"{name}.npy", np.concatenate([old, np.asarray(rows[name])[mine - before].astype(old.dtype)]))
        index = faiss.read_index(str(src / "search_index.faiss"))
        index.add_with_ids(np.ascontiguousarray(np.asarray(vectors)[mine - before], dtype=np.float32), mine)
        faiss.write_index(index, str(out / "search_index.faiss"))
        keys, counts = read_pairs(src)
        here = pairs[0] % AB % shards == i  # pairs whose later song this shard holds
        keys, counts = with_pairs(keys, counts, before, pairs[0][here], np.asarray(pairs[1])[here], songs)
        save_pairs(out, keys, counts, songs)
        sizes.append(len(ids))
        for p in sorted(out.iterdir()):
            checksums[f"{out.relative_to(work)}/{p.name}"] = file_fingerprint(p)
    since = added_songs(parent, base)
    since = {f: since[f] + [v.item() if hasattr(v, "item") else v for v in added[f]] for f in FIELDS}
    (work / "added_songs.json").write_text(json.dumps(since, ensure_ascii=False))
    (work / "settings.json").write_text(json.dumps({**settings, "songs": songs}, indent=2))
    listens = np.asarray(listens).astype(np.int32)
    np.save(work / "song_playlists.npy", listens)
    names = json.loads((base / "songs.json").read_text())
    song_artist = np.concatenate([np.load(base / "song_artist.npy"), np.asarray(since["artist"], dtype=np.int32)])
    by_track, by_artist, _ = name_index(names["tracks"] + since["tracks"], names["artist_names"], song_artist, listens)
    np.save(work / "songs_by_track.npy", by_track)
    np.save(work / "songs_by_artist.npy", by_artist)
    made = datetime.now(timezone.utc)
    (work / "update.json").write_text(json.dumps({
        "base": version_of(base), "parent": version_of(parent), "songs": songs, "added": new,
        "added_since_base": len(since["uris"]), "pairs_added": int(len(pairs[0])), "note": note,
        "made": made.isoformat(timespec="seconds"), "slices": checksums}, indent=2))
    version = write_manifest(work)
    for i, held in enumerate(sizes):  # last: they carry the version, which covers everything above
        (work / f"shards-{shards}" / f"shard-{i}" / "info.json").write_text(json.dumps(
            {"version": version, "index": i, "shards": shards, "songs": held, "catalog": songs}))
    folder = root / f"{made:%Y-%m-%d}-{version}"
    if folder.exists():  # the very same files were written before (within the same second): keep those
        shutil.rmtree(work)
    else:
        work.rename(folder)
    return folder


def fake_update(bundle, copies, parent=None, note="", whole=None):
    """A test update of a whole bundle (one with every file recsys/export.py writes): copies of its `copies` most
    played songs, as new songs by the same artists ("<track> (copy)", URI "fake:track:<ID>"), their search vectors
    a little shorter (so no score ties the original's), and each copy played 2 times right after its song and
    right before it. With whole, also writes the same catalog as a whole bundle there, for the one-process
    recommender (only on top of the base)."""
    base = Path(bundle)
    parent = Path(parent) if parent else base
    names = json.loads((base / "songs.json").read_text())
    base_listens = np.load(base / "song_playlists.npy")
    before = json.loads((parent / "settings.json").read_text())["songs"]
    source = np.argsort(-base_listens, kind="stable")[:copies]
    ids = np.arange(before, before + copies, dtype=np.int64)
    song_artist = np.load(base / "song_artist.npy")
    albums = json.loads((base / "song_albums.json").read_text())
    added = {"uris": [f"fake:track:{j}" for j in ids], "tracks": [f"{names['tracks'][s]} (copy)" for s in source],
             "artist": song_artist[source].tolist(), "albums": [albums[s] for s in source]}
    rows = {name: np.load(base / f"{name}.npy", mmap_mode="r")[source] for name in PER_SONG}
    vectors = np.array(np.load(base / "search_vectors.npy", mmap_mode="r")[source], dtype=np.float32)
    vectors[:, :-1] *= 0.99
    listens = np.concatenate([np.load(parent / "song_playlists.npy"), base_listens[source] // 2])
    a, b = np.concatenate([source, ids]), np.concatenate([ids, source])
    order = np.argsort(a * AB + b)
    pairs = ((a * AB + b)[order], np.full(len(a), 2, dtype=np.int64))
    folder = write_update(base, parent, added, rows, vectors, listens, pairs, note or f"fake: {copies} copies")
    if whole:
        write_whole(base, Path(whole), added, rows, vectors, listens, pairs)
    return folder


def write_whole(base, out, added, rows, vectors, listens, pairs):
    """The base with the songs added, as a whole bundle in out (what one update on top of the base serves)."""
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    settings = json.loads((base / "settings.json").read_text())
    before, songs = settings["songs"], settings["songs"] + len(added["uris"])
    names = json.loads((base / "songs.json").read_text())
    for name in ("retriever_shared.pt", "ranker_shared.pt", "name_words.json"):  # the rest is written below
        os.link(base / name, out / name)
    (out / "settings.json").write_text(json.dumps({**settings, "songs": songs}, indent=2))
    (out / "songs.json").write_text(json.dumps({"uris": names["uris"] + added["uris"],
                                                "tracks": names["tracks"] + added["tracks"],
                                                "artist_names": names["artist_names"]}))
    albums = json.loads((base / "song_albums.json").read_text())
    (out / "song_albums.json").write_text(json.dumps(albums + added["albums"]))
    for name in PER_SONG:
        old = np.load(base / f"{name}.npy")
        np.save(out / f"{name}.npy", np.concatenate([old, np.asarray(rows[name]).astype(old.dtype)]))
    old = np.load(base / "search_vectors.npy")
    np.save(out / "search_vectors.npy", np.concatenate([old, vectors.astype(old.dtype)]))
    index = faiss.read_index(str(base / "search_index.faiss"))
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    faiss.write_index(index, str(out / "search_index.faiss"))
    keys, counts = with_pairs(np.load(base / "pair_keys.npy"), np.load(base / "pair_counts.npy"), before,
                              pairs[0], pairs[1], songs)
    np.save(out / "pair_keys.npy", keys)
    np.save(out / "pair_counts.npy", counts.astype(np.load(base / "pair_counts.npy").dtype))
    listens = np.asarray(listens).astype(np.int32)
    np.save(out / "song_playlists.npy", listens)
    song_artist = np.load(out / "song_artist.npy")
    by_track, by_artist, keys = name_index(names["tracks"] + added["tracks"], names["artist_names"], song_artist, listens)
    np.save(out / "songs_by_track.npy", by_track)
    np.save(out / "songs_by_artist.npy", by_artist)
    (out / "name_keys.json").write_text(json.dumps(keys))
    write_manifest(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Bundle updates (recsys/updates.py).")
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fake", help="a test update: copies of the most played songs")
    f.add_argument("--bundle", required=True, help="a whole bundle (every file recsys/export.py writes)")
    f.add_argument("--copies", type=int, required=True)
    f.add_argument("--parent", help="the update to build on (default: the bundle itself)")
    f.add_argument("--note", default="")
    f.add_argument("--whole", help="also write the same catalog as a whole bundle here")
    args = ap.parse_args()
    print(fake_update(args.bundle, args.copies, args.parent, args.note, args.whole))
