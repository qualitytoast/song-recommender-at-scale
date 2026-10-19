"""Export a trained retriever + ranker pair as a serving bundle: one folder the server
(recsys/recommender.py, recsys/serve.py) loads in seconds, without the training data or the full
checkpoints.

    python -m recsys.export --ranker configs/ranker_1m_any_length.toml

Writes bundles/<ranker config name>-<version>/:
  settings.json                  shapes and settings (input length, shortlist, IVF, extra inputs, ...)
  retriever_shared.pt            the retriever's shared layers: transformer, positions, name words
  ranker_shared.pt               the ranker's: the same for its fine-tuned copy, plus its new weights
  retriever_song_inputs.npy      one input vector per song: song + artist + album + length + genres,
  ranker_song_inputs.npy           added up once here, so serving reads one row instead of five tables
  search_vectors.npy             each song's output vector with its bias ([h, 1] . vector = its score)
  search_index.faiss             the IVF index over them, saved so the server doesn't rebuild it
  song_artist.npy, song_album.npy, song_genres.npy   per-song IDs for the ranker's overlap inputs
  pair_keys.npy, pair_counts.npy the appeared-together table (sorted keys a * songs + b; 2-byte counts)
  songs.json                     Spotify URIs, track names and artist names
  song_playlists.npy             how many of the MPD playlists contain each song (the find-songs menu's order)
  songs_by_track.npy, songs_by_artist.npy, name_keys.json   song IDs sorted by track / artist name and the
                                 names as compared (recommender.name_key), for finding songs as the user types
  name_words.json                the playlist-name words (word i has ID i + 1)
  manifest.json                  every file's SHA-256 and size, and the version
The version is the first 12 hex digits of a SHA-256 over the files' own SHA-256s, so a change to any
file gives a new version, and the same pair always exports to the same version.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from recsys.data import concat_playlists
from recsys.mpd_store import load_store
from recsys.rank_features import GROUPS, pair_counts
from recsys.ranker import CandidateRanker, check_vocab, file_fingerprint, load_ranker_config, load_retriever
from recsys.recommender import SERVED_GROUPS, name_key, shared_state
from recsys.search import SearchWorker, song_vectors
from recsys.train import pick_device, run_dir_for


@torch.no_grad()
def song_inputs(model, device, batch=65536):
    """(songs, d) float32: every song's input vector (model.embed_songs), a batch of songs at a time."""
    model.eval()
    songs = model.song_embedding.num_embeddings
    return torch.cat([model.embed_songs(torch.arange(i, min(i + batch, songs), device=device)).float().cpu()
                      for i in range(0, songs, batch)]).numpy()


def write_manifest(folder):
    """manifest.json for every other file in folder; returns the version."""
    files = {p.name: {"sha256": file_fingerprint(p), "bytes": p.stat().st_size}
             for p in sorted(folder.iterdir()) if p.name != "manifest.json"}
    version = hashlib.sha256("".join(f"{n} {f['sha256']}\n" for n, f in files.items()).encode()).hexdigest()[:12]
    (folder / "manifest.json").write_text(json.dumps({"version": version, "files": files}, indent=2))
    return version


def name_index(tracks, artist_names, song_artist, song_playlists):
    """(songs_by_track, songs_by_artist, name_keys): song IDs sorted by track name key and by artist
    name key, each in the most playlists first among equal names, and the keys themselves."""
    track_keys = [name_key(t) for t in tracks]
    artist_keys = [name_key(a) for a in artist_names]
    by_track = sorted(range(len(tracks)), key=lambda j: (track_keys[j], -song_playlists[j], j))
    by_artist = sorted(range(len(tracks)), key=lambda j: (artist_keys[song_artist[j]], -song_playlists[j], j))
    return (np.array(by_track, dtype=np.int32), np.array(by_artist, dtype=np.int32),
            {"tracks": track_keys, "artists": artist_keys})


def playlist_counts(cfg, vocab):
    """How many of the config's playlists (all of them, held-out ones too) contain each vocab song:
    the find-songs menu's popularity order. Display only: no model sees it."""
    d = cfg.data
    store = load_store(d.folder)
    kept = np.flatnonzero(np.diff(store.playlist_offsets) >= d.min_playlist_len)[:d.max_playlists]
    tracks, offsets = concat_playlists(store, kept)
    vocab_id = {uri: i for i, uri in enumerate(vocab)}
    songs = np.array([vocab_id.get(u, -1) for u in store.track_uris], dtype=np.int64)[tracks]
    playlist = np.repeat(np.arange(len(kept)), np.diff(offsets))
    known = songs >= 0
    once = np.unique(playlist[known] * len(vocab) + songs[known])  # each (playlist, song) once
    return np.bincount(once % len(vocab), minlength=len(vocab))


def write_bundle(out_root, name, retriever, ranker, settings, uris, tracks, artist_names, name_words, pairs,
                 song_playlists, device):
    """Write the bundle for a trained retriever and ranker (see the module docstring) to
    out_root/<name>-<version> and return that path. settings: settings.json's contents (see
    export); uris, tracks: per song; artist_names: per artist ID; name_words: the name-word list;
    pairs: (keys, counts) from rank_features.pair_counts; song_playlists: per song (playlist_counts)."""
    out_root = Path(out_root)
    partial = out_root / f".{name}-partial"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    save = lambda file, a: np.save(partial / f"{file}.npy", a)
    retriever, ranker = retriever.to(device).eval(), ranker.to(device).eval()
    settings = {**settings, "songs": retriever.song_embedding.num_embeddings,
                "embed_dim": retriever.song_embedding.embedding_dim, "num_layers": len(retriever.blocks),
                "scale_attention": retriever.blocks[0].attention.scale, "causal": retriever.blocks[0].attention.causal,
                "input_length": retriever.position_embedding.num_embeddings,
                # None: the retriever has no playlist-name input (the model's own convention; 0 would make
                # serving build a 1-row name table the bundle has no weights for)
                "name_word_count": None if retriever.name_words is None else len(name_words),
                "n_features": sum(len(GROUPS[g]) for g in settings["features"])}
    (partial / "settings.json").write_text(json.dumps(settings, indent=2))
    torch.save(shared_state(retriever), partial / "retriever_shared.pt")
    torch.save(shared_state(ranker), partial / "ranker_shared.pt")
    save("retriever_song_inputs", song_inputs(retriever, device))
    save("ranker_song_inputs", song_inputs(ranker.retriever, device))
    vectors = song_vectors(retriever)
    save("search_vectors", vectors)
    with SearchWorker() as worker:
        worker.build("songs", vectors, kind="ivf", nlist=settings["ivf_nlist"], nprobe=settings["ivf_nprobe"])
        worker.save("songs", (partial / "search_index.faiss").resolve())
    song_artist = retriever.song_feature_ids("artist").cpu().numpy().astype(np.int32)
    save("song_artist", song_artist)
    save("song_album", retriever.song_feature_ids("album").cpu().numpy().astype(np.int32))
    save("song_genres", retriever.song_genres.cpu().numpy().astype(np.int16))
    keys, counts = pairs
    save("pair_keys", np.asarray(keys, dtype=np.int64))
    save("pair_counts", np.asarray(counts).astype(np.int16 if len(counts) == 0 or counts.max() < 2 ** 15 else np.int32))
    (partial / "songs.json").write_text(json.dumps({"uris": list(uris), "tracks": list(tracks),
                                                    "artist_names": list(artist_names)}))
    song_playlists = np.asarray(song_playlists, dtype=np.int32)
    save("song_playlists", song_playlists)
    by_track, by_artist, keys = name_index(list(tracks), list(artist_names), song_artist, song_playlists)
    save("songs_by_track", by_track)
    save("songs_by_artist", by_artist)
    (partial / "name_keys.json").write_text(json.dumps(keys))
    (partial / "name_words.json").write_text(json.dumps(list(name_words)))
    version = write_manifest(partial)
    final = out_root / f"{name}-{version}"
    shutil.rmtree(final, ignore_errors=True)
    partial.rename(final)
    return final


def export(ranker_config, seed=1, out_root="bundles"):
    """Bundle the ranker config's best checkpoint and its retriever's (see write_bundle)."""
    rc = load_ranker_config(ranker_config)
    if rc.search != "ivf":
        raise ValueError("serving searches with the IVF index; this ranker config uses exact search")
    if set(rc.features) - set(SERVED_GROUPS):
        raise ValueError(f"serving supplies only the extra inputs {SERVED_GROUPS}; this ranker uses {rc.features}")
    device = pick_device()
    cfg, ds, retriever = load_retriever(rc, seed, device)
    ranker_path, retriever_path = run_dir_for(ranker_config, seed) / "best.pt", run_dir_for(rc.retriever, seed) / "best.pt"
    checkpoint = torch.load(ranker_path, weights_only=True)
    check_vocab(checkpoint, ds.vocab, ranker_path)
    ranker = CandidateRanker(retriever, sum(len(GROUPS[g]) for g in rc.features), rc.correction_hidden,
                             rc.predict_steps).to(device)
    ranker.load_state_dict(checkpoint["model"])
    settings = {"shortlist": rc.shortlist, "exclude_input": rc.exclude_input, "ivf_nlist": rc.ivf_nlist,
                "ivf_nprobe": rc.ivf_nprobe, "features": rc.features, "correction_hidden": rc.correction_hidden,
                "windows": rc.windows, "sources": {"ranker_config": str(ranker_config), "retriever_config": rc.retriever,
                                                   "ranker_checkpoint_sha256": file_fingerprint(ranker_path),
                                                   "retriever_checkpoint_sha256": file_fingerprint(retriever_path)}}
    pairs = pair_counts(ds.fit_songs, ds.fit_offsets, len(ds.vocab))
    return write_bundle(out_root, Path(ranker_config).stem, retriever, ranker, settings, ds.vocab, ds.names,
                        ds.artist_names, ds.name_words, pairs, playlist_counts(cfg, ds.vocab), device)



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export a retriever + ranker pair as a serving bundle.")
    ap.add_argument("--ranker", required=True, help="ranker config")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="bundles")
    args = ap.parse_args()
    print(export(args.ranker, args.seed, args.out))
