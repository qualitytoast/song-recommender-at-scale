"""Report what the genre vocabulary cutoff does, before genre is used for training.

    python scripts/genre_report.py                   # cutoffs 5 and 3, side by side
    python scripts/genre_report.py --cutoffs 5 4 3

For each cutoff ("a genre must be listed for at least N vocab artists"), shows how
many genres survive and sorts the vocab's artists into: never matched on
MusicBrainz, matched but no genres, had genres but lost them all to the cutoff,
and kept at least one. "Training data" is measured two ways: the share of
training targets (next songs) and of songs in training windows by those artists.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys.config import load_config  # noqa: E402
from recsys.data import (GENRES_PER_ARTIST, artist_genre_ids, build_dataset,  # noqa: E402
                         build_genre_vocab, load_artist_genres)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="What the genre cutoff does to the vocab's artists.")
    ap.add_argument("--config", default=str(ROOT / "configs" / "p2_name.toml"))
    ap.add_argument("--genres", default=str(ROOT / "data" / "genres" / "musicbrainz_artists.jsonl"))
    ap.add_argument("--cutoffs", type=int, nargs="+", default=[5, 3])
    args = ap.parse_args()

    ds = build_dataset(load_config(args.config))
    artist_genres = load_artist_genres(args.genres)
    uris = ds.artist_uris
    missing = [u for u in uris if u not in artist_genres]
    if missing:
        sys.exit(f"{len(missing):,} vocab artists are missing from {args.genres}; fetch them first")

    song_artist = ds.song_features["artist"][0]
    as_target = np.bincount(song_artist[ds.Y_train], minlength=len(uris))
    as_input = np.bincount(song_artist[ds.X_train].ravel(), minlength=len(uris))
    has_genres = np.array([bool(artist_genres[u]) for u in uris])
    with open(args.genres, encoding="utf-8") as f:
        matched_uris = {r["artist_uri"] for r in map(json.loads, f) if r["mbids"]}
    matched = np.array([u in matched_uris for u in uris])

    def row(label, mask):
        return (f"  {label:44}{mask.sum():>7,}{mask.mean():>7.1%}"
                f"{as_target[mask].sum() / as_target.sum():>12.1%}{as_input[mask].sum() / as_input.sum():>12.1%}")

    print(f"{len(uris):,} vocab artists; {len({g for gs in artist_genres.values() for g in gs}):,} distinct genres "
          f"before any cutoff; up to {GENRES_PER_ARTIST} kept per artist\n")
    for cutoff in args.cutoffs:
        vocab = build_genre_vocab(artist_genres, uris, cutoff)
        ids = artist_genre_ids(artist_genres, uris, {g: i + 1 for i, g in enumerate(vocab)}, GENRES_PER_ARTIST)
        kept = (ids > 0).sum(axis=1)
        print(f"=== cutoff {cutoff}: genres listed for >= {cutoff} vocab artists -> {len(vocab):,} genres ===")
        print(f"  {'artists':44}{'count':>7}{'share':>7}{'of targets':>12}{'of inputs':>12}")
        print(row("never matched on MusicBrainz", ~matched))
        print(row("matched, but no genres on MusicBrainz", matched & ~has_genres))
        print(row("had genres, lost ALL to the cutoff", has_genres & (kept == 0)))
        print(row("kept at least one genre", kept > 0))
        dist = Counter(kept[kept > 0])
        print(f"  genres kept per artist (of those with any): "
              + ", ".join(f"{k}: {dist[k]:,}" for k in range(1, GENRES_PER_ARTIST + 1))
              + f" | median {int(np.median(kept[kept > 0]))}")
        before_cap = np.array([sum(g in set(vocab) for g in artist_genres[u]) for u in uris])
        print(f"  artists with more than {GENRES_PER_ARTIST} vocab genres (cap trims them): "
              f"{(before_cap > GENRES_PER_ARTIST).sum():,}")
        lost = [i for i in np.argsort(-as_target) if has_genres[i] and kept[i] == 0][:5]
        print("  biggest artists that lost all genres: " + "; ".join(
            f"{ds.artist_names[i]} ({as_target[i]} targets: {', '.join(artist_genres[uris[i]][:3])})" for i in lost))
        print(f"  most widely used genres: {', '.join(vocab[:12])}\n")
