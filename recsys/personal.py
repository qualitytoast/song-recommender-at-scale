"""Personalization A, the data side: one listener's Spotify Extended Streaming History, matched to a serving
bundle's catalog and cut into sessions with the ListenBrainz rules (recsys.lb_sessions).

    python -m recsys.personal plays --export ~/Downloads/my_spotify_data.zip \\
        --bundle bundles/<bundle> --out data/personal

Reads only the export's Streaming_History_Audio_*.json files, and from each play only its time, ms played,
Spotify track URI, track and artist names, and skipped flag (the files also hold IP addresses and devices,
which are never read into anything). Writes data/personal/<bundle version>/plays.npz (ignored by git: it is
the listener's history) and prints only totals.

- A play counts as a listen at LISTENED_MS or more (Spotify's own threshold for a stream; the ListenBrainz
  store dropped listens followed by the next within 30 s the same way); "skipped" also covers Spotify's flag.
- Songs match the catalog by Spotify URI, else by cleaned artist + title (recsys.listenbrainz.song_key, the
  key that made ListenBrainz songs): song ID -1 if neither finds it.
- Sessions follow the ListenBrainz rules on the listens (catalog songs or not, so a missing song doesn't
  join two sessions): a new one after 30 min without a listen, back-to-back repeats dropped, 4+ songs.
"""
import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from recsys.lb_sessions import sessions
from recsys.listenbrainz import song_key

LISTENED_MS = 30_000


def read_export(path):
    """The export's music plays in time order: dict of arrays seconds (Unix time), ms_played, skipped, and lists
    uri, track, artist. Plays without a track URI (podcasts, audiobooks) are left out."""
    plays = []
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            if Path(name).name.startswith("Streaming_History_Audio_") and name.endswith(".json"):
                for r in json.loads(z.read(name)):
                    if r.get("spotify_track_uri"):
                        plays.append((int(datetime.fromisoformat(r["ts"].replace("Z", "+00:00")).timestamp()),
                                      int(r.get("ms_played") or 0), bool(r.get("skipped")), r["spotify_track_uri"],
                                      r.get("master_metadata_track_name") or "",
                                      r.get("master_metadata_album_artist_name") or ""))
    plays.sort(key=lambda p: p[0])
    seconds, ms, skipped, uri, track, artist = zip(*plays) if plays else ((),) * 6
    return {"seconds": np.array(seconds, dtype=np.int64), "ms_played": np.array(ms, dtype=np.int64),
            "skipped": np.array(skipped, dtype=bool), "uri": list(uri), "track": list(track), "artist": list(artist)}


def catalog_lookup(bundle):
    """(song ID by URI, song ID by song_key) for a serving bundle's catalog."""
    bundle = Path(bundle)
    songs = json.loads((bundle / "songs.json").read_text())
    artist = np.load(bundle / "song_artist.npy")
    by_uri = {u: i for i, u in enumerate(songs["uris"])}
    by_key = {}
    for i, (track, a) in enumerate(zip(songs["tracks"], artist)):
        by_key.setdefault(song_key(songs["artist_names"][a], track), i)
    return by_uri, by_key


def match(plays, by_uri, by_key):
    """(song IDs, how each matched: 1 URI, 2 artist + title, 0 not found)."""
    ids, how = np.full(len(plays["uri"]), -1, dtype=np.int64), np.zeros(len(plays["uri"]), dtype=np.int8)
    for k, (uri, track, artist) in enumerate(zip(plays["uri"], plays["track"], plays["artist"])):
        if uri in by_uri:
            ids[k], how[k] = by_uri[uri], 1
        elif (j := by_key.get(song_key(artist, track))) is not None:
            ids[k], how[k] = j, 2
    return ids, how


def listening_sessions(plays):
    """(listened, kept, session) per play: listened = LISTENED_MS or more; kept = a listen that stays under the
    ListenBrainz session rules (not a back-to-back repeat, in a session of 4+ songs); session numbers per play
    (meaningful where kept)."""
    listened = plays["ms_played"] >= LISTENED_MS
    idx = np.flatnonzero(listened)
    raw = np.unique(np.array(plays["uri"], dtype=object)[idx], return_inverse=True)[1] if len(idx) else idx
    kept_l, session_l = sessions(np.zeros(len(idx), dtype=np.int64), plays["seconds"][idx], raw)
    kept, session = np.zeros(len(listened), dtype=bool), np.full(len(listened), -1, dtype=np.int64)
    kept[idx], session[idx] = kept_l, session_l
    return listened, kept, session


def build_plays(export, bundle, out):
    """Read, match and sessionize; write <out>/<bundle version>/plays.npz; return the totals."""
    version = json.loads((Path(bundle) / "manifest.json").read_text())["version"]
    plays = read_export(export)
    ids, how = match(plays, *catalog_lookup(bundle))
    listened, kept, session = listening_sessions(plays)
    folder = Path(out) / version
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / "plays.npz", seconds=plays["seconds"], ms_played=plays["ms_played"], skipped=plays["skipped"],
             song=ids, matched_by=how, listened=listened, kept=kept, session=session)
    known = listened & (ids >= 0)
    return {"bundle": version, "plays": int(len(ids)), "listens": int(listened.sum()),
            "skipped_flag": round(float(plays["skipped"].mean()), 3) if len(ids) else None,
            "listens_matched": round(float(known.sum() / max(1, listened.sum())), 3),
            "matched_by_uri": int(((how == 1) & listened).sum()), "matched_by_name": int(((how == 2) & listened).sum()),
            "distinct_songs_listened": len(set(np.array(plays["uri"], dtype=object)[listened])),
            "distinct_catalog_songs": int(len(np.unique(ids[known]))),
            "sessions": int(len(np.unique(session[kept]))), "session_listens": int(kept.sum()),
            "session_listens_matched": int((kept & (ids >= 0)).sum()),
            "first": day(plays["seconds"][0]) if len(ids) else None,
            "last": day(plays["seconds"][-1]) if len(ids) else None}


def day(seconds):
    return datetime.fromtimestamp(int(seconds), timezone.utc).date().isoformat()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("plays", help="read, match and sessionize the export")
    p.add_argument("--export", required=True, help="the Extended Streaming History zip")
    p.add_argument("--bundle", required=True)
    p.add_argument("--out", default="data/personal")
    args = ap.parse_args()
    print(json.dumps(build_plays(args.export, args.bundle, args.out), indent=2))
