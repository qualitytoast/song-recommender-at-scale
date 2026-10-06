"""Fetch MusicBrainz genres for every artist in the Phase 2 vocabulary.

This calls the MusicBrainz web API, so it is kept apart from training: training
only reads the file it writes.

    python scripts/fetch_genres.py --sample 100   # 100 random artists not yet done
    python scripts/fetch_genres.py                # every artist not yet done

Appends one JSON line per artist to data/genres/musicbrainz_artists.jsonl as
soon as that artist is finished, including artists with no match:
    {"artist_uri": ..., "artist_name": ..., "mbids": [...], "genres": [{"name": ..., "count": ...}]}
"mbids" is [] when no MusicBrainz artist links to that Spotify URL. "count" is
how many MusicBrainz users voted for that genre. On restart, artists already
in the file are skipped.

Requests, per batch of up to 100 artists:
  1. one URL lookup with every artist's Spotify URL (inc=artist-rels), which
     maps Spotify artists to MusicBrainz artist IDs (MBIDs); unmatched URLs are
     left out of the response
  2. one artist *search* for all matched MBIDs at once (arid:<id> OR arid:<id> ...)
Search returns every user tag, not just genres, so an artist's genres are its
tags that are on MusicBrainz's official genre list with positive votes. That
reproduces the per-artist lookup with inc=genres exactly (checked on 127
artists, 2026-10-06). The genre list itself (~2,200 names) is fetched once and
saved next to the output. An MBID missing from the search index falls back to
a per-artist lookup.

MusicBrainz allows one request per second per IP and answers 503 when that is
exceeded, so requests are spaced 1 second apart and 503s are retried with
increasing waits.
"""
import argparse
import http.client
import json
import random
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys.config import load_config  # noqa: E402
from recsys.data import build_dataset  # noqa: E402

API = "https://musicbrainz.org/ws/2/"
USER_AGENT = "song-recommender-at-scale/0.1 ( nathan.m.hung@gmail.com )"  # MusicBrainz requires contact info
OUT = ROOT / "data" / "genres" / "musicbrainz_artists.jsonl"
GENRE_LIST = ROOT / "data" / "genres" / "musicbrainz_genre_names.json"
BATCH = 100        # most Spotify URLs MusicBrainz accepts in one URL lookup
SEARCH_BATCH = 100  # most results one search returns
GAP_SECONDS = 1.0  # wait between requests
RETRIES = 6        # on 503 or a network error, wait 2, 4, 8, ... 64 seconds and try again

# python.org's macOS Python ships without trusted certificates; use the system's bundle.
SSL_CONTEXT = ssl.create_default_context(cafile="/etc/ssl/cert.pem")
_last_request_end = 0.0

# Connection problems worth retrying. URLError covers failures while connecting; a
# timeout or dropped connection while waiting for the reply (e.g. after the laptop
# wakes from sleep) raises one of the others instead.
NETWORK_ERRORS = (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException)


def get(path, params):
    """GET a MusicBrainz API path as JSON; None on 404. Spaced and retried as described above."""
    global _last_request_end
    url = API + path + "?" + urllib.parse.urlencode({**params, "fmt": "json"}, doseq=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(RETRIES + 1):
        time.sleep(max(0.0, _last_request_end + GAP_SECONDS - time.monotonic()))
        try:
            with urllib.request.urlopen(request, timeout=30, context=SSL_CONTEXT) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code != 503 or attempt == RETRIES:
                raise
            reason = "503 (rate limited)"
        except NETWORK_ERRORS as e:
            if attempt == RETRIES:
                raise
            reason = f"network error ({e!r})"
        finally:
            _last_request_end = time.monotonic()
        wait = 2 ** (attempt + 1)
        print(f"  {reason}, retrying in {wait}s", flush=True)
        time.sleep(wait)


def spotify_url(artist_uri):
    return "https://open.spotify.com/artist/" + artist_uri.rsplit(":", 1)[-1]


def mbids_by_spotify_id(url_lookup):
    """{spotify artist id: [MBIDs]} from a URL-lookup response. A URL can link to more than one artist."""
    out = {}
    for url in (url_lookup or {}).get("urls", []):
        spotify_id = url["resource"].rstrip("/").rsplit("/", 1)[-1]
        out[spotify_id] = [r["artist"]["id"] for r in url.get("relations", []) if "artist" in r]
    return out


def merge_genres(genre_lists):
    """Combine genres from several MusicBrainz artists, adding up votes; most votes first."""
    votes = {}
    for genres in genre_lists:
        for g in genres:
            votes[g["name"]] = votes.get(g["name"], 0) + g["count"]
    return [{"name": n, "count": c} for n, c in sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))]


def read_rows(path=None):
    """Every complete line of the output file (default OUT). A line cut off by the
    program being killed mid-write is skipped, so that artist is simply fetched again."""
    path = path or OUT  # looked up at call time, so tests can point OUT elsewhere
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if line.strip():
                print(f"  skipping an incomplete line in {path.name}; that artist will be fetched again")
    return rows


def official_genres():
    """Names of every MusicBrainz genre. Fetched once (100 per request), then read from GENRE_LIST."""
    if GENRE_LIST.exists():
        return set(json.loads(GENRE_LIST.read_text()))
    names, total = [], None
    while total is None or len(names) < total:
        page = get("genre/all", {"limit": 100, "offset": len(names)})
        total = page["genre-count"]
        names += [g["name"] for g in page["genres"]]
    GENRE_LIST.parent.mkdir(parents=True, exist_ok=True)
    GENRE_LIST.write_text(json.dumps(sorted(names)))
    return set(names)


def genres_from_tags(tags, genre_names):
    """An artist's genres: its tags that are official genre names with positive votes, most votes first."""
    return merge_genres([[t for t in tags if t["name"] in genre_names and t["count"] > 0]])


def search_genres(mbids, genre_names):
    """{mbid: genres} for up to SEARCH_BATCH MBIDs, in one search request."""
    found = get("artist", {"query": " OR ".join(f"arid:{m}" for m in mbids), "limit": SEARCH_BATCH})
    return {a["id"]: genres_from_tags(a.get("tags", []), genre_names) for a in (found or {}).get("artists", [])}


def genres_by_mbid(mbids, genre_names):
    """{mbid: genres} for any number of MBIDs: searched in groups; any the search
    index doesn't return are looked up one at a time with inc=genres."""
    out = {}
    for i in range(0, len(mbids), SEARCH_BATCH):
        out |= search_genres(mbids[i:i + SEARCH_BATCH], genre_names)
    for mbid in mbids:
        if mbid not in out:
            artist = get(f"artist/{mbid}", {"inc": "genres"})
            out[mbid] = merge_genres([artist.get("genres", [])]) if artist else []
    return out


def done_artists():
    return {r["artist_uri"] for r in read_rows()}


def open_for_append(path=None):
    """Open the output file (default OUT) for appending, first ending any cut-off
    last line, so the next line written starts on a line of its own."""
    path = path or OUT  # looked up at call time, so tests can point OUT elsewhere
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n")
    return open(path, "a", encoding="utf-8")


def fetch(artists, genre_names):
    """artists: [(uri, name)]. Appends one line per artist to OUT after each batch of BATCH."""
    with open_for_append() as out:
        for start in range(0, len(artists), BATCH):
            batch = artists[start:start + BATCH]
            matches = mbids_by_spotify_id(get("url", {"resource": [spotify_url(u) for u, _ in batch],
                                                      "inc": "artist-rels"}))
            batch_mbids = list(dict.fromkeys(m for ids in matches.values() for m in ids))
            genres = genres_by_mbid(batch_mbids, genre_names)
            for uri, name in batch:
                mbids = matches.get(uri.rsplit(":", 1)[-1], [])
                out.write(json.dumps({"artist_uri": uri, "artist_name": name, "mbids": mbids,
                                      "genres": merge_genres([genres[m] for m in mbids])}) + "\n")
            out.flush()
            print(f"{start + len(batch):,}/{len(artists):,} artists done", flush=True)


def report(uris):
    """Match and genre coverage for the given artists, from OUT."""
    rows = [r for r in read_rows() if r["artist_uri"] in uris]
    matched = [r for r in rows if r["mbids"]]
    with_genre = [r for r in matched if r["genres"]]
    counts = sorted(len(r["genres"]) for r in with_genre)
    print(f"\n{len(rows)} artists | matched by Spotify ID: {len(matched)} | "
          f"with at least one genre: {len(with_genre)}")
    if counts:
        print(f"genres per artist with any: min {counts[0]}, median {counts[len(counts) // 2]}, max {counts[-1]}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fetch MusicBrainz genres for the vocab's artists.")
    ap.add_argument("--config", default=str(ROOT / "configs" / "p2_name.toml"))
    ap.add_argument("--sample", type=int, help="only this many random artists not yet done")
    ap.add_argument("--sample-seed", type=int, default=0)
    args = ap.parse_args()

    ds = build_dataset(load_config(args.config))
    done = done_artists()
    todo = [(u, n) for u, n in zip(ds.artist_uris, ds.artist_names) if u not in done]
    if args.sample:
        todo = random.Random(args.sample_seed).sample(todo, min(args.sample, len(todo)))
    print(f"{len(ds.artist_uris):,} artists in the vocab, {len(done):,} already done, fetching {len(todo):,}")
    try:
        fetch(todo, official_genres())
    except KeyboardInterrupt:  # Ctrl-C: every finished artist is already saved
        print(f"\nStopped. {len(done_artists()):,} of {len(ds.artist_uris):,} artists saved; "
              f"run the same command again to continue.")
        sys.exit(130)
    report({u for u, _ in todo})
