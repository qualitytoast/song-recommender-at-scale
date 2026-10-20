"""Typo-tolerant search for the find-songs menu, by Meilisearch: a search engine in its own container
(docker-compose.yml), reached only by the front desks.

    # once per bundle (BUNDLE): send the bundle's songs to Meilisearch (index songs-<version>)
    docker compose run --rm search-index
    # after switching to a new bundle: also delete the other bundles' indexes
    docker compose run --rm search-index python -m recsys.song_search --bundle /bundle \\
        --url http://meilisearch:7700 --drop-others

Each song is one document: its ID (the bundle's song ID), its name (track and artist names), its album name,
and popularity (the bundle's count: listens or playlists). Meilisearch matches the typed words in any order
across name and album, the last word as the start of a word, with typos (one in words of 4+ letters, two in 8+),
and orders the matches by its ranking rules (SETTINGS): more of the words matched, fewer typos, a match in the
name before an album-only one, then the most played (then the words closer together, earlier in the name, and
exact rather than typo or prefix matches). Each bundle version has its own index, so a front desk only ever
gets IDs of its own bundle's songs.

SongSearch is the front desk's side. It must never make the menu fail: every search has a short timeout
(250 ms; warm, Meilisearch answers the server's searches in 9 ms, 70 ms at worst; 157 ms once right after a
restart), any error or unusable answer returns None, and the caller then uses the bundle's own name search
(recommender.Catalog._find_songs: names starting with the text, no typos);
after an error Meilisearch is skipped for `cooldown` seconds, so a down or slow Meilisearch costs one timeout,
not one per keystroke.
"""
import argparse
import itertools
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np

SETTINGS = {
    "searchableAttributes": ["name", "album"],  # a match in the name (track and artist) before an album-only one
    "displayedAttributes": ["id"],              # a search returns IDs only; the front desk has the rest
    # Popularity right after "all the words, fewest typos, name before album": with Meilisearch's own order
    # (popularity last) obscure exact titles ("A") and YouTube-style copies ("Taylor Swift - ... (Official
    # Video)") came first; with popularity before the field, album-only matches crowded out titles ("love").
    "rankingRules": ["words", "typo", "attributeRank", "popularity:desc", "proximity", "wordPosition", "exactness"],
    "typoTolerance": {"minWordSizeForTypos": {"oneTypo": 4, "twoTypos": 8}},  # Meilisearch's own: 5 and 9
}
BATCH = 100_000  # documents per upload (~10 MB; Meilisearch takes up to 100 MB per request)


def index_name(version):
    return f"songs-{version}"


def call(method, url, body=None, key=None, timeout=10.0, ndjson=False):
    """One request to Meilisearch: the answer as JSON (None if empty). body: a JSON value, or bytes (ndjson)."""
    headers = {"content-type": "application/x-ndjson" if ndjson else "application/json"}
    if key:
        headers["authorization"] = f"Bearer {key}"
    data = body if ndjson else None if body is None else json.dumps(body).encode()
    with urllib.request.urlopen(urllib.request.Request(url, data, headers, method=method), timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


def documents(bundle):
    """The bundle's songs as search documents, one at a time: {id, name ("track artist"), album, popularity}."""
    bundle = Path(bundle)
    songs = json.loads((bundle / "songs.json").read_text())
    albums = json.loads((bundle / "song_albums.json").read_text())
    artist = np.load(bundle / "song_artist.npy")
    popularity = np.load(bundle / "song_playlists.npy")
    for i, track in enumerate(songs["tracks"]):
        yield {"id": i, "name": f"{track} {songs['artist_names'][artist[i]]}", "album": albums[i],
               "popularity": int(popularity[i])}


def wait(url, key, task, poll=0.5, limit=3600.0):
    """Wait for a Meilisearch task (an accepted change, done in the background) to finish; raise if it failed."""
    end = time.monotonic() + limit
    while time.monotonic() < end:
        info = call("GET", f"{url}/tasks/{task['taskUid']}", key=key)
        if info["status"] == "succeeded":
            return info
        if info["status"] in ("failed", "canceled"):
            raise RuntimeError(f"Meilisearch task {info['taskUid']} {info['status']}: {info.get('error')}")
        time.sleep(poll)
    raise TimeoutError(f"Meilisearch task {task['taskUid']} not done after {limit:.0f} s")


def build_index(bundle, url, key=None, drop_others=False, report=print):
    """Index the bundle's songs as songs-<version> (replacing an index of that name); with drop_others, delete
    every other songs-* index. Returns the index name."""
    url = url.rstrip("/")
    version = json.loads((Path(bundle) / "manifest.json").read_text())["version"]
    name, began = index_name(version), time.perf_counter()
    existing = {i["uid"] for i in call("GET", f"{url}/indexes?limit=1000", key=key)["results"]}
    if name in existing:
        wait(url, key, call("DELETE", f"{url}/indexes/{name}", key=key))
    wait(url, key, call("POST", f"{url}/indexes", {"uid": name, "primaryKey": "id"}, key=key))
    wait(url, key, call("PATCH", f"{url}/indexes/{name}/settings", SETTINGS, key=key))
    docs, sent = documents(bundle), 0
    while batch := list(itertools.islice(docs, BATCH)):
        lines = "\n".join(json.dumps(d, ensure_ascii=False) for d in batch).encode()
        wait(url, key, call("POST", f"{url}/indexes/{name}/documents", lines, key=key, timeout=60.0, ndjson=True))
        sent += len(batch)
        report(f"{sent:,} songs indexed ({time.perf_counter() - began:.0f} s)")
    if drop_others:
        for other in sorted(existing - {name}):
            if other.startswith("songs-"):
                wait(url, key, call("DELETE", f"{url}/indexes/{other}", key=key))
                report(f"deleted {other}")
    return name


class SongSearch:
    """The menu's search for one bundle version (see the module docstring): search() returns song IDs, best
    first, or None when Meilisearch can't answer (the caller falls back)."""

    def __init__(self, url, version, songs, key=None, timeout=0.25, cooldown=1.0):
        self.url, self.index, self.songs, self.key = url.rstrip("/"), index_name(version), songs, key
        self.timeout, self.cooldown = timeout, cooldown
        self.skip_until = 0.0
        self.stats = {"answered": 0, "errors": 0, "skipped": 0}
        self.last_error = None

    def search(self, query, limit):
        if time.monotonic() < self.skip_until:
            self.stats["skipped"] += 1
            return None
        try:
            answer = call("POST", f"{self.url}/indexes/{self.index}/search",
                          {"q": query, "limit": limit, "attributesToRetrieve": ["id"]}, self.key, self.timeout)
            ids = [int(hit["id"]) for hit in answer["hits"]]
            if any(not 0 <= j < self.songs for j in ids):
                raise ValueError("a song ID outside this bundle")
        except (OSError, ValueError, KeyError, TypeError) as e:  # urllib's errors and timeouts are OSErrors
            self.stats["errors"] += 1
            self.last_error = f"{type(e).__name__}: {e}"
            self.skip_until = time.monotonic() + self.cooldown
            return None
        self.stats["answered"] += 1
        return ids

    def report(self):
        return {"index": self.index, **self.stats, "last_error": self.last_error}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Index a bundle's songs in Meilisearch for the find-songs menu.")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:7700")
    ap.add_argument("--drop-others", action="store_true", help="also delete other bundles' song indexes")
    args = ap.parse_args()
    print(build_index(args.bundle, args.url, os.environ.get("MEILI_MASTER_KEY"), args.drop_others))
