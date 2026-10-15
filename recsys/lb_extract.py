"""Step 1 of the ListenBrainz plan: stream a full listen dump once and keep the listens of a time window.

The full dump is one ~253 GB .tar.zst file, one file per month of listening, oldest first. This reads it as it
downloads (never stored): months outside the window are skipped unread, and the window's months are cut into
blocks that a pool of worker processes turns into compact rows. Output, in --out:

  per listen, one binary file per column (numpy dtypes in extract.json; open with load_extract):
    user.bin (int32), timestamp.bin (int64, seconds), song.bin (int32: line of songs.jsonl),
    release.bin (int32: row of releases.json, -1 none), spotify.bin (int32: row of spotify_tracks.json, -1 none),
    duration_ms.bin (int32, 0 unknown)
  songs.jsonl: per song (recsys.listenbrainz.song_key), one line: [key, artist, title] of its first listen
  song_artist_mbid.bin, song_spotify_artist.bin: per song (int32), the first MusicBrainz / Spotify artist ID seen
              for it, as a row of artist_mbids.json / spotify_artists.json (-1 none)
  releases.json, spotify_tracks.json: album names, Spotify track IDs
  extract.json: what was read and kept, the window, the checksum result; written last (a folder without it is an
              unfinished extract)

A dropped connection is resumed where it stopped (HTTP Range request); every byte is hashed and checked against
the server's SHA-256 at the end.

    python -m recsys.lb_extract --source https://data.metabrainz.org/pub/musicbrainz/listenbrainz/fullexport/\
listenbrainz-dump-2692-20261001-000003-full/listenbrainz-listens-dump-2692-20261001-000003-full.tar.zst \
        --start 2025-10-01 --end 2026-10-01 --out data/lb_extract
"""
import argparse
import array
import calendar
import hashlib
import io
import json
import multiprocessing
import os
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import urllib3

from recsys.listenbrainz import listen_files, parse_listen, song_key

MARKER = "extract.json"
COLUMNS = {"user": "int32", "timestamp": "int64", "song": "int32", "release": "int32", "spotify": "int32",
           "duration_ms": "int32"}
BLOCK = 32 << 20  # bytes of listens per worker task
NETWORK_ERRORS = (OSError, requests.RequestException, urllib3.exceptions.HTTPError)


def log(message):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


class ResumableHTTP(io.RawIOBase):
    """A download read like a file, front to back. If the connection drops or stalls, it reconnects and asks the
    server for the rest (an HTTP Range request: "bytes from N on"), so whoever reads it never notices. It gives up
    after `retries` failed attempts in a row. Every byte goes through SHA-256, to check against the server's."""

    def __init__(self, url, retries=30, timeout=60, wait=5):
        self.url, self.retries, self.timeout, self.wait = url, retries, timeout, wait
        self.pos, self.size, self.sha256, self.reconnects = 0, None, hashlib.sha256(), 0
        self.response = None
        self._connect()

    def _connect(self):
        headers = {"Range": f"bytes={self.pos}-"} if self.pos else {}
        response = requests.get(self.url, headers=headers, stream=True, timeout=self.timeout)
        response.raise_for_status()
        if self.pos and response.status_code != 206:
            response.close()
            raise OSError(f"the server didn't resume from byte {self.pos} (status {response.status_code})")
        if self.size is None:
            self.size = int(response.headers.get("Content-Length", 0)) or None
        self.response = response

    def readable(self):
        return True

    def readinto(self, buffer):
        failures = 0
        while True:
            try:
                n = self.response.raw.readinto(buffer)
                if n == 0 and self.size is not None and self.pos < self.size:
                    raise OSError(f"the connection ended at byte {self.pos:,} of {self.size:,}")
                self.sha256.update(memoryview(buffer)[:n])
                self.pos += n
                return n
            except NETWORK_ERRORS as e:
                failures += 1
                if failures > self.retries:
                    raise
                wait = min(self.wait * 2 ** (failures - 1), 300)
                log(f"download interrupted at byte {self.pos:,} ({type(e).__name__}: {e}); "
                    f"resuming in {wait:.0f}s (attempt {failures} of {self.retries})")
                time.sleep(wait)
                self.response.close()
                try:
                    self._connect()
                    self.reconnects += 1
                except NETWORK_ERRORS as reconnect_error:
                    log(f"reconnect failed ({type(reconnect_error).__name__}: {reconnect_error})")

    def close(self):
        if self.response is not None:
            self.response.close()
        super().close()


class LocalFile(io.RawIOBase):
    """A local dump, hashed as it's read like ResumableHTTP (for tests and small dumps)."""

    def __init__(self, path):
        self.file, self.size, self.pos = open(path, "rb"), os.path.getsize(path), 0
        self.sha256, self.reconnects = hashlib.sha256(), 0

    def readable(self):
        return True

    def readinto(self, buffer):
        n = self.file.readinto(buffer)
        self.sha256.update(memoryview(buffer)[:n])
        self.pos += n
        return n

    def close(self):
        self.file.close()
        super().close()


def open_source(source):
    return ResumableHTTP(source) if source.startswith(("http://", "https://")) else LocalFile(source)


def expected_sha256(source):
    """The server's (or a local .sha256 file's) checksum of the dump, or None if there isn't one."""
    try:
        if source.startswith(("http://", "https://")):
            response = requests.get(source + ".sha256", timeout=60)
            text = response.text if response.ok else ""
        else:
            path = Path(source + ".sha256")
            text = path.read_text() if path.exists() else ""
    except NETWORK_ERRORS:
        return None
    digest = text.split()[0].lower() if text.split() else ""
    return digest if len(digest) == 64 else None


def month_in_window(year, month, start, end):
    """True if any second of the month falls in [start, end) (Unix seconds)."""
    first = calendar.timegm((year, month, 1, 0, 0, 0))
    after = calendar.timegm((year + month // 12, month % 12 + 1, 1, 0, 0, 0))
    return first < end and after > start


def blocks(lines_file, size=BLOCK):
    """The file's bytes in blocks of about `size` that end at the end of a line."""
    while True:
        block = lines_file.read(size)
        if not block:
            return
        if not block.endswith(b"\n"):
            block += lines_file.readline()
        yield block


def parse_block(block, start, end):
    """Worker: a block of listen lines as compact rows. Strings are numbered within the block (local tables);
    the main process turns those numbers into the extract's own."""
    users, times, songs = array.array("i"), array.array("q"), array.array("i")
    releases, spotify, durations = array.array("i"), array.array("i"), array.array("i")
    song_index, song_rows, release_index, spotify_index = {}, [], {}, {}
    counts = Counter()
    for line in block.split(b"\n"):
        if not line.strip():
            continue
        try:
            listen = parse_listen(line)
        except (ValueError, KeyError, TypeError, AttributeError):
            counts["unreadable"] += 1
            continue
        if not start <= listen.timestamp < end:
            counts["outside the window"] += 1
            continue
        if not listen.artist.strip() or not listen.title.strip():
            counts["no artist or title"] += 1
            continue
        key = song_key(listen.first_artist, listen.title)
        if not key.split("|")[0] or not key.split("|")[1]:  # nothing left after cleaning (punctuation only)
            counts["no artist or title"] += 1
            continue
        s = song_index.get(key)
        if s is None:
            s = song_index[key] = len(song_rows)
            song_rows.append([key, listen.artist, listen.title, listen.artist_mbid, listen.spotify_artist_id])
        else:
            row = song_rows[s]
            row[3] = row[3] or listen.artist_mbid
            row[4] = row[4] or listen.spotify_artist_id
        users.append(listen.user_id)
        times.append(listen.timestamp)
        songs.append(s)
        releases.append(release_index.setdefault(listen.release, len(release_index)) if listen.release else -1)
        spotify.append(spotify_index.setdefault(listen.spotify_id, len(spotify_index)) if listen.spotify_id else -1)
        durations.append(listen.duration_ms)
        counts["kept"] += 1
    return {"user": users.tobytes(), "timestamp": times.tobytes(), "song": songs.tobytes(),
            "release": releases.tobytes(), "spotify": spotify.tobytes(), "duration_ms": durations.tobytes(),
            "songs": song_rows, "releases": list(release_index), "spotify_tracks": list(spotify_index),
            "counts": counts}


class Tables:
    """The extract's numbering of songs, albums, Spotify tracks and artist IDs, built up block by block, and the
    files it writes. Only lookups stay in memory: a new song's names go straight to songs.jsonl (line = number),
    so tens of millions of songs fit."""

    def __init__(self, out):
        self.out = Path(out)
        self.song_index, self.release_index, self.spotify_index = {}, {}, {}
        self.artist_mbid_index, self.spotify_artist_index = {}, {}
        self.song_artist_mbid, self.song_spotify_artist = array.array("i"), array.array("i")  # per song, -1 none
        self.files = {name: open(self.out / f"{name}.bin", "wb") for name in COLUMNS}
        self.songs_file = open(self.out / "songs.jsonl", "w", encoding="utf-8")
        self.counts = Counter()

    @staticmethod
    def code(index, name):
        return index.setdefault(name, len(index)) if name else -1

    def song_codes(self, rows):
        """The extract's number for each of a block's songs ([key, artist, title, artist MBID, Spotify artist]);
        a song's artist IDs are the first ones seen for it."""
        codes = np.empty(len(rows), dtype=np.int32)
        for i, (key, artist, title, mbid, spotify_artist) in enumerate(rows):
            s = self.song_index.get(key)
            if s is None:
                s = self.song_index[key] = len(self.song_artist_mbid)
                self.songs_file.write(json.dumps([key, artist, title], ensure_ascii=False) + "\n")
                self.song_artist_mbid.append(self.code(self.artist_mbid_index, mbid))
                self.song_spotify_artist.append(self.code(self.spotify_artist_index, spotify_artist))
            else:
                if self.song_artist_mbid[s] < 0 and mbid:
                    self.song_artist_mbid[s] = self.code(self.artist_mbid_index, mbid)
                if self.song_spotify_artist[s] < 0 and spotify_artist:
                    self.song_spotify_artist[s] = self.code(self.spotify_artist_index, spotify_artist)
            codes[i] = s
        return codes

    @staticmethod
    def codes(index, names):
        return np.array([index.setdefault(name, len(index)) for name in names], dtype=np.int32)

    @staticmethod
    def recode(local, codes):
        """Block numbers to the extract's (-1 stays -1)."""
        if not len(codes):
            return local
        return np.where(local >= 0, codes[np.maximum(local, 0)], -1).astype(np.int32)

    def add(self, result):
        """Write one block's rows with the extract's numbering."""
        self.counts.update(result["counts"])
        local = {name: np.frombuffer(result[name], dtype=dtype) for name, dtype in COLUMNS.items()}
        if not len(local["user"]):
            return
        local["song"] = self.song_codes(result["songs"])[local["song"]]
        local["release"] = self.recode(local["release"], self.codes(self.release_index, result["releases"]))
        local["spotify"] = self.recode(local["spotify"], self.codes(self.spotify_index, result["spotify_tracks"]))
        for name, values in local.items():
            values.astype(COLUMNS[name], copy=False).tofile(self.files[name])

    def finish(self):
        """Close the column files and write the tables."""
        for f in self.files.values():
            f.close()
        self.songs_file.close()
        np.frombuffer(self.song_artist_mbid, dtype=np.int32).tofile(self.out / "song_artist_mbid.bin")
        np.frombuffer(self.song_spotify_artist, dtype=np.int32).tofile(self.out / "song_spotify_artist.bin")
        for name, index in (("releases", self.release_index), ("spotify_tracks", self.spotify_index),
                            ("artist_mbids", self.artist_mbid_index), ("spotify_artists", self.spotify_artist_index)):
            (self.out / f"{name}.json").write_text(json.dumps(list(index), ensure_ascii=False))


def extract(source, out, start, end, workers, block=BLOCK, progress_every=60):
    """Stream `source` (a URL or a local .tar.zst) and write the window's listens to `out`. Returns the
    extract.json contents."""
    out = Path(out)
    if (out / MARKER).exists():
        raise FileExistsError(f"{out} already holds a finished extract")
    out.mkdir(parents=True, exist_ok=True)
    began = time.time()
    expected = expected_sha256(source)
    stream = open_source(source)
    tables = Tables(out)
    months_read, months_skipped = [], 0
    last_report = time.monotonic()

    def report(month):
        nonlocal last_report
        if time.monotonic() - last_report < progress_every:
            return
        last_report = time.monotonic()
        rate = stream.pos / max(time.time() - began, 1e-9)
        left = f", ~{(stream.size - stream.pos) / rate / 3600:.1f} h left" if stream.size and rate else ""
        total = f" of {stream.size / 1e9:,.1f}" if stream.size else ""
        log(f"{stream.pos / 1e9:,.1f}{total} GB read ({rate / 1e6:.1f} MB/s{left}) | at {month} | "
            f"{tables.counts['kept']:,} listens kept, {len(tables.song_index):,} songs")

    context = multiprocessing.get_context("spawn")
    with context.Pool(workers) as pool:
        pending = deque()
        for year, month, lines in listen_files(stream):
            label = f"{year}-{month:02d}"
            if not month_in_window(year, month, start, end):
                months_skipped += 1
                report(f"{label} (skipped)")
                continue
            months_read.append(label)
            log(f"reading {label}")
            for chunk in blocks(lines, block):
                pending.append(pool.apply_async(parse_block, (chunk, start, end)))
                while len(pending) >= 2 * workers:  # don't read ahead of the workers
                    tables.add(pending.popleft().get())
                report(label)
        while pending:
            tables.add(pending.popleft().get())
    tables.finish()
    stream.close()
    digest = stream.sha256.hexdigest()
    meta = {"source": source, "window": {"start": iso(start), "end": iso(end)},
            "listens": tables.counts["kept"], "songs": len(tables.song_index), "releases": len(tables.release_index),
            "spotify_tracks": len(tables.spotify_index), "artist_mbids": len(tables.artist_mbid_index),
            "spotify_artists": len(tables.spotify_artist_index), "users": None,
            "skipped": {k: v for k, v in tables.counts.items() if k != "kept"},
            "months_read": months_read, "months_skipped": months_skipped,
            "bytes_read": stream.pos, "reconnects": stream.reconnects,
            "sha256": digest, "sha256_expected": expected,
            "sha256_ok": None if expected is None else digest == expected,
            "columns": COLUMNS, "seconds": round(time.time() - began), "finished": iso(time.time())}
    if meta["listens"]:
        meta["users"] = int(len(np.unique(np.fromfile(out / "user.bin", dtype=np.int32))))
    (out / MARKER).write_text(json.dumps(meta, indent=2))  # last: marks the extract as finished
    return meta


def iso(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


@dataclass
class Extract:
    meta: dict
    user: np.ndarray        # memory-mapped columns, one row per listen
    timestamp: np.ndarray
    song: np.ndarray
    release: np.ndarray
    spotify: np.ndarray
    duration_ms: np.ndarray
    song_artist_mbid: np.ndarray     # per song: row of artist_mbids, -1 none
    song_spotify_artist: np.ndarray  # per song: row of spotify_artists, -1 none
    releases: list
    spotify_tracks: list
    artist_mbids: list
    spotify_artists: list
    folder: Path

    def songs(self):
        """Each song's [key, artist, title] (first listen's names), in song order, read from disk."""
        with open(self.folder / "songs.jsonl", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def load_extract(folder):
    folder = Path(folder)
    meta = json.loads((folder / MARKER).read_text())
    load = lambda name, dtype, n: np.memmap(folder / name, dtype=dtype, mode="r") if n else np.zeros(0, dtype=dtype)
    columns = {name: load(f"{name}.bin", dtype, meta["listens"]) for name, dtype in meta["columns"].items()}
    per_song = {name: load(f"{name}.bin", "int32", meta["songs"]) for name in ("song_artist_mbid", "song_spotify_artist")}
    read = lambda name: json.loads((folder / f"{name}.json").read_text())
    return Extract(meta=meta, **columns, **per_song, releases=read("releases"), spotify_tracks=read("spotify_tracks"),
                   artist_mbids=read("artist_mbids"), spotify_artists=read("spotify_artists"), folder=folder)


def utc_seconds(day):
    return calendar.timegm(datetime.strptime(day, "%Y-%m-%d").timetuple())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True, help="the dump's URL, or a local .tar.zst")
    ap.add_argument("--start", required=True, help="first day kept, YYYY-MM-DD (UTC)")
    ap.add_argument("--end", required=True, help="first day not kept, YYYY-MM-DD (UTC)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    start, end = utc_seconds(args.start), utc_seconds(args.end)
    log(f"extracting {args.start} to {args.end} from {args.source} with {args.workers} workers into {args.out}")
    meta = extract(args.source, args.out, start, end, args.workers)
    log(json.dumps({k: v for k, v in meta.items() if k != "columns"}, indent=2))
    if meta["sha256_ok"] is False:
        log("WARNING: the download's checksum doesn't match the server's; the extract may be damaged")


if __name__ == "__main__":
    main()
