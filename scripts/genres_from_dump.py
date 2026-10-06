"""Get MusicBrainz genres for the vocab's artists from MusicBrainz's JSON data dump.

The alternative to fetch_genres.py for large artist lists: no rate limit, and the
same ~20 minutes whether we need 7,500 artists or 300,000. Writes the same
format as fetch_genres.py, one JSON line per vocab artist:
    {"artist_uri": ..., "artist_name": ..., "mbids": [...], "genres": [{"name": ..., "count": ...}]}
plus <out>.meta.json recording which dump was used.

    python scripts/genres_from_dump.py                        # stream the latest dump
    python scripts/genres_from_dump.py --file artist.tar.xz   # use a downloaded copy

How it works: artist.tar.xz (~1.7 GB compressed, ~18 GB of JSON) holds one
artist per line, in the same format the API returns, including the artist's
URL relationships (its Spotify page) and its genres with vote counts. The file
is read as a stream: bytes are decompressed as they arrive, each line is checked
for a Spotify artist link we need, and everything else is discarded, so nothing
large is stored. Several MusicBrainz artists can link to one Spotify page; their
genres are combined, as fetch_genres.py does.

A streamed run can't resume partway: if it is interrupted it starts over. The
output is written to a temporary file and only renamed into place once the whole
dump has been read, so an interrupted run never leaves a partial file behind.
For a resumable download, fetch the file first (curl -C - resumes), then --file:
    curl -C - -o artist.tar.xz https://data.metabrainz.org/pub/musicbrainz/data/json-dumps/LATEST_DATE/artist.tar.xz

License: MusicBrainz genre and tag data is CC BY-NC-SA; credit MusicBrainz.
"""
import argparse
import json
import re
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys.config import load_config  # noqa: E402
from recsys.data import build_dataset  # noqa: E402
from scripts.fetch_genres import SSL_CONTEXT, USER_AGENT, merge_genres  # noqa: E402

DUMPS = "https://data.metabrainz.org/pub/musicbrainz/data/json-dumps/"
OUT = ROOT / "data" / "genres" / "musicbrainz_artists_from_dump.jsonl"
ARTIST_FILE = "mbdump/artist"  # the member of artist.tar.xz holding the records
SPOTIFY_ARTIST = re.compile(r"open\.spotify\.com/artist/([0-9A-Za-z]+)")


def latest_dump():
    """Name of the newest dump folder, e.g. "20261003-001001"."""
    request = urllib.request.Request(DUMPS + "LATEST", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30, context=SSL_CONTEXT) as response:
        return response.read().decode().strip()


class Progress:
    """Wraps the compressed input to print progress every ~100 MB read."""

    def __init__(self, raw, total):
        self.raw, self.total, self.read_so_far, self.next_report = raw, total, 0, 100e6
        self.start = time.monotonic()

    def read(self, n=-1):
        chunk = self.raw.read(n)
        self.read_so_far += len(chunk)
        if self.read_so_far >= self.next_report:
            self.next_report += 100e6
            of = f" of {self.total / 1e6:,.0f}" if self.total else ""
            print(f"  {self.read_so_far / 1e6:,.0f}{of} MB read ({time.monotonic() - self.start:.0f}s)", flush=True)
        return chunk


def scan(lines, wanted):
    """{spotify id: {"mbids": [...], "genres": [...]}} for the wanted Spotify IDs.

    lines: the dump's artist records (bytes, one JSON object each). Only lines
    mentioning a Spotify artist page are parsed. Genres with no net votes are
    dropped, matching what the API's inc=genres returns.
    """
    found = {}
    for line in lines:
        if b"open.spotify.com/artist/" not in line:
            continue
        record = json.loads(line)
        links = {m.group(1) for rel in record.get("relations", [])
                 for m in [SPOTIFY_ARTIST.search((rel.get("url") or {}).get("resource", ""))] if m}
        genres = [g for g in record.get("genres", []) if g["count"] > 0]
        for spotify_id in links & wanted:
            entry = found.setdefault(spotify_id, {"mbids": [], "genre_lists": []})
            entry["mbids"].append(record["id"])
            entry["genre_lists"].append(genres)
    return {sid: {"mbids": e["mbids"], "genres": merge_genres(e["genre_lists"])} for sid, e in found.items()}


def artist_lines(fileobj):
    """The artist records from a (streamed) artist.tar.xz, one line at a time."""
    with tarfile.open(fileobj=fileobj, mode="r|xz") as tar:  # "|" = read as a stream, in order
        for member in tar:
            if member.name == ARTIST_FILE:
                yield from tar.extractfile(member)
                return
    raise ValueError(f"{ARTIST_FILE} not found in the archive")


def write_output(artists, found, out, meta):
    """One line per artist (unmatched ones included), written atomically, plus out.meta.json."""
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for uri, name in artists:
            entry = found.get(uri.rsplit(":", 1)[-1], {"mbids": [], "genres": []})
            f.write(json.dumps({"artist_uri": uri, "artist_name": name, **entry}) + "\n")
    tmp.replace(out)  # only now does the finished file appear under its real name
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MusicBrainz genres for the vocab's artists, from the JSON dump.")
    ap.add_argument("--config", default=str(ROOT / "configs" / "p2_name.toml"))
    ap.add_argument("--file", help="a downloaded artist.tar.xz (default: stream the latest dump)")
    ap.add_argument("--dump", help="dump folder to stream, e.g. 20261003-001001 (default: latest)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    ds = build_dataset(load_config(args.config))
    artists = list(zip(ds.artist_uris, ds.artist_names))
    wanted = {uri.rsplit(":", 1)[-1] for uri, _ in artists}
    start = time.monotonic()
    if args.file:
        source = str(Path(args.file).resolve())
        print(f"reading {source} for {len(wanted):,} artists")
        with open(args.file, "rb") as raw:
            found = scan(artist_lines(Progress(raw, Path(args.file).stat().st_size)), wanted)
    else:
        dump = args.dump or latest_dump()
        source = f"{DUMPS}{dump}/artist.tar.xz"
        print(f"streaming {source} for {len(wanted):,} artists")
        request = urllib.request.Request(source, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60, context=SSL_CONTEXT) as response:
            total = int(response.headers.get("Content-Length") or 0)
            found = scan(artist_lines(Progress(response, total)), wanted)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_output(artists, found, out, {"source": source, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                                       "seconds": round(time.monotonic() - start), "artists": len(artists)})
    with_genres = sum(1 for e in found.values() if e["genres"])
    print(f"done in {(time.monotonic() - start) / 60:.1f} min | {len(artists):,} artists | "
          f"matched {len(found):,} | with genres {with_genres:,} | wrote {out}")
