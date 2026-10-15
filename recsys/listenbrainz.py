"""Reading ListenBrainz listen dumps, and the song key that decides which listens are the same song.

A dump is a tar archive (many files bundled into one) compressed with zstd. Inside, listens are split by
month, <dump name>/listens/<year>/<month>.listens, with one JSON listen per line:

  {"user_id": 192771, "user_name": "...", "timestamp": 1791935633, "recording_msid": "...",
   "track_metadata": {"artist_name": "...", "track_name": "...", "release_name": "...",
                      "additional_info": {"duration_ms": 351000, "spotify_id": "https://open.spotify.com/track/...",
                                          "artist_names": [...], "recording_mbid": "...", "artist_mbids": [...]}}}

listen_files streams an archive from any file object (a local file, or a download as it arrives): it is
decompressed and unbundled piece by piece and never unpacked to disk.

One song = its cleaned-up artist + title (song_key): lowercase, accents and punctuation removed, featured
artists and version notes ("- Remastered 2011", "(Radio Edit)") dropped, so a single, its album version and its
remasters are one song. Live versions, remixes and other recordings that sound different stay separate.
"""
import functools
import html
import json
import re
import tarfile
import unicodedata
from typing import NamedTuple

import zstandard

MONTH_FILE = re.compile(r"(?:^|/)listens/(\d{4})/(\d{1,2})\.listens$")
SPOTIFY_TRACK = re.compile(r"(?:open\.spotify\.com/(?:intl-[a-z]+/)?track/|spotify:track:)([A-Za-z0-9]{22})")
# Featured artists: "(feat. X)", "[with X]", or a trailing "feat. X" / "ft. X".
FEATURING = re.compile(r"\s*[\(\[](?:feat\.?|ft\.?|featuring|with)\s[^\)\]]*[\)\]]|\s+(?:feat\.?|ft\.?|featuring)\s.*$")
# Version notes that don't change the recording much: after " - " or in brackets.
VERSION_WORDS = (r"remaster(?:ed)?(?:\s+version)?|\d{4}\s+remaster(?:ed)?|"
                 r"(?:single|album|lp|single/lp|lp/single|original|soundtrack|film|movie|radio|mono|stereo)\s+version|"
                 r"radio\s+edit|mono|stereo|explicit|edited|full\s+length|bonus\s+track")
VERSION_NOTE = re.compile(rf"\s+-\s+(?:[^-]*\s)?(?:{VERSION_WORDS})(?:\s[^-]*)?$|\s*[\(\[](?:[^\)\]]*\s)?(?:{VERSION_WORDS})"
                          rf"(?:\s[^\)\]]*)?[\)\]]", re.IGNORECASE)
YEAR_LEFT = re.compile(r"\s+-\s+\d{4}$")  # left by "- 1997 - Remastered" once the note is gone


class Listen(NamedTuple):
    user_id: int
    timestamp: int
    artist: str           # the artist credit as submitted, e.g. "Calvin Harris, Dua Lipa"
    first_artist: str     # the main artist: additional_info.artist_names[0] when given, else the credit
    title: str
    release: str          # album name, "" if missing
    spotify_id: str       # 22-character Spotify track ID, "" if missing
    recording_mbid: str   # MusicBrainz recording ID, "" if missing
    duration_ms: int      # 0 if missing


def month_of(member_name):
    """(year, month) of a dump member "…/listens/2026/10.listens", or None for other members."""
    m = MONTH_FILE.search(member_name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def listen_files(fileobj):
    """Each month file of a .tar.zst listen dump, in archive order: yields (year, month, binary file of lines).
    Each file must be read before asking for the next (the archive is read once, front to back)."""
    reader = zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(fileobj, read_size=1 << 20)
    with tarfile.open(fileobj=reader, mode="r|") as tar:
        for member in tar:
            month = member.isfile() and month_of(member.name)
            if month:
                yield month[0], month[1], tar.extractfile(member)


def spotify_track_id(info):
    """The Spotify track ID in a listen's additional_info (spotify_id, else origin_url), or ""."""
    for field in ("spotify_id", "origin_url"):
        value = info.get(field)
        if isinstance(value, str):
            m = SPOTIFY_TRACK.search(value)
            if m:
                return m.group(1)
    return ""


def listen_duration_ms(info):
    """additional_info's duration_ms, or duration (seconds) x 1000; 0 if missing or not a positive number."""
    for field, scale in (("duration_ms", 1), ("duration", 1000)):
        value = info.get(field)
        if isinstance(value, (int, float)) and value > 0:
            return int(value * scale)
        if isinstance(value, str) and value.isdigit() and int(value) > 0:
            return int(value) * scale
    return 0


def parse_listen(line):
    """One dump line (JSON) as a Listen."""
    d = json.loads(line)
    meta = d["track_metadata"]
    info = meta.get("additional_info") or {}
    artist = meta.get("artist_name") or ""
    names = info.get("artist_names")
    first = names[0] if isinstance(names, list) and names and isinstance(names[0], str) else artist
    mbid = info.get("recording_mbid")
    return Listen(int(d["user_id"]), int(d["timestamp"]), artist, first, meta.get("track_name") or "",
                  meta.get("release_name") or "", spotify_track_id(info), mbid if isinstance(mbid, str) else "",
                  listen_duration_ms(info))


def strip_accents(text):
    """Accents off Latin letters ("é" -> "e", full-width "Ａ" -> "A"); other scripts keep their marks, which can
    change the letter (Japanese "ジ" is not "シ", Russian "й" is not "и")."""
    if text.isascii():
        return text
    out = []
    for c in unicodedata.normalize("NFC", text):
        base = "".join(x for x in unicodedata.normalize("NFKD", c) if not unicodedata.combining(x))
        out.append(base if base.isascii() else unicodedata.normalize("NFKC", c))
    return "".join(out)


def clean_text(text):
    """HTML codes decoded ("&amp;" -> "&"), lowercase, accents off Latin letters, "&" as "and", punctuation as
    spaces, single spaces."""
    text = re.sub(r"[^\w\s]", " ", strip_accents(html.unescape(text).lower()).replace("&", " and "))
    return " ".join(text.split())


def clean_title(title):
    """A title without featured artists or version notes, cleaned."""
    previous = None
    while previous != title:  # notes can be stacked: "Song (feat. X) - Remastered 2011"
        previous = title
        title = YEAR_LEFT.sub("", VERSION_NOTE.sub("", FEATURING.sub("", title)))
    return clean_text(title)


def clean_artist(artist):
    """The main artist's name without featured artists, cleaned."""
    return clean_text(FEATURING.sub("", artist))


@functools.lru_cache(maxsize=1 << 20)
def song_key(artist, title):
    """What makes two listens (or a listen and an MPD track) the same song: "artist|title", both cleaned.
    Remembers recent answers: the same songs come up again and again."""
    return f"{clean_artist(artist)}|{clean_title(title)}"
