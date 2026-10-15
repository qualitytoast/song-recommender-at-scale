import io
import json
import tarfile

import zstandard

from recsys.listenbrainz import (clean_artist, clean_text, clean_title, listen_duration_ms, listen_files, month_of,
                                 parse_listen, song_key, spotify_track_id)


def listen_line(user=1, ts=100, artist="Drake", title="One Dance", info=None, release="Views"):
    meta = {"artist_name": artist, "track_name": title, "additional_info": info or {}}
    if release is not None:
        meta["release_name"] = release
    return json.dumps({"user_id": user, "user_name": f"u{user}", "timestamp": ts, "recording_msid": "x",
                       "track_metadata": meta}).encode()


def dump_bytes(files):
    """A .tar.zst archive holding {member name: bytes}."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    return zstandard.ZstdCompressor().compress(raw.getvalue())


def test_month_of_reads_month_files_only():
    assert month_of("listenbrainz-listens-dump-1-full/listens/2026/10.listens") == (2026, 10)
    assert month_of("listens/2005/1.listens") == (2005, 1)
    assert month_of("listenbrainz-listens-dump-1-full/COPYING") is None
    assert month_of("listens/2026/10.listens.bak") is None


def test_listen_files_streams_each_month_in_order():
    data = dump_bytes({"d/COPYING": b"license", "d/listens/2025/12.listens": listen_line(ts=1) + b"\n",
                       "d/listens/2026/1.listens": listen_line(ts=2) + b"\n" + listen_line(ts=3) + b"\n"})
    seen = [(y, m, [parse_listen(line).timestamp for line in f]) for y, m, f in listen_files(io.BytesIO(data))]
    assert seen == [(2025, 12, [1]), (2026, 1, [2, 3])]


def test_spotify_track_id_from_spotify_id_or_origin_url():
    track = "1rrgWMXGCGHru5bIRxGFV0"
    assert spotify_track_id({"spotify_id": f"http://open.spotify.com/track/{track}"}) == track
    assert spotify_track_id({"origin_url": f"https://open.spotify.com/intl-de/track/{track}?si=abc"}) == track
    assert spotify_track_id({"spotify_id": f"spotify:track:{track}"}) == track
    assert spotify_track_id({"origin_url": "https://www.youtube.com/watch?v=abc"}) == ""
    assert spotify_track_id({"spotify_id": None}) == ""


def test_listen_duration_ms_accepts_ms_or_seconds():
    assert listen_duration_ms({"duration_ms": 351000}) == 351000
    assert listen_duration_ms({"duration": 245}) == 245000
    assert listen_duration_ms({"duration_ms": "1000"}) == 1000
    assert listen_duration_ms({"duration_ms": -5}) == 0
    assert listen_duration_ms({}) == 0


def test_parse_listen_fields_and_main_artist():
    mbid = "7fa7fc04-1011-4876-8095-ecd232edea87"
    info = {"spotify_id": "https://open.spotify.com/track/1rrgWMXGCGHru5bIRxGFV0", "duration_ms": 1000,
            "artist_names": ["Calvin Harris", "Dua Lipa"], "recording_mbid": "rec", "artist_mbids": [mbid.upper()],
            "spotify_artist_ids": ["https://open.spotify.com/artist/6UZ0ba50XreR4TM8u322gs"]}
    listen = parse_listen(listen_line(user=7, ts=9, artist="Calvin Harris, Dua Lipa", title="One Kiss", info=info))
    assert listen == (7, 9, "Calvin Harris, Dua Lipa", "Calvin Harris", "One Kiss", "Views",
                      "1rrgWMXGCGHru5bIRxGFV0", "rec", 1000, mbid, "6UZ0ba50XreR4TM8u322gs")
    bare = parse_listen(listen_line(release=None))
    assert (bare.first_artist, bare.release, bare.spotify_id, bare.recording_mbid, bare.duration_ms,
            bare.artist_mbid, bare.spotify_artist_id) == ("Drake", "", "", "", 0, "", "")
    lastfm = parse_listen(listen_line(info={"lastfm_artist_mbid": mbid, "spotify_artist_ids": ["junk"]}))
    assert (lastfm.artist_mbid, lastfm.spotify_artist_id) == (mbid, "")
    assert parse_listen(listen_line(info={"lastfm_artist_mbid": "not-an-id"})).artist_mbid == ""


def test_clean_text_case_accents_punctuation():
    assert clean_text("  Beyoncé & JAY-Z!! ") == "beyonce and jay z"
    assert clean_text("Sigur Rós") == "sigur ros"
    assert clean_text("Dumb &amp; Poetic") == clean_text("Dumb & Poetic") == "dumb and poetic"
    assert clean_text("ジブリ") == "ジブリ"  # other scripts keep their marks
    assert clean_text("Ａｄｅｌｅ") == "adele"  # full-width letters


def test_clean_title_drops_featuring_and_version_notes():
    assert clean_title("Let It Be - Remastered 2009") == "let it be"
    assert clean_title("Let It Be - 2009 Remaster") == "let it be"
    assert clean_title("Wonderwall (Remastered)") == "wonderwall"
    assert clean_title("Can't Hold Us (feat. Ray Dalton)") == "can t hold us"
    assert clean_title("Mask Off [with Kendrick Lamar]") == "mask off"
    assert clean_title("Song feat. Someone") == "song"
    assert clean_title("Song (feat. X) - Single Version") == "song"
    assert clean_title("Blinding Lights - Radio Edit") == "blinding lights"
    assert clean_title("Return To Me (Ritorna-Me) - 1997 - Remastered") == "return to me ritorna me"
    assert clean_title("Wouldn't It Be Nice - 2000 - Remaster") == "wouldn t it be nice"
    assert clean_title("Stand By Me - Single/LP Version") == "stand by me"
    assert clean_title("Decode - Twilight Soundtrack Version") == "decode"
    assert clean_title("Magic Stick - Edited") == "magic stick"
    assert clean_title("Skyfall - Full Length") == "skyfall"


def test_clean_title_keeps_versions_that_sound_different():
    assert clean_title("Wonderwall - Live") == "wonderwall live"
    assert clean_title("Song - Acoustic") == "song acoustic"
    assert clean_title("Song (Remix)") == "song remix"
    assert clean_title("Song - Acoustic Version") == "song acoustic version"
    assert clean_title("Stereo Hearts") == "stereo hearts"  # version words only count as notes


def test_clean_artist_and_song_key():
    assert clean_artist("Drake feat. Rihanna") == "drake"
    assert clean_artist("Simon & Garfunkel") == "simon and garfunkel"
    assert song_key("Macklemore & Ryan Lewis", "Can't Hold Us - feat. Ray Dalton") == (
        "macklemore and ryan lewis|can t hold us")
    assert song_key("The Beatles", "Let It Be - Remastered 2009") == song_key("the beatles", "Let It Be")
