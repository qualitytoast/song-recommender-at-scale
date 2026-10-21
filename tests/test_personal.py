import json
import zipfile

import numpy as np

from recsys.personal import build_plays, catalog_lookup, listening_sessions, match, read_export
from tests.serving_helpers import write_tiny_bundle

T0 = 1_760_000_000  # a Unix time; plays below are minutes apart


def play(minute, uri, track="", artist="", ms=200_000, skipped=False):
    ts = np.datetime64(T0 + minute * 60, "s").astype(str) + "Z"
    return {"ts": ts, "ms_played": ms, "skipped": skipped, "spotify_track_uri": uri, "ip_addr": "0.0.0.0",
            "master_metadata_track_name": track, "master_metadata_album_artist_name": artist}


def write_export(path, audio, other=None):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("Spotify Extended Streaming History/Streaming_History_Audio_2026.json", json.dumps(audio))
        z.writestr("Spotify Extended Streaming History/Streaming_History_Video_2026.json", json.dumps(other or []))
    return path


def test_read_export_keeps_music_plays_from_the_audio_files_in_time_order(tmp_path):
    podcast = {**play(1, None), "spotify_episode_uri": "spotify:episode:x"}
    path = write_export(tmp_path / "e.zip", [play(5, "spotify:track:2"), play(0, "spotify:track:1"), podcast],
                        other=[play(3, "spotify:track:9")])
    plays = read_export(path)
    assert plays["uri"] == ["spotify:track:1", "spotify:track:2"]   # sorted; no podcast, no video file
    assert list(plays["seconds"]) == [T0, T0 + 300] and set(plays) == {"seconds", "ms_played", "skipped", "uri",
                                                                       "track", "artist"}


def test_songs_match_by_uri_then_by_artist_and_title(tmp_path):
    bundle = write_tiny_bundle(tmp_path)[0]
    plays = read_export(write_export(tmp_path / "e.zip", [
        play(0, "spotify:track:3"),                                          # in the catalog by URI
        play(1, "spotify:track:other", "Nightcall (feat. Someone)", "Kavinsky"),  # by cleaned artist + title
        play(2, "spotify:track:nope", "Unknown Song", "Nobody")]))
    ids, how = match(plays, *catalog_lookup(bundle))
    assert list(ids) == [3, 0, -1] and list(how) == [1, 2, 0]


def test_sessions_follow_the_listenbrainz_rules():
    minutes = [0, 4, 8, 12, 13, 16, 60, 64, 68]       # a 44-minute gap starts a new session
    uris = ["a", "b", "b", "c", "x", "d", "e", "f", "g"]
    ms = [200_000] * 9
    ms[4] = 10_000                                     # "x": under 30 s, not a listen
    plays = {"seconds": T0 + np.array(minutes) * 60, "ms_played": np.array(ms), "uri": uris}
    listened, kept, session = listening_sessions(plays)
    assert list(listened) == [True] * 4 + [False] + [True] * 4
    # session 1: a b (b repeat dropped) c d = 4 songs, kept; session 2: e f g = 3 songs, too short
    assert list(kept) == [True, True, False, True, False, True, False, False, False]
    assert len(set(session[kept])) == 1


def test_build_plays_writes_the_matched_sessions_and_reports_totals(tmp_path):
    bundle = write_tiny_bundle(tmp_path)[0]
    export = write_export(tmp_path / "e.zip", [play(m, f"spotify:track:{m % 7}") for m in range(0, 40, 4)])
    totals = build_plays(export, bundle, tmp_path / "personal")
    saved = np.load(next((tmp_path / "personal").glob("*/plays.npz")))
    assert totals["plays"] == totals["listens"] == 10 and totals["listens_matched"] == 1.0
    assert totals["sessions"] == 1 and saved["kept"].sum() == totals["session_listens"] == 10
    assert "ip_addr" not in saved.files
