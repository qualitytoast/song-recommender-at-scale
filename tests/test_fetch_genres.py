import io
import json

import scripts.fetch_genres as fg
from scripts.fetch_genres import mbids_by_spotify_id, merge_genres, spotify_url


def test_spotify_url_from_uri():
    assert spotify_url("spotify:artist:abc123") == "https://open.spotify.com/artist/abc123"


def test_url_lookup_maps_spotify_ids_to_mbids_and_skips_missing():
    # Shape of a real response: unmatched URLs are simply absent.
    response = {"urls": [
        {"resource": "https://open.spotify.com/artist/aaa",
         "relations": [{"artist": {"id": "mb-1"}}]},
        {"resource": "https://open.spotify.com/artist/bbb",
         "relations": [{"artist": {"id": "mb-2"}}, {"artist": {"id": "mb-3"}}]},
    ]}
    assert mbids_by_spotify_id(response) == {"aaa": ["mb-1"], "bbb": ["mb-2", "mb-3"]}
    assert mbids_by_spotify_id(None) == {}


def test_merge_genres_adds_votes_most_first():
    merged = merge_genres([[{"name": "rap", "count": 2}, {"name": "pop", "count": 1}],
                           [{"name": "pop", "count": 3}]])
    assert merged == [{"name": "pop", "count": 4}, {"name": "rap", "count": 2}]


# --- resuming and retrying ---


def test_cut_off_last_line_is_skipped_and_next_line_starts_fresh(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text('{"artist_uri": "a", "mbids": [], "genres": []}\n{"artist_uri": "b", "mbi')  # killed mid-write
    assert [r["artist_uri"] for r in fg.read_rows(path)] == ["a"]
    with fg.open_for_append(path) as out:
        out.write('{"artist_uri": "c", "mbids": [], "genres": []}\n')
    assert [r["artist_uri"] for r in fg.read_rows(path)] == ["a", "c"]  # "b" will be fetched again


def test_timeout_while_waiting_for_reply_is_retried(monkeypatch):
    calls = []

    def flaky_urlopen(request, timeout, context):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise TimeoutError("timed out")  # not a URLError: the case that used to crash
        return io.BytesIO(json.dumps({"ok": True}).encode())

    monkeypatch.setattr(fg.urllib.request, "urlopen", flaky_urlopen)
    monkeypatch.setattr(fg.time, "sleep", lambda s: None)  # don't actually wait
    assert fg.get("artist/x", {"inc": "genres"}) == {"ok": True}
    assert len(calls) == 2
