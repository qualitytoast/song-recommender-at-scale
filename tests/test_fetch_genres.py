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


# --- batched search ---

GENRES = {"hip hop", "pop rap", "rock"}


def test_genres_from_tags_keeps_official_genres_with_positive_votes():
    tags = [{"name": "hip hop", "count": 10}, {"name": "sxsw", "count": 3},     # not a genre
            {"name": "pop rap", "count": 3}, {"name": "rock", "count": 0},      # no net votes
            {"name": "lesbian", "count": -1}]
    assert fg.genres_from_tags(tags, GENRES) == [{"name": "hip hop", "count": 10}, {"name": "pop rap", "count": 3}]


def test_batch_uses_one_url_lookup_and_one_search_with_fallback_for_missing(monkeypatch, tmp_path):
    calls = []

    def fake_get(path, params):
        calls.append(path)
        if path == "url":
            return {"urls": [{"resource": "https://open.spotify.com/artist/aaa", "relations": [{"artist": {"id": "m1"}}]},
                             {"resource": "https://open.spotify.com/artist/bbb", "relations": [{"artist": {"id": "m2"}}]}]}
        if path == "artist":  # search: m2 is missing from the index
            return {"artists": [{"id": "m1", "tags": [{"name": "rock", "count": 2}, {"name": "sxsw", "count": 5}]}]}
        if path == "artist/m2":  # fallback lookup
            return {"genres": [{"name": "pop rap", "count": 1}]}
        raise AssertionError(path)

    monkeypatch.setattr(fg, "get", fake_get)
    monkeypatch.setattr(fg, "OUT", tmp_path / "out.jsonl")
    fg.fetch([("spotify:artist:aaa", "A"), ("spotify:artist:bbb", "B"), ("spotify:artist:ccc", "C")], GENRES)
    rows = {r["artist_uri"]: r for r in fg.read_rows(tmp_path / "out.jsonl")}
    assert calls == ["url", "artist", "artist/m2"]
    assert rows["spotify:artist:aaa"]["genres"] == [{"name": "rock", "count": 2}]
    assert rows["spotify:artist:bbb"]["genres"] == [{"name": "pop rap", "count": 1}]
    assert rows["spotify:artist:ccc"] == {"artist_uri": "spotify:artist:ccc", "artist_name": "C", "mbids": [], "genres": []}
