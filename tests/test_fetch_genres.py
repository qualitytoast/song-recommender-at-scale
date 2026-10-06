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
