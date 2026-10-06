import io
import json
import tarfile

from scripts.genres_from_dump import artist_lines, scan, write_output


def record(mbid, spotify_ids, genres):
    rels = [{"target-type": "url", "url": {"resource": f"https://open.spotify.com/artist/{s}"}} for s in spotify_ids]
    rels.append({"target-type": "url", "url": {"resource": "https://en.wikipedia.org/wiki/X"}})
    return json.dumps({"id": mbid, "relations": rels,
                       "genres": [{"name": n, "count": c} for n, c in genres]}).encode()


LINES = [
    record("m1", ["aaa"], [("rock", 3), ("pop", 0)]),   # "pop" has no net votes: dropped
    record("m2", ["zzz"], [("jazz", 5)]),               # not an artist we need
    record("m3", ["bbb"], [("rap", 1)]),
    record("m4", ["bbb"], [("rap", 2), ("trap", 1)]),   # second MusicBrainz artist on the same Spotify page
    json.dumps({"id": "m5", "relations": [], "genres": [{"name": "folk", "count": 9}]}).encode(),
]


def test_scan_keeps_wanted_spotify_artists_and_merges_shared_pages():
    found = scan(LINES, {"aaa", "bbb", "ccc"})
    assert found == {"aaa": {"mbids": ["m1"], "genres": [{"name": "rock", "count": 3}]},
                     "bbb": {"mbids": ["m3", "m4"],
                             "genres": [{"name": "rap", "count": 3}, {"name": "trap", "count": 1}]}}


def test_artist_lines_reads_the_artist_member_of_a_streamed_tar_xz():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz") as tar:
        for name, data in [("README", b"hello"), ("mbdump/artist", b"\n".join(LINES) + b"\n")]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    assert scan(artist_lines(buf), {"aaa"}) == {"aaa": {"mbids": ["m1"], "genres": [{"name": "rock", "count": 3}]}}


def test_output_has_every_artist_and_appears_only_when_finished(tmp_path):
    out = tmp_path / "genres.jsonl"
    found = {"aaa": {"mbids": ["m1"], "genres": [{"name": "rock", "count": 3}]}}
    write_output([("spotify:artist:aaa", "A"), ("spotify:artist:ccc", "C")], found, out, {"source": "test"})
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows == [{"artist_uri": "spotify:artist:aaa", "artist_name": "A", "mbids": ["m1"],
                     "genres": [{"name": "rock", "count": 3}]},
                    {"artist_uri": "spotify:artist:ccc", "artist_name": "C", "mbids": [], "genres": []}]
    assert not out.with_suffix(".tmp").exists()
    assert json.loads(out.with_suffix(".meta.json").read_text()) == {"source": "test"}
