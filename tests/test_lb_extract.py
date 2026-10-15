import calendar
import hashlib
import io
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from recsys.lb_extract import (ResumableHTTP, Tables, blocks, extract, load_extract, month_in_window, parse_block)
from tests.test_listenbrainz import dump_bytes, listen_line

MBID = "7fa7fc04-1011-4876-8095-ecd232edea87"
JAN = calendar.timegm((2026, 1, 1, 0, 0, 0))
FEB = calendar.timegm((2026, 2, 1, 0, 0, 0))
MAR = calendar.timegm((2026, 3, 1, 0, 0, 0))


def lines(*listens):
    return b"".join(listen_line(**kw) + b"\n" for kw in listens)


def test_month_in_window():
    assert month_in_window(2026, 1, JAN, FEB)
    assert not month_in_window(2025, 12, JAN, FEB)
    assert not month_in_window(2026, 2, JAN, FEB)        # the end is not kept
    assert month_in_window(2025, 12, JAN - 1, FEB)       # one second of December
    assert month_in_window(2026, 12, JAN, calendar.timegm((2027, 1, 1, 0, 0, 0)))


def test_blocks_end_at_line_ends_and_keep_every_byte():
    data = b"".join(b"line %d\n" % i for i in range(1000))
    parts = list(blocks(io.BytesIO(data), size=100))
    assert b"".join(parts) == data
    assert all(p.endswith(b"\n") for p in parts) and len(parts) > 10


def test_parse_block_keeps_the_window_and_numbers_strings_locally():
    block = lines(dict(user=1, ts=JAN + 5, artist="Drake", title="One Dance", release="Views"),
                  dict(user=2, ts=JAN + 6, artist="Drake", title="One Dance (feat. Wizkid)", release="",
                       info={"artist_mbids": [MBID],
                             "spotify_id": "https://open.spotify.com/track/1rrgWMXGCGHru5bIRxGFV0"}),
                  dict(user=3, ts=FEB, artist="Drake", title="Late"),               # after the window
                  dict(user=4, ts=JAN + 7, artist="", title="No Artist"),
                  dict(user=5, ts=JAN + 8, artist="!!!", title="???"),                # nothing left once cleaned
                  dict(user=6, ts=JAN + 9, artist="Adele", title="Hello", release="25",
                       info={"duration_ms": 295000})) + b"not json\n"
    r = parse_block(block, JAN, FEB)
    assert list(np.frombuffer(r["user"], dtype=np.int32)) == [1, 2, 6]
    assert list(np.frombuffer(r["song"], dtype=np.int32)) == [0, 0, 1]   # "feat. Wizkid" is the same song
    assert list(np.frombuffer(r["release"], dtype=np.int32)) == [0, -1, 1]
    assert r["releases"] == ["Views", "25"]
    assert list(np.frombuffer(r["spotify"], dtype=np.int32)) == [-1, 0, -1]
    assert list(np.frombuffer(r["duration_ms"], dtype=np.int32)) == [0, 0, 295000]
    assert r["songs"] == [["drake|one dance", "Drake", "One Dance", MBID, ""], ["adele|hello", "Adele", "Hello", "", ""]]
    assert r["spotify_tracks"] == ["1rrgWMXGCGHru5bIRxGFV0"]
    assert r["counts"] == {"kept": 3, "outside the window": 1, "no artist or title": 2, "unreadable": 1}


def test_tables_give_the_same_song_the_same_number_across_blocks(tmp_path):
    tables = Tables(tmp_path)
    first = lines(dict(user=1, ts=JAN, artist="Drake", title="One Dance", release="Views"),
                  dict(user=1, ts=JAN + 1, artist="Adele", title="Hello", release=""))
    second = lines(dict(user=2, ts=JAN + 2, artist="Adele", title="Hello", release="25",
                        info={"spotify_artist_ids": ["https://open.spotify.com/artist/6UZ0ba50XreR4TM8u322gs"]}),
                   dict(user=2, ts=JAN + 3, artist="Drake", title="One Dance", release="Views"))
    for block in (first, second):
        tables.add(parse_block(block, JAN, FEB))
    tables.finish()
    song = np.fromfile(tmp_path / "song.bin", dtype=np.int32)
    release = np.fromfile(tmp_path / "release.bin", dtype=np.int32)
    assert list(song) == [0, 1, 1, 0] and list(release) == [0, -1, 1, 0]
    assert json.loads((tmp_path / "releases.json").read_text()) == ["Views", "25"]
    assert [json.loads(line) for line in (tmp_path / "songs.jsonl").read_text().splitlines()] == [
        ["drake|one dance", "Drake", "One Dance"], ["adele|hello", "Adele", "Hello"]]
    assert list(np.fromfile(tmp_path / "song_spotify_artist.bin", dtype=np.int32)) == [-1, 0]  # filled in later
    assert json.loads((tmp_path / "spotify_artists.json").read_text()) == ["6UZ0ba50XreR4TM8u322gs"]


def fake_dump(tmp_path):
    data = dump_bytes({
        "d/COPYING": b"license",
        "d/listens/2025/12.listens": lines(dict(user=9, ts=JAN - 10, artist="Old", title="Song")),
        "d/listens/2026/1.listens": lines(*[dict(user=u, ts=JAN + u, artist=f"Artist {u % 3}", title="Song",
                                                 release="Album") for u in range(50)]),
        "d/listens/2026/2.listens": lines(dict(user=1, ts=FEB + 1, artist="New", title="Song")),
    })
    path = tmp_path / "dump.tar.zst"
    path.write_bytes(data)
    (tmp_path / "dump.tar.zst.sha256").write_text(hashlib.sha256(data).hexdigest() + "\n")
    return path, data


def test_extract_end_to_end_from_a_local_dump(tmp_path):
    path, _ = fake_dump(tmp_path)
    meta = extract(str(path), tmp_path / "out", JAN, FEB, workers=2, block=200)
    assert meta["listens"] == 50 and meta["songs"] == 3 and meta["users"] == 50
    assert meta["months_read"] == ["2026-01"] and meta["months_skipped"] == 2
    assert meta["sha256_ok"] is True and meta["bytes_read"] == path.stat().st_size
    e = load_extract(tmp_path / "out")
    assert list(e.user) == list(range(50)) and list(e.timestamp) == [JAN + u for u in range(50)]
    assert sorted(set(e.song.tolist())) == [0, 1, 2] and set(e.release.tolist()) == {0}
    assert [s[0] for s in e.songs()] == ["artist 0|song", "artist 1|song", "artist 2|song"] and e.releases == ["Album"]
    assert list(e.song_artist_mbid) == [-1, -1, -1] and e.artist_mbids == []
    with pytest.raises(FileExistsError):
        extract(str(path), tmp_path / "out", JAN, FEB, workers=1)


class Server(BaseHTTPRequestHandler):
    payload, cut_after, honor_range, starts = b"", None, True, []

    def do_GET(self):
        if self.path.endswith(".sha256"):
            body = hashlib.sha256(Server.payload).hexdigest().encode()
            start = 0
        else:
            ranged = self.headers.get("Range")
            start = int(ranged.split("=")[1].split("-")[0]) if ranged and Server.honor_range else 0
            body = Server.payload[start:]
            Server.starts.append(start)
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if Server.cut_after is not None and not self.path.endswith(".sha256"):
            n, Server.cut_after = Server.cut_after, None
            self.wfile.write(body[:n])
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_RDWR)  # the connection drops mid-download
            return
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        pass  # the client hanging up on a cut connection is expected


@pytest.fixture
def server():
    httpd = QuietServer(("127.0.0.1", 0), Server)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    Server.starts, Server.cut_after, Server.honor_range = [], None, True
    yield f"http://127.0.0.1:{httpd.server_address[1]}/dump.tar.zst"
    httpd.shutdown()


def test_resumable_http_picks_up_where_the_connection_dropped(server):
    Server.payload = bytes(range(256)) * 4000
    Server.cut_after = 300_000
    reader = ResumableHTTP(server, retries=3, wait=0)
    data = reader.read()
    assert data == Server.payload and reader.reconnects == 1
    assert Server.starts[0] == 0 and 0 < Server.starts[1] <= 300_000
    assert reader.sha256.hexdigest() == hashlib.sha256(Server.payload).hexdigest()


def test_resumable_http_gives_up_if_the_server_cannot_resume(server):
    Server.payload, Server.cut_after, Server.honor_range = b"x" * 500_000, 100_000, False
    reader = ResumableHTTP(server, retries=2, wait=0)
    with pytest.raises(Exception):
        reader.read()


def test_extract_streams_a_dump_over_a_dropping_connection(server, tmp_path):
    _, Server.payload = fake_dump(tmp_path)
    Server.cut_after = len(Server.payload) // 2
    meta = extract(server, tmp_path / "out", JAN, FEB, workers=1)
    assert meta["listens"] == 50 and meta["reconnects"] == 1 and meta["sha256_ok"] is True
