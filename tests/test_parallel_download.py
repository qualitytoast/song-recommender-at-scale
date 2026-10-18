import hashlib
import json
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from scripts.parallel_download import download, piece_ranges

PAYLOAD = bytes(range(256)) * 200  # 51,200 bytes
PIECE = 1000


class RangeServer(BaseHTTPRequestHandler):
    """Serves PAYLOAD with byte ranges; can drop one piece's connection halfway, or publish a wrong checksum."""
    cut_once, starts, wrong_checksum, slow = set(), [], False, 0.0

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(PAYLOAD)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):
        if self.path.endswith(".sha256"):
            body = (hashlib.sha256(b"other" if RangeServer.wrong_checksum else PAYLOAD).hexdigest() + "  f\n").encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        first, last = self.headers["Range"].split("=")[1].split("-")
        first, last = int(first), int(last or len(PAYLOAD) - 1)
        body = PAYLOAD[first:last + 1]
        RangeServer.starts.append(first)
        time.sleep(RangeServer.slow)
        self.send_response(206)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Range", f"bytes {first}-{last}/{len(PAYLOAD)}")
        self.end_headers()
        if first in RangeServer.cut_once:
            RangeServer.cut_once.discard(first)
            self.wfile.write(body[:len(body) // 2])
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_RDWR)
            return
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        pass


@pytest.fixture
def url():
    httpd = QuietServer(("127.0.0.1", 0), RangeServer)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    RangeServer.cut_once, RangeServer.starts, RangeServer.wrong_checksum, RangeServer.slow = set(), [], False, 0.0
    yield f"http://127.0.0.1:{httpd.server_address[1]}/dump.bin"
    httpd.shutdown()


def run(url, out, **kw):
    return download(url, out, connections=4, piece=PIECE, report_every=0.01, **kw)


def test_piece_ranges_cover_the_file_exactly():
    assert piece_ranges(2500, 1000) == [(0, 999), (1000, 1999), (2000, 2499)]
    assert piece_ranges(2000, 1000) == [(0, 999), (1000, 1999)]


def test_downloads_in_pieces_and_checks_the_checksum(url, tmp_path):
    out = tmp_path / "dump.bin"
    assert run(url, out) is True
    assert out.read_bytes() == PAYLOAD
    assert len(RangeServer.starts) == 52 and not (tmp_path / "dump.bin.pieces.json").exists()


def test_continues_a_file_another_program_downloaded_from_the_start(url, tmp_path):
    out = tmp_path / "dump.bin"
    out.write_bytes(PAYLOAD[:12_345])  # like curl -C - stopped partway
    assert run(url, out) is True
    assert out.read_bytes() == PAYLOAD
    assert min(RangeServer.starts) == 12_000  # pieces wholly before byte 12,345 aren't fetched again


def test_continues_from_its_progress_file(url, tmp_path):
    out = tmp_path / "dump.bin"
    done = [0, 1, 7, 51]
    data = bytearray(len(PAYLOAD))
    for i in done:
        start, end = piece_ranges(len(PAYLOAD), PIECE)[i]
        data[start:end + 1] = PAYLOAD[start:end + 1]
    out.write_bytes(bytes(data))
    (tmp_path / "dump.bin.pieces.json").write_text(json.dumps({"total": len(PAYLOAD), "piece": PIECE, "done": done}))
    assert run(url, out) is True
    assert out.read_bytes() == PAYLOAD
    assert not {i * PIECE for i in done} & set(RangeServer.starts)


def test_a_dropped_piece_is_fetched_again(url, tmp_path):
    out = tmp_path / "dump.bin"
    RangeServer.cut_once = {5000, 30000}
    assert run(url, out) is True
    assert out.read_bytes() == PAYLOAD
    assert RangeServer.starts.count(5000) == 2 and RangeServer.starts.count(30000) == 2


def test_a_checksum_mismatch_is_reported_and_progress_kept(url, tmp_path):
    out = tmp_path / "dump.bin"
    RangeServer.wrong_checksum = True
    assert run(url, out) is False
    assert (tmp_path / "dump.bin.pieces.json").exists()


def test_ctrl_c_stops_quickly_keeps_finished_pieces_and_resumes(url, tmp_path):
    out = tmp_path / "dump.bin"
    RangeServer.slow = 0.2  # seconds per piece: 52 pieces over 2 connections would take ~5 s
    code = (f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r}); "
            f"from scripts.parallel_download import download; "
            f"download({url!r}, {str(out)!r}, connections=2, piece={PIECE}, report_every=0.05)")
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(1.5)
    pressed = time.monotonic()
    proc.send_signal(signal.SIGINT)  # what Ctrl-C sends
    proc.wait(timeout=10)
    assert time.monotonic() - pressed < 3  # stops within a piece or so, not after every queued piece
    done = json.loads((tmp_path / "dump.bin.pieces.json").read_text())["done"]
    assert 0 < len(done) < 52
    RangeServer.slow, RangeServer.starts = 0.0, []
    assert run(url, out) is True and out.read_bytes() == PAYLOAD
    assert len(RangeServer.starts) == 52 - len(done)  # only the missing pieces
