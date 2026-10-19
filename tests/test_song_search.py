import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from recsys.recommender import Recommender
from recsys.song_search import SongSearch, documents, index_name
from tests.serving_helpers import ARTIST_OF, ARTISTS, PLAYLISTS, TRACKS, V, write_tiny_bundle


class FakeMeilisearch:
    """A stand-in for Meilisearch's search endpoint: answers with self.hits (or, per self.mode, slowly, with
    an error status, or with nonsense) and records each request."""

    def __init__(self):
        self.hits, self.mode, self.requests = [], "ok", []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                fake.requests.append((self.path, self.headers.get("authorization"), body))
                if fake.mode == "slow":
                    time.sleep(0.5)
                if fake.mode == "error":
                    self.send_response(500)
                    self.end_headers()
                    return
                answer = b"not json" if fake.mode == "nonsense" else json.dumps(
                    {"hits": [{"id": j} for j in fake.hits]}).encode()
                self.send_response(200)
                self.send_header("content-length", str(len(answer)))
                self.end_headers()
                self.wfile.write(answer)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def meili():
    fake = FakeMeilisearch()
    yield fake
    fake.close()


def test_documents_hold_each_songs_names_and_count(tmp_path):
    path = write_tiny_bundle(tmp_path, albums=[f"Album {i % 3}" for i in range(V)])[0]
    docs = list(documents(path))
    assert len(docs) == V
    assert docs[1] == {"id": 1, "name": f"{TRACKS[1]} {ARTISTS[ARTIST_OF[1]]}", "album": "Album 1",
                       "popularity": int(PLAYLISTS[1])}


def test_a_search_returns_song_ids_from_its_bundles_index(meili):
    meili.hits = [3, 0]
    search = SongSearch(meili.url, "abc123", songs=V, key="secret")
    assert search.search("night drive", 5) == [3, 0]
    path, auth, body = meili.requests[0]
    assert path == f"/indexes/{index_name('abc123')}/search" and auth == "Bearer secret"
    assert body == {"q": "night drive", "limit": 5, "attributesToRetrieve": ["id"]}


@pytest.mark.parametrize("mode", ["slow", "error", "nonsense", "bad id"])
def test_a_search_that_cant_answer_gives_none_then_rests(meili, mode):
    meili.mode, meili.hits = ("ok", [V]) if mode == "bad id" else (mode, [1])
    search = SongSearch(meili.url, "abc123", songs=V, timeout=0.1, cooldown=60)
    assert search.search("night", 5) is None
    assert search.search("night", 5) is None  # skipped: no second request while resting
    assert len(meili.requests) == 1 and search.stats == {"answered": 0, "errors": 1, "skipped": 1}


class DictCache:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value


def test_the_menu_asks_the_search_first_and_falls_back_to_names(tmp_path, meili):
    path = write_tiny_bundle(tmp_path)[0]
    rec = Recommender(path, "cpu")
    try:
        names_only = rec.find_songs("nightcall")
        rec.song_search, rec.cache = SongSearch(meili.url, rec.version, V, cooldown=0), DictCache()
        meili.hits = [4, 1]
        assert [(s["track"], s["artist"]) for s in rec.find_songs("nite call")] == [
            (TRACKS[4], ARTISTS[ARTIST_OF[4]]), (TRACKS[1], ARTISTS[ARTIST_OF[1]])]
        assert len(rec.cache.store) == 1                       # the search's answer is cached
        meili.mode = "error"
        assert rec.find_songs("nightcall") == names_only       # the name search, as without one
        assert len(rec.cache.store) == 1                       # a fallback answer isn't cached
    finally:
        rec.close()
