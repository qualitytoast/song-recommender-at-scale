import time

import numpy as np
import pytest
import redis

from recsys.cache import Cache
from recsys.recommender import Recommender
from serving_helpers import write_tiny_bundle


class FakeRedis:
    """The two Redis calls the cache makes, on a dict; or failing, like an unreachable Redis."""

    def __init__(self, broken=False):
        self.store, self.calls, self.broken = {}, 0, broken

    def get(self, key):
        self.calls += 1
        if self.broken:
            raise redis.ConnectionError("down")
        return self.store.get(key)

    def set(self, key, value, ex=None):
        self.calls += 1
        if self.broken:
            raise redis.ConnectionError("down")
        self.store[key] = value.encode()


@pytest.fixture(scope="module")
def rec(tmp_path_factory):
    path, *_ = write_tiny_bundle(tmp_path_factory.mktemp("bundles"))
    rec = Recommender(path, "cpu")
    yield rec
    rec.close()


def test_repeated_requests_come_from_the_cache_keyed_by_version(rec):
    fake = FakeRedis()
    rec.cache = Cache(None, client=fake)
    try:
        first = rec.recommend(["Nightcall - Kavinsky", "spotify:track:9"], "road trip")
        again = rec.recommend(["spotify:track:0", "spotify:track:9"], "Road Trip")  # the same songs and words
        assert first["cached"] is False and again["cached"] is True
        assert again["recommendations"] == first["recommendations"] and again["used"] == first["used"]
        assert rec.recommend(["spotify:track:0", "spotify:track:9"], "gym")["cached"] is False  # another name
        assert all(k.startswith((f"rec:{rec.version}:", f"songs:{rec.version}:")) for k in fake.store)
        menu = rec.find_songs("night")
        assert rec.find_songs("NIGHT") == menu and rec.cache.report()["hits"] == 2
    finally:
        rec.cache = None


def test_a_partial_answer_is_not_cached(rec, monkeypatch):
    fake = FakeRedis()
    rec.cache = Cache(None, client=fake)
    real = rec.recommend_ids
    monkeypatch.setattr(rec, "recommend_ids",
                        lambda *a: (*real(*a)[:3], {"partial": True, "missing_shards": [1]}))
    try:
        out = rec.recommend(["spotify:track:5"])
        assert out["partial"] is True and not any(k.startswith("rec:") for k in fake.store)
    finally:
        rec.cache = None


def test_a_broken_redis_never_fails_a_request_and_is_skipped_for_a_while(rec):
    fake = FakeRedis(broken=True)
    rec.cache = Cache(None, client=fake, cooldown=60)
    try:
        expected = rec.compute([("x", 5)], rec.encode_name(""))[1]
        out = rec.recommend(["spotify:track:5"])
        assert out["cached"] is False and out["recommendations"] == expected
        calls = fake.calls
        rec.recommend(["spotify:track:6"])
        rec.find_songs("night")
        assert fake.calls == calls  # skipped during the cooldown: no time spent on a down Redis
        assert rec.cache.report()["errors"] == 1
    finally:
        rec.cache = None


def test_an_unreachable_redis_costs_one_short_timeout():
    cache = Cache("redis://127.0.0.1:1/0", timeout=0.05)
    start = time.perf_counter()
    assert cache.get("k") is None and cache.get("k") is None
    cache.set("k", [1])
    assert time.perf_counter() - start < 0.5 and cache.report()["errors"] == 1
