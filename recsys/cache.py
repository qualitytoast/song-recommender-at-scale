"""The shared answer cache (Redis) for every front-desk copy (stage 6).

A front desk asks the cache before computing: the menu's answers (by version, limit and the typed text
as compared) and recommendations (by version, the request's resolved song IDs and name-word IDs, so a
song given by URI or by "Track - Artist" text is the same entry). The version is in every key, so after
an update old answers are never served; entries also expire after `ttl` seconds. Partial answers (a
shard was missing) are never stored.

Redis must never make a request fail: every call has a short timeout (100 ms; Redis answers in well
under 1 ms, so it only matters when Redis or the machine is in trouble), any Redis error counts as a miss
(the answer is computed as usual), and after an error the cache is skipped for `cooldown` seconds, so a
down or slow Redis costs one timeout, not one per request.
"""
import json
import time

import redis


class Cache:
    def __init__(self, url, ttl=3600, timeout=0.1, cooldown=1.0, client=None):
        self.client = client or redis.Redis.from_url(url, socket_timeout=timeout, socket_connect_timeout=timeout)
        self.ttl, self.cooldown = ttl, cooldown
        self.skip_until = 0.0
        self.stats = {"hits": 0, "misses": 0, "errors": 0}

    def _failed(self):
        self.stats["errors"] += 1
        self.skip_until = time.monotonic() + self.cooldown

    def get(self, key):
        """The stored value, or None (not stored, or Redis unavailable)."""
        if time.monotonic() < self.skip_until:
            return None
        try:
            value = self.client.get(key)
        except redis.RedisError:
            self._failed()
            return None
        self.stats["hits" if value is not None else "misses"] += 1
        return None if value is None else json.loads(value)

    def set(self, key, value):
        if time.monotonic() < self.skip_until:
            return
        try:
            self.client.set(key, json.dumps(value), ex=self.ttl)
        except redis.RedisError:
            self._failed()

    def report(self):
        asked = self.stats["hits"] + self.stats["misses"]
        return {**self.stats, "hit_rate": round(self.stats["hits"] / asked, 4) if asked else None}
