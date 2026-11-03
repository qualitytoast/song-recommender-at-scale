"""The recommender as a REST API (FastAPI, run by uvicorn).

    python -m recsys.serve --bundle bundles/ranker_1m_any_length-<version> --device cpu
    python -m recsys.serve --bundle ... --shards shard-0:50051,shard-1:50051,...   # songs held by shards
    python -m recsys.serve --bundle ... --shards ... --switch <bundle>/switch.json  # versions switched live

With --switch the version served is the one the bundle's switch file names (recsys/switch.py): the bundle itself
or an update of it (recsys/updates.py), changed without a restart. A request with the X-Recsys-Standby header
(nginx removes it from the public's requests) is answered by the standby version instead, to check it first;
one with X-Recsys-Fresh is computed even if the answer cache has it.

  POST /recommend   {"songs": ["spotify:track:...", "Nightcall - Kavinsky", ...], "name": "road trip"}
                    (songs: Spotify URIs or "Track - Artist" text; name optional; "max_per_artist": at most
                    this many of the 10 by one artist, 0 for no limit; "mix": how much the model's guesses
                    after earlier songs count, 0 to 1, 0 for the last song's alone; both default to the
                    bundle's own, see /health; "profile": a taste profile's key (--profiles) for an answer
                    nudged toward that listener's taste, recsys.taste; an unknown key gets the usual answer)
                    -> {"recommendations": [{"uri", "track", "artist", "score"} x 10],
                        "used": [{"uri", "track", "artist"}], "ignored": [...], "version": ..., "timings_ms": {...}}
                    400 if no songs are given or none of them is in the catalog
  GET  /songs?q=nightc&limit=10
                    -> {"songs": [{"uri", "track", "artist", "listens" or "playlists"}]}: for a type-ahead
                       menu, with --search the typo-tolerant matches of q's words in any order across track,
                       artist and album (recsys.song_search); without it, or when the search can't answer,
                       songs whose track or artist name starts with q (or "track - artist"); most played
                       first (the bundle's count: listens for a ListenBrainz model, playlists for an MPD one)
  GET  /health      -> {"status": "ok" or "degraded", "version", "base_version", "songs", "switch", "defaults",
                        "device", "shards", ...}
  GET  /live        -> {"ok": true}: the container health check; answered on the event loop, so it never
                       waits behind busy requests (a busy front desk mustn't look frozen and be restarted)
  GET  /            the webpage (web/)
  GET  /docs        FastAPI's page for trying requests in a browser

Overload (recsys.admission): at most max_pending recommendation requests per front desk at once, and none
started after waiting max_wait_ms; the rest get 503 with Retry-After: 1 right away.

FastAPI turns each request's JSON into a RecommendRequest (refusing malformed ones) and the returned
dict back into JSON; uvicorn listens on the port and hands requests to it. The work is
recsys.recommender.Recommender (everything in this process and its search process) or, with --shards,
recsys.frontdesk.ShardedRecommender (per-song data in the shard servers), loaded once when the server starts. Endpoints are plain functions,
which FastAPI runs in worker threads; the Recommender answers one request at a time.
"""
import argparse
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from recsys import switch
from recsys.admission import Admission, Overloaded
from recsys.recommender import TOP, NoKnownSongs, Recommender, version_of

WEB = Path(__file__).resolve().parents[1] / "web"


class RecommendRequest(BaseModel):
    songs: list[str]          # Spotify track URIs or "Track - Artist" text, oldest first; the last 10 known are read
    name: str | None = None   # the playlist's name, if it has one
    # None: the bundle's default (settings.json; recommender.MAX_PER_ARTIST and MIX if it has none)
    max_per_artist: int | None = Field(None, ge=0, le=TOP)  # at most this many of the 10 by one artist (0: any)
    mix: float | None = Field(None, ge=0.0, le=1.0)  # how much the guesses after earlier songs count
    profile: str | None = Field(None, max_length=128)  # a taste profile's key: personalized (recsys.taste)


def busy(message="busy: too many requests in progress; try again"):
    return HTTPException(503, {"error": message}, headers={"Retry-After": "1"})


class Switcher:
    """The front desk's side of switching versions (recsys/switch.py): a recommender for each version the switch
    file names, the live one answering requests and the standby built and ready, following the file every second.
    A standby is built only once every shard holds it, sharing the live one's parsed names. The switch itself is
    one reference swapped (app.state.recommender): a request already running keeps the one it started with. A
    recommender the file stops naming is closed `grace` seconds later, when nothing can still be using it."""

    def __init__(self, app, bundle, path, build, grace=30.0, retry=60.0):
        self.app, self.bundle, self.path, self.build = app, Path(bundle), Path(path), build
        self.grace, self.retry = grace, retry
        self.built, self.retiring, self.failed = {}, [], {}  # folder -> recommender; (when, recommender); folder -> when
        self.error = None
        live = switch.read(self.path)["live"]
        self.built[live] = app.state.recommender = build(self.update_of(live), None)  # at startup: the live one first
        app.state.standby = None

    def update_of(self, folder):
        return None if folder == switch.BASE else self.path.parent / folder

    def start(self):
        switch.follow(self.path, self.step, log=self.log)

    def log(self, message):
        self.error = message
        switch.say(message)

    def step(self, state):
        live = self.app.state.recommender
        now = time.monotonic()
        problems = []  # raised at the end, so one bad folder doesn't stop the rest of the step
        for folder in switch.wanted(state):
            if folder in self.built or now < self.failed.get(folder, 0.0):
                continue
            update = self.update_of(folder)
            try:
                if not live.shards_hold(version_of(update or self.bundle)):
                    continue  # the shards are still loading it
                self.built[folder] = self.build(update, live)
            except Exception as e:  # tried again `retry` seconds later
                self.failed[folder] = now + self.retry
                problems.append(f"couldn't build {folder}: {type(e).__name__}: {e}")
                continue
            switch.say(f"front desk: version {self.built[folder].version} ({folder}) ready")
        target = self.built.get(state["live"])
        if target is not None and target is not live:
            # checked again at the switch, not only when built: a shard restarted since then loads it later
            if target.shards_hold(target.version):
                self.app.state.recommender = target  # the switch: every new request uses this one
                switch.say(f"front desk: now serving version {target.version} ({state['live']})")
            else:
                problems.append(f"not switching to {state['live']} yet: a shard doesn't hold it")
        self.app.state.standby = self.built.get(state["standby"]) if state["standby"] else None
        current = self.app.state.recommender
        for folder in [f for f, r in self.built.items() if f not in switch.wanted(state) and r is not current]:
            self.retiring.append((now + self.grace, self.built.pop(folder)))
        for when, rec in [r for r in self.retiring if r[0] <= now]:
            rec.close()
            switch.say(f"front desk: closed version {rec.version}")
        self.retiring = [r for r in self.retiring if r[0] > now]
        if problems:
            raise RuntimeError("front desk: " + "; ".join(problems))

    def recommenders(self):
        return list(self.built.values()) + [rec for _, rec in self.retiring]

    def report(self):
        folder_of = {id(r): f for f, r in self.built.items()}
        live, standby, now = self.app.state.recommender, self.app.state.standby, time.monotonic()
        return {"live": folder_of.get(id(live)),
                "standby": {"folder": folder_of.get(id(standby)), "version": standby.version} if standby else None,
                "retiring": [rec.version for _, rec in self.retiring],
                "failed": {f: round(when - now, 1) for f, when in self.failed.items() if when > now},
                "last_error": self.error}


def create_app(bundle, device="cpu", shards=None, deadline=2.0, redis_url=None, batch=0, batch_wait_ms=2.0,
               max_pending=64, max_wait_ms=1000.0, max_pending_menu=512, search_url=None, profiles=None,
               switch_file=None):
    """The FastAPI app serving bundle (a recsys.export folder) with the models on device; with shards
    (their addresses, shard i at position i), the songs' data comes from the shard servers; with
    redis_url, answers are shared through a Redis cache (recsys.cache); with batch > 1 (sharded only),
    requests arriving within batch_wait_ms are answered together, up to batch at a time (recsys.batching);
    with search_url, the menu searches with Meilisearch there (recsys.song_search; its key from the
    MEILI_MASTER_KEY environment variable); with profiles (a folder), the taste profiles in it built for this
    bundle can be named by requests (recsys.taste); with switch_file (sharded only), the version served is the
    one it names (recsys/switch.py), switched without a restart (Switcher). max_pending, max_wait_ms: overload
    limits for recommendations; max_pending_menu: for /songs."""
    admission, menu_admission = Admission(max_pending, max_wait_ms), Admission(max_pending_menu)
    if switch_file and not shards:
        raise ValueError("switching versions needs shards: the one-process recommender loads one version")

    def build(update=None, share=None):
        """A recommender for one version of the bundle (update: an update's folder; None: the bundle itself)."""
        if shards:
            from recsys.frontdesk import ShardedRecommender
            rec = ShardedRecommender(bundle, shards, device, deadline, update=update, share=share)
            if batch > 1:
                from recsys.batching import Batcher
                rec.batcher = Batcher(rec.recommend_batch, batch, batch_wait_ms)
        else:
            rec = Recommender(bundle, device)
        try:
            rec.cache = app.state.cache
            if search_url:
                from recsys.song_search import SongSearch
                rec.song_search = SongSearch(search_url, rec.base_version, len(rec.uris),
                                             os.environ.get("MEILI_MASTER_KEY"))
            if profiles:
                rec.load_profiles(profiles)
            rec.admission = admission
        except Exception:
            rec.close()  # its connections and batching thread, or a failed build tried again leaks them
            raise
        return rec

    @asynccontextmanager
    async def lifespan(app):  # load once at startup, close the search process or connections at shutdown
        if redis_url:
            from recsys.cache import Cache
            app.state.cache = Cache(redis_url)
        else:
            app.state.cache = None
        if switch_file:
            app.state.switcher = Switcher(app, bundle, switch_file, build)
            app.state.switcher.start()
        else:
            app.state.switcher, app.state.standby = None, None
            app.state.recommender = build()
        # Plain (def) endpoints run in worker threads: enough for every admitted request, so a request
        # never waits for a thread instead of for its answer.
        anyio.to_thread.current_default_thread_limiter().total_tokens = max_pending + max_pending_menu + 16
        yield
        for rec in app.state.switcher.recommenders() if app.state.switcher else [app.state.recommender]:
            rec.close()

    app = FastAPI(title="Next-song recommender", lifespan=lifespan)

    @app.post("/recommend")
    def recommend(request: RecommendRequest, x_recsys_standby: str | None = Header(None),
                  x_recsys_fresh: str | None = Header(None)):
        """Two headers for the daily job's checks (recsys/daily_job.py), which nginx removes from the public's
        requests: X-Recsys-Standby answers with the standby version, to check it before switching to it (409 if there
        is none); X-Recsys-Fresh computes the answer even if the cache has it, so the check reaches the shards."""
        arrived = time.monotonic()
        if not request.songs:
            raise HTTPException(400, {"error": "no songs given: send at least one Spotify track URI"})
        rec = app.state.standby if x_recsys_standby else app.state.recommender
        if rec is None:
            raise HTTPException(409, {"error": "no standby version is loaded"})
        if not admission.enter():
            raise busy()
        try:
            return rec.recommend(request.songs, request.name, arrived, request.max_per_artist, request.mix,
                                 request.profile, fresh=bool(x_recsys_fresh))
        except NoKnownSongs as e:
            raise HTTPException(400, {"error": str(e), "ignored": e.ignored})
        except Overloaded as e:
            raise busy(str(e))
        except RuntimeError as e:  # frontdesk.ShardsUnavailable: too few shards answered
            if type(e).__name__ != "ShardsUnavailable":
                raise
            raise HTTPException(503, {"error": str(e)})
        finally:
            admission.leave()

    @app.get("/songs")
    def songs(q: str, limit: int = Query(10, ge=1, le=50)):
        if not menu_admission.enter():
            raise busy()
        try:
            return {"songs": app.state.recommender.find_songs(q, limit)}
        finally:
            menu_admission.leave()

    @app.get("/live")
    async def live():
        return {"ok": True}

    @app.get("/health")
    def health():
        rec = app.state.recommender
        shards = rec.shard_states() if hasattr(rec, "shard_states") else None
        status = "ok" if shards is None or all(s["up"] for s in shards) else "degraded"
        return {"status": status, "version": rec.version, "base_version": rec.base_version, "songs": len(rec.uris),
                "switch": app.state.switcher.report() if app.state.switcher else None,
                "defaults": {"max_per_artist": rec.max_per_artist, "mix": rec.mix}, "device": str(rec.device),
                "shards": shards, "cache": rec.cache.report() if rec.cache else None,
                "search": rec.song_search.report() if rec.song_search else None, "profiles": len(rec.profiles),
                "batching": rec.batcher.report() if rec.batcher else None, "answers": rec.answers,
                "admission": {"recommend": admission.report(), "songs": menu_admission.report()}}

    if WEB.is_dir():  # the webpage, after the API routes so they take precedence
        app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
    return app


def latest_bundle(root="bundles"):
    """The most recently written bundle folder."""
    bundles = [p for p in Path(root).iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not bundles:
        raise FileNotFoundError(f"no bundles in {root}/: run python -m recsys.export first")
    return max(bundles, key=lambda p: p.stat().st_mtime)


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser(description="Serve recommendations over HTTP.")
    ap.add_argument("--bundle", help="bundle folder (default: the newest in bundles/)")
    ap.add_argument("--device", default="cpu", help="where the models run: cpu or mps")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--shards", help="comma-separated shard addresses (host:port), shard 0 first")
    ap.add_argument("--deadline", type=float, default=2.0, help="seconds each call to a shard may take")
    ap.add_argument("--redis", help="a shared answer cache, e.g. redis://redis:6379/0")
    ap.add_argument("--search", help="Meilisearch for the menu (recsys.song_search), e.g. http://meilisearch:7700")
    ap.add_argument("--profiles", help="a folder of taste profiles (recsys.taste) requests may name")
    ap.add_argument("--batch", type=int, default=0, help="answer up to this many requests together (sharded)")
    ap.add_argument("--batch-wait-ms", type=float, default=2.0, help="how long a request waits for company")
    ap.add_argument("--max-pending", type=int, default=64, help="recommendation requests admitted at once")
    ap.add_argument("--max-wait-ms", type=float, default=1000.0, help="a request not started by then gets 503")
    ap.add_argument("--switch", help="the bundle's switch.json to follow (recsys/switch.py; sharded)")
    args = ap.parse_args()
    shards = args.shards.split(",") if args.shards else None
    uvicorn.run(create_app(args.bundle or latest_bundle(), args.device, shards, args.deadline, args.redis, args.batch,
                           args.batch_wait_ms, args.max_pending, args.max_wait_ms, search_url=args.search,
                           profiles=args.profiles, switch_file=args.switch),
                host=args.host, port=args.port)
