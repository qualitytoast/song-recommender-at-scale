"""The recommender as a REST API (FastAPI, run by uvicorn).

    python -m recsys.serve --bundle bundles/ranker_1m_any_length-<version> --device cpu
    python -m recsys.serve --bundle ... --shards shard-0:50051,shard-1:50051,...   # songs held by shards

  POST /recommend   {"songs": ["spotify:track:...", "Nightcall - Kavinsky", ...], "name": "road trip"}
                    (songs: Spotify URIs or "Track - Artist" text; name optional; "max_per_artist": at most
                    this many of the 10 by one artist, default 2, 0 for no limit; "mix": how much the
                    model's guesses after earlier songs count, 0 to 1, default 0.5, 0 for the last song's alone)
                    -> {"recommendations": [{"uri", "track", "artist", "score"} x 10],
                        "used": [{"uri", "track", "artist"}], "ignored": [...], "version": ..., "timings_ms": {...}}
                    400 if no songs are given or none of them is in the catalog
  GET  /songs?q=nightc&limit=10
                    -> {"songs": [{"uri", "track", "artist", "playlists"}]}: songs whose track or artist name
                       starts with q (or "track - artist"), in the most playlists first, for a type-ahead menu
  GET  /health      -> {"status": "ok" or "degraded", "version", "songs", "device", "shards", "cache", ...}
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
import time
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from recsys.admission import Admission, Overloaded
from recsys.recommender import MAX_PER_ARTIST, MIX, TOP, NoKnownSongs, Recommender

WEB = Path(__file__).resolve().parents[1] / "web"


class RecommendRequest(BaseModel):
    songs: list[str]          # Spotify track URIs or "Track - Artist" text, oldest first; the last 10 known are read
    name: str | None = None   # the playlist's name, if it has one
    max_per_artist: int = Field(MAX_PER_ARTIST, ge=0, le=TOP)  # at most this many of the 10 by one artist (0: any)
    mix: float = Field(MIX, ge=0.0, le=1.0)  # how much the guesses after earlier songs count (recommender.MIX)


def busy(message="busy: too many requests in progress; try again"):
    return HTTPException(503, {"error": message}, headers={"Retry-After": "1"})


def create_app(bundle, device="cpu", shards=None, deadline=2.0, redis_url=None, batch=0, batch_wait_ms=2.0,
               max_pending=64, max_wait_ms=1000.0, max_pending_menu=512):
    """The FastAPI app serving bundle (a recsys.export folder) with the models on device; with shards
    (their addresses, shard i at position i), the songs' data comes from the shard servers; with
    redis_url, answers are shared through a Redis cache (recsys.cache); with batch > 1 (sharded only),
    requests arriving within batch_wait_ms are answered together, up to batch at a time (recsys.batching).
    max_pending, max_wait_ms: overload limits for recommendations; max_pending_menu: for /songs."""
    admission, menu_admission = Admission(max_pending, max_wait_ms), Admission(max_pending_menu)

    @asynccontextmanager
    async def lifespan(app):  # load once at startup, close the search process or connections at shutdown
        if shards:
            from recsys.frontdesk import ShardedRecommender
            app.state.recommender = ShardedRecommender(bundle, shards, device, deadline)
            if batch > 1:
                from recsys.batching import Batcher
                rec = app.state.recommender
                app.state.recommender.batcher = Batcher(rec.recommend_batch, batch, batch_wait_ms)
        else:
            app.state.recommender = Recommender(bundle, device)
        if redis_url:
            from recsys.cache import Cache
            app.state.recommender.cache = Cache(redis_url)
        app.state.recommender.admission = admission
        # Plain (def) endpoints run in worker threads: enough for every admitted request, so a request
        # never waits for a thread instead of for its answer.
        anyio.to_thread.current_default_thread_limiter().total_tokens = max_pending + max_pending_menu + 16
        yield
        app.state.recommender.close()

    app = FastAPI(title="Next-song recommender", lifespan=lifespan)

    @app.post("/recommend")
    def recommend(request: RecommendRequest):
        arrived = time.monotonic()
        if not request.songs:
            raise HTTPException(400, {"error": "no songs given: send at least one Spotify track URI"})
        if not admission.enter():
            raise busy()
        try:
            return app.state.recommender.recommend(request.songs, request.name, arrived, request.max_per_artist,
                                                   request.mix)
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
        return {"status": status, "version": rec.version, "songs": len(rec.uris), "device": str(rec.device),
                "shards": shards, "cache": rec.cache.report() if rec.cache else None,
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
    ap.add_argument("--batch", type=int, default=0, help="answer up to this many requests together (sharded)")
    ap.add_argument("--batch-wait-ms", type=float, default=2.0, help="how long a request waits for company")
    ap.add_argument("--max-pending", type=int, default=64, help="recommendation requests admitted at once")
    ap.add_argument("--max-wait-ms", type=float, default=1000.0, help="a request not started by then gets 503")
    args = ap.parse_args()
    shards = args.shards.split(",") if args.shards else None
    uvicorn.run(create_app(args.bundle or latest_bundle(), args.device, shards, args.deadline, args.redis, args.batch,
                           args.batch_wait_ms, args.max_pending, args.max_wait_ms), host=args.host, port=args.port)
