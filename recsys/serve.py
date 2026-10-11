"""The recommender as a REST API (FastAPI, run by uvicorn).

    python -m recsys.serve --bundle bundles/ranker_1m_any_length-<version> --device cpu
    python -m recsys.serve --bundle ... --shards shard-0:50051,shard-1:50051,...   # songs held by shards

  POST /recommend   {"songs": ["spotify:track:...", "Nightcall - Kavinsky", ...], "name": "road trip"}
                    (songs: Spotify URIs or "Track - Artist" text; name optional)
                    -> {"recommendations": [{"uri", "track", "artist", "score"} x 10],
                        "used": [{"uri", "track", "artist"}], "ignored": [...], "version": ..., "timings_ms": {...}}
                    400 if no songs are given or none of them is in the catalog
  GET  /songs?q=nightc&limit=10
                    -> {"songs": [{"uri", "track", "artist", "playlists"}]}: songs whose track or artist name
                       starts with q (or "track - artist"), in the most playlists first, for a type-ahead menu
  GET  /health      -> {"status": "ok", "version", "songs", "device"}
  GET  /docs        FastAPI's page for trying requests in a browser

FastAPI turns each request's JSON into a RecommendRequest (refusing malformed ones) and the returned
dict back into JSON; uvicorn listens on the port and hands requests to it. The work is
recsys.recommender.Recommender (everything in this process and its search process) or, with --shards,
recsys.frontdesk.ShardedRecommender (per-song data in the shard servers), loaded once when the server starts. Endpoints are plain functions,
which FastAPI runs in worker threads; the Recommender answers one request at a time.
"""
import argparse
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from recsys.recommender import NoKnownSongs, Recommender


class RecommendRequest(BaseModel):
    songs: list[str]          # Spotify track URIs or "Track - Artist" text, oldest first; the last 10 known are read
    name: str | None = None   # the playlist's name, if it has one


def create_app(bundle, device="cpu", shards=None, deadline=2.0):
    """The FastAPI app serving bundle (a recsys.export folder) with the models on device; with shards
    (their addresses, shard i at position i), the songs' data comes from the shard servers."""

    @asynccontextmanager
    async def lifespan(app):  # load once at startup, close the search process or connections at shutdown
        if shards:
            from recsys.frontdesk import ShardedRecommender
            app.state.recommender = ShardedRecommender(bundle, shards, device, deadline)
        else:
            app.state.recommender = Recommender(bundle, device)
        yield
        app.state.recommender.close()

    app = FastAPI(title="Next-song recommender", lifespan=lifespan)

    @app.post("/recommend")
    def recommend(request: RecommendRequest):
        if not request.songs:
            raise HTTPException(400, {"error": "no songs given: send at least one Spotify track URI"})
        try:
            return app.state.recommender.recommend(request.songs, request.name)
        except NoKnownSongs as e:
            raise HTTPException(400, {"error": str(e), "ignored": e.ignored})
        except RuntimeError as e:  # frontdesk.ShardsUnavailable: too few shards answered
            if type(e).__name__ != "ShardsUnavailable":
                raise
            raise HTTPException(503, {"error": str(e)})

    @app.get("/songs")
    def songs(q: str, limit: int = Query(10, ge=1, le=50)):
        return {"songs": app.state.recommender.find_songs(q, limit)}

    @app.get("/health")
    def health():
        rec = app.state.recommender
        shards = rec.shard_states() if hasattr(rec, "shard_states") else None
        status = "ok" if shards is None or all(s["up"] for s in shards) else "degraded"
        return {"status": status, "version": rec.version, "songs": len(rec.uris), "device": str(rec.device),
                "shards": shards}

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
    args = ap.parse_args()
    shards = args.shards.split(",") if args.shards else None
    uvicorn.run(create_app(args.bundle or latest_bundle(), args.device, shards, args.deadline), host=args.host,
                port=args.port)
