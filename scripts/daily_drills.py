"""Part D's failure drills for the daily updates (recsys/daily_job.py): steady traffic logged request by request
while the drills run, a mark at each drill's start and end, a summary of each drill's stretch of the traffic,
personalized requests that reach an update's new songs, and bad updates for the switch's checks to stop.

    # steady traffic straight to the front desk, every answer computed (X-Recsys-Fresh), until the stop file is
    # there: from inside the compose network, where nginx can't remove the header. <dir>: a scratch folder (the
    # log has a line per request, too big to keep)
    docker compose run --rm --no-deps -v ./scripts:/app/scripts -v <dir>:/out daily \\
        python scripts/daily_drills.py traffic --url http://frontdesk:8000 --cases /bundle/daily/cases.json \\
        --out /out/traffic.jsonl --stop /out/stop
    python scripts/daily_drills.py mark <dir>/marks.jsonl "undo: start"   # then "undo: end"
    python scripts/daily_drills.py summary --traffic <dir>/traffic.jsonl --marks <dir>/marks.jsonl --out <dir>/drills.json
    # in the same kind of container (they read /bundle):
    python scripts/daily_drills.py personal --url http://frontdesk:8000 --bundle /bundle   # profiles/owner.npz
    python scripts/daily_drills.py bad --bundle /bundle --break counts   # or vector; then daily_job switch to it

Every answer in the traffic is computed, so one that mixed two versions would differ from the other answers to
the same case from the same version: the summary counts those cases. Imports numpy and FAISS only, never PyTorch.
"""
import argparse
import hashlib
import json
import math
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys import switch, updates  # noqa: E402
from recsys.daily_job import SliceTable, daily_dir  # noqa: E402
from recsys.pair_table import AB  # noqa: E402
from recsys.shard_slices import PER_SONG  # noqa: E402
from recsys.taste import load_profile  # noqa: E402
from recsys.updates import FIELDS, added_songs, write_update  # noqa: E402

SLOW_MS = 500.0


def ask(url, songs, profile=None, timeout=10.0):
    """(HTTP status, the answer or None, milliseconds) for songs from the front desk, computed despite the cache."""
    body = json.dumps({"songs": songs, **({"profile": profile} if profile else {})}).encode()
    request = urllib.request.Request(f"{url}/recommend", data=body,
                                     headers={"content-type": "application/json", "X-Recsys-Fresh": "1"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as r:
            answer, code = json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        answer, code = None, e.code
    except (urllib.error.URLError, OSError):
        answer, code = None, 0
    return code, answer, (time.perf_counter() - start) * 1000


def read_cases(path):
    cases = json.loads(Path(path).read_text())
    return cases["cases"] if isinstance(cases, dict) else cases


def traffic(url, cases, out, stop, threads, pause, seconds):
    """Threads asking for the cases in turn until the stop file is there (or `seconds` have passed), one line per
    request in out: when it was sent (Unix seconds), how long it took, its status, and the answer's version, whether
    it was partial (and which shards it missed) and its top 10's fingerprint."""
    cases, url, lock, end = read_cases(cases), url.rstrip("/"), threading.Lock(), time.time() + seconds

    def loop(k, f):
        i = k
        while not Path(stop).exists() and time.time() < end:
            case, sent = i % len(cases), time.time()
            code, answer, ms = ask(url, cases[case]["songs"])
            row = {"t": round(sent, 3), "ms": round(ms, 1), "code": code, "case": case}
            if answer:  # the top 10 as a fingerprint: equal answers, equal fingerprints
                top = hashlib.blake2b(" ".join(r["uri"] for r in answer["recommendations"]).encode(), digest_size=6)
                row.update(version=answer["version"], partial=answer["partial"], missing=answer.get("missing_shards"),
                           top=top.hexdigest())
            with lock:
                f.write(json.dumps(row) + "\n")
                f.flush()
            i += threads
            time.sleep(pause)

    with open(out, "a") as f:
        workers = [threading.Thread(target=loop, args=(k, f)) for k in range(threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()


def mark(path, what):
    """Note the time of a drill's step: "<drill>: start", "<drill>: <step>", ..., "<drill>: end"."""
    with open(path, "a") as f:
        f.write(json.dumps({"t": round(time.time(), 3), "what": what}) + "\n")
    print(f"{time.strftime('%H:%M:%S')} {what}", flush=True)


def stretch(rows, start):
    """What the requests in rows (sent from `start` on) got."""
    ok = [r for r in rows if r["code"] == 200]
    tops = {}
    for r in ok:
        if not r["partial"]:  # a partial answer differs anyway: it lacks a shard's songs
            tops.setdefault((r["version"], r["case"]), set()).add(r["top"])
    ms = np.array([r["ms"] for r in rows]) if rows else np.zeros(0)
    pct = lambda q: round(float(np.percentile(ms, q)), 1) if len(ms) else None
    bad = [r for r in rows if r["code"] != 200 or r.get("partial")]
    return {"requests": len(rows), "failed": {str(c): sum(r["code"] == c for r in rows)
                                              for c in sorted({r["code"] for r in rows if r["code"] != 200})},
            "partial": sum(bool(r.get("partial")) for r in ok),
            "versions": {v: sum(r.get("version") == v for r in ok) for v in sorted({r["version"] for r in ok})},
            "cases_answered_two_ways_by_one_version": sum(len(t) > 1 for t in tops.values()),
            "ms": {"p50": pct(50), "p99": pct(99), "max": pct(100)}, "over_500_ms": int((ms > SLOW_MS).sum()),
            "problems": [{"at_s": round(r["t"] - start, 1), "code": r["code"], "missing_shards": r.get("missing")}
                         for r in bad][:30],
            "slow": [{"at_s": round(r["t"] - start, 1), "ms": r["ms"]} for r in rows if r["ms"] > SLOW_MS][:30]}


def summary(traffic_file, marks_file, out):
    """For each drill (its marks "<drill>: start" to "<drill>: end"): its steps' times after its start, and what the
    traffic sent in that stretch got (stretch); also all the traffic together."""
    rows = [json.loads(line) for line in open(traffic_file)]
    drills = {}
    for m in (json.loads(line) for line in open(marks_file)):
        drill, _, step = m["what"].partition(": ")
        drills.setdefault(drill, {})[step] = m["t"]
    report = {"all": stretch(rows, rows[0]["t"] if rows else 0.0)}
    for drill, steps in drills.items():
        start, end = steps.get("start", -math.inf), steps.get("end", math.inf)
        report[drill] = {"steps_s": {s: round(t - start, 1) for s, t in steps.items()},
                         **stretch([r for r in rows if start <= r["t"] < end], start)}
    Path(out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({d: {k: v for k, v in r.items() if k not in ("problems", "slow")} for d, r in report.items()},
                     indent=2))


def personal(url, bundle, profile_file, n):
    """n requests personalized by the profile in profile_file (its key, never printed: it's the owner's link), half
    of them seeded by a new song of the live update (after an old song), half by the checks' cases: each must be
    answered (200, not partial, personalized); also how many answers hold a new song."""
    bundle, url = Path(bundle), url.rstrip("/")
    profile = load_profile(profile_file).key
    live = switch.read(bundle / "switch.json")["live"]
    if live == switch.BASE:
        raise SystemExit("the base is live: no new songs to reach")
    new = added_songs(bundle / live, bundle)["uris"]
    cases = read_cases(daily_dir(bundle) / "cases.json")
    rng = np.random.default_rng(0)
    picked = rng.choice(len(new), size=n // 2, replace=False)
    asks = ([("new song", [cases[k]["songs"][-1], new[j]]) for k, j in enumerate(picked)]
            + [("old songs", c["songs"]) for c in cases[:n - n // 2]])
    new, out = set(new), {}
    for seed, songs in asks:
        code, answer, ms = ask(url, songs, profile)
        o = out.setdefault(seed, {"requests": 0, "answered": 0, "partial": 0, "personalized": 0,
                                  "with_a_new_song": 0, "ms": []})
        o["requests"] += 1
        if code == 200:
            uris = {r["uri"] for r in answer["recommendations"]}
            o["answered"] += 1
            o["partial"] += answer["partial"]
            o["personalized"] += answer["personalized"]
            o["with_a_new_song"] += bool(uris & new)
        o["ms"].append(ms)
    for o in out.values():
        o["ms"] = {"p50": round(float(np.percentile(o["ms"], 50)), 1), "max": round(max(o["ms"]), 1)}
    print(json.dumps({"live": live, "profile": Path(profile_file).name, **out}, indent=2))


def bad_update(bundle, how, copies=20):
    """An update on top of the live version, as a bug the build's own checks missed could have made it, for the
    switch's checks to stop: "counts" adds no songs but every appeared-together count is zero; "vector" adds
    copies of the most played songs, one of them with its search vector 3x as long as any song's. Returns its
    folder, relative to the bundle."""
    bundle = Path(bundle)
    live = switch.read(bundle / "switch.json")["live"]
    parent = bundle / live
    settings = json.loads((bundle / "settings.json").read_text())
    base_songs, before = settings["songs"], json.loads((parent / "settings.json").read_text())["songs"]
    listens = np.load(parent / "song_playlists.npy")
    tables = {name: SliceTable(bundle, name, base_songs) for name in PER_SONG}
    vectors_of = np.load(daily_dir(bundle) / "search_vectors.npy", mmap_mode="r")
    source = np.argsort(-listens[:base_songs], kind="stable")[:copies if how == "vector" else 0]
    ids = np.arange(before, before + len(source), dtype=np.int64)
    rows = {name: tables[name][source] for name in PER_SONG}
    names = json.loads((bundle / "songs.json").read_text())["tracks"]
    albums = json.loads((bundle / "song_albums.json").read_text())
    added = {"uris": [f"fake:track:{j}" for j in ids], "tracks": [f"{names[s]} (drill copy)" for s in source],
             "artist": rows["song_artist"].tolist(), "albums": [albums[s] for s in source],
             "keys": [f"fake|{j}" for j in ids], "album_ids": rows["song_album"].tolist(),
             "listeners": [0] * len(source), "joined": [0.0] * len(source)}
    assert set(added) == set(FIELDS)
    vectors = np.array(vectors_of[source], dtype=np.float32)
    if how == "vector":
        step = 1 << 18
        longest = max(float(np.linalg.norm(np.asarray(vectors_of[i:i + step, :-1], dtype=np.float32), axis=1).max())
                      for i in range(0, len(vectors_of), step))
        vectors[0, :-1] *= 3 * longest / np.linalg.norm(vectors[0, :-1])
    a, b = np.concatenate([source, ids]), np.concatenate([ids, source])
    order = np.argsort(a * AB + b)
    pairs = ((a * AB + b)[order], np.full(len(a), 2, dtype=np.int64))
    listens = np.concatenate([listens, listens[source] // 2])
    grow = updates.grow
    if how == "counts":
        def zeroed(src, out, *args):  # the pair table grown as always, then its counts zeroed: the bug
            grow(src, out, *args)
            for name in ("pair_small.npy", "pair_big_counts.npy"):
                np.save(out / name, np.zeros_like(np.load(out / name)))
        updates.grow = zeroed
    try:
        folder = write_update(bundle, parent, added, rows, vectors, listens, pairs, note=f"drill: bad {how}",
                              info={"drill": how})
    finally:
        updates.grow = grow
    rel = str(folder.relative_to(bundle))
    print(rel)
    return rel


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("traffic")
    t.add_argument("--url", default="http://frontdesk:8000")
    t.add_argument("--cases", required=True, help="the checks' cases.json (<bundle>/daily/cases.json)")
    t.add_argument("--out", required=True)
    t.add_argument("--stop", required=True, help="the traffic ends once this file is there")
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--pause", type=float, default=0.0, help="seconds each thread waits between requests")
    t.add_argument("--seconds", type=float, default=4 * 3600.0, help="the traffic ends after this long anyway")
    m = sub.add_parser("mark")
    m.add_argument("file")
    m.add_argument("what")
    s = sub.add_parser("summary")
    s.add_argument("--traffic", required=True)
    s.add_argument("--marks", required=True)
    s.add_argument("--out", required=True)
    p = sub.add_parser("personal")
    p.add_argument("--url", default="http://frontdesk:8000")
    p.add_argument("--bundle", required=True)
    p.add_argument("--profile-file", help="a taste profile (default <bundle>/profiles/owner.npz)")
    p.add_argument("-n", type=int, default=400)
    b = sub.add_parser("bad")
    b.add_argument("--bundle", required=True)
    b.add_argument("--break", dest="how", choices=["counts", "vector"], required=True)
    args = ap.parse_args()
    if args.command == "traffic":
        traffic(args.url, args.cases, args.out, args.stop, args.threads, args.pause, args.seconds)
    elif args.command == "mark":
        mark(args.file, args.what)
    elif args.command == "summary":
        summary(args.traffic, args.marks, args.out)
    elif args.command == "personal":
        personal(args.url, args.bundle, args.profile_file or Path(args.bundle) / "profiles" / "owner.npz", args.n)
    else:
        bad_update(args.bundle, args.how)


if __name__ == "__main__":
    main()
