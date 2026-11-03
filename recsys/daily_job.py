"""The daily update: each day's ListenBrainz dump becomes a new version of the served bundle (recsys/updates.py),
switched to without a restart (recsys/switch.py), checked before and after the switch, and switched back at once
if anything fails. Nothing is retrained (recsys/daily.py says what a dump adds, and why).

    # once, on the Mac (it needs the store the pair was trained on):
    python -m recsys.daily_job starter --bundle bundles/<bundle> --store data/lb_store_n3_1.5x \\
        --cases runs/serving/lb_n3_16_cases --data-end 2026-10-15
    # in the `daily` container on the server (docker compose run --rm daily ...):
    python -m recsys.daily_job run --bundle /bundle       # build, then switch: what the timer runs, unless paused
    python -m recsys.daily_job build --bundle /bundle [--dumps <URL or folder>]   # a new update, not switched to
    python -m recsys.daily_job switch --bundle /bundle --update updates/<folder>   # check it, switch, watch
    python -m recsys.daily_job undo --bundle /bundle [--to <folder>]   # back to the parent (or a kept folder); pauses
    python -m recsys.daily_job skip-dump --bundle /bundle <dump>       # never use that dump (step past a bad one)
    python -m recsys.daily_job resume --bundle /bundle                # let the timer run again
    python -m recsys.daily_job status --bundle /bundle

build, on top of the live version (the base or an update):
  1. the dumps not used yet: chained by their windows after the live version's (recsys.daily.chain); for the base,
     after where its training data ends; skip-dump's ones are stepped past, not used
  2. each one's fresh listens as an extract, its checksum checked (recsys.daily.fresh_extract, streamed: the dump
     itself is never stored), then what it adds (recsys.daily.apply_dump): new songs at 3+ listeners (most first,
     2,000 a dump at most), every catalog song's listens, every pair's count
  3. checks before writing anything (a day's dump, not the tail dump seconds after it): fresh listens per day
     within 0.3-3x of the median of the last 14 days, no song over 5% of the day's listens
  4. the new songs' vectors: the lengthened average of their artist's songs (recsys.daily.standins "artist_norm"),
     each finite and no longer than the longest song's
  5. the update (recsys.updates.write_update), with who played songs not yet served (seen.npz: tomorrow's
     new-song rule counts 30 days), then read back: old rows unchanged, old counts never lower
switch, to an update built on the live version:
  1. its new songs into the song search index (recsys.song_search.add_songs)
  2. prepare: every shard loads it, and the front desk builds a recommender for it
  3. checks on the standby, through the real shards (the X-Recsys-Standby header, straight to the front desk:
     nginx removes it): the held-out cases answered by both versions; none may fail or be partial, NDCG@10 at most
     0.003 below live, p99 at most 1.5x live's, no new song in more than 1% of the lists. A failure drops the update
     (finish) and nothing changes for the site.
  4. flip, then 10 minutes of watching (a request every 3 s, /health): two failures in a row flip back at once,
     drop the update and pause the timer
  5. finish (the old version dropped), and old updates and extracts deleted (the last 3 updates kept, for undo)

Files (made by starter, kept in <bundle>/daily/): lookups.npz, artist_of.json, artist_names.json (every song found by
its key's hash, recsys.daily.LeanServed: its keys come from the store it was trained on), search_vectors.npy,
song_album.npy, song_genres.npy (the stand-ins' sources and the new songs' rows; the server's bundle has these only in
its slices), cases.json (the checks' held-out cases), state.json (where the base's data ends, paused or not, dumps to
skip, the last runs), extracts/ (each used dump's fresh listens, the last few kept), job.lock (one job at a time).
Imports numpy and FAISS only, never PyTorch (they can't share a process, see recsys/ann.py).
"""
import argparse
import fcntl
import json
import math
import os
import re
import shutil
import time
import urllib.error
import urllib.request
from calendar import timegm
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from recsys import switch
from recsys.daily import (DAY, MAX_NEW, MIN_USERS, LeanServed, Seen, Update, apply_dump, chain, fresh_extract,
                          lean_files, read_window, standins)
from recsys.pair_table import AB, PairTable
from recsys.shard_slices import PER_SONG
from recsys.updates import added_songs, slice_count, version_of, write_update

DUMPS = "https://data.metabrainz.org/pub/musicbrainz/listenbrainz/incremental/"
DUMP_NAME = re.compile(r"listenbrainz-dump-(\d+)-(\d{8})-(\d{6})-incremental")
MODE = "artist_norm"     # the new songs' vectors (RESULTS.md, "Daily updates": the replay's choice)
HISTORY_DAYS = 14        # the size check compares a day's dump with the median of this many before it
SIZE_RANGE = (0.3, 3.0)  # ... and wants it within this range of that median
MAX_TOP_SHARE = 0.05     # no song may be more than this share of a day's catalog listens
MAX_NDCG_DROP = 0.003    # the standby's NDCG@10 on the held-out cases at most this much below live's
MAX_P99_RATIO = 1.5      # and its p99 at most this many times live's
MAX_NEW_SHARE = 0.01     # no new song in more than this share of the lists
WATCH_S, PROBE_S = 600.0, 3.0  # after the flip: how long to watch, and a request this often
KEEP_UPDATES, KEEP_EXTRACTS = 3, 4
MIN_MEMORY_GB, MIN_DISK_GB = 4.0, 5.0  # a run won't start with less free (a build peaks at ~3 GB; see RESULTS.md)


class Stop(RuntimeError):
    """A check failed: nothing was switched (or it was switched back)."""


def log(message):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def iso(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat(timespec="seconds")


# --- state ---------------------------------------------------------------------------------------------------------

def daily_dir(bundle):
    return Path(bundle) / "daily"


def read_state(bundle):
    return json.loads((daily_dir(bundle) / "state.json").read_text())


def write_state(bundle, state):
    path = daily_dir(bundle) / "state.json"
    temporary = path.with_name(f".state.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    os.replace(temporary, path)


def update_info(bundle, folder):
    """An update's update.json ({} for the base itself)."""
    return {} if folder == switch.BASE else json.loads((Path(bundle) / folder / "update.json").read_text())


def lineage(bundle, folder):
    """folder, its parent, its parent's parent, ... down to the base (not included)."""
    out = []
    while folder != switch.BASE:
        out.append(folder)
        folder = update_info(bundle, folder)["parent_folder"]
    return out


@contextmanager
def one_job(bundle):
    """Hold the job lock (a second job at the same time gets Stop)."""
    with open(daily_dir(bundle) / "job.lock", "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Stop("another daily job is running") from None
        yield


def record(bundle, run):
    state = read_state(bundle)
    state["runs"] = (state.get("runs", []) + [run])[-30:]
    write_state(bundle, state)


def free_memory_gb():
    """MemAvailable (Linux), or None where there's no /proc/meminfo."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1 << 20)
    except OSError:
        return None
    return None


# --- the starter files (once, on the Mac) ------------------------------------------------------------------------

def starter(bundle, store, cases, data_end):
    """<bundle>/daily/ (see the module docstring). store: the ListenBrainz store the pair was trained on."""
    from recsys.lb_store import store_song_keys
    from recsys.mpd_store import load_store
    bundle, out = Path(bundle), daily_dir(bundle)
    out.mkdir(exist_ok=True)
    uris = json.loads((bundle / "songs.json").read_text())["uris"]
    loaded = load_store(store)
    key_of = dict(zip(loaded.track_uris, store_song_keys(loaded)))
    keys = [key_of[u] for u in uris]
    if len(set(keys)) != len(keys):
        raise ValueError("two songs with the same key")
    lean_files(bundle, keys, out)
    shutil.copyfile(bundle / "search_vectors.npy", out / "search_vectors.npy")
    held = json.loads((Path(cases) / "cases.json").read_text())["cases"]
    (out / "cases.json").write_text(json.dumps([{"songs": c["songs"], "target": c["target"]} for c in held]))
    end = timegm(datetime.strptime(data_end, "%Y-%m-%d").timetuple())
    if not (out / "state.json").exists():
        write_state(bundle, {"data_end": end, "paused": False, "paused_why": None, "skip": [], "runs": []})
    log(f"starter files in {out}: {len(keys):,} song keys, {len(held):,} cases; the base's data ends {iso(end)}")


# --- build -----------------------------------------------------------------------------------------------------------

def list_dumps(source):
    """{name: its .tar.zst} for every daily dump at source: ListenBrainz's listing (a URL) or a folder of folders."""
    if str(source).startswith(("http://", "https://")):
        source = source.rstrip("/") + "/"
        with urllib.request.urlopen(source, timeout=60) as r:
            names = sorted(set(DUMP_NAME.findall(r.read().decode())))
        return {f"listenbrainz-dump-{n}-{d}-{t}-incremental":
                f"{source}listenbrainz-dump-{n}-{d}-{t}-incremental/listenbrainz-listens-dump-{n}-{d}-{t}-incremental.tar.zst"
                for n, d, t in names}
    return {d.name: str(next(d.glob("*.tar.zst"))) for d in sorted(Path(source).iterdir())
            if d.is_dir() and DUMP_NAME.fullmatch(d.name) and any(d.glob("*.tar.zst"))}


def named_time(name):
    _, d, t = DUMP_NAME.fullmatch(name).groups()
    return timegm(datetime.strptime(d + t, "%Y%m%d%H%M%S").timetuple())


class SliceTable:
    """A per-song table of the base's songs read from the base's slices as if it were one array (song j is row
    j // N of shard j % N), memory-mapped: only the rows asked for are read."""

    def __init__(self, bundle, name, songs):
        n = slice_count(bundle)
        folders = [Path(bundle) / f"shards-{n}" / f"shard-{i}" for i in range(n)]
        for i, folder in enumerate(folders):
            ids = np.load(folder / "ids.npy", mmap_mode="r")
            if len(ids) != len(range(i, songs, n)) or (len(ids) and (ids[0], ids[-1]) != (i, range(i, songs, n)[-1])):
                raise ValueError(f"{folder}: not the base's songs i, i + N, ...")
        self.n, self.parts = n, [np.load(f / f"{name}.npy", mmap_mode="r") for f in folders]
        self.shape, self.dtype = (songs,) + self.parts[0].shape[1:], self.parts[0].dtype

    def __getitem__(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        out = np.empty((len(ids),) + self.shape[1:], dtype=self.dtype)
        for i, part in enumerate(self.parts):
            at = ids % self.n == i
            out[at] = part[ids[at] // self.n]
        return out


def served_at(bundle, folder):
    """The served songs (recsys.daily.Served) at a version: the base's, plus the ones its updates added."""
    bundle = Path(bundle)
    served = LeanServed(bundle, daily_dir(bundle))
    if folder != switch.BASE:
        served.restore(added_songs(bundle / folder, bundle))
    return served


def history(bundle, folder):
    """Fresh listens per day of the day dumps (windows of 12 h or more) folder and its parents used, oldest first."""
    rates = []
    for f in lineage(bundle, folder):
        for d in reversed(update_info(bundle, f).get("dumps", [])):
            if d["end"] - d["start"] >= DAY / 2:
                rates.append(d["fresh_listens"] / ((d["end"] - d["start"]) / DAY))
        if len(rates) >= HISTORY_DAYS:
            break
    return rates[:HISTORY_DAYS][::-1]


def check_dump(report, rates):
    """Stop if a day's dump looks wrong (see the module docstring); rates: the days before it."""
    days = (report["end"] - report["start"]) / DAY
    if days < 0.5:
        return  # the nearly empty dump seconds after a day's: nothing to compare
    rate = report["fresh_listens"] / days
    if len(rates) >= 3:
        median = float(np.median(rates))
        if not SIZE_RANGE[0] * median <= rate <= SIZE_RANGE[1] * median:
            raise Stop(f"{report['dump']}: {rate:,.0f} fresh listens a day, {rate / median:.2f}x the median of the "
                       f"{len(rates)} days before ({median:,.0f}); allowed {SIZE_RANGE[0]}-{SIZE_RANGE[1]}x")
    if report["top_song_share"] > MAX_TOP_SHARE:
        raise Stop(f"{report['dump']}: one song is {report['top_song_share']:.1%} of the day's catalog listens "
                   f"(at most {MAX_TOP_SHARE:.0%})")


def check_vectors(vectors, reference):
    """Stop unless every new search vector is finite and no longer than the longest of the base's (reference)."""
    if not np.isfinite(vectors).all():
        raise Stop("a new song's vector isn't finite")
    step = 1 << 18
    longest = max(float(np.linalg.norm(np.asarray(reference[i:i + step, :-1], dtype=np.float32), axis=1).max())
                  for i in range(0, len(reference), step))
    lengths = np.linalg.norm(vectors[:, :-1], axis=1)
    if len(lengths) and lengths.max() > longest * 1.0001:
        raise Stop(f"a new song's vector is {lengths.max():.3f} long, longer than any song's ({longest:.3f})")


def verify(parent, folder, songs_before, pairs, sample=20_000, seed=0):
    """Read the written update back: every old row unchanged, the new IDs in order, and old counts (a sample) never
    lower, the update's own counts at least what it added."""
    n = slice_count(folder)
    rng = np.random.default_rng(seed)
    for i in range(n):
        old, new = parent / f"shards-{n}" / f"shard-{i}", folder / f"shards-{n}" / f"shard-{i}"
        old_ids, new_ids = np.load(old / "ids.npy"), np.load(new / "ids.npy")
        if not (np.array_equal(new_ids[:len(old_ids)], old_ids) and (np.diff(new_ids) > 0).all()
                and (new_ids[len(old_ids):] % n == i).all() and (new_ids[len(old_ids):] >= songs_before).all()):
            raise Stop(f"{new}: song IDs aren't the old ones followed by new ones of this shard")
        for name in PER_SONG:
            if not np.array_equal(np.load(new / f"{name}.npy", mmap_mode="r")[:len(old_ids)],
                                  np.load(old / f"{name}.npy", mmap_mode="r")):
                raise Stop(f"{new}: an old song's {name} row changed")
        old_table = PairTable(old, songs_before)
        new_table = PairTable(new, json.loads((new / "info.json").read_text())["catalog"])
        if len(old_table):
            rows = np.sort(rng.choice(len(old_table), size=min(sample, len(old_table)), replace=False))
            a = np.searchsorted(old_table.starts, rows, side="right") - 1  # each row's earlier song
            b = old_table.later[rows].astype(np.int64)
            if (new_table.lookup(a * new_table.songs + b) < old_table.lookup(a * songs_before + b)).any():
                raise Stop(f"{new}: an old pair's count went down")
    keys, counts = pairs
    if len(keys):
        pick = rng.choice(len(keys), size=min(sample, len(keys)), replace=False)
        a, b = keys[pick] // AB, keys[pick] % AB
        songs = json.loads((folder / "settings.json").read_text())["songs"]
        for i in range(n):
            here = b % n == i
            table = PairTable(folder / f"shards-{n}" / f"shard-{i}", songs)
            if (table.lookup(a[here] * songs + b[here]) < counts[pick][here]).any():
                raise Stop(f"shard {i}: a pair's count is below what this update added")


def build(bundle, source=DUMPS, extracts=None, workers=1, until=None):
    """A new update on top of the live version from the dumps not used yet (see the module docstring); returns its
    folder (relative to the bundle), or None if there was nothing new."""
    bundle = Path(bundle)
    state, sw = read_state(bundle), switch.read(bundle / "switch.json")
    if sw["standby"] is not None:
        raise Stop(f"a switch is under way (standby {sw['standby']}): finish it or undo first")
    parent_folder = sw["live"]
    parent, info = bundle / parent_folder, update_info(bundle, parent_folder)
    after = info.get("chain_end", state["data_end"])
    began = time.perf_counter()
    found = list_dumps(source)
    recent = {n: loc for n, loc in found.items()  # a dump is named for its window's end
              if named_time(n) > after - 2 * DAY and (until is None or named_time(n) <= until + 3600)}
    windows = {n: read_window(loc) for n, loc in recent.items()}
    used, skipped, gaps = chain(windows, after)
    if until is not None:
        used = [n for n in used if windows[n][1] <= until + 3600]
    if not used:
        log(f"nothing new: {len(found)} dumps listed, none after {iso(after)}")
        return None
    log(f"on {parent_folder} ({version_of(parent)}): {len(used)} dumps to use, from {used[0]} to {used[-1]}"
        + (f"; gaps {[(iso(a), iso(b)) for a, b in gaps]}" if gaps else ""))
    served, seen = served_at(bundle, parent_folder), (Seen.load(parent / "seen.npz") if (parent / "seen.npz").exists()
                                                       else Seen())
    before, added_before = len(served), len(served.added)
    update, rates, reports, stepped = Update(), history(bundle, parent_folder), [], []
    extracts = Path(extracts) if extracts else daily_dir(bundle) / "extracts"
    for name in used:
        if name in state.get("skip", []):
            stepped.append(name)
            log(f"{name}: skipped (skip-dump)")
            continue
        t = time.perf_counter()
        ext, window = fresh_extract(recent[name], extracts / name, workers)
        report = apply_dump(served, update, seen, ext, window, name, MIN_USERS, MAX_NEW)
        check_dump(report, rates)
        if window[1] - window[0] >= DAY / 2:
            rates = (rates + [report["fresh_listens"] / ((window[1] - window[0]) / DAY)])[-HISTORY_DAYS:]
        reports.append(report)
        log(f"{name}: {report['fresh_listens']:,} fresh listens, {report['catalog_listens']:,} in catalog sessions, "
            f"{report['new_songs']:,} new songs, {report['pairs']:,} pairs ({time.perf_counter() - t:.0f} s)")
    if not reports:
        raise Stop(f"every dump to use is skipped ({stepped}): nothing to build on them")
    n = len(served)
    tables = {"retriever_song_inputs": SliceTable(bundle, "retriever_song_inputs", served.base),
              "ranker_song_inputs": SliceTable(bundle, "ranker_song_inputs", served.base),
              "search_vectors": np.load(daily_dir(bundle) / "search_vectors.npy", mmap_mode="r")}
    made = standins(served, update, tables, MODE, start=added_before)
    check_vectors(made["search_vectors"], tables["search_vectors"])
    rows = {"retriever_song_inputs": made["retriever_song_inputs"], "ranker_song_inputs": made["ranker_song_inputs"],
            "song_artist": served.song_artist()[before:], "song_album": served.song_album()[before:],
            "song_genres": served.song_genres()[before:]}
    listens = np.zeros(n, dtype=np.int64)
    listens[:before] = np.load(parent / "song_playlists.npy")
    listens[:len(update.listens)] += update.listens
    seen_file = daily_dir(bundle) / f".seen.{os.getpid()}.npz"
    seen.save(seen_file)
    windows_used = [windows[r["dump"]] for r in reports] + [windows[s] for s in stepped]
    folder = write_update(bundle, parent, served.fields(added_before), rows, made["search_vectors"], listens,
                          (update.pairs, update.counts), note=f"daily: {len(reports)} dumps",
                          info={"dumps": reports, "skipped_dumps": stepped, "gaps": gaps,
                                "chain_end": max(end for _, end in windows_used), "mode": MODE,
                                "min_users": MIN_USERS, "max_new": MAX_NEW},
                          files={"seen.npz": seen_file})
    verify(parent, folder, before, (update.pairs, update.counts))
    rel = str(folder.relative_to(bundle))
    log(f"built {rel}: {n - before:,} new songs ({n:,} in all), {len(update.pairs):,} pairs counted, "
        f"{int(update.listens.sum()):,} listens ({time.perf_counter() - began:.0f} s)")
    return rel


# --- switch ----------------------------------------------------------------------------------------------------------

class FrontDesk:
    """A front desk reached directly (not through nginx, which removes the standby header)."""

    def __init__(self, url, timeout=10.0):
        self.url, self.timeout = url.rstrip("/"), timeout

    def health(self):
        with urllib.request.urlopen(f"{self.url}/health", timeout=self.timeout) as r:
            return json.loads(r.read())

    def recommend(self, songs, standby=False):
        """(HTTP status, the answer or None, milliseconds)."""
        request = urllib.request.Request(f"{self.url}/recommend", data=json.dumps({"songs": songs}).encode(),
                                         headers={"content-type": "application/json",
                                                  **({"X-Recsys-Standby": "1"} if standby else {})})
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as r:
                body, code = json.loads(r.read()), r.status
        except urllib.error.HTTPError as e:
            body, code = None, e.code
        except (urllib.error.URLError, OSError):
            body, code = None, 0
        return code, body, (time.perf_counter() - start) * 1000


def wait_for(condition, timeout, what):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            if condition():
                return time.monotonic() - start
        except (urllib.error.URLError, OSError, KeyError, TypeError):
            pass
        time.sleep(1.0)
    raise Stop(f"waited {timeout:.0f} s for {what}")


def ndcg10(top, target):
    return next((1 / math.log2(r + 2) for r, uri in enumerate(top[:10]) if uri == target), 0.0)


def check_standby(desk, cases, version, new_uris, warm=20):
    """Live and standby answer every case (live first, so both are computed or both cached): (report, problems)."""
    def answers(standby):
        for case in cases[:warm]:  # a version's first answers pay one-off costs
            desk.recommend(case["songs"], standby)
        return [desk.recommend(case["songs"], standby) for case in cases]
    out = {}
    for side, standby in (("live", False), ("standby", True)):
        got = answers(standby)
        ok = [b for code, b, _ in got if code == 200 and b is not None]
        tops = [[r["uri"] for r in b["recommendations"]] if code == 200 and b else [] for code, b, _ in got]
        computed = [ms for code, b, ms in got if code == 200 and b and not b["cached"]]  # a cached one isn't timed
        out[side] = {"failed": len(got) - len(ok), "partial": sum(b["partial"] for b in ok),
                     "versions": sorted({b["version"] for b in ok}),
                     "ndcg@10": float(np.mean([ndcg10(top, c["target"]) for top, c in zip(tops, cases)])),
                     "p99_ms": float(np.percentile(computed, 99)) if len(computed) >= 500 else None,
                     "computed": len(computed), "tops": tops}
    new = set(new_uris)
    lists = out["standby"].pop("tops")
    out["live"].pop("tops")
    per_song = {}
    for top in lists:
        for uri in set(top) & new:
            per_song[uri] = per_song.get(uri, 0) + 1
    worst = max(per_song.items(), key=lambda kv: kv[1], default=(None, 0))
    out["new_songs"] = {"lists_with_one": round(sum(bool(set(t) & new) for t in lists) / max(len(lists), 1), 4),
                        "most_lists": worst[1], "most_lists_song": worst[0]}
    s, live = out["standby"], out["live"]
    problems = [f"{side}: {out[side]['failed']} failed, {out[side]['partial']} partial answers"
                for side in ("live", "standby") if out[side]["failed"] or out[side]["partial"]]
    if s["versions"] != [version]:
        problems.append(f"the standby answered as {s['versions']}, not {version}")
    if s["ndcg@10"] < live["ndcg@10"] - MAX_NDCG_DROP:
        problems.append(f"NDCG@10 {s['ndcg@10']:.4f}, live {live['ndcg@10']:.4f}: more than {MAX_NDCG_DROP} lower")
    if s["p99_ms"] and live["p99_ms"] and s["p99_ms"] > MAX_P99_RATIO * live["p99_ms"]:
        problems.append(f"p99 {s['p99_ms']:.0f} ms, live {live['p99_ms']:.0f} ms: over {MAX_P99_RATIO}x")
    if worst[1] > MAX_NEW_SHARE * len(lists):
        problems.append(f"a new song ({worst[0]}) is in {worst[1]} of {len(lists)} lists (at most {MAX_NEW_SHARE:.0%})")
    return out, problems


def watch(desk, version, cases, seconds=WATCH_S, every=PROBE_S):
    """Ask the live version every `every` seconds for `seconds`: Stop after two failures in a row (one alone can be
    a hiccup: a shard missing one deadline is skipped for a second)."""
    end, i, in_row, failures = time.monotonic() + seconds, 0, 0, 0
    while time.monotonic() < end:
        code, body, _ = desk.recommend(cases[i % len(cases)]["songs"])
        ok = code == 200 and body is not None and not body["partial"] and body["version"] == version
        if ok and i % 5 == 0:
            try:
                h = desk.health()
                ok = h["status"] == "ok" and h["version"] == version
            except (urllib.error.URLError, OSError, KeyError):
                ok = False
        in_row, failures = (0, failures) if ok else (in_row + 1, failures + 1)
        if in_row >= 2:
            raise Stop(f"two failed checks in a row {i * every:.0f} s after the flip (status {code})")
        i += 1
        time.sleep(every)
    return {"probes": i, "failures": failures}


def new_song_documents(bundle, folder):
    """The search documents (recsys.song_search.documents' fields) of the songs this update added."""
    bundle = Path(bundle)
    added, info = added_songs(bundle / folder, bundle), update_info(bundle, folder)
    names = json.loads((daily_dir(bundle) / "artist_names.json").read_text())
    listens = np.load(bundle / folder / "song_playlists.npy")
    first = info["songs"] - info["added"]
    k = len(added["uris"]) - info["added"]
    return [{"id": first + i, "name": f"{added['tracks'][k + i]} {names[added['artist'][k + i]]}",
             "album": added["albums"][k + i], "popularity": int(listens[first + i])} for i in range(info["added"])]


def switch_to(bundle, folder, desk, search=None, search_key=None, watch_s=WATCH_S):
    """Switch to an update built on the live version (see the module docstring); returns the checks' report."""
    from recsys.song_search import add_songs
    bundle, path = Path(bundle), Path(bundle) / "switch.json"
    sw, info = switch.read(path), update_info(bundle, folder)
    if sw["live"] == folder:
        raise Stop(f"{folder} is live already")
    if sw["standby"] not in (None, folder):
        raise Stop(f"another standby is loaded ({sw['standby']}): finish or undo first")
    if info["parent_folder"] != sw["live"]:
        raise Stop(f"{folder} was built on {info['parent_folder']}, not on what's live ({sw['live']})")
    version, old = version_of(bundle / folder), version_of(bundle / sw["live"])
    report, began = {"update": folder, "version": version}, time.perf_counter()
    if search:
        docs = new_song_documents(bundle, folder)
        add_songs(search, search_key, info["base"], docs)
        log(f"{len(docs):,} new songs in the search index")
    switch.prepare(path, folder)
    report["standby_ready_s"] = round(wait_for(lambda: (desk.health()["switch"]["standby"] or {}).get("version")
                                               == version, 900, "every shard and the front desk to load it"), 1)
    log(f"standby {version} ready after {report['standby_ready_s']} s")
    cases = json.loads((daily_dir(bundle) / "cases.json").read_text())
    added = added_songs(bundle / folder, bundle)["uris"]
    checks, problems = check_standby(desk, cases, version, added[len(added) - info["added"]:])
    report["checks"] = checks
    log(f"checks: live NDCG@10 {checks['live']['ndcg@10']:.4f}, standby {checks['standby']['ndcg@10']:.4f}; "
        f"p99 {checks['live']['p99_ms']} / {checks['standby']['p99_ms']} ms; new songs in "
        f"{checks['new_songs']['lists_with_one']:.1%} of lists, at most {checks['new_songs']['most_lists']} for one")
    if problems:
        switch.finish(path)  # the update is dropped; live as it was
        raise Stop("standby checks failed, not switched: " + "; ".join(problems))
    switch.flip(path)
    report["flip_seen_s"] = round(wait_for(lambda: desk.health()["version"] == version, 120, "the flip"), 1)
    log(f"live: {version} (watching {watch_s:.0f} s)")
    try:
        report["watch"] = watch(desk, version, cases, watch_s)
    except Stop as e:
        switch.flip(path)
        wait_for(lambda: desk.health()["version"] == old, 120, "the flip back")
        switch.finish(path)
        pause(bundle, f"switched back from {folder}: {e}")
        raise Stop(f"switched back to {old}: {e}") from None
    switch.finish(path)
    report["seconds"] = round(time.perf_counter() - began)
    log(f"done: {version} live, the old version let go ({report['seconds']} s)")
    prune(bundle)
    return report


def pause(bundle, why):
    state = read_state(bundle)
    state["paused"], state["paused_why"] = True, f"{iso(time.time())}: {why}"
    write_state(bundle, state)
    log(f"paused: {why}")


def prune(bundle, keep=KEEP_UPDATES, keep_extracts=KEEP_EXTRACTS):
    """Delete updates other than the live one's last `keep` (it, its parent, ...) and the standby, and all but the
    newest `keep_extracts` extracts."""
    bundle = Path(bundle)
    sw = switch.read(bundle / "switch.json")
    kept = set(lineage(bundle, sw["live"])[:keep]) | ({sw["standby"]} if sw["standby"] else set())
    for folder in sorted((bundle / "updates").glob("*")) if (bundle / "updates").exists() else []:
        rel = str(folder.relative_to(bundle))
        if rel not in kept and folder.is_dir():
            shutil.rmtree(folder)
            log(f"deleted {rel}")
    extracts = sorted(p for p in (daily_dir(bundle) / "extracts").glob("listenbrainz-dump-*")) \
        if (daily_dir(bundle) / "extracts").exists() else []
    for folder in extracts[:-keep_extracts] if keep_extracts else extracts:
        shutil.rmtree(folder)


def undo(bundle, desk, to=None):
    """Switch back to the live version's parent (or `to`, a kept folder), and pause the timer: tomorrow's run would
    otherwise build the same update again (skip-dump a bad dump, then resume)."""
    bundle, path = Path(bundle), Path(bundle) / "switch.json"
    sw = switch.read(path)
    target = to or (update_info(bundle, sw["live"]).get("parent_folder") if sw["live"] != switch.BASE else None)
    if target is None:
        raise Stop("the base is live: nothing to undo")
    if not (bundle / target).exists():
        raise Stop(f"{target} isn't kept any more")
    if target == sw["live"]:
        raise Stop(f"{target} is live already")
    version = version_of(bundle / target)
    pause(bundle, f"undo from {sw['live']} to {target}")
    if sw["standby"] != target:
        switch.prepare(path, target)
    wait_for(lambda: (desk.health()["switch"]["standby"] or {}).get("version") == version, 900, "the old version to load")
    switch.flip(path)
    wait_for(lambda: desk.health()["version"] == version, 120, "the flip")
    switch.finish(path)
    log(f"undone: {target} ({version}) live again; the timer is paused (resume to restart it)")


def status(bundle):
    bundle = Path(bundle)
    state, sw = read_state(bundle), switch.read(bundle / "switch.json")
    info = update_info(bundle, sw["live"])
    lines = [f"live: {sw['live']} ({version_of(bundle / sw['live'])}), standby: {sw['standby']}",
             f"dumps used up to {iso(info.get('chain_end', state['data_end']))}",
             f"paused: {state['paused']}" + (f" ({state['paused_why']})" if state["paused"] else ""),
             f"skipped dumps: {state.get('skip', [])}",
             f"kept updates: {sorted(p.name for p in (bundle / 'updates').glob('*'))}" if (bundle / "updates").exists()
             else "kept updates: none"]
    lines += [f"run {r['at']}: {r['result']}" for r in state.get("runs", [])[-5:]]
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="The daily update (recsys/daily_job.py).")
    ap.add_argument("command", choices=["starter", "build", "switch", "run", "undo", "skip-dump", "resume", "status"])
    ap.add_argument("dump", nargs="?", help="skip-dump: the dump's name")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--dumps", default=DUMPS, help="ListenBrainz's listing (a URL) or a folder of dump folders")
    ap.add_argument("--extracts", help="where the dumps' fresh listens go (default <bundle>/daily/extracts)")
    ap.add_argument("--workers", type=int, default=1, help="processes reading a dump")
    ap.add_argument("--until", help="build: only dumps ending by this day's start, UTC (YYYY-MM-DD)")
    ap.add_argument("--update", help="switch: the update's folder, relative to the bundle")
    ap.add_argument("--to", help="undo: the folder to go back to (default: the live one's parent)")
    ap.add_argument("--frontdesk", default="http://frontdesk:8000")
    ap.add_argument("--search", default="http://meilisearch:7700", help="Meilisearch ('' for none)")
    ap.add_argument("--watch-s", type=float, default=WATCH_S)
    ap.add_argument("--store", help="starter: the store the pair was trained on")
    ap.add_argument("--cases", help="starter: a folder with the checks' cases.json (scripts/serving_check.py)")
    ap.add_argument("--data-end", help="starter: the day the base's training data ends (YYYY-MM-DD)")
    args = ap.parse_args()
    bundle, desk = Path(args.bundle), FrontDesk(args.frontdesk)
    key = os.environ.get("MEILI_MASTER_KEY")
    until = timegm(datetime.strptime(args.until, "%Y-%m-%d").timetuple()) if args.until else None
    if args.command == "starter":
        return starter(bundle, args.store, args.cases, args.data_end)
    if args.command == "status":
        return status(bundle)
    if args.command in ("skip-dump", "resume"):
        state = read_state(bundle)
        if args.command == "skip-dump":
            state["skip"] = sorted(set(state.get("skip", [])) | {args.dump})
        else:
            state["paused"], state["paused_why"] = False, None
        write_state(bundle, state)
        return status(bundle)
    began, run = time.perf_counter(), {"at": iso(time.time()), "command": args.command}
    try:
        with one_job(bundle):
            if args.command == "undo":
                undo(bundle, desk, args.to)
                run["result"] = "undone"
            elif args.command == "switch":
                run["report"] = switch_to(bundle, args.update, desk, args.search or None, key, args.watch_s)
                run["result"] = f"switched to {args.update}"
            else:
                if args.command == "run":
                    state = read_state(bundle)
                    if state["paused"]:
                        log(f"paused ({state['paused_why']}): not running; resume to restart")
                        return
                    memory, disk = free_memory_gb(), shutil.disk_usage(bundle).free / (1 << 30)
                    if (memory is not None and memory < MIN_MEMORY_GB) or disk < MIN_DISK_GB:
                        raise Stop(f"too little free: {memory} GB of memory (at least {MIN_MEMORY_GB}), {disk:.1f} GB of "
                                   f"disk (at least {MIN_DISK_GB})")
                folder = build(bundle, args.dumps, args.extracts, args.workers, until)
                run["update"] = folder
                run["result"] = f"built {folder}" if folder else "nothing new"
                if folder and args.command == "run":
                    run["report"] = switch_to(bundle, folder, desk, args.search or None, key, args.watch_s)
                    run["result"] = f"switched to {folder}"
    except Stop as e:
        run["result"] = f"stopped: {e}"
        log(f"STOPPED: {e}")
        raise SystemExit(1)
    finally:
        run["seconds"] = round(time.perf_counter() - began)
        if "result" in run and (daily_dir(bundle) / "state.json").exists():
            record(bundle, run)


if __name__ == "__main__":
    main()
