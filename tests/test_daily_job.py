"""The daily job (recsys/daily_job.py) on the tiny bundle: two days of dumps built into updates, a switch with its
checks through real shard and front-desk processes, undo, and the checks on their own."""
import calendar
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pytest

from recsys import daily_job as dj
from recsys import switch
from recsys.daily import lean_files
from recsys.listenbrainz import song_key
from recsys.updates import version_of
from serving_helpers import ROOT, free_ports, shard_servers, slices, write_tiny_bundle
from tests.test_listenbrainz import dump_bytes, listen_line

DATA_END = calendar.timegm((2026, 10, 15, 0, 0, 0))
NEW = ("Kavinsky", "Brand New")  # a song the catalog doesn't have, by an artist it does


def names(bundle):
    songs = json.loads((bundle / "songs.json").read_text())
    artist = np.load(bundle / "song_artist.npy")
    return [(songs["artist_names"][a], t) for a, t in zip(artist, songs["tracks"])]


def write_dump(folder, number, start, end, sessions):
    """A daily dump in its own folder, named as ListenBrainz names them (for its window's end); sessions: {user:
    [(artist, title), ...]}, played in the 6 hours before the end."""
    stamp = lambda t: f"{datetime.fromtimestamp(t, timezone.utc):%Y-%m-%d %H:%M:%S.%f}+00:00".encode()
    name = f"listenbrainz-dump-{number:04d}-{datetime.fromtimestamp(end, timezone.utc):%Y%m%d-%H%M%S}-incremental"
    lines = [listen_line(user=u, ts=end - 6 * 3600 + u * 1800 + i * 120, artist=a, title=t, release="")
             for u, songs in sessions.items() for i, (a, t) in enumerate(songs)]
    data = dump_bytes({"d/START_TIMESTAMP": stamp(start), "d/END_TIMESTAMP": stamp(end),
                       "d/listens/2026/10.listens": b"\n".join(lines) + b"\n"})
    (folder / name).mkdir(parents=True)
    path = folder / name / f"{name.replace('-dump-', '-listens-dump-')}.tar.zst"
    path.write_bytes(data)
    path.with_name(path.name + ".sha256").write_text(hashlib.sha256(data).hexdigest())
    return name


def build(bundle, dumps, *extra):
    """recsys.daily_job build in its own process (writing an update needs FAISS, this one has PyTorch): the
    update's folder, or None if there was nothing new."""
    out = subprocess.run([sys.executable, "-m", "recsys.daily_job", "build", "--bundle", str(bundle), "--dumps",
                          str(dumps), *extra], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    built = [line.split("built ", 1)[1].split(":", 1)[0] for line in out.stdout.splitlines() if " built " in line]
    return built[0] if built else None


def every_song_twice(songs, first_user):
    """8 listeners' sessions of 10 songs that play each of the 40 songs twice (so none is over 5% of the day)."""
    return {first_user + u: [songs[(u * 10 + k) % 40] for k in range(10)] for u in range(8)}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """(bundle, dumps folder, the dumps' names): the tiny bundle with its daily files, and two days of dumps (each
    catalog song played twice a day, so none is over 5% of a day) in which the new song has 2 listeners on day 1
    and a third on day 2."""
    root = tmp_path_factory.mktemp("daily")
    bundle = write_tiny_bundle(root)[0]
    slices(bundle, 2)
    songs = names(bundle)
    daily = dj.daily_dir(bundle)
    daily.mkdir()
    lean_files(bundle, [song_key(a, t) for a, t in songs], daily)
    shutil.copyfile(bundle / "search_vectors.npy", daily / "search_vectors.npy")
    uris = json.loads((bundle / "songs.json").read_text())["uris"]
    (daily / "cases.json").write_text(json.dumps([{"songs": [uris[j]], "target": uris[j + 1]} for j in range(30)]))
    dj.write_state(bundle, {"data_end": DATA_END, "paused": False, "paused_why": None, "skip": [], "runs": []})
    day1, day2 = every_song_twice(songs, 1), every_song_twice(songs, 11)
    day1[1], day1[2] = day1[1] + [NEW], [NEW] + day1[2]
    day2[11] = day2[11][:5] + [NEW] + day2[11][5:]
    dumps = root / "dumps"
    names_ = [write_dump(dumps, 1, DATA_END + 3, DATA_END + 86403, day1),
              write_dump(dumps, 2, DATA_END + 86403, DATA_END + 2 * 86403, day2)]
    return bundle, dumps, names_


def test_two_days_build_on_each_other_and_the_new_song_joins_on_the_second(tiny):
    bundle, dumps, (day1, day2) = tiny
    sw = bundle / "switch.json"
    first = build(bundle, dumps, "--until", "2026-10-16")
    info = dj.update_info(bundle, first)
    assert [d["dump"] for d in info["dumps"]] == [day1] and info["added"] == 0  # 2 listeners: not yet
    assert info["parent_folder"] == "." and info["chain_end"] == pytest.approx(DATA_END + 86403)
    assert info["dumps"][0]["top_song_share"] <= dj.MAX_TOP_SHARE and (bundle / first / "seen.npz").exists()
    switch.write(sw, first)  # as if switched to (the switch has its own test)
    try:
        second = build(bundle, dumps)
        info = dj.update_info(bundle, second)
        assert [d["dump"] for d in info["dumps"]] == [day2] and info["parent_folder"] == first
        added = json.loads((bundle / second / "added_songs.json").read_text())
        assert added["keys"] == [song_key(*NEW)] and added["listeners"] == [3]  # day 1's two, remembered, and day 2's
        assert added["artist"] == [0] and info["songs"] == 41
        switch.write(sw, second)
        assert build(bundle, dumps) is None  # nothing new
        per_day = [r["fresh_listens"] * 86400 / (r["end"] - r["start"]) for r in
                   dj.update_info(bundle, first)["dumps"] + info["dumps"]]
        assert dj.history(bundle, second) == pytest.approx(per_day)  # oldest first
    finally:
        sw.unlink()


def test_a_skipped_dump_is_stepped_past(tiny):
    bundle, dumps, (day1, day2) = tiny
    state = dj.read_state(bundle)
    dj.write_state(bundle, {**state, "skip": [day1]})
    try:
        info = dj.update_info(bundle, build(bundle, dumps))
        assert info["skipped_dumps"] == [day1] and [d["dump"] for d in info["dumps"]] == [day2]
        assert info["chain_end"] == pytest.approx(DATA_END + 2 * 86403) and info["added"] == 0  # 1 listener of it
    finally:
        dj.write_state(bundle, state)


def test_a_tail_dump_is_used_once(tiny, tmp_path):
    """As 2761 (2026-11-02): the nearly empty dump after a day's spans 0.31 s, less than the chain's 1 s tolerance,
    so as an update's last dump its window alone doesn't mark it as used: its name keeps it out of the next build."""
    bundle, _, _ = tiny
    songs, sw, dumps = names(bundle), bundle / "switch.json", tmp_path / "dumps"
    day1 = write_dump(dumps, 1, DATA_END + 3.3, DATA_END + 86402.91, every_song_twice(songs, 1))
    tail = write_dump(dumps, 2, DATA_END + 86402.91, DATA_END + 86403.22, {9: songs[:1]})
    day2 = write_dump(dumps, 3, DATA_END + 86403.22, DATA_END + 2 * 86403, every_song_twice(songs, 11))
    first = build(bundle, dumps, "--until", "2026-10-16")
    assert [d["dump"] for d in dj.update_info(bundle, first)["dumps"]] == [day1, tail]
    switch.write(sw, first)
    try:
        second = build(bundle, dumps)
        assert [d["dump"] for d in dj.update_info(bundle, second)["dumps"]] == [day2]
    finally:
        sw.unlink()


def test_a_damaged_dump_stops_the_build_and_leaves_nothing(tiny, tmp_path):
    bundle, dumps, (day1, _) = tiny
    shutil.copytree(dumps / day1, tmp_path / "dumps" / day1)
    path = next((tmp_path / "dumps" / day1).glob("*.tar.zst"))
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF  # one byte damaged: as in the drill (2026-11-04), zstd's own checksum fails
    path.write_bytes(bytes(data))
    updates = sorted((bundle / "updates").glob("*")) if (bundle / "updates").exists() else []
    out = subprocess.run([sys.executable, "-m", "recsys.daily_job", "build", "--bundle", str(bundle), "--dumps",
                          str(tmp_path / "dumps"), "--extracts", str(tmp_path / "extracts")],
                         cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 1 and f"STOPPED: {day1} can't be read" in out.stdout, out.stdout + out.stderr
    assert dj.read_state(bundle)["runs"][-1]["result"].startswith(f"stopped: {day1} can't be read")
    assert not any((tmp_path / "extracts").glob("*"))  # no half an extract
    assert (sorted((bundle / "updates").glob("*")) if (bundle / "updates").exists() else []) == updates


def test_a_day_out_of_line_with_the_ones_before_stops_the_build():
    day = {"dump": "d", "start": 0.0, "end": 86400.0, "fresh_listens": 4_000_000, "top_song_share": 0.02}
    with pytest.raises(dj.Stop, match="3.33x the median"):
        dj.check_dump(day, [1.2e6, 1.0e6, 1.3e6])
    dj.check_dump({**day, "fresh_listens": 1_100_000}, [1.2e6, 1.0e6, 1.3e6])
    dj.check_dump(day, [1.2e6, 1.0e6])  # too few days to compare with
    dj.check_dump({**day, "end": 0.3, "top_song_share": 0.5}, [1.2e6] * 3)  # the nearly empty dump after a day's
    with pytest.raises(dj.Stop, match="one song is 6.0%"):
        dj.check_dump({**day, "fresh_listens": 1_100_000, "top_song_share": 0.06}, [1.2e6] * 3)
    with pytest.raises(dj.Stop, match="isn't finite"):
        dj.check_vectors(np.array([[np.nan, 1.0, 0.0]]), np.ones((4, 3)))
    with pytest.raises(dj.Stop, match="longer than any"):
        dj.check_vectors(np.array([[3.0, 0.0, 0.0]]), np.ones((4, 3)))
    dj.check_vectors(np.array([[1.0, 1.0, 9.0]]), np.ones((4, 3)))  # the bias (last) isn't a length


class FakeDesk:
    """Answers each case's song with a fixed list per version: live's and the standby's."""

    def __init__(self, live, standby, version="v2", partial=()):
        self.lists, self.version, self.partial = {False: live, True: standby}, version, set(partial)

    def recommend(self, songs, standby=False):
        top = self.lists[standby](songs[0])
        return 200, {"recommendations": [{"uri": u} for u in top], "partial": standby and songs[0] in self.partial,
                     "cached": False, "version": self.version if standby else "v1"}, 5.0


CASES = [{"songs": [f"s{i}"], "target": f"t{i}"} for i in range(200)]
GOOD = lambda song: [f"t{song[1:]}"] + [f"x{k}" for k in range(9)]  # the target first


def test_the_standby_checks_pass_a_good_update_and_stop_a_bad_one():
    report, problems = dj.check_standby(FakeDesk(GOOD, GOOD), CASES, "v2", ["new0"])
    assert problems == [] and report["standby"]["ndcg@10"] == report["live"]["ndcg@10"] == 1.0
    flood = lambda song: ["new0"] + GOOD(song)[:9]  # a new song in every list, everything else one place down
    _, problems = dj.check_standby(FakeDesk(GOOD, flood), CASES, "v2", ["new0"])
    assert any("new0" in p and "200 of 200 lists" in p for p in problems)
    assert any("NDCG@10" in p for p in problems)
    _, problems = dj.check_standby(FakeDesk(GOOD, GOOD, partial={"s100", "s101", "s102"}), CASES, "v2", [])
    assert problems == ["standby: 0 failed, 3 partial answers"]
    _, problems = dj.check_standby(FakeDesk(GOOD, GOOD, version="v3"), CASES, "v2", [])
    assert problems == ["the standby answered as ['v3'], not v2"]


def test_watching_forgives_one_failure_but_not_two_in_a_row():
    class Flaky:
        def __init__(self, failures):
            self.failures, self.calls = failures, 0

        def recommend(self, songs, standby=False):
            self.calls += 1
            fail = self.calls in self.failures
            return (503 if fail else 200), None if fail else {"partial": False, "version": "v2"}, 1.0

        def health(self):
            return {"status": "ok", "version": "v2"}
    assert dj.watch(Flaky({2, 4}), "v2", CASES, seconds=0.2, every=0.01)["failures"] == 2
    with pytest.raises(dj.Stop, match="two failed checks in a row"):
        dj.watch(Flaky({3, 4}), "v2", CASES, seconds=5, every=0.01)


def test_pruning_keeps_the_live_ones_last_three_and_the_standby(tmp_path):
    bundle = tmp_path
    names_ = ["u1", "u2", "u3", "u4", "u5", "other"]
    parents = {"u1": ".", "u2": "updates/u1", "u3": "updates/u2", "u4": "updates/u3", "u5": "updates/u4", "other": "."}
    for n in names_:
        (bundle / "updates" / n).mkdir(parents=True)
        (bundle / "updates" / n / "update.json").write_text(json.dumps({"parent_folder": parents[n]}))
    (bundle / "daily" / "extracts").mkdir(parents=True)
    for k in range(6):
        (bundle / "daily" / "extracts" / f"listenbrainz-dump-{k:04d}").mkdir()
    switch.write(bundle / "switch.json", "updates/u5", "updates/other")
    dj.prune(bundle)
    assert sorted(p.name for p in (bundle / "updates").iterdir()) == ["other", "u3", "u4", "u5"]
    assert sorted(p.name for p in (bundle / "daily" / "extracts").iterdir()) == \
        [f"listenbrainz-dump-{k:04d}" for k in range(2, 6)]


def test_a_switch_checks_flips_watches_and_undoes(tiny, monkeypatch, tmp_path):
    """Through real shards and a real front desk: the update's checks pass, it goes live, then undo takes the
    base back and pauses the timer."""
    monkeypatch.setattr(dj, "MAX_NDCG_DROP", 1.0)  # the tiny random model's answers: the mechanics, not the bar
    monkeypatch.setattr(dj, "MAX_NEW_SHARE", 1.0)
    bundle, dumps, _ = tiny
    path = bundle / "switch.json"
    path.unlink(missing_ok=True)
    folder = build(bundle, dumps)
    base_v, new_v = version_of(bundle), version_of(bundle / folder)
    log = open(tmp_path / "frontdesk.log", "w")
    with shard_servers([bundle / "shards-2" / f"shard-{i}" for i in range(2)], switch_file=path) as addresses:
        port = free_ports(1)[0]
        desk_proc = subprocess.Popen([sys.executable, "-m", "recsys.serve", "--bundle", str(bundle), "--shards",
                                      ",".join(addresses), "--port", str(port), "--switch", str(path)], cwd=ROOT,
                                     stdout=log, stderr=log)
        try:
            desk = dj.FrontDesk(f"http://127.0.0.1:{port}")
            dj.wait_for(lambda: desk.health()["status"] == "ok", 60, "the front desk")
            report = dj.switch_to(bundle, folder, desk, watch_s=2.0)
            assert desk.health()["version"] == new_v and switch.read(path) == {"live": folder, "standby": None}
            assert report["checks"]["standby"]["versions"] == [new_v] and report["watch"]["failures"] == 0
            with pytest.raises(dj.Stop, match="live already"):
                dj.switch_to(bundle, folder, desk, watch_s=0.1)
            dj.undo(bundle, desk)
            assert desk.health()["version"] == base_v and switch.read(path) == {"live": ".", "standby": None}
            state = dj.read_state(bundle)
            assert state["paused"] and "undo from" in state["paused_why"]
        finally:
            desk_proc.terminate()
            desk_proc.wait(timeout=30)
            log.close()
            path.unlink(missing_ok=True)
            dj.write_state(bundle, {**dj.read_state(bundle), "paused": False, "paused_why": None})
