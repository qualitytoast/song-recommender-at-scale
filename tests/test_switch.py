"""Switching versions without a restart (recsys/switch.py, recsys/updates.py): updates of the tiny bundle, shards
that hold two versions, and a front desk that flips between them under traffic."""
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import grpc
import numpy as np
import pytest

from recsys import switch
from recsys.bundle import write_manifest
from recsys.frontdesk import ShardMismatch, ShardedRecommender
from recsys.proto import shard_pb2, shard_pb2_grpc
from recsys.recommender import Recommender, version_of
from recsys.taste import adjust, make_profile, save_profile
from recsys.wire import pack, unpack
from serving_helpers import ARTISTS, ROOT, V, free_ports, shard_servers, slices, write_tiny_bundle

COPIES = 3  # the test update's new songs: copies of the base's 3 most played songs (IDs V, V + 1, V + 2)


def fake(bundle, copies, *extra):
    out = subprocess.run([sys.executable, "-m", "recsys.updates", "fake", "--bundle", str(bundle), "--copies",
                          str(copies), *extra], cwd=ROOT, check=True, capture_output=True, text=True)
    return type(bundle)(out.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def updated(tmp_path_factory):
    """(base bundle, an update adding COPIES songs, the same catalog as a whole bundle, an update adding none,
    an update of the first adding 2 more)."""
    root = tmp_path_factory.mktemp("bundles")
    base = write_tiny_bundle(root)[0]
    slices(base, 2)
    whole = root / "whole"
    update = fake(base, COPIES, "--whole", str(whole))
    empty = fake(base, 0)
    second = fake(base, 2, "--parent", str(update))
    return base, update, whole, empty, second


def wait_for(condition, timeout=60.0, what="it"):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return
        time.sleep(0.1)
    raise AssertionError(f"waited {timeout:.0f} s for {what}")


def test_the_switch_file_steps(tmp_path):
    path = tmp_path / "switch.json"
    assert switch.read(path) == {"live": ".", "standby": None}  # no file: the bundle itself
    switch.prepare(path, "updates/u1")
    assert switch.read(path) == {"live": ".", "standby": "updates/u1"}
    switch.flip(path)
    assert switch.read(path) == {"live": "updates/u1", "standby": "."}
    switch.flip(path)  # undo
    assert switch.read(path) == {"live": ".", "standby": "updates/u1"}
    switch.flip(path)
    switch.finish(path)
    assert switch.read(path) == {"live": "updates/u1", "standby": None}
    with pytest.raises(ValueError, match="no standby"):
        switch.flip(path)
    assert [p.name for p in tmp_path.iterdir()] == ["switch.json"]  # written whole: no temporary file left
    path.write_text('{"live": 3}')
    with pytest.raises(ValueError, match="not a switch file"):
        switch.read(path)


def test_a_manifest_covers_files_not_folders(tmp_path):
    (tmp_path / "a.json").write_text("1")
    (tmp_path / "shards-2").mkdir()
    version = write_manifest(tmp_path)
    assert list(json.loads((tmp_path / "manifest.json").read_text())["files"]) == ["a.json"]
    (tmp_path / "shards-2" / "x").write_text("2")
    assert write_manifest(tmp_path) == version


def test_an_update_appends_songs_and_keeps_everything_else(updated):
    base, update, whole, empty, second = updated
    info = json.loads((update / "update.json").read_text())
    assert info["base"] == info["parent"] == version_of(base) and info["songs"] == V + COPIES
    assert update.parent == base / "updates" and update.name.endswith(version_of(update))
    added = json.loads((update / "added_songs.json").read_text())
    assert added["uris"] == [f"fake:track:{j}" for j in range(V, V + COPIES)]
    names = json.loads((base / "songs.json").read_text())
    top = np.argsort(-np.load(base / "song_playlists.npy"), kind="stable")[:COPIES]
    assert added["tracks"] == [f"{names['tracks'][s]} (copy)" for s in top]
    for i in range(2):
        old, new = base / "shards-2" / f"shard-{i}", update / "shards-2" / f"shard-{i}"
        ids = np.load(new / "ids.npy")
        np.testing.assert_array_equal(ids[:len(np.load(old / "ids.npy"))], np.load(old / "ids.npy"))  # old rows first
        assert set(ids[len(np.load(old / "ids.npy")):]) == {j for j in range(V, V + COPIES) if j % 2 == i}
        slice_info = json.loads((new / "info.json").read_text())
        assert slice_info == {"version": version_of(update), "index": i, "shards": 2, "songs": len(ids),
                              "catalog": V + COPIES}
    assert json.loads((second / "update.json").read_text())["parent"] == version_of(update)
    assert len(json.loads((second / "added_songs.json").read_text())["uris"]) == COPIES + 2  # since the base
    assert json.loads((empty / "settings.json").read_text())["songs"] == V and version_of(empty) != version_of(base)


@pytest.mark.parametrize("which", ["update", "empty"])
def test_an_update_answers_as_the_whole_bundle_it_stands_for(updated, which):
    base, update, whole, empty, _ = updated
    folder, reference = (update, whole) if which == "update" else (empty, base)
    path = base / "switch.json"
    switch.write(path, str(folder.relative_to(base)))
    single = Recommender(reference, "cpu")
    try:
        with shard_servers(slices_of(base), switch_file=path) as addresses:
            rec = ShardedRecommender(base, addresses, "cpu", startup_wait=60, update=folder)
            try:
                assert (rec.version, rec.base_version, len(rec.uris)) == (version_of(folder), version_of(base),
                                                                          len(single.uris))
                name = single.encode_name("road trip")
                new = list(range(V, len(single.uris)))  # the update's songs, as inputs too
                for songs in ([7], [3, 17, 9], [1, 2, 3, 4, 5], *([[V], [3, V + 1, 9], [V - 1] + new] if new else [])):
                    for mix in (0.0, 0.5):
                        ids, scores, _, notes = rec.recommend_ids(songs, name, mix)
                        expected_ids, expected_scores, _, _ = single.recommend_ids(songs, name, mix)
                        np.testing.assert_array_equal(ids, expected_ids)
                        np.testing.assert_allclose(scores, expected_scores, rtol=1e-5, atol=1e-5)
                        assert notes == {"partial": False}
                request = ([f"fake:track:{V}", "Nightcall - Kavinsky", "nope"], "drive")
                out, expected = rec.recommend(*request), single.recommend(*request)
                assert {**out, "timings_ms": None, "version": None} == {**expected, "timings_ms": None, "version": None}
                assert out["version"] == version_of(folder)
                assert rec.find_songs("nightcall") == single.find_songs("nightcall")  # copies in the menu too
            finally:
                rec.close()
    finally:
        single.close()
        path.unlink()


def slices_of(base):
    return [base / "shards-2" / f"shard-{i}" for i in range(2)]


def test_a_shard_holds_what_the_switch_file_names(updated, tmp_path):
    base, update, _, _, _ = updated
    path = base / "switch.json"
    path.unlink(missing_ok=True)
    u, b = version_of(update), version_of(base)
    with shard_servers(slices_of(base)[:1], switch_file=path) as (address,):
        stub = shard_pb2_grpc.ShardStub(grpc.insecure_channel(address))
        info = lambda version="": stub.Info(shard_pb2.InfoRequest(version=version), timeout=60, wait_for_ready=True)
        lookup = lambda version, ids: stub.Lookup(shard_pb2.LookupRequest(version=version, ids=pack(np.array(ids))),
                                                  timeout=10)
        assert list(info().versions) == [b] and info().catalog == V
        switch.prepare(path, str(update.relative_to(base)))
        wait_for(lambda: list(info().versions) == [b, u], what="the update to load")
        assert info().version == b and info(u).catalog == V + COPIES  # the live one still described first
        assert unpack(lookup(u, [V]).retriever_inputs).shape == (1, 8)  # song V (shard 0 of 2) is the update's
        with pytest.raises(grpc.RpcError) as error:
            lookup(b, [V])
        assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        switch.flip(path)
        wait_for(lambda: info().version == u, what="the flip")
        assert list(info().versions) == [u, b]
        switch.finish(path)
        wait_for(lambda: list(info().versions) == [u], what="the base to be dropped")
        with pytest.raises(grpc.RpcError) as error:
            lookup(b, [0])
        assert error.value.code() == grpc.StatusCode.FAILED_PRECONDITION and f"not {b}" in error.value.details()
        switch.prepare(path, "updates/missing")  # a folder that can't load: logged, everything else unchanged
        time.sleep(1.5)
        assert list(info().versions) == [u]
        assert unpack(lookup(u, [V]).retriever_inputs).shape == (1, 8)
    path.unlink()


def test_a_front_desk_reports_shards_without_its_version_as_down(updated):
    base, update, _, _, _ = updated
    path = base / "switch.json"
    switch.write(path, ".", str(update.relative_to(base)))
    try:
        with shard_servers(slices_of(base), switch_file=path) as addresses:
            old = ShardedRecommender(base, addresses, "cpu", startup_wait=60)
            wait_for(lambda: old.shards_hold(version_of(update)), what="the update to load")
            new = ShardedRecommender(base, addresses, "cpu", startup_wait=60, update=update)
            try:
                assert all(s["up"] for s in old.shard_states()) and all(s["up"] for s in new.shard_states())
                switch.write(path, str(update.relative_to(base)))  # the shards let the base go
                wait_for(lambda: not old.shards_hold(version_of(base)), what="the base to be dropped")
                states = old.shard_states()
                assert [s["up"] for s in states] == [False, False]
                assert f"not {version_of(base)}" in states[0]["refuses"]
                assert all(s["up"] for s in new.shard_states())
            finally:
                old.close()
                new.close()
            with pytest.raises(ShardMismatch, match=f"not {version_of(base)}"):
                ShardedRecommender(base, addresses, "cpu", startup_wait=60)
    finally:
        path.unlink()


def test_a_profile_still_fits_an_update_and_reaches_its_new_songs(updated):
    base, update, whole, _, _ = updated
    path = base / "switch.json"
    switch.write(path, str(update.relative_to(base)))
    profile = make_profile("me", version_of(base), np.zeros(9), np.ones(len(ARTISTS)), np.full(V, 0.5), w=0.0,
                           alpha=1.0, beta=1.0)
    folder = base / "profiles"
    folder.mkdir(exist_ok=True)
    save_profile(folder / "me.npz", profile)
    try:
        with shard_servers(slices_of(base), switch_file=path) as addresses:
            rec = ShardedRecommender(base, addresses, "cpu", startup_wait=60, update=update)
            try:
                assert rec.load_profiles(folder) == 1  # built for the base: it fits the update
                out = rec.recommend([f"fake:track:{V}", "spotify:track:3"], profile="me", max_per_artist=0)
                assert out["personalized"] and len(out["recommendations"]) == 10
            finally:
                rec.close()
    finally:
        path.unlink()
        (folder / "me.npz").unlink()
    ids = np.array([V + 1, 2, V])  # songs past the profile's: no skips, their artists' bonus as usual
    got, scores = adjust(profile, ids, np.zeros(3, np.float32), np.array([0, 1, 2] * 15))
    assert got.tolist() == [V + 1, V, 2]
    np.testing.assert_allclose(scores, [1.0, 1.0, 0.5])


def post(port, songs, headers=None, timeout=10):
    body = json.dumps({"songs": songs, "max_per_artist": 0}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}/recommend", data=body,
                                     headers={"content-type": "application/json", **(headers or {})})
    with urllib.request.urlopen(request, timeout=timeout) as r:
        return json.loads(r.read())


def health(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as r:
        return json.loads(r.read())


def test_a_front_desk_flips_and_flips_back_under_traffic(updated, tmp_path):
    """Every answer, while the version switches and switches back, is one whole version's: exactly what that
    version alone answers, never a mix, never an error."""
    base, update, whole, _, _ = updated
    path = base / "switch.json"
    path.unlink(missing_ok=True)
    u, b = version_of(update), version_of(base)
    requests = [[f"spotify:track:{j}"] for j in (3, 4, 7, 9)] + [["spotify:track:1", "spotify:track:2"]]
    expected = {}
    for version, folder in ((b, base), (u, whole)):
        single = Recommender(folder, "cpu")
        try:
            for i, songs in enumerate(requests):
                expected[version, i] = [r["uri"] for r in single.recommend(songs, max_per_artist=0)["recommendations"]]
        finally:
            single.close()
    assert any(expected[u, i] != expected[b, i] for i in range(len(requests)))  # the update's answers differ
    log = open(tmp_path / "frontdesk.log", "w")
    with shard_servers(slices_of(base), switch_file=path) as addresses:
        port = free_ports(1)[0]
        desk = subprocess.Popen([sys.executable, "-m", "recsys.serve", "--bundle", str(base), "--shards", ",".join(addresses),
                                 "--port", str(port), "--batch", "4", "--switch", str(path)], cwd=ROOT, stdout=log,
                                stderr=log)
        try:
            wait_for(lambda: alive(port), what="the front desk")
            answers, stop = [], threading.Event()

            def traffic(k):
                i = k
                while not stop.is_set():
                    try:
                        r = post(port, requests[i % len(requests)])
                        answers.append((i % len(requests), r["version"], [x["uri"] for x in r["recommendations"]],
                                        r["partial"]))
                    except (urllib.error.URLError, OSError) as e:
                        answers.append((i % len(requests), "error", str(e), None))
                    i += 1
            workers = [threading.Thread(target=traffic, args=(k,)) for k in range(4)]
            for w in workers:
                w.start()
            with pytest.raises(urllib.error.HTTPError) as error:
                post(port, requests[0], {"X-Recsys-Standby": "1"})
            assert error.value.code == 409  # no standby yet
            switch.prepare(path, str(update.relative_to(base)))
            wait_for(lambda: (health(port)["switch"]["standby"] or {}).get("version") == u, what="the standby")
            assert post(port, requests[0], {"X-Recsys-Standby": "1"})["version"] == u
            switch.flip(path)
            wait_for(lambda: health(port)["version"] == u, what="the flip")
            time.sleep(1)
            switch.flip(path)  # undo
            wait_for(lambda: health(port)["version"] == b, what="the flip back")
            time.sleep(1)
            switch.finish(path)
            wait_for(lambda: health(port)["switch"]["standby"] is None, what="the standby to be let go")
            stop.set()
            for w in workers:
                w.join()
            report = health(port)
            assert report["status"] == "ok" and report["switch"]["retiring"] == [u]  # closed 30 s on
        finally:
            desk.terminate()
            desk.wait(timeout=30)
            log.close()
    path.unlink()
    assert not [a for a in answers if a[1] == "error"], [a for a in answers if a[1] == "error"][:3]
    by_version = {v: sum(a[1] == v for a in answers) for v in (b, u)}
    assert by_version[b] > 0 and by_version[u] > 0, by_version
    assert all(a[1] in (b, u) and a[2] == expected[a[1], a[0]] and a[3] is False for a in answers)


def alive(port):
    try:
        return health(port)["status"] == "ok"
    except (urllib.error.URLError, OSError):
        return False


class FakeRecommender:
    """Stands in for a front desk's recommender: its version, and the versions the shards hold (shared)."""

    def __init__(self, version, held):
        self.version, self.held, self.closed = version, held, False

    def shards_hold(self, version):
        return version in self.held

    def close(self):
        self.closed = True


def test_the_switcher_keeps_going_past_a_bad_folder_and_waits_for_every_shard(tmp_path):
    from types import SimpleNamespace

    from recsys.serve import Switcher
    for folder, version in ((tmp_path, "base"), (tmp_path / "updates" / "x", "x"), (tmp_path / "updates" / "y", "y")):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "manifest.json").write_text(json.dumps({"version": version}))
    held = {"base", "x"}
    app = SimpleNamespace(state=SimpleNamespace())
    path = tmp_path / "switch.json"
    built = []
    build = lambda update, share: built.append(FakeRecommender(version_of(update or tmp_path), held)) or built[-1]
    switcher = Switcher(app, tmp_path, path, build, grace=0.0)
    base = app.state.recommender
    assert base.version == "base" and app.state.standby is None
    switch.prepare(path, "updates/x")
    switcher.step(switch.read(path))
    assert app.state.standby.version == "x" and app.state.recommender is base
    held.discard("x")  # a shard restarted: it loads the standby only after the live one
    switch.flip(path)
    with pytest.raises(RuntimeError, match="not switching to updates/x yet"):
        switcher.step(switch.read(path))
    assert app.state.recommender is base  # still the old one: no request goes to a shard without x
    held.add("x")
    switcher.step(switch.read(path))
    assert app.state.recommender.version == "x" and app.state.standby is base
    switch.prepare(path, "updates/typo")  # no such folder: logged, and the rest of the step still happens
    with pytest.raises(RuntimeError, match="couldn't build updates/typo"):
        switcher.step(switch.read(path))
    assert app.state.standby is None and base.closed  # the old version let go and closed (grace 0)
    assert "updates/typo" in switcher.report()["failed"]
    held.add("y")
    switch.prepare(path, "updates/y")
    switcher.step(switch.read(path))
    assert app.state.standby.version == "y" and not app.state.recommender.closed
