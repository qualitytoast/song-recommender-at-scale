import json
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import grpc
import numpy as np
import pytest

from recsys.frontdesk import ShardMismatch, ShardedRecommender, ShardsUnavailable, merge_top
from recsys.proto import shard_pb2, shard_pb2_grpc
from recsys.recommender import Recommender
from recsys.wire import pack, unpack
from serving_helpers import write_tiny_bundle

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def slices(bundle, n):
    """Cut the bundle into n shard slices (in its own process: FAISS and PyTorch can't share one)."""
    subprocess.run([sys.executable, "-m", "recsys.shard_slices", "--bundle", str(bundle), "--shards", str(n)],
                   cwd=ROOT, check=True, capture_output=True)
    return [Path(bundle) / f"shards-{n}" / f"shard-{i}" for i in range(n)]


@contextmanager
def shard_servers(folders, with_processes=False):
    """A shard server process per slice folder; yields their addresses (and processes)."""
    procs, addresses = [], []
    for folder in folders:
        port = free_port()
        procs.append(subprocess.Popen([sys.executable, "-m", "recsys.shard", "--slice", str(folder), "--port", str(port),
                                       "--host", "127.0.0.1"], cwd=ROOT, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL))
        addresses.append(f"127.0.0.1:{port}")
    try:
        yield (addresses, procs) if with_processes else addresses
    finally:
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGCONT)
                p.terminate()
            p.wait(timeout=30)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    path, *_ = write_tiny_bundle(tmp_path_factory.mktemp("bundles"))
    single = Recommender(path, "cpu")
    yield path, single
    single.close()


def test_merging_keeps_the_best_and_breaks_ties_by_song_id():
    ids, scores = merge_top([np.array([[7, 3, -1]]), np.array([[2, 9, 4]])],
                            [np.array([[5.0, 1.0, -3e38]]), np.array([[5.0, 4.0, 0.5]])], 4)
    np.testing.assert_array_equal(ids, [[2, 7, 9, 3]])
    np.testing.assert_array_equal(scores, [[5.0, 5.0, 4.0, 1.0]])


@pytest.mark.parametrize("n", [1, 2, 3])
def test_sharded_answers_equal_the_single_process_ones(bundle, n):
    path, single = bundle
    with shard_servers(slices(path, n)) as addresses:
        rec = ShardedRecommender(path, addresses, "cpu", startup_wait=60)
        try:
            name = single.encode_name("road trip")
            for songs in ([7], [7, 21], [3, 17, 9], [1, 2, 3, 4, 5], [5, 5, 8]):
                ids, scores, times, notes = rec.recommend_ids(songs, name)
                expected_ids, expected_scores, _, _ = single.recommend_ids(songs, name)
                np.testing.assert_array_equal(ids, expected_ids)
                np.testing.assert_allclose(scores, expected_scores, rtol=1e-5, atol=1e-5)
            assert set(times) == {"shards: input songs", "retriever", "shards: search", "shards: candidates",
                                  "ranker inputs", "ranker"}
            request = (["Nightcall - Kavinsky", "nope", "spotify:track:9"], "drive")
            out, expected = rec.recommend(*request), single.recommend(*request)
            assert {**out, "timings_ms": None} == {**expected, "timings_ms": None}
        finally:
            rec.close()


def test_the_front_desk_refuses_shards_in_the_wrong_order_or_of_another_version(bundle, tmp_path):
    path, _ = bundle
    folders = slices(path, 2)
    with shard_servers(folders[::-1]) as addresses:
        with pytest.raises(ShardMismatch, match="shard 1 of 2, not 0 of 2"):
            ShardedRecommender(path, addresses, startup_wait=60)
    stale = tmp_path / "shard-0"
    shutil.copytree(folders[0], stale)
    info = json.loads((stale / "info.json").read_text())
    (stale / "info.json").write_text(json.dumps({**info, "version": "000000000000"}))
    with shard_servers([stale, folders[1]]) as addresses:
        with pytest.raises(ShardMismatch, match="version 000000000000"):
            ShardedRecommender(path, addresses, startup_wait=60)


def test_a_shard_refuses_another_version_and_songs_it_does_not_hold(bundle):
    path, _ = bundle
    folders = slices(path, 2)
    with shard_servers(folders[:1]) as (address,):
        stub = shard_pb2_grpc.ShardStub(grpc.insecure_channel(address))
        info = stub.Info(shard_pb2.InfoRequest(), timeout=60, wait_for_ready=True)
        with pytest.raises(grpc.RpcError) as error:
            stub.Search(shard_pb2.SearchRequest(version="nope", queries=pack(np.zeros((1, 9), np.float32)), k=3), timeout=10)
        assert error.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        with pytest.raises(grpc.RpcError) as error:
            stub.Lookup(shard_pb2.LookupRequest(version=info.version, ids=pack(np.array([1], np.int64))), timeout=10)
        assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT  # song 1 lives on shard 1
        reply = stub.Lookup(shard_pb2.LookupRequest(version=info.version, ids=pack(np.array([4, 0], np.int64))), timeout=10)
        assert unpack(reply.retriever_inputs).shape == (2, 8)


def test_a_crashed_shard_gives_a_flagged_partial_answer_without_its_songs(bundle):
    path, single = bundle
    with shard_servers(slices(path, 3), with_processes=True) as (addresses, procs):
        rec = ShardedRecommender(path, addresses, "cpu", deadline=1.0, startup_wait=60)
        try:
            name = single.encode_name("")
            ids, _, _, notes = rec.recommend_ids([3, 4, 5], name)
            assert notes == {"partial": False}
            procs[1].terminate()
            procs[1].wait(timeout=30)
            ids, _, _, notes = rec.recommend_ids([3, 4, 5], name)  # song 4 lives on shard 1
            assert notes == {"partial": True, "missing_shards": [1]}
            assert len(ids) == 10 and not (ids % 3 == 1).any()  # nothing from shard 1
            assert [s["up"] for s in rec.shard_states()] == [True, False, True]
            with pytest.raises(ShardsUnavailable):
                rec.recommend_ids([4, 7], name)  # every input song lives on shard 1
        finally:
            rec.close()


def test_a_hung_shard_is_cut_off_by_the_deadline(bundle):
    path, single = bundle
    with shard_servers(slices(path, 3), with_processes=True) as (addresses, procs):
        rec = ShardedRecommender(path, addresses, "cpu", deadline=0.3, startup_wait=60)
        try:
            name = single.encode_name("")
            procs[2].send_signal(signal.SIGSTOP)  # alive but not answering
            start = time.perf_counter()
            _, _, _, notes = rec.recommend_ids([3, 4], name)
            assert notes == {"partial": True, "missing_shards": [2]}
            assert time.perf_counter() - start < 2.0  # waited for the deadlines, not forever
            procs[2].send_signal(signal.SIGCONT)
            ids, _, _, notes = rec.recommend_ids([3, 4], name)
            assert notes == {"partial": False}
            np.testing.assert_array_equal(ids, single.recommend_ids([3, 4], name)[0])
        finally:
            rec.close()


def test_a_shard_with_two_copies_survives_losing_one(bundle):
    path, single = bundle
    folders = slices(path, 2)
    with shard_servers([folders[0], folders[0], folders[1]], with_processes=True) as (addresses, procs):
        shards = [f"{addresses[0]}|{addresses[1]}", addresses[2]]  # shard 0 twice
        rec = ShardedRecommender(path, shards, "cpu", deadline=1.0, startup_wait=60)
        try:
            name = single.encode_name("road trip")
            expected = single.recommend_ids([2, 3, 4], name)[0]
            procs[0].terminate()
            procs[0].wait(timeout=30)
            for _ in range(4):  # each call goes to a live copy (retried there if it hit the dead one)
                ids, _, _, notes = rec.recommend_ids([2, 3, 4], name)
                assert notes == {"partial": False}
                np.testing.assert_array_equal(ids, expected)
        finally:
            rec.close()
