import json
import shutil
import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import grpc
import numpy as np
import pytest

from recsys.frontdesk import ShardMismatch, ShardedRecommender, merge_top
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
def shard_servers(folders):
    """A shard server process per slice folder; yields their addresses."""
    procs, addresses = [], []
    for folder in folders:
        port = free_port()
        procs.append(subprocess.Popen([sys.executable, "-m", "recsys.shard", "--slice", str(folder), "--port", str(port),
                                       "--host", "127.0.0.1"], cwd=ROOT, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL))
        addresses.append(f"127.0.0.1:{port}")
    try:
        yield addresses
    finally:
        for p in procs:
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
                ids, scores, times = rec.recommend_ids(songs, name)
                expected_ids, expected_scores, _ = single.recommend_ids(songs, name)
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
