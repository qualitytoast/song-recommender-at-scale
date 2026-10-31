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
from recsys.pair_table import PairTable, save_pairs
from recsys.proto import shard_pb2, shard_pb2_grpc
from recsys.recommender import Recommender
from recsys.wire import pack, unpack
from serving_helpers import write_tiny_bundle

ROOT = Path(__file__).resolve().parents[1]


def free_ports(n):
    """n distinct free ports (all held open while choosing, so none is handed out twice)."""
    sockets = [socket.socket() for _ in range(n)]
    for s in sockets:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in sockets]
    for s in sockets:
        s.close()
    return ports


def slices(bundle, n):
    """Cut the bundle into n shard slices (in its own process: FAISS and PyTorch can't share one)."""
    subprocess.run([sys.executable, "-m", "recsys.shard_slices", "--bundle", str(bundle), "--shards", str(n)],
                   cwd=ROOT, check=True, capture_output=True)
    return [Path(bundle) / f"shards-{n}" / f"shard-{i}" for i in range(n)]


@contextmanager
def shard_servers(folders, with_processes=False):
    """A shard server process per slice folder; yields their addresses (and processes)."""
    procs, addresses = [], []
    for folder, port in zip(folders, free_ports(len(folders))):
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


def test_the_pair_table_keeps_every_count_exact(tmp_path):
    songs = 10
    pairs = {(0, 3): 1, (0, 8): 254, (2, 1): 255, (2, 5): 300, (2, 9): 262_470, (9, 0): 7, (9, 9): 256}
    keys = np.array([a * songs + b for a, b in pairs], dtype=np.int64)
    save_pairs(tmp_path, keys, np.array(list(pairs.values()), dtype=np.int32), songs)
    table = PairTable(tmp_path, songs)
    assert len(table) == 7
    assert table.small.dtype == np.uint8 and table.later.dtype == np.int32
    assert len(table.big_keys) == 4  # only the counts a byte can't hold are kept twice
    shuffled = np.random.RandomState(0).permutation(keys)
    np.testing.assert_array_equal(table.lookup(shuffled), [pairs[(k // songs, k % songs)] for k in shuffled])
    never = [(0, 4), (0, 9), (2, 0), (2, 6), (9, 5), (1, 3), (5, 5)]  # between, past and before a song's rows; no rows
    keys = np.array([a * songs + b for a, b in never], dtype=np.int64)
    np.testing.assert_array_equal(table.lookup(keys), np.zeros(len(keys)))
    assert table.lookup(np.zeros(0, dtype=np.int64)).shape == (0,)


def test_an_empty_pair_table_finds_nothing(tmp_path):
    save_pairs(tmp_path, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32), 10)
    table = PairTable(tmp_path, 10)
    assert len(table) == 0
    np.testing.assert_array_equal(table.lookup(np.array([0, 99], dtype=np.int64)), [0, 0])


@pytest.mark.parametrize("n", [1, 2, 3])
def test_sharded_answers_equal_the_single_process_ones(bundle, n):
    path, single = bundle
    with shard_servers(slices(path, n)) as addresses:
        rec = ShardedRecommender(path, addresses, "cpu", startup_wait=60)
        try:
            name = single.encode_name("road trip")
            for songs in ([7], [7, 21], [3, 17, 9], [1, 2, 3, 4, 5], [5, 5, 8]):
                for mix in (0.0, 0.5):
                    ids, scores, times, notes = rec.recommend_ids(songs, name, mix)
                    expected_ids, expected_scores, _, _ = single.recommend_ids(songs, name, mix)
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
            assert len(ids) >= 10 and not (ids % 3 == 1).any()  # nothing from shard 1
            assert [s["up"] for s in rec.shard_states()] == [True, False, True]
            with pytest.raises(ShardsUnavailable):
                rec.recommend_ids([4, 7], name)  # every input song lives on shard 1
        finally:
            rec.close()


def test_a_hung_shard_is_cut_off_by_the_deadline(bundle):
    path, single = bundle
    with shard_servers(slices(path, 3), with_processes=True) as (addresses, procs):
        rec = ShardedRecommender(path, addresses, "cpu", deadline=0.3, startup_wait=60, cooldown=1.0)
        try:
            name = single.encode_name("")
            procs[2].send_signal(signal.SIGSTOP)  # alive but not answering
            start = time.perf_counter()
            _, _, _, notes = rec.recommend_ids([3, 4], name)
            assert notes == {"partial": True, "missing_shards": [2]}
            assert time.perf_counter() - start < 2.0  # waited for a deadline, not forever
            start = time.perf_counter()
            _, _, _, notes = rec.recommend_ids([3, 4], name)  # the breaker skips shard 2: no waiting
            assert notes == {"partial": True, "missing_shards": [2]} and time.perf_counter() - start < 0.2
            procs[2].send_signal(signal.SIGCONT)
            time.sleep(1.1)  # the cooldown passes; the next call tries shard 2 again
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
            for address in addresses[:2]:  # both copies up (not still starting) before one is lost
                with grpc.insecure_channel(address) as channel:
                    shard_pb2_grpc.ShardStub(channel).Info(shard_pb2.InfoRequest(), timeout=60, wait_for_ready=True)
            time.sleep(0.3)  # the front desk's channel reconnects within 0.1-1 s
            procs[0].terminate()
            procs[0].wait(timeout=30)
            for _ in range(4):  # each call goes to a live copy (retried there if it hit the dead one)
                ids, _, _, notes = rec.recommend_ids([2, 3, 4], name)
                assert notes == {"partial": False}
                np.testing.assert_array_equal(ids, expected)
        finally:
            rec.close()


@pytest.fixture(scope="module")
def bundle16(tmp_path_factory):
    path, *_ = write_tiny_bundle(tmp_path_factory.mktemp("bundles16"), settings={"vector_bits": 16})
    single = Recommender(path, "cpu")
    yield path, single
    single.close()


def test_a_16_bit_bundle_keeps_its_vectors_at_16_bits_and_its_shards_answer_as_one_process(bundle16, bundle):
    path, single = bundle16
    assert json.loads((path / "settings.json").read_text())["vector_bits"] == 16
    folders = slices(path, 2)
    for folder in (path, *folders):
        for name in ("retriever_song_inputs", "ranker_song_inputs"):
            assert np.load(folder / f"{name}.npy").dtype == np.float16
    kinds = subprocess.run(  # FAISS in its own process (it can't share one with PyTorch)
        [sys.executable, "-c", "import faiss, sys; print(*(type(faiss.downcast_index(faiss.read_index(p))).__name__ "
         "for p in sys.argv[1:]))", *(str(f / "search_index.faiss") for f in (path, *folders))],
        cwd=ROOT, check=True, capture_output=True, text=True).stdout.split()
    assert set(kinds) == {"IndexIVFScalarQuantizer"}
    name = single.encode_name("road trip")
    requests = ([7], [7, 21], [3, 17, 9], [1, 2, 3, 4, 5])
    with shard_servers(folders) as addresses:
        rec = ShardedRecommender(path, addresses, "cpu", startup_wait=60)
        try:
            for songs in requests:  # the shards' 16-bit codes equal the whole index's: the same answers
                ids, scores, _, _ = rec.recommend_ids(songs, name, 0.0)
                expected_ids, expected_scores, _, _ = single.recommend_ids(songs, name, 0.0)
                np.testing.assert_array_equal(ids, expected_ids)
                np.testing.assert_allclose(scores, expected_scores, rtol=1e-5, atol=1e-5)
        finally:
            rec.close()
    full = bundle[1]  # the same model at 32 bits: nearly the same top 10s
    shared = [len(set(single.recommend_ids(s, name, 0.0)[0][:10]) & set(full.recommend_ids(s, name, 0.0)[0][:10]))
              for s in requests]
    assert np.mean(shared) >= 9
