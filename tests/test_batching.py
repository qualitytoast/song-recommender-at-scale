import threading

import numpy as np
import pytest

from recsys.batching import Batcher
from recsys.frontdesk import ShardedRecommender, ShardsUnavailable
from recsys.recommender import Recommender
from serving_helpers import write_tiny_bundle
from test_sharding import shard_servers, slices

REQUESTS = [([7], "road trip"), ([7, 21], "gym"), ([3, 17, 9], ""), ([1, 2, 3, 4, 5], "chill mix"), ([5, 5, 8], "trip"),
            ([30], "")]


@pytest.fixture(scope="module")
def sharded(tmp_path_factory):
    path, *_ = write_tiny_bundle(tmp_path_factory.mktemp("bundles"))
    single = Recommender(path, "cpu")
    with shard_servers(slices(path, 3), with_processes=True) as (addresses, procs):
        rec = ShardedRecommender(path, addresses, "cpu", deadline=1.0, startup_wait=60)
        yield rec, single, procs
        rec.close()
    single.close()


def test_a_batch_gives_each_request_its_own_answer(sharded):
    rec, single, _ = sharded
    requests = [(songs, single.encode_name(name), mix) for (songs, name), mix in zip(REQUESTS, [0.0, 0.5, 1.0] * 2)]
    batched = rec.recommend_batch(requests)  # each request with its own mix
    for (songs, name_ids, mix), (ids, scores, times, notes) in zip(requests, batched):
        expected_ids, expected_scores, _, _ = single.recommend_ids(songs, name_ids, mix)
        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_allclose(scores, expected_scores, rtol=1e-4, atol=1e-4)
        assert notes == {"partial": False} and times["batch size"] == len(requests)


def test_requests_arriving_together_are_answered_together(sharded):
    rec, single, _ = sharded
    batcher = Batcher(rec.recommend_batch, max_batch=8, wait_ms=50)
    requests = [(songs, single.encode_name(name)) for songs, name in REQUESTS]
    answers = [None] * len(requests)

    def ask(i):
        answers[i] = batcher.submit(requests[i])
    threads = [threading.Thread(target=ask, args=(i,)) for i in range(len(requests))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert max(batcher.sizes) > 1 and sum(batcher.sizes) == len(requests)
    for (songs, name_ids), (ids, *_) in zip(requests, answers):
        np.testing.assert_array_equal(ids, single.recommend_ids(songs, name_ids)[0])


def test_in_a_batch_a_dead_shard_only_fails_the_requests_it_holds_every_song_of(sharded):
    rec, single, procs = sharded
    procs[1].terminate()  # shard 1: songs 1, 4, 7, ...
    procs[1].wait(timeout=30)
    results = rec.recommend_batch([([4, 7], single.encode_name("")), ([3, 4], single.encode_name(""))])
    assert isinstance(results[0], ShardsUnavailable)  # both its songs live on shard 1
    ids, _, _, notes = results[1]
    assert notes == {"partial": True, "missing_shards": [1]} and len(ids) >= 10 and not (ids % 3 == 1).any()
