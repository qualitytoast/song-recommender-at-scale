import time

import pytest
from fastapi.testclient import TestClient

from recsys.admission import Admission, Overloaded
from recsys.batching import Batcher
from recsys.recommender import Recommender
from recsys.serve import create_app
from serving_helpers import write_tiny_bundle


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    path, *_ = write_tiny_bundle(tmp_path_factory.mktemp("bundles"))
    return path


def test_admission_caps_requests_in_progress():
    a = Admission(max_pending=2)
    assert a.enter() and a.enter() and not a.enter()  # the third is turned away
    a.leave()
    assert a.enter()
    assert a.report()["rejected_full"] == 1 and a.report()["pending"] == 2


def test_a_request_that_waited_too_long_is_dropped():
    a = Admission(max_pending=8, max_wait_ms=100)
    a.check_started(time.monotonic())  # just arrived: fine
    with pytest.raises(Overloaded):
        a.check_started(time.monotonic() - 0.5)
    assert a.report()["rejected_waited"] == 1


def test_the_batcher_drops_requests_past_their_deadline_without_computing_them():
    computed = []
    batcher = Batcher(lambda requests: computed.extend(requests) or [r * 10 for r in requests], wait_ms=1)
    assert batcher.submit(3, deadline=time.monotonic() + 5) == 30
    with pytest.raises(Overloaded):
        batcher.submit(4, deadline=time.monotonic() - 1)
    assert computed == [3] and batcher.report()["dropped_waited_too_long"] == 1


def test_one_at_a_time_a_request_that_waited_too_long_is_dropped(bundle):
    rec = Recommender(bundle, "cpu")
    try:
        rec.admission = Admission(64, max_wait_ms=100)
        assert len(rec.recommend(["spotify:track:5"], arrived=time.monotonic())["recommendations"]) == 10
        with pytest.raises(Overloaded):
            rec.recommend(["spotify:track:5"], arrived=time.monotonic() - 1)
    finally:
        rec.close()


def test_a_full_front_desk_answers_503_at_once_and_the_menu_has_its_own_limit(bundle):
    with TestClient(create_app(bundle, "cpu", max_pending=0)) as client:  # every recommendation slot taken
        start = time.perf_counter()
        busy = client.post("/recommend", json={"songs": ["spotify:track:5"]})
        assert busy.status_code == 503 and busy.headers["retry-after"] == "1"
        assert time.perf_counter() - start < 0.5  # turned away, not queued
        assert client.get("/songs", params={"q": "night"}).status_code == 200  # typing still works
        assert client.get("/live").json() == {"ok": True}
        admission = client.get("/health").json()["admission"]["recommend"]
        assert admission["rejected_full"] == 1


def test_the_webpage_is_served_next_to_the_api(bundle):
    with TestClient(create_app(bundle, "cpu")) as client:
        page = client.get("/")
        assert page.status_code == 200 and "<title>Song recommender</title>" in page.text
        assert client.get("/app.js").status_code == 200 and client.get("/style.css").status_code == 200
        assert client.get("/health").json()["status"] == "ok"  # the API still wins over the static files
