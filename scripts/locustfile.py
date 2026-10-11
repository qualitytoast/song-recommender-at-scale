"""Simulated users for load tests (locust): each types a song name into the menu (/songs, a few
prefixes as they type) and asks for recommendations for a held-out playlist (/recommend).

    locust -f scripts/locustfile.py --headless -u 32 -r 32 --run-time 30s --host http://localhost:8000

CASES points at serving_check's cases (default runs/serving/cases/cases.json). THINK_MIN and THINK_MAX
(seconds, default 0 and 0.05) set the pause between a user's actions: near 0 for a stress test, a few
seconds for people browsing. OUTCOMES, if set, is a file that gets every request's (endpoint, HTTP status,
milliseconds) at the end, so answered requests and fast "busy" (503) refusals can be told apart.
"""
import json
import os
import random

from locust import HttpUser, between, events, task

CASES = json.load(open(os.environ.get("CASES", "runs/serving/cases/cases.json")))["cases"]
OUTCOMES = []


@events.request.add_listener
def record(request_type, name, response_time, response_length, response=None, exception=None, **kwargs):
    OUTCOMES.append((name, getattr(response, "status_code", 0) or 0, response_time))


@events.quitting.add_listener
def save(environment, **kwargs):
    if os.environ.get("OUTCOMES"):
        with open(os.environ["OUTCOMES"], "w") as f:
            json.dump(OUTCOMES, f)


WORDS = ["nightcall", "humble", "beyonce", "mr. brightside", "drake", "kavinsky", "taylor", "dancing queen"]


class Listener(HttpUser):
    wait_time = between(float(os.environ.get("THINK_MIN", 0.0)), float(os.environ.get("THINK_MAX", 0.05)))

    @task(3)
    def type_a_name(self):
        word = random.choice(WORDS)
        for n in (2, 4, len(word)):  # the menu as the user types
            self.client.get("/songs", params={"q": word[:n]}, name="/songs")

    @task(1)
    def recommend(self):
        case = random.choice(CASES)
        with self.client.post("/recommend", json={"songs": case["songs"], "name": case["name"]}, name="/recommend",
                              catch_response=True) as response:
            if response.status_code == 503:
                response.failure("busy (503)")
