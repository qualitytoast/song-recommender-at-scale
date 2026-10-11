"""Simulated users for load tests (locust): each types a song name into the menu (/songs, a few
prefixes as they type) and asks for recommendations for a held-out playlist (/recommend).

    locust -f scripts/locustfile.py --headless -u 32 -r 32 --run-time 30s --host http://localhost:8000

CASES points at serving_check's cases (default runs/serving/cases/cases.json). THINK_MIN and THINK_MAX
(seconds, default 0 and 0.05) set the pause between a user's actions: near 0 for a stress test, a few
seconds for people browsing.
"""
import json
import os
import random

from locust import HttpUser, between, task

CASES = json.load(open(os.environ.get("CASES", "runs/serving/cases/cases.json")))["cases"]
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
        self.client.post("/recommend", json={"songs": case["songs"], "name": case["name"]}, name="/recommend")
