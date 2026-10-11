"""A zero-downtime update under load (blue/green), on Docker (stage 5).

    python scripts/bluegreen.py --bundle bundles/<bundle> --cases runs/serving/cases --out runs/serving/bluegreen.json

  1. "blue": the running version, a complete stack (4 shards + front desk, docker/bluegreen/stack.yml),
     behind the entrance nginx (edge.yml) on port 8000; requests start flowing and keep flowing
  2. "green": a new version is made (the bundle copied as a new release: its files hard-linked, so it takes
     no disk, settings.json noting the release, so the version differs) and started as a second stack
  3. once green's front desk is healthy, the entrance is switched to it (nginx reload: requests already
     in progress finish on blue, new ones go to green)
  4. blue is stopped once it has drained
Every response carries the version that answered it; the summary counts answers per version and
failures, and when blue answered last and green first. The main docker-compose.yml system is stopped
first (it uses the same port) and started again at the end.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recsys.export import write_manifest  # noqa: E402

ENV = {**os.environ, "PATH": f"{Path.home()}/.docker/bin:{os.environ['PATH']}"}
STACK, EDGE = "docker/bluegreen/stack.yml", "docker/bluegreen/edge.yml"


def run(*cmd, env=None, check=True):
    return subprocess.run(cmd, cwd=ROOT, env={**ENV, **(env or {})}, check=check, capture_output=True, text=True)


def stack(name, bundle, *args):
    return run("docker", "compose", "-p", name, "-f", STACK, *args, env={"STACK": name, "BUNDLE": str(bundle.resolve())})


def new_release(bundle):
    """bundle as a new version: hard-linked files, settings.json with a release note, its own manifest
    and version, and 4 shard slices."""
    staging = bundle.parent / f".{bundle.name}-release"
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(bundle, staging, copy_function=os.link, ignore=shutil.ignore_patterns("shards-*", "manifest.json"))
    settings = json.loads((staging / "settings.json").read_text())
    (staging / "settings.json").unlink()  # a new file, not the hard-linked original
    (staging / "settings.json").write_text(json.dumps({**settings, "release": time.strftime("%Y-%m-%d %H:%M:%S")},
                                                      indent=2))
    version = write_manifest(staging)
    final = bundle.parent / f"{bundle.name.rsplit('-', 1)[0]}-{version}"
    shutil.rmtree(final, ignore_errors=True)
    staging.rename(final)
    run(sys.executable, "-m", "recsys.shard_slices", "--bundle", str(final), "--shards", "4")
    return final, version


def write_edge(active):
    """The entrance's nginx config, sending everything to the `active` stack's front desk. Written in place
    (same file), as the container sees the file Docker mounted, not a replacement."""
    with open(ROOT / "docker/bluegreen/edge.conf", "w") as f:
        f.write(f"""events {{}}
http {{
  upstream active {{ server {active}-frontdesk:8000; keepalive 32; }}
  server {{
    listen 80;
    location / {{
      proxy_pass http://active;
      proxy_http_version 1.1;
      proxy_set_header Connection "";
      proxy_read_timeout 10s;
    }}
  }}
}}
""")


def wait_healthy(container, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        r = run("docker", "inspect", "--format", "{{.State.Health.Status}}", container, check=False)
        if r.stdout.strip() == "healthy":
            return time.time() - start
        time.sleep(1)
    raise TimeoutError(f"{container} not healthy")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    blue_bundle = Path(args.bundle)
    blue_version = json.loads((blue_bundle / "manifest.json").read_text())["version"]
    cases = json.loads((Path(args.cases) / "cases.json").read_text())["cases"]

    run("docker", "compose", "down")  # the main system uses port 8000
    run("docker", "network", "create", "recsys-edge", check=False)
    green_bundle, green_version = new_release(blue_bundle)
    print(f"blue = {blue_version}, green = {green_version}", flush=True)
    stack("blue", blue_bundle, "up", "-d")
    wait_healthy("blue-frontdesk-1")
    write_edge("blue")
    run("docker", "compose", "-p", "edge", "-f", EDGE, "up", "-d")
    while True:
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
            break
        except OSError:
            time.sleep(0.5)

    log, stop, t0 = [], threading.Event(), time.perf_counter()

    def traffic():
        i = 0
        while not stop.is_set():
            case = cases[i % len(cases)]
            body = json.dumps({"songs": case["songs"], "name": case["name"]}).encode()
            start = time.perf_counter()
            try:
                r = json.loads(urllib.request.urlopen(urllib.request.Request(
                    "http://127.0.0.1:8000/recommend", data=body, headers={"content-type": "application/json"}),
                    timeout=10).read())
                status, version = ("partial" if r.get("partial") else "ok"), r["version"]
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                status, version = f"error {getattr(e, 'code', type(e).__name__)}", None
            log.append({"t": start - t0, "ms": (time.perf_counter() - start) * 1000, "status": status, "version": version})
            i += 1
    worker = threading.Thread(target=traffic)
    worker.start()
    marks = {}
    time.sleep(10)
    marks["green starting"] = time.perf_counter() - t0
    stack("green", green_bundle, "up", "-d")
    marks["green healthy after s"] = wait_healthy("green-frontdesk-1")
    marks["switch"] = time.perf_counter() - t0
    write_edge("green")
    run("docker", "compose", "-p", "edge", "-f", EDGE, "exec", "-T", "edge", "nginx", "-s", "reload")
    time.sleep(5)  # blue finishes what it was answering
    marks["blue stopping"] = time.perf_counter() - t0
    stack("blue", blue_bundle, "down")
    time.sleep(10)
    stop.set()
    worker.join()

    ms = np.array([r["ms"] for r in log])
    by_version = {v: sum(r["version"] == v for r in log) for v in (blue_version, green_version)}
    errors = [r for r in log if r["status"].startswith("error")]
    near = np.array([r["ms"] for r in log if abs(r["t"] - marks["switch"]) < 2])
    summary = {"requests": len(log), "answers_by_version": {f"blue {blue_version}": by_version[blue_version],
                                                            f"green {green_version}": by_version[green_version]},
               "errors": len(errors), "partial": sum(r["status"] == "partial" for r in log),
               "last_blue_answer_s": max(r["t"] for r in log if r["version"] == blue_version) - marks["switch"],
               "first_green_answer_s": min(r["t"] for r in log if r["version"] == green_version) - marks["switch"],
               "p50_ms": float(np.percentile(ms, 50)), "p99_ms": float(np.percentile(ms, 99)),
               "within_2s_of_switch": {"requests": len(near), "p50_ms": float(np.percentile(near, 50)),
                                       "p99_ms": float(np.percentile(near, 99)), "max_ms": float(near.max())},
               "marks": marks}
    Path(args.out).write_text(json.dumps({"summary": summary, "log": log}, indent=1))
    print(json.dumps(summary, indent=2))

    run("docker", "compose", "-p", "green", "-f", STACK, "down", env={"STACK": "green", "BUNDLE": str(green_bundle.resolve())})
    run("docker", "compose", "-p", "edge", "-f", EDGE, "down")
    run("docker", "compose", "up", "-d")  # the main system again


if __name__ == "__main__":
    main()
