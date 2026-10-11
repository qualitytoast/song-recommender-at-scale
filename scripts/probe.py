"""What one visitor sees during a load test: requests sent one at a time from a separate, unhurried
client, timed and grouped by HTTP status (a load generator that is itself overloaded adds its own delays
to every timing; this probe does not).

    python scripts/probe.py 30      # 30 seconds, while scripts/load_test.py runs
"""
import json, sys, time, urllib.error, urllib.request
import numpy as np
cases = json.load(open("runs/serving/cases/cases.json"))["cases"]
seconds, out = float(sys.argv[1]), {}
end, i = time.time() + seconds, 0
while time.time() < end:
    case = cases[i % len(cases)]; i += 1
    body = json.dumps({"songs": case["songs"], "name": case["name"]}).encode()
    start = time.perf_counter()
    try:
        urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/recommend", data=body,
                               headers={"content-type": "application/json"}), timeout=10).read()
        status = 200
    except urllib.error.HTTPError as e:
        status = e.code
    out.setdefault(status, []).append((time.perf_counter() - start) * 1000)
    time.sleep(0.05)
for status, ms in sorted(out.items()):
    print(f"probe: HTTP {status}: {len(ms)} requests, p50 {np.percentile(ms, 50):.1f} ms, p99 {np.percentile(ms, 99):.1f} ms")
