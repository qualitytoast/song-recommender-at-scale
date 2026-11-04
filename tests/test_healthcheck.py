import http.server
import re
import signal
import subprocess
import sys
import threading
import time

import pytest

from recsys.healthcheck import SelfCheck, http_ok
from serving_helpers import ROOT, free_ports, shard_servers, slices, write_tiny_bundle


def wait_for(condition, seconds=30.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.05)
    return False


def alive_check(compose):
    """The containers' health check command from a compose file, as the shell gets it ($$ is compose's literal $)."""
    line = next(line for line in (ROOT / compose).read_text().splitlines() if line.startswith("x-alive-check:"))
    return re.search(r'"CMD-SHELL", "(.*)"\]', line).group(1).replace("$$", "$")


def test_a_self_check_writes_the_time_only_while_its_check_passes(tmp_path):
    path, state = tmp_path / "healthy", {"ok": False}

    def check():
        if state["ok"] is None:
            raise ConnectionError("no answer")
        return state["ok"]
    SelfCheck(check, path, every=0.02)
    time.sleep(0.2)
    assert not path.exists()  # not serving yet: nothing written
    state["ok"] = True
    assert wait_for(path.exists)
    assert abs(int(path.read_text()) - time.time()) <= 2
    for failing in (False, None):  # says not serving, or raises: the file is left as it was
        state["ok"] = failing
        time.sleep(0.1)
        before = path.stat().st_mtime_ns
        time.sleep(0.2)
        assert path.stat().st_mtime_ns == before
    state["ok"] = True  # and it keeps going after a failure
    assert wait_for(lambda: path.stat().st_mtime_ns != before)
    assert not (tmp_path / "healthy.new").exists()


@pytest.mark.parametrize("compose", ["docker-compose.yml", "docker/bluegreen/stack.yml"])
def test_dockers_check_passes_only_on_a_file_written_in_the_last_10_seconds(compose, tmp_path):
    command = alive_check(compose)
    assert "/dev/shm/healthy" in command
    path = tmp_path / "healthy"
    run = lambda: subprocess.run(["sh", "-c", command.replace("/dev/shm/healthy", str(path))],
                                 capture_output=True).returncode
    assert run() != 0  # no file yet: a container that just started
    path.write_text(f"{int(time.time())}\n")
    assert run() == 0
    path.write_text(f"{int(time.time()) - 20}\n")  # last written 20 s ago: stuck
    assert run() != 0


def test_every_server_in_compose_writes_the_file_dockers_check_reads():
    for compose, servers in (("docker-compose.yml", 5), ("docker/bluegreen/stack.yml", 5)):
        text = (ROOT / compose).read_text()
        assert text.count("--health-file /dev/shm/healthy") == servers, compose
        assert "recsys.healthcheck" not in text  # no Python started per check
        assert text.count("test: *alive-check") == 2, compose  # the shards' anchor and the front desk


def test_a_shard_checks_itself_over_grpc_and_a_one_off_check_still_works(tmp_path):
    path, *_ = write_tiny_bundle(tmp_path / "bundles")
    with shard_servers(slices(path, 1), with_processes=True, health_dir=tmp_path) as (addresses, procs):
        healthy = tmp_path / "healthy-0"
        assert wait_for(healthy.exists, 60), "the shard never wrote its health file"
        assert abs(int(healthy.read_text()) - time.time()) <= 3
        one_off = subprocess.run([sys.executable, "-m", "recsys.healthcheck", "grpc", addresses[0]], cwd=ROOT)
        assert one_off.returncode == 0
        procs[0].send_signal(signal.SIGSTOP)  # frozen: no more writes
        time.sleep(0.5)
        frozen = healthy.stat().st_mtime_ns
        time.sleep(2.5)
        assert healthy.stat().st_mtime_ns == frozen


def test_the_front_desks_check_asks_its_live_page(tmp_path):
    class Live(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/live" else 404)
            self.end_headers()

        def log_message(self, *args):
            pass
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Live)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        assert http_ok(f"http://127.0.0.1:{port}/live")() is True
        SelfCheck(http_ok(f"http://127.0.0.1:{port}/live"), tmp_path / "healthy", every=0.02)
        assert wait_for((tmp_path / "healthy").exists)
    finally:
        server.shutdown()
    nobody = free_ports(1)[0]
    with pytest.raises(OSError):
        http_ok(f"http://127.0.0.1:{nobody}/live", timeout=1)()  # nothing answering: no write
