"""Download one big file over several connections at once, resumable, then check it against its SHA-256.

A slow server often gives each connection only so much speed; several connections that each fetch a different
piece of the file (HTTP Range requests: "bytes N to M") add up. Pieces are written straight into their place in
the output file, and a small progress file (<out>.pieces.json) records which ones are done, so stopping (Ctrl-C,
sleep, shutdown) loses at most the pieces in flight: run the same command again to continue. A file that another
program (e.g. curl -C -) downloaded from the start counts as already done up to its size.

At the end the whole file is checked against the server's .sha256 (or a .sha256 file next to it): the same
bytes as a one-connection download, just sooner.

    python scripts/parallel_download.py <url> --out ~/Downloads/<file> [--connections 8]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

PIECE = 64 << 20   # bytes per piece
CHUNK = 1 << 20    # bytes per write


def piece_ranges(total, piece=PIECE):
    """[(start, end)] covering bytes 0..total-1, end inclusive."""
    return [(start, min(start + piece, total) - 1) for start in range(0, total, piece)]


def load_done(progress_path, out_path, ranges):
    """Indices of finished pieces: from the progress file, or, for a file without one (downloaded front to back by
    another program), every piece wholly inside its current size."""
    if progress_path.exists():
        return set(json.loads(progress_path.read_text())["done"])
    if out_path.exists():
        size = out_path.stat().st_size
        return {i for i, (start, end) in enumerate(ranges) if end < size}
    return set()


def save_done(progress_path, done, total, piece):
    tmp = progress_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"total": total, "piece": piece, "done": sorted(done)}))
    tmp.replace(progress_path)  # all at once: never a half-written progress file


def other_writer(out_path):
    """True if another download (curl, wget) is still running on the same file name."""
    try:
        found = subprocess.run(["pgrep", "-f", f"(curl|wget).*{out_path.name}"], capture_output=True, text=True)
    except FileNotFoundError:
        return False
    return bool(found.stdout.strip())


def fetch_piece(session, url, fd, start, end, stop, counter, lock, retries=20, timeout=60):
    """Download bytes start..end into the file at the same position; retry the piece on errors."""
    for attempt in range(retries + 1):
        if stop.is_set():
            return False
        written = 0
        try:
            with session.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=timeout) as r:
                if r.status_code != 206:
                    raise OSError(f"the server didn't send the requested piece (status {r.status_code})")
                for chunk in r.iter_content(CHUNK):
                    if stop.is_set():
                        return False
                    chunk = chunk[:end - start + 1 - written]  # never past the piece, even if sent more
                    os.pwrite(fd, chunk, start + written)
                    written += len(chunk)
                    with lock:
                        counter[0] += len(chunk)
                    if written == end - start + 1:
                        break
            if written != end - start + 1:
                raise OSError(f"piece ended early ({written:,} of {end - start + 1:,} bytes)")
            return True
        except (OSError, requests.RequestException) as e:
            with lock:
                counter[0] -= written  # those bytes will come again
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 60))
            print(f"\npiece at {start:,}: {type(e).__name__}: {e}; retrying", flush=True)
    return False


def expected_sha256(url, out_path):
    local = out_path.with_name(out_path.name + ".sha256")
    try:
        text = local.read_text() if local.exists() else requests.get(url + ".sha256", timeout=60).text
    except (OSError, requests.RequestException):
        return None
    digest = text.split()[0].lower() if text.split() else ""
    return digest if len(digest) == 64 else None


def file_sha256(path, block=16 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, out, connections=8, piece=PIECE, check=True, report_every=2.0):
    """Download url to out (resuming) and check its SHA-256. Returns True if complete and it matches (or there's
    no checksum to compare with), False if stopped or it doesn't match."""
    out_path = Path(out).expanduser()
    progress_path = out_path.with_name(out_path.name + ".pieces.json")
    if other_writer(out_path):
        raise SystemExit(f"another download is still writing {out_path.name}: stop it first (Ctrl-C in its window)")
    session = requests.Session()
    for scheme in ("https://", "http://"):
        session.mount(scheme, requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=connections))
    head = session.head(url, timeout=60, allow_redirects=True)
    head.raise_for_status()
    total = int(head.headers["Content-Length"])
    if head.headers.get("Accept-Ranges", "").lower() != "bytes":
        raise SystemExit("the server doesn't support downloading in pieces (no Accept-Ranges: bytes)")
    ranges = piece_ranges(total, piece)
    done = load_done(progress_path, out_path, ranges)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_done(progress_path, done, total, piece)  # before growing the file: its size no longer says what's done
    fd = os.open(out_path, os.O_RDWR | os.O_CREAT)
    os.ftruncate(fd, total)  # full size up front; unwritten parts take no disk space until filled
    already = sum(end - start + 1 for i, (start, end) in enumerate(ranges) if i in done)
    todo = [i for i in range(len(ranges)) if i not in done]
    print(f"{out_path.name}: {total / 1e9:,.1f} GB, {already / 1e9:,.1f} GB already here, "
          f"{len(todo):,} pieces of {piece >> 20} MB to fetch over {connections} connections", flush=True)

    stop, lock, counter = threading.Event(), threading.Lock(), [0]
    finished_ok = True

    def run(i):
        start, end = ranges[i]
        if fetch_piece(session, url, fd, start, end, stop, counter, lock):
            with lock:
                done.add(i)
                save_done(progress_path, done, total, piece)

    began, samples = time.monotonic(), [(time.monotonic(), 0)]
    try:
        with ThreadPoolExecutor(connections) as pool:
            futures = [pool.submit(run, i) for i in todo]
            try:
                while not all(f.done() for f in futures):
                    time.sleep(report_every)
                    now, got = time.monotonic(), counter[0]
                    samples.append((now, got))
                    samples = [s for s in samples if now - s[0] <= 30] or samples[-1:]
                    speed = (got - samples[0][1]) / max(now - samples[0][0], 1e-9)
                    have = already + got
                    left = (total - have) / speed if speed > 0 else float("inf")
                    eta = time.strftime("%H:%M:%S", time.gmtime(left)) if left < 360_000 else "--:--:--"
                    print(f"\r{have / total:6.1%}  {have / 1e9:,.1f} of {total / 1e9:,.1f} GB  "
                          f"{speed / 1e6:5.1f} MB/s  {eta} left   ", end="", flush=True)
            except KeyboardInterrupt:  # inside the pool, so waiting pieces are cancelled, not run
                stop.set()
                finished_ok = False
                for f in futures:
                    f.cancel()
                print("\nstopping: finished pieces are saved; run the same command to continue", flush=True)
        if finished_ok:
            for f in futures:
                f.result()  # raise a piece's error, if any
    finally:
        os.close(fd)
    if not finished_ok or len(done) < len(ranges):
        return False
    print(f"\ndownloaded in {(time.monotonic() - began) / 3600:.1f} h", flush=True)
    if not check:
        progress_path.unlink()
        return True
    expected = expected_sha256(url, out_path)
    if expected is None:
        print("no .sha256 to check against; done")
        progress_path.unlink()
        return True
    print("checking the SHA-256 of the whole file (a few minutes)...", flush=True)
    actual = file_sha256(out_path)
    if actual != expected:
        print(f"CHECKSUM MISMATCH: got {actual}, expected {expected}. Don't use this file; tell Claude.")
        return False
    progress_path.unlink()
    print(f"checksum OK ({actual}): the file is complete and intact")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("url")
    ap.add_argument("--out", required=True, help="where to save it (an existing partial file is continued)")
    ap.add_argument("--connections", type=int, default=8)
    args = ap.parse_args()
    sys.exit(0 if download(args.url, args.out, args.connections) else 1)


if __name__ == "__main__":
    main()
