"""Which version of a bundle the system serves: the switch file, <bundle>/switch.json, the only pointer.

    {"live": ".", "standby": "updates/2026-11-03-<version>"}

Each value is a folder relative to the bundle: "." is the bundle itself (the base), an update is a folder under
updates/ (recsys/updates.py). The shards hold the live and the standby version; the front desk answers with the
live one and keeps the standby ready. A switch is two steps, so all shards change at once:
  1. prepare: name the new folder as standby. Every shard loads it beside the live one, and the front desk builds
     a recommender for it. Nothing changes for requests.
  2. flip: swap live and standby. The front desk swaps one reference, so every new request uses the new version
     on all shards, while a request already running finishes on the old one (the shards still hold it).
Then, once the new version has proved itself, finish: standby null, and the old version is dropped. Undo before
that is another flip; later, it's a prepare and a flip back to an older folder.

No file means the base alone ({"live": ".", "standby": null}), as before switching existed. Shards and front
desks read the file every second (follow). It's written whole, by renaming a finished file over it, so a reader
never sees half of one. Imports nothing heavy (no numpy, FAISS or PyTorch).
"""
import ctypes
import json
import os
import threading
import time
from pathlib import Path

BASE = "."


def say(message):
    print(message, flush=True)  # a container's log: unbuffered, so lines show when they happen


def give_back():
    """Return freed memory to the system after a version is dropped: glibc keeps part of it otherwise (~85 MB of a
    shard's ~0.5 GB stayed after a switch). Nothing where there's no glibc (a Mac)."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def read(path):
    """{"live": folder, "standby": folder or None} from the switch file (the base alone if there is none)."""
    try:
        state = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {"live": BASE, "standby": None}
    if not isinstance(state, dict) or not isinstance(state.get("live"), str) or \
            not isinstance(state.get("standby"), (str, type(None))):
        raise ValueError(f"{path}: not a switch file: {state!r}")
    return {"live": state["live"], "standby": state["standby"]}


def write(path, live, standby=None):
    """Replace the switch file in one step (a finished file renamed over it)."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps({"live": live, "standby": standby}) + "\n")
    os.replace(temporary, path)


def prepare(path, folder):
    """Step 1: folder becomes the standby (the old standby, if any, is dropped)."""
    write(path, read(path)["live"], folder)


def flip(path):
    """Step 2: the standby becomes live and the live one the standby."""
    state = read(path)
    if state["standby"] is None:
        raise ValueError(f"{path}: no standby to flip to")
    write(path, state["standby"], state["live"])


def finish(path):
    """Drop the standby: the shards and front desks let it go."""
    write(path, read(path)["live"], None)


def wanted(state):
    """The folders a state needs held, live first."""
    return [f for f in (state["live"], state["standby"]) if f is not None]


def follow(path, step, every=1.0, log=say):
    """Call step(state) every `every` seconds in a background thread, with the switch file's state (an error,
    from reading the file or from step, is logged and tried again next time). Returns the thread."""
    def loop():
        last_error = None
        while True:
            time.sleep(every)
            try:
                step(read(path))
                last_error = None
            except Exception as e:  # keep following: a bad file or a failed load mustn't stop the thread
                message = f"{type(e).__name__}: {e}"
                if message != last_error:
                    log(f"switch: {message}")
                last_error = message
    thread = threading.Thread(target=loop, daemon=True, name="switch")
    thread.start()
    return thread
