---
name: audit
description: Bloat audit for song-recommender-at-scale on the Mac and the Oracle server - delete leftovers Claude made that aren't needed, list everything else and ask. Use after every training run, and when the user says they're logging off or asks for an audit or cleanup.
---

# Audit

CLAUDE.md asks for this after every training run. Look first, then act. Delete only what Claude made (this
session or earlier ones) that isn't needed; ask before deleting anything else; leave alone anything that isn't
from this project.

**Never touch:** the v1 server on port 8765 (`../transformer-song-recommender`, the user's), `~/.oci`, `.env`
files and any keys (never print them), the user's Downloads, Claude Code's session transcripts, the bundle the
server's `.env` points to.

## Mac

- **Processes:** `ps` for `run_queue`, `recsys`, `caffeinate`, `uvicorn`, `search_worker`, leftover `ssh`
  sessions to the server, Docker Desktop (if Claude started it and it's not needed, quit it).
- **Temp files:** `/tmp/__KMP_REGISTERED_LIB_<pid>` whose pid isn't running (OpenMP leaves one per
  process); `$(getconf DARWIN_USER_TEMP_DIR)`'s `torchinductor_*` and `pytest-of-*`; this project's folders
  under `/private/tmp/claude-*/` (empty ones from old sessions, finished task logs).
- **Repo:** `__pycache__`, `.pytest_cache`, `runs/shortlist_cache` (rebuilt when needed), scratch files.
- **Big files to judge:** `bundles/` (keep the one the site serves), checkpoints (`runs/**/best.pt`; a ranker
  needs its retriever's), `data/` (ask; the user keeps `data/lb_extract` and `data/mpd.tar.xz`).
- **Caches:** the pip cache (may be other projects': ask), Docker (`docker system df` if it's running; the real
  size of Docker.raw with `du`, not its apparent size).

## Server (`ssh -i ~/.ssh/oracle_songrec -o StrictHostKeyChecking=accept-new ubuntu@songrec.duckdns.org`)

- `/tmp` files Claude left; `~/bundles` (keep the one `.env`'s BUNDLE names); `docker system df`, dangling
  images (`docker image prune -f` is safe); `git status --short --ignored` in `~/song-recommender-at-scale`
  (check files copied over for a check, like `runs/serving/*/cases.json`, block `git pull` later); leftover
  processes (`docker stats`, `bash -s`, `rsync`).

## Finish

Report three lists: **deleted** (with sizes), **kept** (and why), **your call** (with a recommendation each).
Free disk on both machines. Update the auto-memory's pending-cleanup note with what was done and asked.
