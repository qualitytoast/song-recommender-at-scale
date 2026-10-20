---
name: site-status
description: Read-only health check of the live site https://songrec.duckdns.org and its Oracle server (model version, shards, song search, cache, containers, memory, disk). Changes nothing. Use when the user asks whether the site is up, healthy, slow, or what it's serving.
---

# Live site status (read-only)

Change nothing: no restarts, no deletes, no edits. Never print `.env`'s Meilisearch key.

1. **From the Mac, over HTTPS** (use `curl`: this Mac's Python can't verify HTTPS certificates):
   - `curl -s https://songrec.duckdns.org/health`: `status` (ok / degraded), `version` (the model bundle),
     `shards` (all `up`), `search` (`answered`, `errors`, `skipped`, `last_error`), `cache` (hit rate),
     `answers`, `defaults`.
   - The page loads: `curl -s -o /dev/null -w '%{http_code} %{time_total}s' https://songrec.duckdns.org/`, and
     one search: `.../songs?q=one+more+time&limit=3`.
2. **On the server** (`ssh -i ~/.ssh/oracle_songrec -o StrictHostKeyChecking=accept-new ubuntu@songrec.duckdns.org`):
   - `docker ps --format '{{.Names}} {{.Status}}'`: every container up; shards and front desk `(healthy)`;
     `docker inspect -f '{{.RestartCount}}'` for any that restarted.
   - `free -m` (available memory; the stack uses ~4.3 GB of 7.7), `df -h /`, `uptime` (load on 2 cores).
   - `grep ^BUNDLE ~/song-recommender-at-scale/.env` (only that line), and `git log --oneline -1` there.
3. **Report** in a few lines: up or not, the model version, search and cache health, memory, disk and load.
   Flag anything degraded, unhealthy, restarting, under ~1 GB of memory available or under ~5 GB of disk.
   Suggest a fix, but don't apply it.
