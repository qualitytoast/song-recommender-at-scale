#!/usr/bin/env bash
# Run configs one after another: the Mac has one GPU, and two runs at once are slower in total than one
# after the other (RESULTS.md). Each run's output is appended to its own run.out, next to its seed folders
# (runs/<group>/[<size>/]<config name>/run.out). A retriever config is trained, then evaluated; a ranker config
# ([ranker] table) is trained and evaluated by recsys.ranker, and skipped if its retriever has no checkpoint or
# failed earlier in this queue (a run that crashes partway leaves its best checkpoint so far, half-trained). A
# failed run is logged and the queue moves on. Progress: runs/queue.log.
#
#   caffeinate -i scripts/run_queue.sh configs/retriever_50k_1m_settings.toml configs/ranker_50k_1m_settings.toml ...
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONUNBUFFERED=1
mkdir -p runs

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a runs/queue.log; }
run_dir() { $PY -c "import sys; from recsys.train import run_dir_for; print(run_dir_for(sys.argv[1], 1).parent)" "$1"; }

log "queue of $#: $*"
failed=" "   # run folders of the runs that failed in this queue
for config in "$@"; do
  name=$(basename "$config" .toml)
  dir=$(run_dir "$config")
  mkdir -p "$dir"
  if grep -q '^\[ranker\]' "$config"; then
    retriever=$($PY -c "import sys; from recsys.ranker import load_ranker_config; print(load_ranker_config(sys.argv[1]).retriever)" "$config")
    if [[ "$failed" == *" $(run_dir "$retriever") "* ]]; then
      log "SKIPPED $name: $retriever failed in this queue (its checkpoint may be half-trained)"
      continue
    fi
    if [ ! -f "$(run_dir "$retriever")/seed1/best.pt" ]; then
      log "SKIPPED $name: $retriever has no checkpoint"
      continue
    fi
    cmd=("$PY" -m recsys.ranker --config "$config")
  else
    cmd=("$PY" -m recsys.train --config "$config")
  fi
  log "start $name"
  start=$(date +%s)
  if "${cmd[@]}" >> "$dir/run.out" 2>&1 \
     && { grep -q '^\[ranker\]' "$config" || "$PY" -m recsys.evaluate --config "$config" >> "$dir/run.out" 2>&1; }; then
    log "done $name in $(( ($(date +%s) - start) / 60 )) min"
  else
    log "FAILED $name after $(( ($(date +%s) - start) / 60 )) min (see $dir/run.out)"
    failed+="$dir "
  fi
done
log "queue finished"
