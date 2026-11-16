#!/usr/bin/env bash
# Wait for the weight download to finish, then run the full benchmark matrix on
# all three databases. Safe to interrupt: every model writes a summary record and
# a per-image score array, and bench.runner skips anything already on disk.
set -u
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
mkdir -p logs
LOG=logs/download_models.log

# The downloader prints one OK/FAIL line per model; 17 models are queued.
while [ "$(grep -cE '^(OK|FAIL)' "$LOG" 2>/dev/null || echo 0)" -lt 17 ]; do
    sleep 60
done

for ds in agiqa3k aigciqa2023 agiqa1k; do
    echo "=== $(date '+%F %T') start $ds ===" >> "logs/run_${ds}_full.log"
    "$PY" -u -m bench.runner --dataset "$ds" --models all >> "logs/run_${ds}_full.log" 2>&1
    echo "=== $(date '+%F %T') done $ds ===" >> "logs/run_${ds}_full.log"
done

"$PY" -u -m bench.leaderboard > logs/leaderboard_full.md 2>&1
echo "DONE $(date '+%F %T')" > logs/overnight.done
