#!/usr/bin/env bash
# Run the task queue in the foreground (one task at a time on one GPU; finished tasks are skipped, so the script
# can be restarted after a stop). Create runs/STOP to stop after the current task.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"
exec nice -n 15 ionice -c3 "$PYTHON" -m mlopd ${MLOPD_CONFIG:+--config "$MLOPD_CONFIG"} queue run "$@"
