#!/usr/bin/env bash
# OPD multilingual experiments on H100 nodes. The same command on every node, no arguments:
#   bash scripts/h100/run.sh            setup (once, shared) -> GPU check -> smoke (once) -> workers -> analysis
#   bash scripts/h100/run.sh status     per block: done / running / ready / waiting / failed, running tasks, failures
#   bash scripts/h100/run.sh smoke      the smoke queue only (real models, few items, loose gates; not results)
#   bash scripts/h100/run.sh workers    the real queue only (skips setup checks that already passed)
#   bash scripts/h100/run.sh analyze    runs/analysis/verdicts.json from the finished runs
#   bash scripts/h100/run.sh rl-check       RL diagnosis per run (start -> end accuracy, reward groups, truncation)
#   bash scripts/h100/run.sh retry-failed   clear failure records (a task is skipped after 2 failures)
#   bash scripts/h100/run.sh stop       every worker on every node stops after its current task
# Storage: code here (user-volume), everything else under $OPD_WORK (default /group-volume/<cluster user>/opd-multilingual-acl2027).
# Each GPU runs $WORKERS_PER_GPU worker processes (default 2; the models are 0.6B-1.7B). Workers take tasks under
# leases in runs/.leases, so any number of nodes can join; a finished task is skipped and a stopped training run
# resumes from its latest.pt. Ctrl-C stops this node's workers and frees the GPUs.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
[ -f "$REPO/.cluster_env.local.sh" ] && . "$REPO/.cluster_env.local.sh"
ACTION=${1:-all}
GV=${GROUP_VOLUME:-/group-volume}
# The group-volume folder is named after the cluster user; if no folder has that name, use the one this user owns.
if [ -z "${OPD_USER:-}" ]; then
  OPD_USER=$(id -un)
  if [ ! -d "$GV/$OPD_USER" ]; then
    own=$(find "$GV" -mindepth 1 -maxdepth 1 -type d -user "$(id -u)" 2>/dev/null | head -1)
    [ -n "$own" ] && OPD_USER=$(basename "$own")
  fi
fi
WORK=${OPD_WORK:-$GV/$OPD_USER/opd-multilingual-acl2027}
case "$WORK" in
  "$GV"/*) ;;
  *) [ "${OPD_ALLOW_ANY_WORK:-0}" = 1 ] || { echo "refusing OPD_WORK=$WORK: outputs must live under $GV" >&2; exit 2; } ;;
esac
HOST=$(hostname -s)
RT=$WORK/runtime/$(id -un)/$HOST
mkdir -p "$WORK/runs" "$WORK/smoke-runs" "$WORK/logs" "$WORK/assets" "$WORK/hf" "$RT/home" "$RT/tmp" "$RT/cache"
export HOME=$RT/home TMPDIR=$RT/tmp XDG_CACHE_HOME=$RT/cache PIP_CACHE_DIR=$RT/cache/pip
export HF_HOME=$WORK/hf HF_HUB_CACHE=$WORK/hf/hub HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export MLOPD_LID_MODEL=$WORK/assets/lid.176.bin
export PYTHONPATH=$REPO/src:$WORK/pylib${PYTHONPATH:+:$PYTHONPATH}
CONFIG=configs/h100.json
SMOKE_CONFIG=configs/h100.json,configs/h100-smoke.json
LID_SHA=7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e
SETUP_VERSION=1
log() { echo "$(date +%H:%M:%S) [$HOST] $*"; }

# runs/ and smoke-runs/ in the checkout must be links into $WORK (a real directory here would put outputs on
# the user volume).
for name in runs smoke-runs; do
  if [ -L "$name" ]; then
    [ "$(readlink -f "$name")" = "$(readlink -f "$WORK/$name")" ] || ln -sfn "$WORK/$name" "$name"
  elif [ -e "$name" ]; then
    echo "$REPO/$name is a real directory; move it into $WORK/$name and rerun" >&2; exit 2
  else
    ln -s "$WORK/$name" "$name"
  fi
done

python_path() {
  if [ -n "${OPD_PYTHON:-}" ]; then echo "$OPD_PYTHON"; elif [ -f "$WORK/.python" ]; then cat "$WORK/.python"; fi
}

setup_python() {
  local py venv=$WORK/.venv-cu126
  py=$(python_path)
  if [ -n "$py" ] && "$py" -c "import torch, transformers, numpy, fasttext" 2>/dev/null; then
    log "python: $py"; return
  fi
  if [ ! -x "$venv/bin/python" ]; then
    log "creating $venv"
    "${BASE_PYTHON:-python3}" -m venv "$venv"
  fi
  if "$venv/bin/pip" install -q --upgrade pip &&
     "$venv/bin/pip" install -q -c constraints/h100-cu126.txt torch --index-url https://download.pytorch.org/whl/cu126 \
        --extra-index-url https://pypi.org/simple &&
     "$venv/bin/pip" install -q -c constraints/h100-cu126.txt transformers numpy huggingface-hub safetensors \
        tokenizers fasttext-wheel pytest; then
    py=$venv/bin/python
  else
    # No package index from this node: reuse the existing cu126 venv of the offpolicy project (read only) and
    # put fastText next to it.
    py=${FALLBACK_PYTHON:-$GV/$OPD_USER/offpolicy-misranking/.venv-cu126/bin/python}
    log "pip failed; using $py"
    "$py" -c "import fasttext" 2>/dev/null ||
      "$py" -m pip install -q --no-deps --target "$WORK/pylib" fasttext-wheel==0.9.2 pybind11
  fi
  "$py" -c "import torch, transformers, numpy, fasttext; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'numpy', numpy.__version__)"
  echo "$py" > "$WORK/.python"
}

fetch_lid() {
  local target=$MLOPD_LID_MODEL
  if [ -f "$target" ] && echo "$LID_SHA  $target" | sha256sum -c --quiet 2>/dev/null; then return; fi
  for url in https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin \
             "$HF_ENDPOINT/julien-c/fasttext-language-id/resolve/9f2e94d2fd6b2615d26f5ec5b34cf7e2f38cb86e/lid.176.bin" \
             https://huggingface.co/julien-c/fasttext-language-id/resolve/9f2e94d2fd6b2615d26f5ec5b34cf7e2f38cb86e/lid.176.bin; do
    log "lid.176 from $url"
    if curl -fsSL --retry 3 --max-time 900 -o "$target.part" "$url" &&
       echo "$LID_SHA  $target.part" | sha256sum -c --quiet; then
      mv "$target.part" "$target"; return
    fi
  done
  echo "could not fetch lid.176.bin (sha256 $LID_SHA); copy it to $target" >&2; exit 1
}

fetch_models() {
  local py=$1 endpoint
  for endpoint in "$HF_ENDPOINT" https://huggingface.co; do
    if HF_ENDPOINT=$endpoint "$py" scripts/prefetch_models.py --config "$CONFIG" main x; then return; fi
    log "model download via $endpoint failed"
  done
  echo "model download failed; copy the snapshots into $HF_HUB_CACHE" >&2; exit 1
}

check_data() {
  "$1" - <<'PY'
import hashlib, json
m = json.load(open("data/processed/manifest.json"))
h = hashlib.sha256(open("data/processed/processed.json", "rb").read()).hexdigest()
assert h == m["processed_sha256"], (h, m["processed_sha256"])
print("data: processed.json matches manifest", h[:16])
PY
}

# runs/ holds results of one processed.json. When the data changes (a new commit), the old results are moved
# aside to $WORK/runs-data-<old sha> (kept, not deleted) and the queue starts again on the new data.
# runs/ without a stamp was made before this check existed, by the data before 2026-11-26 (broken bn/th
# translations in the pool).
data_guard() {
  local py=$1 cur old stamp=$WORK/runs/.data-sha256
  check_data "$py" >/dev/null
  cur=$(sha256sum data/processed/processed.json | cut -c1-64)
  if [ -f "$stamp" ]; then old=$(cat "$stamp"); else old=; fi
  [ "$old" = "$cur" ] && return
  if [ -z "$old" ] && [ -z "$(ls -A "$WORK/runs" 2>/dev/null)" ]; then echo "$cur" > "$stamp"; return; fi
  if [ -n "$(find "$WORK/runs/.leases" -name owner.json -mmin -10 2>/dev/null | head -1)" ]; then
    echo "data changed, but workers still hold leases in $WORK/runs/.leases (another node?); stop them first" >&2
    exit 1
  fi
  local dest=$WORK/runs-data-${old:0:12}
  [ -n "$old" ] || dest=$WORK/runs-data-before-2026-11-26
  [ -e "$dest" ] && dest=$dest-$(date +%Y%m%d%H%M%S)
  mv "$WORK/runs" "$dest"
  mkdir -p "$WORK/runs"
  echo "$cur" > "$stamp"
  log "data changed (processed.json ${cur:0:12}); earlier results moved to $dest, queue starts again"
}

setup() {
  local stamp=$WORK/.setup-v$SETUP_VERSION.done lock=$WORK/.setup.lock
  if [ -f "$stamp" ]; then log "setup already done ($(cat "$stamp"))"; return; fi
  if ! mkdir "$lock" 2>/dev/null; then
    log "another node is running setup ($lock); waiting"
    while [ ! -f "$stamp" ]; do
      [ -d "$lock" ] || { log "setup lock gone without stamp; retrying"; setup; return; }
      if [ -n "$(find "$lock" -maxdepth 0 -mmin +180 2>/dev/null)" ]; then
        log "setup lock older than 3 h (its node died); removing it"; rmdir "$lock" 2>/dev/null || true
      fi
      sleep 30
    done
    return
  fi
  trap 'rmdir "'"$lock"'" 2>/dev/null || true' EXIT
  setup_python
  local py; py=$(python_path)
  fetch_lid
  fetch_models "$py"
  check_data "$py"
  log "CPU tests"
  if ! nice -n 10 "$py" -m pytest -q -p no:cacheprovider -p no:logging tests > "$WORK/logs/cpu-tests-$HOST.log" 2>&1; then
    tail -30 "$WORK/logs/cpu-tests-$HOST.log"; echo "CPU tests failed (log above)" >&2; exit 1
  fi
  tail -1 "$WORK/logs/cpu-tests-$HOST.log"
  echo "$(date -Iseconds) $HOST $py" > "$stamp"
  rmdir "$lock" 2>/dev/null || true
  trap - EXIT
}

gpus() {
  if [ -n "${GPUS:-}" ]; then echo "$GPUS" | tr ',' ' '; return; fi
  nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' '
}

gpu_check() {
  local py=$1 g
  for g in $(gpus); do
    CUDA_VISIBLE_DEVICES=$g "$py" - <<'PY' || { echo "GPU $g failed the check; set GPUS= to skip it" >&2; exit 1; }
import os, time, torch
assert torch.cuda.is_available(), "no CUDA"
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
t = time.time()
for _ in range(20):
    y = x @ x
torch.cuda.synchronize()
assert torch.isfinite(y.float()).all()
free, total = torch.cuda.mem_get_info()
print(f"gpu {os.environ['CUDA_VISIBLE_DEVICES']}: {torch.cuda.get_device_name()} ok, {free/2**30:.0f}/{total/2**30:.0f} GB free, "
      f"{20*2*4096**3/(time.time()-t)/1e12:.0f} TFLOP/s bf16")
PY
  done
}

PIDS=()
# Pre-registration change 2026-11-27 (RL dynamic sampling): results that used the old RL updates are moved once to
# runs-rl-before-dynamic (same layout), so the queue runs them again: S0-d and grid RL runs (main, x), block A,
# washout RL runs, the gates computed from them (b, m3-pilot, m5, m6) and the analysis. OPD, SFT and handoff stay.
rl_guard() {
  local stamp=$WORK/runs/.rl-dynamic-sampling R=$WORK/runs dest=$WORK/runs-rl-before-dynamic p rel moved=0
  [ -f "$stamp" ] && return
  if [ -n "$(find "$R/.leases" -name owner.json -mmin -10 2>/dev/null | head -1)" ]; then
    echo "RL results must be moved, but workers still hold leases in $R/.leases (another node?); stop them first" >&2
    exit 1
  fi
  shopt -s nullglob
  for p in "$R"/main/s0d/rl-* "$R"/main/g/rl-* "$R"/x/g/rl-* "$R"/main/a "$R"/main/w/rl-* \
           "$R"/main/gates/b "$R"/main/gates/m3-pilot "$R"/main/gates/m5 "$R"/main/gates/m6 "$R"/analysis; do
    [ -e "$p" ] || continue
    rel=${p#"$R"/}
    mkdir -p "$dest/$(dirname "$rel")"
    mv "$p" "$dest/$rel"
    moved=$((moved + 1))
  done
  shopt -u nullglob
  mkdir -p "$R"
  date -Iseconds > "$stamp"
  [ "$moved" -gt 0 ] && log "RL dynamic sampling: $moved earlier RL result folders moved to $dest; those runs start again"
  return 0
}

# A task that failed twice is skipped until its failure record is deleted. When the code changed since the last start
# (a fix was pulled), the old failures may be the bug that was fixed, so they are cleared and those tasks run again.
code_guard() {
  local cur old stamp=$WORK/runs/.code-sha256
  cur=$(cat src/mlopd/*.py | sha256sum | cut -c1-64)
  if [ -f "$stamp" ]; then old=$(cat "$stamp"); else old=; fi
  [ "$old" = "$cur" ] && return
  if [ -n "$(ls -A "$WORK/runs/.failures" 2>/dev/null)" ]; then
    queue_cmd "$CONFIG" retry-failed >/dev/null
    log "code changed since the last start; failure records cleared, failed tasks run again"
  fi
  echo "$cur" > "$stamp"
}

stop_workers() {
  trap - INT TERM
  log "stopping workers on this node"
  for pid in "${PIDS[@]:-}"; do [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
  exit 130
}

# Run the queue of config $1 with $WORKERS_PER_GPU workers per GPU, in the foreground; raw lines on screen with a
# [gpu.k] prefix and in $WORK/logs/<name>-<host>-gpu<g>.<k>.log.
run_workers() {
  local config=$1 name=$2 py g k per=${WORKERS_PER_GPU:-2}
  py=$(python_path)
  PIDS=()
  trap stop_workers INT TERM
  for g in $(gpus); do
    for k in $(seq 1 "$per"); do
      local logf=$WORK/logs/$name-$HOST-gpu$g.$k.log
      ( CUDA_VISIBLE_DEVICES=$g exec nice -n 5 "$py" -m mlopd --config "$config" queue worker --wait \
          --stop-file runs/STOP 2>&1 ) > >(tee -a "$logf" | sed -u "s/^/[gpu$g.$k] /") &
      PIDS+=($!)
      sleep 20  # staggered start: model loading and lease checks do not all hit the volume at once
    done
  done
  log "${#PIDS[@]} workers started ($name); logs: $WORK/logs/$name-$HOST-*"
  local rc=0
  for pid in "${PIDS[@]}"; do wait "$pid" || rc=$?; done
  trap - INT TERM
  return $rc
}

queue_cmd() {
  local config=$1; shift
  "$(python_path)" -m mlopd --config "$config" queue "$@"
}

smoke() {
  local stamp=$WORK/.smoke.done
  if [ -f "$stamp" ] && [ "$ACTION" != smoke ]; then log "smoke already passed ($(cat "$stamp"))"; return; fi
  queue_cmd "$SMOKE_CONFIG" retry-failed >/dev/null
  log "smoke: real models, 3 languages, few items (outputs in $WORK/smoke-runs, not results)"
  run_workers "$SMOKE_CONFIG" smoke || true
  queue_cmd "$SMOKE_CONFIG" status --stop-file runs/STOP | tee "$WORK/logs/smoke-status-$HOST.txt"
  if [ -n "$(unfinished "$SMOKE_CONFIG")" ]; then
    echo "smoke did not finish every task: see the status above and smoke-runs/.failures/" >&2
    exit 1
  fi
  "$(python_path)" - <<'PY'
import json, collections
rows = [json.loads(l) for l in open("smoke-runs/queue.log")]
peak = max((r.get("peak_gpu_gb") or 0) for r in rows)
by = collections.defaultdict(float)
for r in rows:
    by[r["task"].split("/")[0]] += r["hours"]
print(f"smoke: {len(rows)} tasks, peak GPU memory {peak} GB per worker; hours by prefix:",
      ", ".join(f"{k} {v:.2f}" for k, v in sorted(by.items())))
PY
  echo "$(date -Iseconds) $HOST" > "$stamp"
}

# Prints the number of unfinished tasks of config $1, or nothing when every listed task is done.
unfinished() {
  "$(python_path)" -c "from mlopd.experiment import Context, load_config, build_queue
c = Context(load_config('$1'))
print(sum(not c.done(t) for t in build_queue(c)) or '')"
}

analyze() {
  "$(python_path)" -m mlopd --config "$CONFIG" analyze
}

case "$ACTION" in
  status) queue_cmd "$CONFIG" status ;;
  retry-failed) queue_cmd "$CONFIG" retry-failed ;;
  rl-check) "$(python_path)" scripts/h100/rl_check.py "$CONFIG" | tee "$WORK/logs/rl-check-$HOST.txt" ;;
  stop) touch runs/STOP; log "runs/STOP created; workers stop after their current task (delete it to run again)" ;;
  analyze) data_guard "$(python_path)"; analyze ;;
  smoke|workers|all)
    rm -f runs/STOP
    setup
    PY=$(python_path)
    gpu_check "$PY"
    if [ "$ACTION" != workers ] && [ "${SKIP_SMOKE:-0}" != 1 ]; then smoke; fi
    [ "$ACTION" = smoke ] && exit 0
    data_guard "$PY"
    rl_guard
    code_guard
    log "real queue (config $CONFIG)"
    run_workers "$CONFIG" queue || true
    queue_cmd "$CONFIG" status
    if [ -z "$(unfinished "$CONFIG")" ]; then
      log "queue finished; analysis"
      analyze
    else
      log "queue not finished on this node's view (failed tasks or gates); see status above"
    fi
    ;;
  *) sed -n '2,13p' "$0"; exit 2 ;;
esac
