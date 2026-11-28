# OPD multilingual: experiment code

Small-student staged post-training (SFT, on-policy distillation, RL with verifiable rewards) across ten languages:
what each method imprints (answer format, reasoning language) and when each method helps (teacher gap, student
solve rate).

| Folder | Contents |
|---|---|
| `src/mlopd/` | experiment code (data, scoring, language ratio, SFT/OPD/RL, evaluation, stage 0, gates, analysis) |
| `configs/` | run settings and pre-registration specs |
| `tests/` | CPU tests (tiny random models) |

## Running the experiments (`src/mlopd/`)

Python 3.10+, `torch`, `transformers`, `numpy` (no peft, no scipy). Latin-script language ID needs
`fasttext-wheel` and `lid.176.bin` (`scripts/fetch_lid.sh`, then `export MLOPD_LID_MODEL=...`).

```bash
export PYTHONPATH=$PWD/src HF_ENDPOINT=https://hf-mirror.com
python -m mlopd data build              # pinned datasets -> data/processed/{processed,manifest}.json
python scripts/prefetch_models.py main  # model weights into the HF cache (x for block X)
python -m mlopd queue dry-run           # every task with its state and estimated GPU hours
scripts/run_queue.sh                    # run tasks one at a time (resumable; touch runs/STOP to stop)
python -m mlopd gates                   # stage-0 decisions (gates and language selections, with file hashes)
python -m mlopd analyze                 # runs/analysis/verdicts.json (original, C-*, X record)
python -m mlopd prompts                 # every prompt text
python -m pytest tests                  # CPU tests on tiny random models, no downloads
```

The queue starts with stage 0 (S0-a, S0-b, scorer, teacher targets on F, handoff SFT, start evaluation,
p-measurement, Latin mass), computes the gates of design §5.4 from those files only, then S0-d, and only then
enumerates the grid blocks (G, G-ref, G-x, I, K) and the conditional blocks (A, W, B, X). Outputs go to
`runs/<pair>/<task id>/` with a `COMPLETE.json` per finished task. `src/mlopd/README.md` maps design sections to code.

**Confirmation block C** (pre-registration amendment 2026-11-28, `confirm` in `configs/default.json`): once every
gate of the original blocks exists, the queue also lists block C on new seeds 401-471: per seed teacher targets,
handoff SFT and start evaluation; OPD T-def / T-priv (C-M4 languages) / T-en (C-M1b languages); OPD and RL on all
included languages for the first 4 seeds (C-M3a-p); washout W/N + English RL (C-M6a/b). No new command: after
`git pull`, `scripts/h100/run.sh` picks the tasks up (`run.sh status` shows a `main/C` row) and `analyze` writes the
C-* verdicts (with `p_holm_confirm`) and the block X record next to the original verdicts in `verdicts.json`.
Finished runs of the original blocks are not moved or re-run; `confirm.enabled: false` or `confirm.blocks` switch
the block or its parts off.

## H100 nodes

The code is also public at https://github.com/33modeling/opd-multilingual (no `paper/`, no `docs/`); update it with
`bash scripts/publish_public.sh`. Nodes can clone that repo without credentials:
`git clone https://github.com/33modeling/opd-multilingual.git`.

```bash
bash scripts/h100/run.sh          # same command on every node; Ctrl-C stops this node's workers
bash scripts/h100/run.sh status   # done / running / ready / waiting / failed per block, failure messages
```

- **Storage**: outputs, venv, model cache, lid.176 and logs live under `$OPD_WORK`
  (default `/group-volume/<cluster user>/opd-multilingual-acl2027`; `OPD_USER` or `OPD_WORK` overrides it). `runs/` and `smoke-runs/` in the checkout are
  links into it, and the script refuses a real directory there. HOME, TMPDIR and caches go to
  `$OPD_WORK/runtime/<user>/<host>`.
- **Setup (once, shared by all nodes under a lock)**:
  - Python: `$OPD_WORK/.venv-cu126` with `constraints/h100-cu126.txt` (torch 2.7.1 cu126). If pip has no index,
    the script uses the offpolicy cu126 venv instead and installs fastText into `$OPD_WORK/pylib`.
  - lid.176 is checked against its sha256 and fetched from fbaipublicfiles or the HF mirror.
  - Models are fetched at the commits in `configs/h100.json`, from `HF_ENDPOINT` (default hf-mirror.com) or
    huggingface.co.
  - The data needs no network: `data/processed/processed.json` is in git and checked against its manifest hash.
  - The CPU tests run before anything else.
- **GPU check**: a bf16 matmul on every GPU. `GPUS=0,1` limits the GPUs used.
- **Smoke (once)**: `configs/h100-smoke.json` runs real models and data with 3 languages, 16 items, 4 updates, a
  16-item evaluation and loose gates. It goes through every task kind and both model pairs, and prints the peak
  GPU memory and the time per block. Its outputs are not results. `SKIP_SMOKE=1` skips it.
- **All blocks at once (`queue.parallel_blocks`, on in `configs/h100.json`)**: after stage 0, G (ten languages, six seeds), G-ref, I, K, A, W, B and X are queued together. Gates are still computed from the same files and decide whether a result is a verdict or `record (gate no-go)`, so a failed gate never leaves GPUs idle. The M3 pilot (`gates/m3-pilot`: highest-p and lowest-p languages, two seeds, |ΔD| >= 4 pp) stops the rest of G only in sequential mode (`configs/default.json`).
- **Real queue (`configs/h100.json`)**: the default settings, except that the teacher stays on the GPU, generation
  batches hold 128 sequences and the vocabulary chunk is 512. The learning settings, seeds, splits and gates are
  unchanged.
  - Each GPU runs `WORKERS_PER_GPU` workers (default 2). A worker takes a task under a lease in `runs/.leases`;
    a lease without a heartbeat for 15 minutes is taken over, and the task resumes from its `latest.pt`.
  - A task that fails twice is skipped (`run.sh retry-failed` clears this). A worker stops after 3 failures in a
    row.
  - `run.sh stop` stops all workers after their current task.
  - When nothing is left, `analyze` writes `runs/analysis/verdicts.json`.
- **Per-node overrides**: `.cluster_env.local.sh` in the checkout (not tracked), e.g. `OPD_WORK`, `GPUS`,
  `WORKERS_PER_GPU`, `OPD_PYTHON`.

## Data licenses

`data/processed/` is built from GSM8K (MIT), MGSM8KInstruct (Apache-2.0) and MGSM (CC BY-SA 4.0); the sources and
revisions are in `data/processed/manifest.json`. The MGSM-derived parts keep CC BY-SA 4.0.
