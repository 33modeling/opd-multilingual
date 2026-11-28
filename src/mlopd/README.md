# mlopd: map from the design (v0.2.0, Part II) to the code

| Design | What | Code |
|---|---|---|
| §3.1 | symbols p_l, q_l, g_l, r^S_l, r^T_l, Acc_l, D_{l,s}, G0 | `gates.stage0_a` (p, q, g, r), `analysis.collect_grid` (D, G0) |
| §3.2 R1 | paired t with "greater" / "less" / "two-sided" | `stats.paired_t` (test: the R1 pairs give mean -0.98, SD 2.81, t -0.99, p 0.82) |
| §3.3 M0-M6 | hypotheses | `analysis.verdict_M1a`, `verdict_M1b`, `verdict_M1c`, `verdict_M3` (M3a-c), `verdict_M4`, `verdict_M5`, `washout_verdicts` (M6a, M6b), `verdict_M2b` |
| prereg amendment 2026-11-28 | confirmation block C (new seeds 401-471): C-M4, C-M1b, C-M3a-p, C-M6a/b, Holm over {C-M4, C-M3a-p, C-M1b, C-M6a}; block X record | `configs/default.json` `confirm`, `experiment.confirm_plan`, `confirm_ready`, `confirm_tasks`, `confirm_hours`; `analysis.confirm_verdicts`, `verdict_C_M4`, `verdict_C_M1b`, `verdict_C_M3ap`, `_washout_contrast`, `record_X` |
| §5.1 | MGSM8KInstruct pool P, splits S0/F/U, E_l = MGSM 250 + MSVAMP 250 | `data.py` (pinned revisions and sha256, `parse_parallel`, `parallel_pool`, `Dataset.splits`) |
| §5.2 | student / teacher, LoRA q/v r4 a8, AdamW 1e-4, clip 1, handoff 32x8, OPD, GRPO, SFT-continue, evaluation | `configs/default.json`, `lora.py`, `runtime.py`, `train.py`, `measure.py` |
| §5.2 prompts | student prompt (English record, unchanged), T-def / T-en / T-priv, S0-b lines | `prompts.py` |
| §5.3 | blocks S0-a, S0-b, S0-c, S0-d, G, G-ref, G-x, I, K, A, W, X, B (+ C) | `experiment.stage0_tasks`, `handoff_chain`, `stage0_gate_tasks`, `grid_tasks`, `washout_tasks`, `x_tasks`, `confirm_tasks` |
| §5.4 | gates and selection rules (file hashes recorded) | `gates.py` (pure rules), `experiment.gate_*` (inputs from files) |
| §5.5 | per-run records (RL group composition, zero-advantage updates; OPD loss, KL by segment, teacher Latin mass; tokens, truncation, `####` lines, collapse) | `train.summarize_updates`, `train.Unit.opd_step` / `rl_step`, `measure.collapse_flag`, `records.summarize` |
| §6.1, App. B | strict, lenient_norm, lenient_ascii, lenient_trim (+ the English legacy rules for the scorer gate) | `scoring.py` |
| §6.2-§6.3 | target-language ratio r, language match, line-level pass, language-consistent accuracy | `langid.py`, `records.response_row` |
| §7 | paired t, non-inferiority, MDE (noncentral t), Spearman permutation test, Holm, item bootstrap, mixed-model slope | `stats.py` |
| §8 | GPU-hour estimates of the dry run | `configs/default.json` `estimates_hours`, `experiment.dry_run` |
| §13 | pre-registration (and its amendments) | `configs/prereg-multilingual.json`, `docs/prereg/OSF-PREREGISTRATION.md` |

Conventions kept from the English code (continual-pair-opd, read only): `stable_seed` (sha256 of ":"-joined
parts, first 8 hex digits), the English pool draw (pool_seed 20261102, 1024 items), the student prompt, the SFT
target rule (first EOS-ended lenient-correct teacher sample, answer line rewritten to `#### N`), token-mean SFT loss,
GRPO advantages (r - mean) / (std + 1e-4), mean-token policy-gradient loss, mean-position forward KL.
