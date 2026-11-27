"""RL diagnosis from finished runs (no GPU): why gate B said English RL is not above start.

Prints, for the S0-d reruns and the grid runs of every language: start accuracy, accuracy after the run, the
change, and the run records (share of all-correct / all-wrong / mixed groups, zero-advantage updates, reward in
the first and last 32 updates, rollout truncation and marker-line rates, mean rollout length).
Usage: python scripts/h100/rl_check.py [config]   (run.sh rl-check)
"""
import json
import sys
from pathlib import Path

from mlopd.experiment import Context, final_stage0, load_config
from mlopd.util import read_jsonl


def pct(x):
    return "   -  " if x is None else f"{100 * x:6.1f}"


def run_line(ctx, run_id, eval_id, start_id):
    c = ctx.complete("main", run_id)
    e = ctx.complete("main", eval_id)
    s = ctx.complete("main", start_id)
    if c is None or e is None or s is None:
        return f"{run_id:28s} missing"
    rec = c.get("records", {})
    rl = rec.get("rl") or {}
    g = rl.get("group_shares", {})
    first = last = None
    rows = read_jsonl(ctx.runs / "main" / run_id / "updates.jsonl") if (ctx.runs / "main" / run_id / "updates.jsonl").exists() else []
    rewards = [r.get("reward_mean", None if not r.get("strict_rewards") else sum(r["strict_rewards"]) / len(r["strict_rewards"]))
               for r in rows]
    rewards = [x for x in rewards if x is not None]
    if rewards:
        first, last = sum(rewards[:32]) / len(rewards[:32]), sum(rewards[-32:]) / len(rewards[-32:])
    sc, ec = s["combined"], e["combined"]
    return (f"{run_id:28s} acc {pct(s['acc'])} -> {pct(e['acc'])} ({100 * (e['acc'] - s['acc']):+5.1f}pp)"
            f"  strict {pct(sc.get('strict_acc'))} -> {pct(ec.get('strict_acc'))}"
            f"  groups ok/wrong/mixed {pct(g.get('all_correct'))}/{pct(g.get('all_wrong'))}/{pct(g.get('mixed'))}"
            f"  zeroA {rl.get('zero_advantage_updates', '-')}"
            f"  reward first32 {pct(first)} last32 {pct(last)}"
            f"  trunc {pct(rec.get('rollout_truncation_rate'))} marker {pct(rec.get('rollout_marker_line_rate'))}"
            f"  len {rec.get('mean_rollout_tokens', 0):.0f}")


def main():
    ctx = Context(load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/h100.json"))
    cfg = ctx.cfg
    a = final_stage0(ctx)
    b = ctx.gate("main", "b")
    print("gate B:", json.dumps({k: b.get(k) for k in ("decision", "rl_sd_pass", "rl_english_pass", "opd_sd_pass",
                                                        "rl_collapses")} if b else None, ensure_ascii=False))
    if b:
        for method in ("rl", "opd"):
            for l, v in b.get(method, {}).items():
                print(f"  {method} {l}: start {pct(v['start'])} reruns {' '.join(pct(x) for x in v['values'])}"
                      f"  sd {v['sd_pp']}  above start {v['above_start']}")
    seed0 = cfg["seeds"]["grid"][0]
    u = a["handoff_updates"]
    print("\n== S0-d reruns (seed", seed0, ")")
    for l in a["selection"]["S0d"]:
        for method in ("rl", "opd"):
            for tag in cfg["stage0"]["s0d_tags"]:
                i = f"s0d/{method}-{l}-{tag}"
                print(run_line(ctx, i, f"{i}/eval-{l}", f"start/s{seed0}-u{u}/{l}"))
    seeds = cfg["seeds"]["grid"] + cfg["seeds"]["grid_extra"]
    for method in ("rl", "opd"):
        print(f"\n== grid {method}")
        for l in a["included"]:
            for s in seeds:
                i = f"g/{method}-{l}-s{s}"
                if ctx.complete("main", i):
                    print(run_line(ctx, i, f"{i}/eval-{l}", f"start/s{s}-u{u}/{l}"))


if __name__ == "__main__":
    main()
