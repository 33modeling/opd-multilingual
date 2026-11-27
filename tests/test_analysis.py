"""Verdict functions on synthetic run files (no models): M1a/M1b reporting condition, M3a-c Spearman, M4, gates."""
import json
import random

from mlopd.records import write_rows
from mlopd.util import L10, write_json


def put_eval(runs, id, acc, r, *, pair="main", items=10):
    folder = runs / pair / id
    rows = []
    for k in range(items):
        rows.append({"id": f"mgsm:{k}", "set": "mgsm", "lenient_norm": float(k < round(acc * items)), "r": r,
                     "language_match": int(r >= 0.5)})
    write_rows(folder / "eval.rows.jsonl.gz", rows)
    combined = {"acc": acc, "r_acc": r, "match_acc": float(r >= 0.5), "strict_acc": acc / 2,
                "lenient_norm_pass_at_k": min(1., acc * 1.5)}
    write_json(folder / "COMPLETE.json", {"acc": acc, "combined": combined, "rows": str(folder / "eval.rows.jsonl.gz")})


def make_runs(tmp_path, *, m1_sign_flip=False):
    from mlopd.experiment import Context, load_config
    cfg = load_config(overrides={"paths": {"runs": "runs", "data": "unused.json"}, "analysis": {"bootstrap": 100}})
    ctx = Context(cfg, root=tmp_path)
    runs = ctx.runs
    rng = random.Random(1)
    g = {l: 0.05 * i for i, l in enumerate(L10)}
    p = {l: 0.6 - 0.05 * i for i, l in enumerate(L10)}
    sel = {"M1": ["es", "fr", "de", "ru"], "M4": ["bn", "sw", "th", "ja"], "M5": {"A": ["bn", "sw"], "B": ["en", "es"]},
           "M6": ["en", "es", "ru", "bn"], "B": ["es", "fr"], "S0d": ["en", "bn"], "X": ["en", "bn"]}
    a = {"handoff_updates": 32, "included": list(L10), "selection": sel, "imprint": {"outcome": "go"},
         "gap": {"outcome": "go"}, "g": g, "p": p, "range": {"pass": True}, "m3_mode": "verdict", "proceed": True}
    write_json(runs / "main/gates/a-u32/COMPLETE.json", a)
    write_json(runs / "main/gates/b/COMPLETE.json", {"grid_seed_count": 4, "rl_sd_pass": True, "rl_english_pass": True})
    write_json(runs / "main/gates/m5/COMPLETE.json", {"go": False})
    write_json(runs / "main/gates/m6/COMPLETE.json", {"go": False})
    seeds = cfg["seeds"]["grid"]
    for l in L10:
        for s in seeds:
            start = 0.4
            put_eval(runs, f"start/s{s}-u32/{l}", start, 0.9)
            rl = start + 0.05 + rng.gauss(0, 0.005)
            opd = rl - 0.05 + g[l] * 0.6 + rng.gauss(0, 0.005)
            put_eval(runs, f"g/rl-{l}-s{s}/eval-{l}", rl, 0.9)
            put_eval(runs, f"g/opd-{l}-s{s}/eval-{l}", opd, 0.9)
            put_eval(runs, f"gref/sft-{l}-s{s}/eval-{l}", 0.45, 0.92)
    for i, l in enumerate(sel["M1"]):
        for s in seeds:
            shift = -0.05 if (m1_sign_flip and i < 2) else 0.2
            put_eval(runs, f"i/sft-ten-{l}-s{s}/eval-{l}", 0.45, 0.92 - 0.3)
            put_eval(runs, f"i/opd-ten-{l}-s{s}/eval-{l}", 0.44, 0.9 - shift + rng.gauss(0, 0.002))
    for l in sel["M4"]:
        for s in seeds:
            base = json.loads((runs / f"main/g/opd-{l}-s{s}/eval-{l}/COMPLETE.json").read_text())["acc"]
            put_eval(runs, f"k/opd-tpriv-{l}-s{s}/eval-{l}", base + 0.03 + rng.gauss(0, 0.004), 0.8)
    return ctx


def test_verdicts_on_synthetic_runs(tmp_path):
    from mlopd.analysis import verdicts
    v = verdicts(make_runs(tmp_path))
    assert v["M1a"]["status"] == "supported" and v["M1b"]["status"] == "supported"
    assert v["M1b"]["test"]["n"] == 16 and v["M1b"]["test"]["df"] == 15
    assert v["M1c"]["transfer_ratio"] is not None
    assert v["M3a"]["status"] == "supported" and v["M3a"]["test"]["method"] == "exact"
    assert v["M3a"]["test"]["rho"] > 0.9 and v["M3a"]["mixed_model_slope"]["beta"] > 0
    assert v["M3c"]["status"] == "supported"
    assert v["M4"]["status"] == "supported" and v["M4"]["item_bootstrap"] is not None
    assert v["M5"]["status"] == "not run (condition not met)"
    assert v["M6a"]["status"] == "not measured (gate no-go)"
    assert v["M2b"]["status"] == "incomplete"          # M1b supported but block B not run
    assert set(v["holm_family"]) == {"M1b", "M3a", "M3a-p", "M4"}
    assert all(v[h]["p_holm"] >= v[h]["test"]["p"] for h in v["holm_family"])


def test_reporting_condition_blocks_support(tmp_path):
    from mlopd.analysis import verdicts
    v = verdicts(make_runs(tmp_path, m1_sign_flip=True))
    m1b = v["M1b"]
    assert m1b["positive_languages"] == 2
    assert m1b["status"] in ("pair test significant; reporting condition not met", "not supported")


def test_allocation_gate_from_files(tmp_path):
    from mlopd.experiment import gate_allocation
    ctx = make_runs(tmp_path)
    out = gate_allocation(ctx, None)
    assert out["m3a_supported"] and out["d_gap_pp"] is not None


def test_allocation_gate_when_gate_b_says_no_go(tmp_path):
    # H100 run 2026-11-27: gate B returned grid_seed_count 0, the seed list came out empty and gates/m5 raised
    # StatisticsError. The grid ran anyway (parallel mode); it is now read on the pre-registered seeds as a record.
    from mlopd.analysis import verdicts
    from mlopd.experiment import gate_allocation
    ctx = make_runs(tmp_path)
    write_json(ctx.runs / "main/gates/b/COMPLETE.json",
               {"grid_seed_count": 0, "decision": "no-go: English RL not above start in 3 of 4 (check the setting)",
                "rl_sd_pass": True, "rl_english_pass": False})
    out = gate_allocation(ctx, None)
    assert out["go"] is False and out["d_gap_pp"] is not None and out["gate_b"].startswith("no-go")
    assert out["M3a"]["status"].startswith("record (gate B")
    v = verdicts(ctx)
    assert v["M3a"]["status"].startswith("record (gate B") and "M3a" not in v["holm_family"]
