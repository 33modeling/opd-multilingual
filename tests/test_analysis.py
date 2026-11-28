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


def make_confirm_runs(tmp_path, *, ratio_check=True, m4_effect=0.03):
    """Synthetic block C on top of make_runs: new seeds, C-M4 (T-priv above T-def), C-M1b (T-en lowers r),
    C-M3a-p (D falls as p rises), C-M6a/b (washout), and an X record."""
    ctx = make_runs(tmp_path)
    runs, cfg = ctx.runs, ctx.cfg
    rng = random.Random(2)
    a = json.loads((runs / "main/gates/a-u32/COMPLETE.json").read_text())
    a["handoff_languages"] = list(L10)
    a["r_teacher"] = {"T-def": {l: 0.9 for l in L10}, "T-en": {l: 0.9 - (0.5 if ratio_check else 0.1) for l in L10}}
    a["imprint"] = {"outcome": "m1a_only", "latin_mass": {l: {"T-def": 0.7, "T-en": 0.75} for l in a["selection"]["M1"]},
                    "latin_ok": {l: False for l in a["selection"]["M1"]},
                    "ratio_ok": {l: ratio_check for l in a["selection"]["M1"]}}
    write_json(runs / "main/gates/a-u32/COMPLETE.json", a)
    seeds = cfg["confirm"]["seeds"]
    grid_seeds = seeds[:cfg["confirm"]["grid_seeds"]]
    for s in seeds:
        for l in L10:
            put_eval(runs, f"start/s{s}-u32/{l}", 0.4, 0.9)
            opd = 0.45 + a["g"][l] * 0.6 + rng.gauss(0, 0.005)
            put_eval(runs, f"c/opd-{l}-s{s}/eval-{l}", opd, 0.9)
            if s in grid_seeds:
                put_eval(runs, f"c/rl-{l}-s{s}/eval-{l}", 0.45 + rng.gauss(0, 0.005), 0.9)
        for l in a["selection"]["M4"]:
            base = json.loads((runs / f"main/c/opd-{l}-s{s}/eval-{l}/COMPLETE.json").read_text())["acc"]
            put_eval(runs, f"c/opd-tpriv-{l}-s{s}/eval-{l}", base + m4_effect + rng.gauss(0, 0.004), 0.8)
        for l in a["selection"]["M1"]:
            put_eval(runs, f"c/opd-ten-{l}-s{s}/eval-{l}", 0.44, 0.6 + rng.gauss(0, 0.01))
    for s in cfg["confirm"]["washout_seeds"]:
        for l in a["selection"]["M6"]:
            for which in ("a", "b"):
                put_eval(runs, f"c/w/sft-W-s{s}/eval-{which}-{l}", 0.40 + rng.gauss(0, 0.003), 0.9)
                put_eval(runs, f"c/w/sft-N-s{s}/eval-{which}-{l}", 0.37 + rng.gauss(0, 0.003), 0.7)
            put_eval(runs, f"c/w/rl-W-s{s}/eval-{l}", 0.45 + rng.gauss(0, 0.003), 0.85)
            put_eval(runs, f"c/w/rl-N-s{s}/eval-{l}", 0.44 + rng.gauss(0, 0.003), 0.7)
    return ctx


def test_confirmation_verdicts(tmp_path):
    from mlopd.analysis import verdicts
    v = verdicts(make_confirm_runs(tmp_path))
    assert v["C-M4"]["status"] == "supported" and v["C-M4"]["test"]["n"] == 4 * 8
    assert v["C-M4"]["seed_aggregate"]["n"] == 8 and v["C-M4"]["seed_aggregate"]["p"] < 0.05
    assert v["C-M1b"]["status"] == "supported" and v["C-M1b"]["manipulation_check"]["pass"]
    assert v["C-M1b"]["latin_mass_record"]["latin_ok"] == {l: False for l in v["C-M1b"]["languages"]}
    m3 = v["C-M3a-p"]
    assert m3["status"] == "supported" and m3["test"]["method"] == "exact" and m3["test"]["rho"] < -0.9
    assert m3["seeds"] == [401, 411, 421, 431] and len(m3["languages"]) == 10
    assert v["C-M6a"]["status"] == "supported" and v["C-M6a"]["test"]["n"] == 8
    assert "C-M6b" in v and "C-M6b" not in v["confirm_holm_family"]
    assert set(v["confirm_holm_family"]) == {"C-M4", "C-M3a-p", "C-M1b", "C-M6a"}
    assert all(v[h]["p_holm_confirm"] >= v[h]["test"]["p"] for h in v["confirm_holm_family"])
    # the original verdicts are untouched by block C (M1b stays unjudged: its Latin-mass gate failed)
    assert v["M1b"]["status"].startswith("not measured (gate no-go)") and v["M4"]["test"]["n"] == 16
    assert set(v["holm_family"]) == {"M3a", "M3a-p", "M4"}


def test_confirmation_m1b_record_when_teacher_ratio_check_fails(tmp_path):
    from mlopd.analysis import verdicts
    v = verdicts(make_confirm_runs(tmp_path, ratio_check=False))
    assert v["C-M1b"]["status"] == "record (gate no-go)"
    assert v["C-M1b"]["verdict_if_gate_had_passed"] == "supported"
    assert "C-M1b" not in v["confirm_holm_family"]


def test_confirmation_m4_not_supported_without_effect(tmp_path):
    from mlopd.analysis import verdicts
    v = verdicts(make_confirm_runs(tmp_path, m4_effect=-0.01))
    assert v["C-M4"]["status"] == "not supported" and "p_holm_confirm" in v["C-M4"]


def test_confirmation_incomplete_and_switched_off(tmp_path):
    from mlopd.analysis import confirm_verdicts
    ctx = make_runs(tmp_path)
    a = json.loads((ctx.runs / "main/gates/a-u32/COMPLETE.json").read_text())
    v = confirm_verdicts(ctx, a)
    assert all(v[h]["status"] == "incomplete" for h in ("C-M4", "C-M1b", "C-M3a-p", "C-M6a", "C-M6b"))
    assert v["confirm_holm_family"] == []
    ctx.cfg["confirm"]["blocks"]["M4"] = False
    assert confirm_verdicts(ctx, a)["C-M4"]["status"].startswith("not run")
    ctx.cfg["confirm"]["enabled"] = False
    assert confirm_verdicts(ctx, a)["C-M3a-p"]["status"].startswith("not run")


def test_record_X(tmp_path):
    from mlopd.analysis import record_X
    ctx = make_runs(tmp_path)
    runs, seeds = ctx.runs, ctx.cfg["seeds"]["x"]
    assert record_X(ctx)["status"] == "incomplete"
    write_json(runs / ("x/gates/handoff-languages-" + "-".join(map(str, seeds))) / "COMPLETE.json",
               {"languages": ["en", "de"]})
    for l in ("en", "de"):
        for s in seeds:
            put_eval(runs, f"start/s{s}-u32/{l}", 0.30, 0.9, pair="x")
            put_eval(runs, f"g/opd-{l}-s{s}/eval-{l}", 0.36, 0.9, pair="x")
            put_eval(runs, f"g/rl-{l}-s{s}/eval-{l}", 0.40, 0.9, pair="x")
    x = record_X(ctx)
    assert x["status"] == "record" and set(x["per_language"]) == {"en", "de"}
    de = x["per_language"]["de"]
    assert abs(de["D_mean_pp"] + 4.0) < 1e-9 and abs(de["G0_pp"]["rl"] - 10.0) < 1e-9
    assert abs(de["start_acc"] - 0.30) < 1e-9 and abs(de["strict"]["opd"] - 0.18) < 1e-9
    assert de["collapses"] == {"opd": 0, "rl": 0} and de["D_test"]["n"] == len(seeds)


def test_confirm_queue_listing(tmp_path):
    """Block C on the H100 stage-0 outcome: new seeds only, shared T-def OPD runs, no id clash with earlier blocks."""
    from mlopd.experiment import Context, confirm_tasks, load_config
    ctx = Context(load_config(), root=tmp_path)
    L = ["en", "es", "fr", "de", "ru", "zh", "ja", "th", "bn"]
    a = {"handoff_updates": 32, "handoff_languages": L, "included": L,
         "selection": {"M4": ["bn", "th", "ja", "de"], "M1": ["de", "ru", "th", "ja"], "M6": ["en", "es", "zh", "bn"]}}
    tasks = {t["id"]: t for t in confirm_tasks(ctx, a)}
    cfg = ctx.cfg
    old = set(cfg["seeds"]["grid"] + cfg["seeds"]["grid_extra"] + cfg["seeds"]["allocation"] + cfg["seeds"]["washout"]
              + cfg["seeds"]["x"])
    assert not old & {t["params"].get("seed") for t in tasks.values()} - {None}
    assert len(tasks) == 688 and all(t["block"] == "C" for t in tasks.values())
    runs = [t for t in tasks.values() if t["kind"] == "train" and t["id"].startswith("c/opd-")]
    assert sum(t["params"]["condition"] == "T-def" for t in runs) == 5 * 8 + 4 * 4
    assert sum(t["params"]["condition"] == "T-priv" for t in runs) == 4 * 8
    assert sum(t["params"]["condition"] == "T-en" for t in runs) == 4 * 8
    assert sum(i.startswith("c/rl-") and i.count("/") == 1 for i in tasks) == 9 * 4
    assert tasks["c/opd-tpriv-de-s401"]["deps"] == ["handoff/s401-u32"]
