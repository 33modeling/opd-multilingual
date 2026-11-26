import json
from pathlib import Path

from mlopd import gates as G

CFG = json.loads((Path(__file__).resolve().parents[1] / "configs" / "default.json").read_text())
LANGS = ["en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn"]


def synthetic(p_values, q_values, r_values, latin_gap=0.2, ratio_gap=0.5, priv_gap=0.2):
    seeds = [1, 2]
    p_strict = {s: dict(p_values) for s in seeds}
    p_len = {s: {l: v / 0.95 for l, v in p_values.items()} for s in seeds}
    r_s = {s: dict(r_values) for s in seeds}
    q = {"T-def": dict(q_values), "T-en": dict(q_values), "T-priv": {l: v + priv_gap for l, v in q_values.items()}}
    r_t = {"T-def": {l: 0.9 for l in LANGS}, "T-en": {l: 0.9 - ratio_gap for l in LANGS}, "T-priv": {l: 0.8 for l in LANGS}}
    latin = {l: {"T-def": 0.1, "T-en": 0.1 + latin_gap} for l in LANGS if l != "en"}
    return dict(q=q, r_teacher=r_t, p_strict=p_strict, p_lenient=p_len, r_student=r_s, latin=latin)


def test_handoff_language_rule():
    out = G.handoff_languages({1: {"en": 100, "bn": 60}, 2: {"en": 99, "bn": 55}}, 56)
    assert out["languages"] == ["en"] and out["excluded"] == ["bn"] and out["min_verified"]["bn"] == 55


def test_stage0_a_inclusion_selection_and_manipulation_gates():
    p = dict(zip(LANGS, [0.5, 0.4, 0.38, 0.36, 0.3, 0.32, 0.2, 0.1, 0.04, 0.08]))
    q = dict(zip(LANGS, [0.8, 0.7, 0.7, 0.7, 0.6, 0.7, 0.6, 0.5, 0.3, 0.5]))
    r = dict(zip(LANGS, [1.0, 0.9, 0.95, 0.8, 0.7, 0.99, 0.6, 0.5, 0.4, 0.3]))
    inputs = synthetic(p, q, r)
    scorer = {"pass": True}
    out = G.stage0_a(CFG, handoff={"languages": LANGS}, scorer=scorer, handoff_updates=32, **inputs)
    assert "sw" not in out["included"]          # p 0.04 < 0.05
    assert out["selection"]["M1"] == ["zh", "fr", "es", "de"]
    assert out["selection"]["M4"] == ["bn", "th", "ja", "ru"]
    assert out["selection"]["M6"][0] == "en" and out["selection"]["M6"][1] == "es" and out["selection"]["M6"][-1] == "bn"
    assert out["selection"]["S0d"] == ["en", "bn"]
    g = out["g"]
    assert out["selection"]["M5"]["A"] == sorted([l for l in out["included"] if l != "en"], key=lambda l: (-g[l], LANGS.index(l)))[:2]
    assert out["imprint"]["outcome"] == "go" and out["gap"]["outcome"] == "go" and out["proceed"]
    assert out["range"]["pass"] == (max(g[l] for l in out["included"]) - min(g[l] for l in out["included"]) >= 0.15)
    weak = G.stage0_a(CFG, handoff={"languages": LANGS}, scorer=scorer, handoff_updates=32,
                      **synthetic(p, q, r, latin_gap=0.0))
    assert weak["imprint"]["outcome"] == "m1a_only"
    none = G.stage0_a(CFG, handoff={"languages": LANGS}, scorer=scorer, handoff_updates=32,
                      **synthetic(p, q, r, latin_gap=0.0, ratio_gap=0.1, priv_gap=0.05))
    assert none["imprint"]["outcome"] == "no-go" and none["gap"]["outcome"] == "no-go"


def test_format_gate_extends_once_then_drops():
    p = {l: 0.3 for l in LANGS}
    q = {l: 0.6 for l in LANGS}
    r = {l: 0.9 for l in LANGS}
    inputs = synthetic(p, q, r)
    inputs["p_lenient"] = {s: {l: (0.6 if l == "th" else 0.32) for l in LANGS} for s in inputs["p_strict"]}
    first = G.stage0_a(CFG, handoff={"languages": LANGS}, scorer={"pass": True}, handoff_updates=32, **inputs)
    assert first["extend_handoff"] and not first["proceed"] and "th" in first["format"]["fail"]
    second = G.stage0_a(CFG, handoff={"languages": LANGS}, scorer={"pass": True}, handoff_updates=48, **inputs)
    assert not second["extend_handoff"] and "th" not in second["included"] and second["proceed"]


def test_scorer_gate():
    rows = [{"lenient_norm": 1, "lenient_legacy": 1}] * 99 + [{"lenient_norm": 1, "lenient_legacy": 0}]
    assert G.scorer_gate(rows, 0.99)["pass"] and not G.scorer_gate(rows[1:] + rows[-1:], 0.99)["pass"]


def test_stage0_b_noise_rules():
    start = {"en": 0.50, "bn": 0.20}
    ok = G.stage0_b(CFG, s0d={"rl": {"en": [0.52, 0.53, 0.55, 0.51], "bn": [0.21, 0.22, 0.20, 0.23]},
                              "opd": {"en": [0.50, 0.505, 0.51, 0.50], "bn": [0.2, 0.2, 0.21, 0.2]}}, start=start)
    assert ok["decision"] == "go" and ok["grid_seed_count"] == 4
    noisy = G.stage0_b(CFG, s0d={"rl": {"en": [0.52, 0.60, 0.55, 0.51], "bn": [0.21, 0.22, 0.20, 0.23]},
                                 "opd": {"en": [0.5] * 4, "bn": [0.2] * 4}}, start=start)
    assert noisy["grid_seed_count"] == 6
    bad = G.stage0_b(CFG, s0d={"rl": {"en": [0.49, 0.48, 0.55, 0.41], "bn": [0.21] * 4},
                               "opd": {"en": [0.5] * 4, "bn": [0.2] * 4}}, start=start)
    assert bad["grid_seed_count"] == 0 and bad["rl_collapses"] == 1


def test_conditional_gates():
    assert G.allocation_gate(CFG, m3a_supported=True, d_mean={"A": 5.0, "B": -2.0})["go"]
    assert not G.allocation_gate(CFG, m3a_supported=True, d_mean={"A": 3.0, "B": -2.0})["go"]
    assert not G.allocation_gate(CFG, m3a_supported=False, d_mean={"A": 9.0, "B": -2.0})["go"]
    assert G.washout_gate(CFG, pilot_diffs_pp=[3, 2, -1, 4])["go"]
    assert not G.washout_gate(CFG, pilot_diffs_pp=[3, -2, -1, 4])["go"]


def test_m3_pilot_gate():
    from mlopd.experiment import load_config
    from mlopd.gates import m3_pilot_gate
    cfg = load_config()
    assert m3_pilot_gate(cfg, d_mean_pp={"high_p": -5.0, "low_p": 1.0})["go"]          # spread +6
    assert m3_pilot_gate(cfg, d_mean_pp={"high_p": 2.0, "low_p": -3.0})["go"]          # spread -5 (g direction)
    out = m3_pilot_gate(cfg, d_mean_pp={"high_p": -2.0, "low_p": 0.5})
    assert not out["go"] and out["spread_pp"] == 2.5
