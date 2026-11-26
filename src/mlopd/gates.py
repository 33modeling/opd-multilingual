"""Stage-0 gates and selection rules (design §5.4, §13 "gates"), computed only from stage-0 files.

Every function here is pure: it takes numbers already read from stage-0 files and returns decisions. The file-level
wrappers in experiment.py record the sha256 of every input file and of this module next to the decisions.

Choices where §5.4 is not explicit (recorded in the output as `rules`):
  * Handoff languages: a language enters the handoff SFT mix when its verified-target count in F is >= 56 for every
    grid seed (the minimum over seeds), so that all grid seeds share one composition.
  * p_l, r^S_l: means over the grid seeds of the handoff student measured on the stage-0 items S0; q_l: initial
    teacher (T-def) on the same S0 items; g_l = q_l - (handoff lenient_norm pass@1 on S0).
  * Format gate: the seed-mean strict pass@1 / seed-mean lenient_norm pass@1 of the handoff student on S0.
  * Imprint gate: both conditions are counted per M1 language and need 3 of 4 (the first condition says so in
    §5.4; the second is given the same form).
  * Ties in selection rules break by the L10 order.
  * M6 middle language: the non-English included languages sorted by p (high to low), the top and bottom removed,
    then the element at index (n - 1) // 2 of the rest.
"""
from __future__ import annotations

import statistics

from .util import L10


def _order(language):
    return L10.index(language)


def _at_least(a, b, gap):
    """a - b >= gap; a missing value (None) fails."""
    return a is not None and b is not None and (a - b) >= gap


def handoff_languages(verified, minimum):
    """verified: {seed: {lang: count of F items with a verified teacher target}}."""
    languages = sorted({l for counts in verified.values() for l in counts}, key=_order)
    low = {l: min(counts.get(l, 0) for counts in verified.values()) for l in languages}
    keep = [l for l in languages if low[l] >= minimum]
    return {"languages": keep, "excluded": [l for l in languages if l not in keep], "min_verified": low,
            "rule": f"min over grid seeds of verified F targets >= {minimum}"}


def scorer_gate(rows, threshold):
    """Response-level agreement of lenient_norm with the English record's lenient on the initial student's E_en."""
    if not rows:
        return {"pass": False, "agreement": None, "responses": 0}
    agree = sum(r["lenient_norm"] == r["lenient_legacy"] for r in rows) / len(rows)
    return {"pass": agree >= threshold, "agreement": agree, "responses": len(rows), "threshold": threshold}


def select_m1(candidates, r_student, p, k):
    return sorted(candidates, key=lambda l: (-r_student[l], -p[l], _order(l)))[:k]


def select_m4(candidates, p, k):
    return sorted(candidates, key=lambda l: (p[l], _order(l)))[:k]


def select_m5(included, g, p, k):
    non_english = [l for l in included if l != "en"]
    a = sorted(non_english, key=lambda l: (-g[l], _order(l)))[:k]
    b = sorted([l for l in included if l not in a], key=lambda l: (-p[l], _order(l)))[:k]
    return {"A": a, "B": b}


def select_m6(candidates, p):
    ranked = sorted(candidates, key=lambda l: (-p[l], _order(l)))
    if len(ranked) < 3:
        return ["en"] + ranked
    middle = ranked[1:-1]
    return ["en", ranked[0], middle[(len(middle) - 1) // 2], ranked[-1]]


def stage0_a(cfg, *, handoff, q, r_teacher, p_strict, p_lenient, r_student, latin, scorer, handoff_updates):
    """q, r_teacher: {condition: {lang: value}} (initial teacher on S0). p_strict, p_lenient, r_student:
    {seed: {lang: value}} (handoff student on S0). latin: {lang: {"T-def": mass, "T-en": mass}}."""
    G = cfg["gates"]
    languages = handoff["languages"]
    seeds = sorted(p_strict)
    p = {l: statistics.fmean(p_strict[s][l] for s in seeds) for l in languages}
    p_len = {l: statistics.fmean(p_lenient[s][l] for s in seeds) for l in languages}
    r_s = {l: statistics.fmean(r_student[s][l] for s in seeds if r_student[s][l] is not None)
           if any(r_student[s][l] is not None for s in seeds) else 0. for l in languages}
    qd = q["T-def"]
    g = {l: qd[l] - p_len[l] for l in languages}
    inclusion = {l: {"p": p[l], "q": qd[l], "pass": p[l] >= G["p_min"] and qd[l] >= G["q_min"]} for l in languages}
    ratio = {l: (p[l] / p_len[l] if p_len[l] else 0.) for l in languages}
    format_fail = [l for l in languages if inclusion[l]["pass"] and ratio[l] < G["format_ratio"]]
    extend = bool(format_fail) and handoff_updates < cfg["train"]["handoff"]["extended_updates"]
    included = [l for l in languages if inclusion[l]["pass"] and (extend or l not in format_fail)]
    non_english = [l for l in included if l != "en"]
    out = {"handoff_updates": handoff_updates, "handoff_languages": languages,
           "scorer": scorer, "inclusion": inclusion,
           "format": {"ratio": ratio, "fail": format_fail, "threshold": G["format_ratio"],
                      "action": ("extend handoff SFT to %d updates and measure again"
                                 % cfg["train"]["handoff"]["extended_updates"]) if extend else
                      ("drop failing languages" if format_fail else "none")},
           "extend_handoff": extend, "included": included,
           "p": p, "p_lenient": p_len, "q": q, "g": g, "r_student": r_s, "r_teacher": r_teacher}
    out["m3_mode"] = "verdict" if len(included) >= G["min_languages_for_m3"] else "record (fewer than %d languages)" % G["min_languages_for_m3"]
    g_incl = [g[l] for l in included]
    out["range"] = {"max_minus_min": (max(g_incl) - min(g_incl)) if g_incl else None, "threshold": G["range_min"]}
    out["range"]["pass"] = out["range"]["max_minus_min"] is not None and out["range"]["max_minus_min"] >= G["range_min"]
    S = cfg["selection"]
    m1 = select_m1(non_english, r_s, p, S["m1_count"])
    m4 = select_m4(non_english, p, S["m4_count"])
    selection = {"M1": m1, "M4": m4, "M5": select_m5(included, g, p, S["m5_group"]), "M6": select_m6(non_english, p),
                 "S0d": ["en"] + select_m4(non_english, p, 1)}
    x = S.get("x_languages")
    selection["X"] = list(x) if x else ["en"] + sorted(non_english, key=lambda l: (-g[l], _order(l)))[:1]
    selection["B"] = m1[:S["b_languages"]]
    out["selection"] = selection
    latin_ok = {l: _at_least(latin[l]["T-en"], latin[l]["T-def"], G["latin_mass_gap"]) for l in m1 if l in latin}
    ratio_ok = {l: _at_least(r_teacher["T-def"][l], r_teacher["T-en"][l], G["teacher_ratio_gap"]) for l in m1}
    need = G["imprint_languages_needed"]
    first, second = sum(latin_ok.values()) >= need, sum(ratio_ok.values()) >= need
    out["imprint"] = {"latin_mass": {l: latin.get(l) for l in m1}, "latin_ok": latin_ok, "ratio_ok": ratio_ok,
                      "first_condition": first, "second_condition": second,
                      "outcome": "go" if first and second else ("m1a_only" if second else "no-go")}
    gap_ok = {l: _at_least(q["T-priv"][l], q["T-def"][l], G["gap_manipulation"]) for l in m4}
    out["gap"] = {"per_language": {l: q["T-priv"][l] - q["T-def"][l] for l in m4}, "ok": gap_ok,
                  "outcome": "go" if sum(gap_ok.values()) >= G["gap_languages_needed"] else "no-go"}
    out["proceed"] = bool(scorer["pass"]) and not extend
    return out


def stage0_b(cfg, *, s0d, start):
    """s0d: {method: {lang: [acc of the reruns (fraction)]}}; start: {lang: handoff acc (fraction)}."""
    G = cfg["gates"]
    out = {"rl": {}, "opd": {}}
    collapses = 0
    for method in ("rl", "opd"):
        for language, values in s0d.get(method, {}).items():
            sd = statistics.stdev(values) * 100 if len(values) > 1 else None
            above = sum(v > start[language] for v in values)
            below = sum(100 * (start[language] - v) > G["collapse_pp"] for v in values)
            if method == "rl":
                collapses += below
            out[method][language] = {"values": values, "sd_pp": sd, "above_start": above, "collapses": below,
                                     "start": start[language]}
    rl_sd_ok = all(v["sd_pp"] is not None and v["sd_pp"] <= G["rl_sd_max_pp"] for v in out["rl"].values())
    english_ok = out["rl"].get("en", {}).get("above_start", 0) >= G["rl_english_above_start"]
    opd_ok = all(v["sd_pp"] is not None and v["sd_pp"] <= G["opd_sd_max_pp"] for v in out["opd"].values())
    out.update(rl_sd_pass=rl_sd_ok, rl_english_pass=english_ok, opd_sd_pass=opd_ok, rl_collapses=collapses,
               collapse_risk=collapses >= 2)
    base = len(cfg["seeds"]["grid"])
    more = base + len(cfg["seeds"]["grid_extra"])
    if not english_ok:
        out["decision"], out["grid_seed_count"] = "no-go: English RL not above start in 3 of 4 (check the setting)", 0
    elif not opd_ok and not G.get("override_opd_noise_gate"):
        out["decision"], out["grid_seed_count"] = "hold: OPD rerun SD above threshold (check the cause; set gates.override_opd_noise_gate to continue)", 0
    elif not rl_sd_ok:
        out["decision"], out["grid_seed_count"] = f"go with {more} grid seeds (RL rerun SD above threshold)", more
    else:
        out["decision"], out["grid_seed_count"] = "go", base
    return out


def allocation_gate(cfg, *, m3a_supported, d_mean):
    """M5 runs only if M3a is supported and mean D over group A - mean D over group B >= 6pp (d_mean in pp)."""
    gap = (d_mean["A"] - d_mean["B"]) if d_mean.get("A") is not None and d_mean.get("B") is not None else None
    go = bool(m3a_supported) and gap is not None and gap >= cfg["gates"]["allocation_d_gap_pp"]
    return {"m3a_supported": bool(m3a_supported), "d_gap_pp": gap, "go": go}


def washout_gate(cfg, *, pilot_diffs_pp):
    """pilot_diffs_pp: SFT-end (W - N) non-English accuracy differences of the pilot seeds, in pp."""
    G = cfg["gates"]
    mean = statistics.fmean(pilot_diffs_pp) if pilot_diffs_pp else None
    positive = sum(v > 0 for v in pilot_diffs_pp)
    go = mean is not None and mean >= G["washout_sft_gap_pp"] and positive >= G["washout_positive_pilots"]
    return {"pilot_diffs_pp": pilot_diffs_pp, "mean_pp": mean, "positive": positive, "go": go}


def m3_pilot_gate(cfg, *, d_mean_pp):
    """Before the full grid: D (OPD - RL, pp) on the highest-p and lowest-p included languages, first two seeds.
    The remaining grid runs only if |D(low p) - D(high p)| >= m3_pilot_spread_pp: a smaller spread between the two
    extremes leaves the language-level correlation of M3 without the range it needs (design §7.3)."""
    values = list(d_mean_pp.values())
    spread = (d_mean_pp["low_p"] - d_mean_pp["high_p"]) if None not in values else None
    threshold = cfg["gates"]["m3_pilot_spread_pp"]
    return {"d_mean_pp": d_mean_pp, "spread_pp": spread, "threshold_pp": threshold,
            "go": spread is not None and abs(spread) >= threshold}
