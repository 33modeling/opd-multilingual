"""Contrasts and verdicts (design §7.1-§7.2, §13). Reads only finished task files under runs/.

Units: accuracies and ratios are fractions in the task files; every contrast here is in points (x 100), so a
difference of target-language ratio r of 0.05 is 5 points. Primary verdict rule: one-sided paired t, alpha 0.05,
no multiplicity correction for verdicts; Holm-adjusted p recorded over the primary verdicts present
(M1b, M2b, M3a, M4, M5, M6a, M6b). Language-level hypotheses (M3) use the Spearman permutation test.

Status values: "supported", "not supported", "not measured (gate no-go)", "not run (condition not met)",
"record" (M3 with fewer than 6 languages), "incomplete" (inputs missing).
Reporting condition (§13): M1a/M1b/M4 are "supported" only if the pair-level test is significant, at least 3 of the 4
language means are positive, and the seed-aggregated mean (languages averaged within seed) is positive; otherwise a
significant pair test is reported as "pair test significant; reporting condition not met".
"""
from __future__ import annotations

import statistics


from . import stats
from .records import read_rows
from .util import write_json

PRIMARY = ("M1b", "M2b", "M3a", "M3a-p", "M4", "M5", "M6a", "M6b")


def _eval(ctx, id, pair="main"):
    return ctx.complete(pair, id)


def _metric(summary, key):
    c = summary["combined"]
    return {"acc": c["acc"], "r": c["r_acc"], "match": c["match_acc"], "strict": c["strict_acc"],
            "pass4": c["lenient_norm_pass_at_k"]}[key]


def _item_vector(rows, key):
    """{(set, id): mean over samples}. A sample whose value is None (r of a response without letters) is left out,
    as in the run-level means; an item whose samples are all None is absent."""
    out = {}
    for r in rows:
        if r.get(key) is None:
            continue
        out.setdefault((r.get("set"), r["id"]), []).append(float(r[key]))
    return {k: sum(v) / len(v) for k, v in out.items()}


def _bootstrap(ctx, pairs, key, cfg):
    """pairs: [(eval id of arm 1, eval id of arm 2)]; item bootstrap of the mean difference in points."""
    column = {"acc": "lenient_norm", "r": "r", "match": "language_match"}.get(key)
    if column is None:
        return None
    vectors = []
    for a, b in pairs:
        vectors.append((_item_vector(read_rows(_eval(ctx, a)["rows"]), column),
                        _item_vector(read_rows(_eval(ctx, b)["rows"]), column)))
    # Items present in every arm of every pair, so that column k is the same item in every row of the matrix.
    common = sorted(set.intersection(*[set(va) & set(vb) for va, vb in vectors])) if vectors else []
    if not common:
        return None
    matrix = [[100 * (va[k] - vb[k]) for k in common] for va, vb in vectors]
    A = cfg["analysis"]
    out = stats.item_bootstrap(matrix, iterations=A["bootstrap"], seed=A["seed"])
    if isinstance(out, dict):
        out["items"] = len(common)
    return out


def paired_contrast(ctx, cfg, pairs, key, *, alternative="greater", reporting=False, bootstrap=True):
    """pairs: [{"lang", "seed", "a": eval id, "b": eval id}] -> contrast a - b in points."""
    missing = [p for p in pairs if _eval(ctx, p["a"]) is None or _eval(ctx, p["b"]) is None]
    if missing or not pairs:
        return {"status": "incomplete", "missing": [(p["a"], p["b"]) for p in missing]}
    values = [100 * (_metric(_eval(ctx, p["a"]), key) - _metric(_eval(ctx, p["b"]), key)) for p in pairs]
    test = stats.paired_t(values, alternative)
    out = {"test": test, "pairs": [{**p, "value": v} for p, v in zip(pairs, values)], "metric": key,
           "mde": stats.mde(test["sd"], test["n"], alpha=cfg["analysis"]["alpha"], power=cfg["analysis"]["power"])}
    out["collapsed_arms"] = sum(int(bool(_eval(ctx, p[k]).get("collapse"))) for p in pairs for k in ("a", "b"))
    for extra in ("strict", "pass4"):
        out[f"{extra}_mean_diff"] = statistics.fmean(
            100 * (_metric(_eval(ctx, p["a"]), extra) - _metric(_eval(ctx, p["b"]), extra)) for p in pairs)
    if bootstrap:
        out["item_bootstrap"] = _bootstrap(ctx, [(p["a"], p["b"]) for p in pairs], key, cfg)
    langs = sorted({p["lang"] for p in pairs if p.get("lang")})
    seeds = sorted({p["seed"] for p in pairs})
    by_lang = {l: statistics.fmean(v for p, v in zip(pairs, values) if p.get("lang") == l) for l in langs}
    out["per_language_mean"] = by_lang
    out["positive_languages"] = sum(v > 0 for v in by_lang.values())
    seed_means = [statistics.fmean(v for p, v in zip(pairs, values) if p["seed"] == s) for s in seeds]
    out["seed_aggregate"] = stats.paired_t(seed_means, alternative) if len(seed_means) > 1 else {"mean": seed_means[0] if seed_means else None}
    significant = test["p"] is not None and test["p"] < cfg["analysis"]["alpha"]
    if not significant:
        out["status"] = "not supported"
    elif reporting:
        agree = out["positive_languages"] >= 3 and (out["seed_aggregate"]["mean"] or 0) > 0
        out["reporting_condition"] = agree
        out["status"] = "supported" if agree else "pair test significant; reporting condition not met"
    else:
        out["status"] = "supported"
    return out


def _runs_seeds(ctx):
    b = ctx.gate("main", "b")
    count = b["grid_seed_count"] if b else len(ctx.cfg["seeds"]["grid"])
    return (ctx.cfg["seeds"]["grid"] + ctx.cfg["seeds"]["grid_extra"])[:count]


def collect_grid(ctx, a):
    seeds = _runs_seeds(ctx)
    updates = a["handoff_updates"]
    acc = {"opd": {}, "rl": {}}
    start, D, G0 = {}, {}, {"opd": {}, "rl": {}}
    complete_langs = []
    for l in a["included"]:
        ok = True
        for method in acc:
            acc[method][l] = {}
            for s in seeds:
                e = _eval(ctx, f"g/{method}-{l}-s{s}/eval-{l}")
                acc[method][l][s] = e["acc"] if e else None
                ok &= e is not None
        start[l] = {}
        for s in seeds:
            e = _eval(ctx, f"start/s{s}-u{updates}/{l}")
            start[l][s] = e["acc"] if e else None
            ok &= e is not None
        if not ok:
            continue
        complete_langs.append(l)
        D[l] = {s: 100 * (acc["opd"][l][s] - acc["rl"][l][s]) for s in seeds}
        for method in G0:
            G0[method][l] = statistics.fmean(100 * (acc[method][l][s] - start[l][s]) for s in seeds)
    return {"seeds": seeds, "languages": complete_langs, "acc": acc, "start": start, "D": D,
            "D_mean": {l: statistics.fmean(D[l].values()) for l in D}, "G0_mean": G0}


def _parallel(cfg):
    return bool(cfg.get("queue", {}).get("parallel_blocks", False))


def _as_record(result, gate):
    """Parallel mode ran a block whose gate said no-go: the result is reported, but only as a record."""
    if isinstance(result, dict) and result.get("test"):
        result = {**result, "status": "record (gate no-go)", "verdict_if_gate_had_passed": result.get("status"),
                  "gate": gate}
    return result


def _spearman(cfg, x, y, alternative="greater"):
    A = cfg["analysis"]
    return stats.spearman_permutation(x, y, alternative=alternative, exact_max_n=A["exact_max_n"],
                                      permutations_count=A["permutations"], seed=A["seed"])


def m3_verdicts(cfg, grid, a, b=None):
    langs = grid["languages"]
    out = {}
    if len(langs) < len(a["included"]):
        for h in ("M3a", "M3a-p", "M3b", "M3c"):
            out[h] = {"status": "incomplete", "languages_complete": langs}
        return out
    g, p = a["g"], a["p"]
    # M3a has two pre-registered predictors of D: the teacher gap g (rho > 0) and the student solve rate p
    # (rho < 0: RL gets no reward signal where p is low, OPD still gets a dense signal). Each is tested on its own;
    # the pair is Holm-adjusted and each is supported when its adjusted p < alpha.
    tests = {"M3a": ([grid["D_mean"][l] for l in langs], [g[l] for l in langs], "greater"),
             "M3a-p": ([grid["D_mean"][l] for l in langs], [p[l] for l in langs], "less"),
             "M3b": ([grid["G0_mean"]["rl"][l] for l in langs], [p[l] for l in langs], "greater"),
             "M3c": ([grid["G0_mean"]["opd"][l] for l in langs], [g[l] for l in langs], "greater")}
    p_incl = [p[l] for l in a["included"]]
    p_range = (max(p_incl) - min(p_incl)) if p_incl else None
    ranges = {"M3a": a["range"]["pass"],
              "M3a-p": p_range is not None and p_range >= cfg["gates"]["range_min"]}
    pilot = (b or {}).get("_m3_pilot")
    for h, (y, x, alternative) in tests.items():
        result = _spearman(cfg, y, x, alternative)
        entry = {"test": result, "languages": langs, "outcome": dict(zip(langs, y)), "predictor": dict(zip(langs, x)),
                 "alternative": alternative}
        if h == "M3a-p":
            entry["predictor_range"] = {"max_minus_min": p_range, "threshold": cfg["gates"]["range_min"]}
        if a["m3_mode"] != "verdict":
            entry["status"] = "record"
        elif h in ranges and not ranges[h]:
            entry["status"] = "not measured (gate no-go)"
        else:
            entry["status"] = "supported" if result["p"] is not None and result["p"] < cfg["analysis"]["alpha"] else "not supported"
            if h.startswith("M3a") and b is not None and not b.get("rl_sd_pass"):
                entry["note"] = "RL rerun SD above the gate threshold; grid seeds were increased to 6 (§5.4)"
        if h.startswith("M3a") and pilot is not None:
            entry["m3_pilot"] = {k: pilot.get(k) for k in ("spread_pp", "threshold_pp", "go")}
        out[h] = entry
    family = {h: out[h]["test"]["p"] for h in ("M3a", "M3a-p")
              if out[h].get("status") in ("supported", "not supported") and out[h]["test"].get("p") is not None}
    for h, adjusted in stats.holm(family).items():
        out[h]["p_holm_m3a_pair"] = adjusted
        out[h]["status"] = "supported" if adjusted < cfg["analysis"]["alpha"] else "not supported"
    ys, xs, la, se = [], [], [], []
    for l in langs:
        for s, v in grid["D"][l].items():
            ys.append(v); xs.append(g[l]); la.append(l); se.append(s)
    try:
        out["M3a"]["mixed_model_slope"] = stats.mixed_slope(ys, xs, la, se)
    except Exception as error:  # recorded secondary; a failed fit never blocks the verdict
        out["M3a"]["mixed_model_slope"] = {"error": repr(error)}
    try:
        out["M3a-p"]["mixed_model_slope"] = stats.mixed_slope(ys, [p[l] for l in la], la, se)
    except Exception as error:  # noqa: BLE001
        out["M3a-p"]["mixed_model_slope"] = {"error": repr(error)}
    return out


def _m1_pairs(a, seeds, arm_a, arm_b):
    return [{"lang": l, "seed": s, "a": arm_a.format(l=l, s=s), "b": arm_b.format(l=l, s=s)}
            for l in a["selection"]["M1"] for s in seeds[:4]]


def verdict_M1a(ctx, a, seeds):
    """r(SFT | T-def targets) - r(SFT | T-en targets), (language, seed) pairs, one-sided t + reporting condition."""
    result = None
    if a["imprint"]["outcome"] in ("go", "m1a_only") or _parallel(ctx.cfg):
        result = paired_contrast(ctx, ctx.cfg, _m1_pairs(a, seeds, "gref/sft-{l}-s{s}/eval-{l}",
                                                         "i/sft-ten-{l}-s{s}/eval-{l}"), "r", reporting=True)
    if a["imprint"]["outcome"] not in ("go", "m1a_only"):
        return _as_record(result, a["imprint"]) if result and result.get("test") else \
            {"status": "not measured (gate no-go)", "gate": a["imprint"]}
    return result


def verdict_M1b(ctx, a, seeds):
    """r(OPD | T-def) - r(OPD | T-en). M1c (transfer ratio) and M1d (accuracy cost) are recorded with it."""
    outcome = a["imprint"]["outcome"]
    if outcome != "go" and not _parallel(ctx.cfg):
        return {"status": "not measured (gate no-go)" if outcome == "no-go" else
                "not measured (gate no-go): manipulation not transmitted on student prefixes", "gate": a["imprint"]}
    pairs = _m1_pairs(a, seeds, "g/opd-{l}-s{s}/eval-{l}", "i/opd-ten-{l}-s{s}/eval-{l}")
    out = paired_contrast(ctx, ctx.cfg, pairs, "r", reporting=True)
    if out.get("test"):
        out["M1d_accuracy_difference"] = paired_contrast(ctx, ctx.cfg, pairs, "acc", alternative="two-sided",
                                                         bootstrap=False).get("test")
    return out if outcome == "go" else _as_record(out, a["imprint"])


def verdict_M1c(m1a, m1b):
    if not (m1a.get("test") and m1b.get("test")):
        return {"status": "incomplete"}
    return {"status": "record", "transfer_ratio": (m1b["test"]["mean"] / m1a["test"]["mean"])
            if m1a["test"]["mean"] else None}


def verdict_M3(ctx, a, b):
    """M3a, M3b, M3c from the grid (Spearman one-sided permutation tests; mixed-model slope recorded for M3a)."""
    grid = collect_grid(ctx, a)
    b = {**(b or {}), "_m3_pilot": ctx.gate("main", "m3-pilot")} if (b or ctx.gate("main", "m3-pilot")) else b
    return m3_verdicts(ctx.cfg, grid, a, b), grid


def verdict_M4(ctx, a, seeds):
    """Acc(OPD | T-priv) - Acc(OPD | T-def) on the M4 languages; target-ratio change recorded."""
    if a["gap"]["outcome"] != "go" and not _parallel(ctx.cfg):
        return {"status": "not measured (gate no-go)", "gate": a["gap"]}
    pairs = [{"lang": l, "seed": s, "a": f"k/opd-tpriv-{l}-s{s}/eval-{l}", "b": f"g/opd-{l}-s{s}/eval-{l}"}
             for l in a["selection"]["M4"] for s in seeds[:4]]
    out = paired_contrast(ctx, ctx.cfg, pairs, "acc", reporting=True)
    if out.get("test"):
        out["target_ratio_change"] = paired_contrast(ctx, ctx.cfg, pairs, "r", alternative="two-sided",
                                                     bootstrap=False)["test"]
    return out if a["gap"]["outcome"] == "go" else _as_record(out, a["gap"])


def verdict_M5(ctx, a):
    """Mean accuracy over the four A/B languages, matched - swapped, 8 new seeds."""
    cfg = ctx.cfg
    m5 = ctx.gate("main", "m5")
    if m5 is None:
        return {"status": "incomplete"}
    if not m5.get("go") and not _parallel(cfg):
        return {"status": "not run (condition not met)", "gate": m5}
    langs = a["selection"]["M5"]["A"] + a["selection"]["M5"]["B"]
    values, missing = [], []
    for s in cfg["seeds"]["allocation"]:
        ea = [_eval(ctx, f"a/matched-s{s}/eval-{l}") for l in langs]
        eb = [_eval(ctx, f"a/swapped-s{s}/eval-{l}") for l in langs]
        if None in ea or None in eb:
            missing.append(s)
            continue
        values.append(100 * (statistics.fmean(e["acc"] for e in ea) - statistics.fmean(e["acc"] for e in eb)))
    if missing:
        return {"status": "incomplete", "missing_seeds": missing}
    out = _simple(cfg, values, "greater")
    return out if m5.get("go") else _as_record(out, m5)


def verdict_M2b(ctx, a, seeds, m1b):
    """Non-inferiority (margin 3 points) of the language-match rate of masked OPD (T-en) against the handoff
    checkpoint; accuracy against plain OPD (T-en) recorded."""
    cfg = ctx.cfg
    par = _parallel(cfg)
    if m1b.get("status") != "supported" and not par:
        return {"status": "not run (condition not met)", "condition": "M1b supported"}
    gate = ctx.gate("main", "m2b")
    if gate is None or not (gate.get("go") or par):
        return {"status": "incomplete"}
    condition_met = m1b.get("status") == "supported" and gate.get("go")
    updates = a["handoff_updates"]
    values, pairs = [], []
    for l in a["selection"]["B"]:
        for s in seeds[:4]:
            e, h = _eval(ctx, f"b/opdmask-ten-{l}-s{s}/eval-{l}"), _eval(ctx, f"start/s{s}-u{updates}/{l}")
            if e is None or h is None:
                return {"status": "incomplete"}
            values.append(100 * (e["combined"]["match_acc"] - h["combined"]["match_acc"]))
            pairs.append({"lang": l, "seed": s})
    test = stats.noninferiority(values, cfg["analysis"]["noninferiority_margin_pp"])
    out = {"test": test, "pairs": pairs, "mde": stats.mde(test["sd"], test["n"]),
            "status": "supported" if test["p"] is not None and test["p"] < cfg["analysis"]["alpha"] else "not supported",
            "acc_vs_plain_opd_T-en": paired_contrast(
                ctx, cfg, [{"lang": l, "seed": s, "a": f"b/opdmask-ten-{l}-s{s}/eval-{l}",
                            "b": f"i/opd-ten-{l}-s{s}/eval-{l}"} for l in a["selection"]["B"] for s in seeds[:4]],
                "acc", alternative="two-sided", bootstrap=False).get("test")}
    return out if condition_met else _as_record(out, {"condition": "M1b supported", "m2b_gate": gate})


def verdicts(ctx):
    from .experiment import final_stage0
    a, b = final_stage0(ctx), ctx.gate("main", "b")
    if a is None:
        return {"status": "stage 0 not finished"}
    seeds = _runs_seeds(ctx)
    out = {"M1a": verdict_M1a(ctx, a, seeds), "M1b": verdict_M1b(ctx, a, seeds)}
    out["M1c"] = verdict_M1c(out["M1a"], out["M1b"])
    m3, grid = verdict_M3(ctx, a, b)
    out.update(m3)
    out["grid"] = {"D_mean": grid["D_mean"], "G0_mean": grid["G0_mean"], "languages": grid["languages"]}
    out["M4"] = verdict_M4(ctx, a, seeds)
    out["M5"] = verdict_M5(ctx, a)
    out.update(washout_verdicts(ctx, a))
    out["M2b"] = verdict_M2b(ctx, a, seeds, out["M1b"])
    pvalues = {h: out[h]["test"]["p"] for h in PRIMARY if isinstance(out.get(h), dict) and out[h].get("test")
               and out[h]["test"].get("p") is not None and out[h].get("status") in ("supported", "not supported")}
    adjusted = stats.holm(pvalues)
    for h, p in adjusted.items():
        out[h]["p_holm"] = p
    out["holm_family"] = sorted(adjusted)
    return out


def _simple(cfg, values, alternative):
    test = stats.paired_t(values, alternative)
    return {"test": test, "mde": stats.mde(test["sd"], test["n"]),
            "status": "supported" if test["p"] is not None and test["p"] < cfg["analysis"]["alpha"] else "not supported"}


def washout_verdicts(ctx, a):
    """M6a (accuracy) and M6b (target-language ratio): (W - N) at SFT end - (W - N) after English RL, mean of the
    three non-English M6 languages, 8 seeds."""
    cfg = ctx.cfg
    gate = ctx.gate("main", "m6")
    if gate is None:
        return {"M6a": {"status": "incomplete"}, "M6b": {"status": "incomplete"}}
    if not gate.get("go") and not _parallel(cfg):
        return {h: {"status": "not measured (gate no-go)", "gate": gate} for h in ("M6a", "M6b")}
    langs = [l for l in a["selection"]["M6"] if l != "en"]
    rows = {"acc": [], "r": []}
    record = []
    for s in cfg["seeds"]["washout"]:
        def mean_of(prefix, key, which=None):
            values = []
            for l in langs:
                id = f"{prefix}/eval-{which}-{l}" if which else f"{prefix}/eval-{l}"
                e = _eval(ctx, id)
                if e is None:
                    return None
                values.append(_metric(e, key))
            return statistics.fmean(values)
        entry = {"seed": s}
        for key in rows:
            sft_w, sft_n = mean_of(f"w/sft-W-s{s}", key, "b"), mean_of(f"w/sft-N-s{s}", key, "b")
            rl_w, rl_n = mean_of(f"w/rl-W-s{s}", key), mean_of(f"w/rl-N-s{s}", key)
            base_w, base_n = mean_of(f"w/sft-W-s{s}", key, "a"), mean_of(f"w/sft-N-s{s}", key, "a")
            if None in (sft_w, sft_n, rl_w, rl_n):
                return {h: {"status": "incomplete"} for h in ("M6a", "M6b")}
            rows[key].append(100 * ((sft_w - sft_n) - (rl_w - rl_n)))
            entry[key] = {"sft_end_diff": 100 * (sft_w - sft_n), "after_rl_diff": 100 * (rl_w - rl_n),
                          "rl_change_W": 100 * (rl_w - base_w) if base_w is not None else None,
                          "rl_change_N": 100 * (rl_n - base_n) if base_n is not None else None}
        record.append(entry)
    out = {"M6a": {**_simple(cfg, rows["acc"], "greater"), "per_seed": record},
           "M6b": {**_simple(cfg, rows["r"], "greater"), "per_seed": record}}
    return out if gate.get("go") else {h: _as_record(v, gate) for h, v in out.items()}


def write_verdicts(ctx, path=None):
    result = verdicts(ctx)
    path = path or ctx.runs / "analysis" / "verdicts.json"
    write_json(path, result)
    return path, result
