"""Run grid of design §5.3 as a resumable queue, and the executor for every task kind.

The queue is rebuilt from the config and the gate files after every task, so blocks that depend on a stage-0
decision appear only once that decision file exists. The confirmation block C (pre-registration amendment
2026-11-28) appears only after every gate of the original blocks exists. One task runs at a time; a task is finished when its
COMPLETE.json exists and is skipped afterwards. Outputs: runs/<pair>/<task id>/.

Task kinds
  measure          generation + scoring on stage-0 items (S0-a, S0-b, handoff p_l / r^S_l)
  teacher_targets  teacher samples on F or U items and the verified SFT targets built from them
  train            one SFT / OPD / RL / mixed unit (train.Unit)
  eval             a checkpoint on E_l (MGSM 250 + MSVAMP 250, 4 samples, temperature 1, 512 tokens)
  latin_mass       teacher Latin-script probability mass at the first 16 positions of handoff-student samples
  gate             stage-0 / conditional decisions (gates.py), CPU only
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import time

from . import gates as G
from .data import Dataset
from .records import LazyIdentifier, read_rows
from .util import read_json, sha256_file, shuffled, stable_seed, write_json

CONDITIONS = ("T-def", "T-en", "T-priv")
GATES_SOURCE = Path(__file__).with_name("gates.py")


def deep_update(base, extra):
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path=None, overrides=None):
    default = Path(__file__).resolve().parents[2] / "configs" / "default.json"
    cfg = read_json(default)
    for part in (str(path).split(",") if path else []):  # "a.json,b.json": merged left to right
        if part and Path(part).resolve() != default.resolve():
            cfg = deep_update(cfg, read_json(part))
    return deep_update(cfg, overrides)


def task(id, block, kind, pair="main", deps=(), hours=0., **params):
    return {"id": id, "block": block, "kind": kind, "pair": pair, "deps": list(deps), "hours": hours,
            "params": params}


class Context:
    def __init__(self, cfg, *, root=None):
        self.cfg = cfg
        self.root = Path(root or ".")
        self.runs = self.root / cfg["paths"]["runs"]
        self._dataset = None
        self._runtime = {}
        self.identifier = LazyIdentifier(cfg.get("langid"))

    # ------------------------------------------------------------ data
    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = Dataset.load(self.root / self.cfg["paths"]["data"])
        return self._dataset

    def splits(self, seed):
        s = self.cfg["splits"]
        return self.dataset.splits(seed, split_seed=s["split_seed"], sizes=s)

    def stage2_order(self, seed):
        return shuffled(self.splits(seed)["U"], seed, "stage2-order")

    # ------------------------------------------------------------ files
    def dir(self, t):
        return self.runs / t["pair"] / t["id"]

    def done(self, t):
        return (self.dir(t) / "COMPLETE.json").exists()

    def complete(self, pair, id):
        path = self.runs / pair / id / "COMPLETE.json"
        return read_json(path) if path.exists() else None

    def gate(self, pair, name):
        return self.complete(pair, f"gates/{name}")

    # ------------------------------------------------------------ models
    def runtime(self, pair):
        if pair not in self._runtime:
            from .runtime import Runtime
            for other in list(self._runtime):
                del self._runtime[other]
            self._gc()
            models = {**self.cfg["model_pairs"][pair], "revisions": self.cfg.get("model_revisions") or {}}
            self._runtime[pair] = Runtime(models, self.cfg["runtime"], adapter_seed=stable_seed(0, "adapter"))
        return self._runtime[pair]

    @staticmethod
    def _gc():
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ================================================================ queue
def _hours(cfg, key, count=1.):
    return cfg["estimates_hours"][key] * count


def _gen_hours(cfg, generations, role):
    per = cfg["estimates_hours"]["teacher_generations_per_hour" if role == "teacher" else "student_generations_per_hour"]
    return generations / per


def final_stage0(ctx, pair="main"):
    """The latest stage-0 decision file (handoff 32, or 48 after a format-gate extension)."""
    cfg = ctx.cfg
    for updates in (cfg["train"]["handoff"]["extended_updates"], cfg["train"]["handoff"]["updates"]):
        decision = ctx.gate(pair, f"a-u{updates}")
        if decision is not None:
            return decision
    return None


def stage0_tasks(ctx, pair="main", languages=None, seeds=None):
    """S0-a, S0-b, scorer, teacher targets on F, handoff languages, handoff SFT, start evaluation, p-measurement,
    Latin mass, gates A, S0-d and gates B (main pair). For pair x a reduced version (S0-a T-def, handoff, start)."""
    cfg = ctx.cfg
    S, E = cfg["stage0"], cfg["eval"]
    langs = list(languages or cfg["languages"])
    seeds = list(seeds or cfg["seeds"]["grid"])
    n_s0 = cfg["splits"]["S0"] * S["samples"]
    out = []
    for l in langs:
        out.append(task(f"s0a/student/{l}", "S0-a", "measure", pair, hours=_gen_hours(cfg, n_s0, "student"),
                        role="initial", condition="T-def", lang=l, items="S0"))
    conditions = CONDITIONS if pair == "main" else ("T-def",)
    for c in conditions:
        for l in langs:
            if c == "T-priv" and l == "en":
                continue
            out.append(task(f"s0a/teacher-{c}/{l}", "S0-a", "measure", pair, hours=_gen_hours(cfg, n_s0, "teacher"),
                            role="teacher", condition=c, lang=l, items="S0"))
    if pair == "main":
        for c in S["s0b_conditions"]:
            for l in langs:
                out.append(task(f"s0b/{c}/{l}", "S0-b", "measure", pair, hours=_gen_hours(cfg, n_s0, "student"),
                                role="initial", condition=c, lang=l, items="S0"))
        out.append(task("s0/scorer-en", "S0-c", "eval", pair, hours=_hours(cfg, "eval_E"), adapter=None, lang="en",
                        eval_seed=E["eval_seed"], sets=E["sets"]))
    out += handoff_chain(ctx, pair, seeds, langs, block="S0-c", gate_seeds=seeds)
    return out


def handoff_chain(ctx, pair, seeds, langs, *, block, gate_seeds, measure=True):
    """Teacher targets on F, the handoff-language decision (from gate_seeds), handoff SFT, start evaluation,
    p-measurement (and Latin mass for the first seed)."""
    cfg = ctx.cfg
    T, E, S = cfg["teacher_targets"], cfg["eval"], cfg["stage0"]
    out = []
    for seed in sorted(set(seeds) | set(gate_seeds)):
        for l in langs:
            out.append(task(f"targets/T-def/F-s{seed}/{l}", block, "teacher_targets", pair,
                            hours=_gen_hours(cfg, cfg["splits"]["F"] * T["samples"], "teacher"),
                            condition="T-def", lang=l, itemset="F", seed=seed))
    hl_id = "gates/handoff-languages-" + "-".join(map(str, gate_seeds))
    out.append(task(hl_id, block, "gate", pair, deps=[f"targets/T-def/F-s{s}/{l}" for s in gate_seeds for l in langs],
                    name="handoff_languages", seeds=list(gate_seeds), langs=langs))
    decision = ctx.complete(pair, hl_id)
    if decision is None:
        return out
    hlangs = decision["languages"]
    updates = handoff_updates(ctx, pair)
    for seed in seeds:
        h = f"handoff/s{seed}-u{updates}"
        out.append(task(h, block, "train", pair, deps=[f"targets/T-def/F-s{seed}/{l}" for l in hlangs],
                        hours=_hours(cfg, "sft_32x8", updates / 32), plan="handoff", seed=seed, langs=hlangs,
                        updates=updates))
        if not measure:
            continue
        for l in hlangs:
            out.append(task(f"start/s{seed}-u{updates}/{l}", block, "eval", pair, deps=[h], hours=_hours(cfg, "eval_E"),
                            adapter=h, lang=l, eval_seed=E["eval_seed"], sets=E["sets"]))
            out.append(task(f"p/s{seed}-u{updates}/{l}", block, "measure", pair, deps=[h],
                            hours=_gen_hours(cfg, cfg["splits"][S["p_items"]] * S["samples"], "student"),
                            role="student", adapter=h, condition="T-def", lang=l, items=S["p_items"], seed=seed,
                            keep_ids=(seed == seeds[0])))
    if pair == "main" and measure:
        seed0 = seeds[0]
        for l in hlangs:
            if l != "en":
                out.append(task(f"latin/s{seed0}-u{updates}/{l}", block, "latin_mass", pair,
                                deps=[f"p/s{seed0}-u{updates}/{l}"], hours=_hours(cfg, "latin_mass"), lang=l,
                                seed=seed0, measure=f"p/s{seed0}-u{updates}/{l}"))
    return out


def handoff_updates(ctx, pair):
    cfg = ctx.cfg
    base = cfg["train"]["handoff"]["updates"]
    first = ctx.gate(pair, f"a-u{base}")
    if first is not None and first.get("extend_handoff"):
        return cfg["train"]["handoff"]["extended_updates"]
    return base


def stage0_gate_tasks(ctx, tasks):
    cfg = ctx.cfg
    updates = handoff_updates(ctx, "main")
    deps = [t["id"] for t in tasks if t["block"] in ("S0-a", "S0-b") or t["id"] == "s0/scorer-en"
            or t["id"].startswith(("start/", "p/", "latin/")) and f"-u{updates}/" in t["id"]]
    deps += [t["id"] for t in tasks if t["id"].startswith("gates/handoff-languages")]
    out = [task(f"gates/a-u{updates}", "S0-c", "gate", "main", deps=deps, name="stage0_a", updates=updates)]
    decision = ctx.gate("main", f"a-u{updates}")
    if decision is None or not decision.get("proceed"):
        return out
    seed0 = cfg["seeds"]["grid"][0]
    handoff = f"handoff/s{seed0}-u{updates}"
    s0d_ids = []
    for l in decision["selection"]["S0d"]:
        for method in ("opd", "rl"):
            for tag in cfg["stage0"]["s0d_tags"]:
                id = f"s0d/{method}-{l}-{tag}"
                out.append(task(id, "S0-d", "train", deps=[handoff], hours=_hours(cfg, f"{method}_128"),
                                plan=method, seed=seed0, lang=l, start=handoff, condition="T-def", tag=tag))
                out.append(task(f"{id}/eval-{l}", "S0-d", "eval", deps=[id], hours=_hours(cfg, "eval_E"), adapter=id,
                                lang=l, eval_seed=cfg["eval"]["eval_seed"], sets=cfg["eval"]["sets"]))
                s0d_ids.append(f"{id}/eval-{l}")
    out.append(task("gates/b", "S0-d", "gate", deps=s0d_ids, name="stage0_b", updates=updates))
    return out


def second_stage(ctx, id, block, plan, *, seed, lang, start, eval_langs, condition="T-def", tag="r0", deps=(),
                 eval_sets=None, extra=None):
    cfg = ctx.cfg
    key = {"opd": "opd_128", "opd_mask": "opd_128", "rl": "rl_128", "sft_continue": "sft_128x8", "mixed": "opd_128",
           "washout_sft": "sft_32x8"}[plan]
    out = [task(id, block, "train", deps=[start] + list(deps) if start else list(deps), hours=_hours(cfg, key),
                plan=plan, seed=seed, lang=lang, start=start, condition=condition, tag=tag, **(extra or {}))]
    sets = eval_sets or cfg["eval"]["sets"]
    for l in eval_langs:
        out.append(task(f"{id}/eval-{l}", block, "eval", deps=[id],
                        hours=_hours(cfg, "eval_E" if len(sets) > 1 else "eval_E_mgsm"), adapter=id, lang=l,
                        eval_seed=cfg["eval"]["eval_seed"], sets=sets))
    return out


def parallel_blocks(cfg):
    """True: after stage 0 every block is queued at once and gates only label results (configs/h100.json)."""
    return bool(cfg.get("queue", {}).get("parallel_blocks", False))


def m3_pilot_plan(ctx, a):
    """The two included languages with the highest and lowest p (ties: language order) and the first seeds."""
    cfg = ctx.cfg
    ranked = sorted(a["included"], key=lambda l: (a["p"][l], cfg["languages"].index(l)))
    seeds = cfg["seeds"]["grid"][:cfg["gates"]["m3_pilot_seeds"]]
    return {"high_p": ranked[-1], "low_p": ranked[0], "seeds": seeds} if len(ranked) >= 2 else None


def grid_tasks(ctx, a, b):
    """Blocks G, G-ref, G-x, I, K, then the conditional blocks A, W, B and X.

    Sequential mode: G starts with the M3 pilot (highest-p and lowest-p languages, two seeds); the rest of G runs
    only after gates/m3-pilot says go, except the OPD runs that M1b and M4 use as their T-def arm. Blocks I, K, A,
    W and B wait for their gates. Parallel mode (queue.parallel_blocks): G covers all languages and the six grid
    seeds, and every block is queued at once; the gate files are still written and the analysis uses them to label
    each result as a verdict or a record."""
    cfg = ctx.cfg
    par = parallel_blocks(cfg)
    updates = a["handoff_updates"]
    sel = a["selection"]
    all_seeds = cfg["seeds"]["grid"] + cfg["seeds"]["grid_extra"]
    count = len(all_seeds) if par else b["grid_seed_count"]
    seeds = all_seeds[:count]
    grid_langs = list(cfg["languages"]) if par else list(a["included"])
    out = []
    if count > len(cfg["seeds"]["grid"]):
        extra = seeds[len(cfg["seeds"]["grid"]):]
        out += handoff_chain(ctx, "main", extra, cfg["languages"], block="G", gate_seeds=cfg["seeds"]["grid"])
    T = cfg["teacher_targets"]
    imprint = a["imprint"]["outcome"]
    gap_go = par or a["gap"]["outcome"] == "go"
    pilot_plan = m3_pilot_plan(ctx, a)
    pilot = ctx.gate("main", "m3-pilot")
    full_grid = par or (pilot is not None and pilot.get("go"))
    keep_opd = set(sel["M1"]) | (set(sel["M4"]) if gap_go else set())

    def in_pilot(l, seed):
        return pilot_plan is not None and l in (pilot_plan["high_p"], pilot_plan["low_p"]) \
            and seed in pilot_plan["seeds"]

    for seed in seeds:
        h = f"handoff/s{seed}-u{updates}"
        for l in grid_langs:
            for method in ("opd", "rl"):
                if not (full_grid or in_pilot(l, seed) or (method == "opd" and l in keep_opd)):
                    continue
                made = second_stage(ctx, f"g/{method}-{l}-s{seed}", "G", method, seed=seed, lang=l, start=h,
                                    eval_langs=[l] + (sel["M1"] if l == "en" else []))
                for x in made:
                    if x["kind"] == "eval" and x["params"]["lang"] != l:
                        x["block"] = "G-x"
                out += made
            if not (full_grid or l in sel["M1"]):
                continue
            tid = f"targets/T-def/U-s{seed}/{l}"
            out.append(task(tid, "G-ref", "teacher_targets",
                            hours=_gen_hours(cfg, cfg["splits"]["U"] * T["samples"], "teacher"),
                            condition="T-def", lang=l, itemset="U", seed=seed))
            out += second_stage(ctx, f"gref/sft-{l}-s{seed}", "G-ref", "sft_continue", seed=seed, lang=l, start=h,
                                eval_langs=[l], deps=[tid])
        for l in sel["M1"]:
            if par or imprint in ("go", "m1a_only"):
                tid = f"targets/T-en/U-s{seed}/{l}"
                out.append(task(tid, "I", "teacher_targets",
                                hours=_gen_hours(cfg, cfg["splits"]["U"] * T["samples"], "teacher"),
                                condition="T-en", lang=l, itemset="U", seed=seed))
                out += second_stage(ctx, f"i/sft-ten-{l}-s{seed}", "I", "sft_continue", seed=seed, lang=l, start=h,
                                    eval_langs=[l], condition="T-en", deps=[tid])
            if par or imprint == "go":
                out += second_stage(ctx, f"i/opd-ten-{l}-s{seed}", "I", "opd", seed=seed, lang=l, start=h,
                                    eval_langs=[l], condition="T-en")
        if gap_go:
            for l in sel["M4"]:
                out += second_stage(ctx, f"k/opd-tpriv-{l}-s{seed}", "K", "opd", seed=seed, lang=l, start=h,
                                    eval_langs=[l], condition="T-priv")
    if pilot is None and pilot_plan is not None:
        deps = [f"g/{m}-{l}-s{s}/eval-{l}" for m in ("opd", "rl") for l in (pilot_plan["high_p"], pilot_plan["low_p"])
                for s in pilot_plan["seeds"]]
        out.append(task("gates/m3-pilot", "G", "gate", deps=deps, name="m3_pilot"))
    if par or pilot is not None:  # gates/m5 needs the whole grid listed, so it waits for the pilot decision
        grid_eval = [t["id"] for t in out if t["block"] == "G" and t["kind"] == "eval"]
        out.append(task("gates/m5", "A", "gate", deps=grid_eval + ["gates/b"], name="allocation"))
    m5 = ctx.gate("main", "m5")
    if par or (m5 and m5.get("go")):
        aseeds = cfg["seeds"]["allocation"]
        out += handoff_chain(ctx, "main", aseeds, cfg["languages"], block="A", gate_seeds=cfg["seeds"]["grid"],
                             measure=False)
        groups = sel["M5"]
        for seed in aseeds:
            h = f"handoff/s{seed}-u{updates}"
            for arm in ("matched", "swapped"):
                out += second_stage(ctx, f"a/{arm}-s{seed}", "A", "mixed", seed=seed, lang=None, start=h,
                                    eval_langs=groups["A"] + groups["B"], extra={"arm": arm, "groups": groups})
    out += washout_tasks(ctx, a)
    m2b = ctx.gate("main", "m2b")
    if m2b is None and (par or imprint == "go"):
        deps = [t["id"] for t in out if t["block"] == "I" and t["kind"] == "eval"]
        deps += [f"g/opd-{l}-s{s}/eval-{l}" for l in sel["M1"] for s in seeds]
        out.append(task("gates/m2b", "B", "gate", deps=deps, name="m2b"))
    if par or (m2b is not None and m2b.get("go")):
        for seed in seeds[:4]:
            for l in sel["B"]:
                out += second_stage(ctx, f"b/opdmask-ten-{l}-s{seed}", "B", "opd_mask", seed=seed, lang=l,
                                    start=f"handoff/s{seed}-u{updates}", eval_langs=[l], condition="T-en")
    out += x_tasks(ctx, a)
    return out


def washout_tasks(ctx, a):
    cfg = ctx.cfg
    sel = a["selection"]
    wseeds = cfg["seeds"]["washout"]
    pilots = wseeds[:cfg["gates"]["washout_pilot_seeds"]]
    langs = sel["M6"]
    E = cfg["eval"]
    out = []
    for seed in wseeds:
        for l in langs:
            out.append(task(f"targets/T-def/F-s{seed}/{l}", "W", "teacher_targets",
                            hours=_gen_hours(cfg, cfg["splits"]["F"] * cfg["teacher_targets"]["samples"], "teacher"),
                            condition="T-def", lang=l, itemset="F", seed=seed))
    gate = ctx.gate("main", "m6")
    go = parallel_blocks(cfg) or bool(gate and gate.get("go"))
    for seed in wseeds:
        if seed not in pilots and not go:
            continue
        for arm, arm_langs in (("W", langs), ("N", ["en"])):
            sft = f"w/sft-{arm}-s{seed}"
            out.append(task(sft, "W", "train", deps=[f"targets/T-def/F-s{seed}/{l}" for l in arm_langs],
                            hours=_hours(cfg, "sft_32x8"), plan="washout_sft", seed=seed, langs=arm_langs))
            for l in langs:
                for which, es in (("a", E["eval_seed"]), ("b", E["second_eval_seed"])):
                    out.append(task(f"{sft}/eval-{which}-{l}", "W", "eval", deps=[sft], hours=_hours(cfg, "eval_E_mgsm"),
                                    adapter=sft, lang=l, eval_seed=es, sets=E["washout_sets"]))
            if go:
                out += second_stage(ctx, f"w/rl-{arm}-s{seed}", "W", "rl", seed=seed, lang="en", start=sft,
                                    eval_langs=langs, eval_sets=E["washout_sets"])
    if gate is None:
        deps = [f"w/sft-{arm}-s{s}/eval-a-{l}" for s in pilots for arm in ("W", "N") for l in langs]
        out.append(task("gates/m6", "W", "gate", deps=deps, name="washout"))
    return out


def x_tasks(ctx, a):
    cfg = ctx.cfg
    langs = a["selection"]["X"]
    seeds = cfg["seeds"]["x"]
    out = stage0_tasks(ctx, "x", languages=langs, seeds=seeds)
    for t in out:
        t["block"] = "X"
    decision = ctx.complete("x", "gates/handoff-languages-" + "-".join(map(str, seeds)))
    if decision is None:
        return out
    updates = cfg["train"]["handoff"]["updates"]
    for seed in seeds:
        for l in decision["languages"]:
            for method in ("opd", "rl"):
                for t in second_stage(ctx, f"g/{method}-{l}-s{seed}", "X", method, seed=seed, lang=l,
                                      start=f"handoff/s{seed}-u{updates}", eval_langs=[l]):
                    t["pair"] = "x"
                    out.append(t)
    return out


def confirm_plan(cfg, a):
    """Seeds and languages of block C (pre-registration amendment 2026-11-28), from the config and the stage-0
    decision file only: C-M4 uses the original M4 selection, C-M1b the original M1 selection, C-M3a-p the included
    languages, C-M6 the original M6 languages. A sub-block switched off in `confirm.blocks` gets no languages."""
    C = cfg.get("confirm") or {}
    on = C.get("blocks") or {}
    sel = a["selection"]
    seeds = [int(s) for s in C.get("seeds") or []]
    return {"seeds": seeds, "grid_seeds": seeds[:int(C.get("grid_seeds", 4))],
            "washout_seeds": [int(s) for s in C.get("washout_seeds") or []],
            "handoff_languages": list(a.get("handoff_languages") or a["included"]),
            "handoff_updates": a["handoff_updates"],
            "M4": list(sel["M4"]) if on.get("M4", True) else [],
            "M1": list(sel["M1"]) if on.get("M1b", True) else [],
            "grid": list(a["included"]) if on.get("M3a-p", True) else [],
            "M6": list(sel["M6"]) if on.get("M6", True) else []}


def confirm_ready(ctx, tasks):
    """Block C is listed only when it is switched on and every gate task of the original blocks (both pairs) is
    finished, and gates b and m6 exist: the confirmation is registered after the original results and must not
    change which original runs exist or how they are labelled."""
    if not (ctx.cfg.get("confirm") or {}).get("enabled"):
        return False
    gates = [t for t in tasks if t["kind"] == "gate"]
    return bool(gates) and all(ctx.done(t) for t in gates) \
        and ctx.gate("main", "b") is not None and ctx.gate("main", "m6") is not None


def confirm_tasks(ctx, a):
    """Block C on new seeds (confirm.seeds, used by no earlier block): teacher targets on F, handoff SFT on the
    original handoff languages and start evaluation per seed; OPD T-def, OPD T-priv (C-M4 languages) and OPD T-en
    (C-M1b languages) on every confirm seed; OPD and RL on every included language for the first grid_seeds seeds
    (C-M3a-p; the OPD T-def runs are shared); washout W/N handoff SFT and English RL on confirm.washout_seeds
    (C-M6a/b, no pilot gate). Task ids start with c/ except the shared per-seed ones (targets/, handoff/, start/),
    whose seeds are new."""
    cfg = ctx.cfg
    plan = confirm_plan(cfg, a)
    E, T = cfg["eval"], cfg["teacher_targets"]
    updates = plan["handoff_updates"]
    hlangs = plan["handoff_languages"]
    target_hours = _gen_hours(cfg, cfg["splits"]["F"] * T["samples"], "teacher")

    def targets(seed, l):
        return task(f"targets/T-def/F-s{seed}/{l}", "C", "teacher_targets", hours=target_hours, condition="T-def",
                    lang=l, itemset="F", seed=seed)
    out = []
    if plan["M4"] or plan["M1"] or plan["grid"]:
        for seed in plan["seeds"]:
            out += [targets(seed, l) for l in hlangs]
            h = f"handoff/s{seed}-u{updates}"
            out.append(task(h, "C", "train", deps=[f"targets/T-def/F-s{seed}/{l}" for l in hlangs],
                            hours=_hours(cfg, "sft_32x8", updates / 32), plan="handoff", seed=seed, langs=hlangs,
                            updates=updates))
            for l in hlangs:
                out.append(task(f"start/s{seed}-u{updates}/{l}", "C", "eval", deps=[h], hours=_hours(cfg, "eval_E"),
                                adapter=h, lang=l, eval_seed=E["eval_seed"], sets=E["sets"]))
            grid = plan["grid"] if seed in plan["grid_seeds"] else []
            opd_langs = [l for l in cfg["languages"] if l in set(plan["M4"]) | set(plan["M1"]) | set(grid)]
            for l in opd_langs:
                out += second_stage(ctx, f"c/opd-{l}-s{seed}", "C", "opd", seed=seed, lang=l, start=h, eval_langs=[l])
            for l in grid:
                out += second_stage(ctx, f"c/rl-{l}-s{seed}", "C", "rl", seed=seed, lang=l, start=h, eval_langs=[l])
            for l in plan["M4"]:
                out += second_stage(ctx, f"c/opd-tpriv-{l}-s{seed}", "C", "opd", seed=seed, lang=l, start=h,
                                    eval_langs=[l], condition="T-priv")
            for l in plan["M1"]:
                out += second_stage(ctx, f"c/opd-ten-{l}-s{seed}", "C", "opd", seed=seed, lang=l, start=h,
                                    eval_langs=[l], condition="T-en")
    langs = plan["M6"]
    if langs:
        for seed in plan["washout_seeds"]:
            out += [targets(seed, l) for l in langs]
            for arm, arm_langs in (("W", langs), ("N", ["en"])):
                sft = f"c/w/sft-{arm}-s{seed}"
                out.append(task(sft, "C", "train", deps=[f"targets/T-def/F-s{seed}/{l}" for l in arm_langs],
                                hours=_hours(cfg, "sft_32x8"), plan="washout_sft", seed=seed, langs=arm_langs))
                for l in langs:
                    for which, es in (("a", E["eval_seed"]), ("b", E["second_eval_seed"])):
                        out.append(task(f"{sft}/eval-{which}-{l}", "C", "eval", deps=[sft],
                                        hours=_hours(cfg, "eval_E_mgsm"), adapter=sft, lang=l, eval_seed=es,
                                        sets=E["washout_sets"]))
                out += second_stage(ctx, f"c/w/rl-{arm}-s{seed}", "C", "rl", seed=seed, lang="en", start=sft,
                                    eval_langs=langs, eval_sets=E["washout_sets"])
    return out


def confirm_hours(c, *, handoff_languages=9, included=9, shared_opd_languages=5):
    """Planning figure of block C (design §8 unit costs) for the H100 stage-0 outcome: 9 handoff and included
    languages, M1 and M4 selections sharing 3 of 4 languages (5 languages with OPD T-def on every seed)."""
    C = c.get("confirm") or {}
    S, Gs, Ws = len(C.get("seeds") or []), int(C.get("grid_seeds", 4)), len(C.get("washout_seeds") or [])
    m4, m1 = c["selection"]["m4_count"], c["selection"]["m1_count"]
    opd = S * shared_opd_languages + Gs * (included - shared_opd_languages) + S * (m4 + m1)
    rl = Gs * included
    return (S * handoff_languages * (_gen_hours(c, c["splits"]["F"] * c["teacher_targets"]["samples"], "teacher")
                                     + _hours(c, "eval_E"))
            + S * _hours(c, "sft_32x8") + opd * _hours(c, "opd_128") + rl * _hours(c, "rl_128")
            + (opd + rl) * _hours(c, "eval_E")
            + Ws * (2 * _hours(c, "sft_32x8") + 2 * _hours(c, "rl_128") + 2 * 4 * 3 * _hours(c, "eval_E_mgsm")))


def build_queue(ctx):
    tasks = stage0_tasks(ctx, "main")
    tasks += stage0_gate_tasks(ctx, tasks)
    a, b = final_stage0(ctx), ctx.gate("main", "b")
    if a is not None and a.get("proceed") and (parallel_blocks(ctx.cfg) or (b is not None and b.get("grid_seed_count"))):
        tasks += grid_tasks(ctx, a, b)
        if confirm_ready(ctx, tasks):
            tasks += confirm_tasks(ctx, a)
    seen, unique = set(), []
    for t in tasks:
        key = (t["pair"], t["id"])
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


PENDING_BLOCKS = {  # design §5.3 run counts, shown by the dry run before the gates exist
    "S0-c+": ("handoff SFT x4, start evaluation and p-measurement x40, Latin mass x9",
              lambda c: 4 * _hours(c, "sft_32x8") + 40 * _hours(c, "eval_E")
              + 40 * _gen_hours(c, c["splits"]["S0"] * c["stage0"]["samples"], "student") + 9 * _hours(c, "latin_mass")),
    "S0-d": ("16 reruns (OPD, RL) + evaluations", lambda c: 8 * (_hours(c, "opd_128") + _hours(c, "rl_128"))
             + 16 * _hours(c, "eval_E")),
    "G": ("80 runs + 80 evals", lambda c: 40 * (_hours(c, "opd_128") + _hours(c, "rl_128")) + 80 * _hours(c, "eval_E")),
    "G-ref": ("40 SFT runs + teacher targets on U", lambda c: 40 * (_hours(c, "sft_128x8") + _hours(c, "eval_E"))
              + 40 * _gen_hours(c, c["splits"]["U"] * 4, "teacher")),
    "G-x": ("8 runs x 4 languages, evaluation only", lambda c: 32 * _hours(c, "eval_E")),
    "I": ("32 runs (OPD T-en, SFT T-en)", lambda c: 16 * (_hours(c, "opd_128") + _hours(c, "sft_128x8")) + 32 * _hours(c, "eval_E")
          + 16 * _gen_hours(c, c["splits"]["U"] * 4, "teacher")),
    "K": ("16 OPD T-priv runs", lambda c: 16 * (_hours(c, "opd_128") + _hours(c, "eval_E"))),
    "A": ("16 mixed runs, 4-language evaluation (conditional)", lambda c: 16 * (_hours(c, "opd_128") + 4 * _hours(c, "eval_E"))),
    "W": ("16 SFT + 16 RL, MGSM only (conditional)", lambda c: 16 * (_hours(c, "sft_32x8") + _hours(c, "rl_128"))
          + 16 * 4 * 3 * _hours(c, "eval_E_mgsm")),
    "X": ("second pair: stage 0 + 16 runs", lambda c: 2. + 16 * (_hours(c, "opd_128") + _hours(c, "eval_E"))),
    "B": ("8 masked-OPD runs (conditional)", lambda c: 8 * (_hours(c, "opd_128") + _hours(c, "eval_E"))),
    "C": ("confirmation (amendment 2026-11-28): 8 new seeds, handoff + start, OPD T-def/T-priv/T-en, "
          "grid OPD/RL on 4 seeds, washout W/N + English RL; listed after every original gate exists",
          lambda c: confirm_hours(c) if (c.get("confirm") or {}).get("enabled") else 0.),
}


def dry_run(ctx):
    tasks = build_queue(ctx)
    lines, total, left = [], 0., 0.
    blocks = {}
    for t in tasks:
        state = "done" if ctx.done(t) else ("ready" if all(ctx.complete(t["pair"], d) for d in t["deps"]) else "waiting")
        lines.append(f"{state:7s} {t['pair']:4s} {t['block']:6s} {t['kind']:15s} {t['hours']:6.2f}h  {t['id']}")
        total += t["hours"]
        left += 0 if state == "done" else t["hours"]
        blocks.setdefault(t["block"], [0, 0.])
        blocks[t["block"]][0] += 1
        blocks[t["block"]][1] += t["hours"]
    pending = []
    ids = {t["id"] for t in tasks}
    for name, (what, hours) in PENDING_BLOCKS.items():
        present = (any(i.startswith("handoff/") for i in ids) if name == "S0-c+" else name in blocks)
        if not present:
            pending.append((name, what, hours(ctx.cfg)))
    return {"tasks": tasks, "lines": lines, "total_hours": total, "remaining_hours": left, "blocks": blocks,
            "pending_blocks": pending}


def next_task(ctx, tasks, leases=None, skip=()):
    for t in tasks:
        if ctx.done(t) or (t["pair"], t["id"]) in skip:
            continue
        if leases is not None and (leases.failed(t) or leases.held(t)):
            continue
        if all(ctx.complete(t["pair"], d) is not None for d in t["deps"]):
            return t
    return None


def run_queue(ctx, *, max_tasks=None, log=print, stop_file=None, leases=None, wait=False, poll=60.,
              max_streak=3):
    """Run ready tasks until none is left. With `leases` (several workers on one runs/ directory) a task is taken
    only under its lease, a failing task is recorded and retried up to the failure limit, and with `wait` the
    worker sleeps while other workers hold tasks that may unlock more work. A worker stops after `max_streak`
    failures in a row (a broken environment would otherwise mark every task failed)."""
    finished, streak, skip = 0, 0, set()
    while max_tasks is None or finished < max_tasks:
        if stop_file and Path(stop_file).exists():
            log(f"stop file {stop_file} found; stopping")
            break
        tasks = build_queue(ctx)
        t = next_task(ctx, tasks, leases, skip)
        lease = None
        if t is not None and leases is not None:
            try:
                lease = leases.acquire(t)
            except OSError as error:
                log(f"lease error on {t['pair']}/{t['id']}: {error}")
            if lease is None:
                skip.add((t["pair"], t["id"]))  # taken by another worker, or not removable: try the next task
                continue
            if ctx.done(t):  # finished by the previous owner just before its lease went away
                lease.release()
                continue
        if t is None:
            waiting = [x for x in tasks if not ctx.done(x)]
            running = leases.live() if leases is not None else []
            if wait and (running or skip):
                log(f"waiting: {len(running)} task(s) running on other workers, {len(waiting)} unfinished")
                skip.clear()
                time.sleep(poll)
                continue
            failed = [x["id"] for x in waiting if leases is not None and leases.failed(x)]
            log(f"queue idle: {len(waiting)} unfinished task(s) wait on gates or missing inputs"
                + (f"; {len(failed)} stopped after repeated failures: {', '.join(failed[:10])}" if failed else ""))
            break
        began = time.monotonic()
        _reset_peak()
        log(f"start {t['pair']}/{t['id']} ({t['kind']}, est {t['hours']:.2f}h)")
        try:
            execute(ctx, t)
        except Exception as error:  # noqa: BLE001 - recorded, then the worker moves on
            if leases is None:
                raise
            count = leases.record_failure(t, error)
            log(f"FAIL  {t['pair']}/{t['id']} ({count}/{leases.max_failures}): {type(error).__name__}: {error}")
            ctx._runtime.clear()
            ctx._gc()
            streak += 1
            if streak >= max_streak:
                log(f"stopping this worker: {streak} failures in a row (details: runs/.failures/)")
                break
            continue
        finally:
            if lease is not None:
                lease.release()
        streak = 0
        skip.clear()
        log(f"done  {t['pair']}/{t['id']} in {(time.monotonic() - began) / 3600:.3f}h")
        with open(ctx.runs / "queue.log", "a") as handle:
            handle.write(json.dumps({"task": t["id"], "pair": t["pair"], "hours": (time.monotonic() - began) / 3600,
                                     "worker": leases.worker if leases is not None else None,
                                     "peak_gpu_gb": _peak_gpu_gb(),
                                     "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")
        finished += 1
    return finished


def _reset_peak():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


def _peak_gpu_gb():
    try:
        import torch
        return round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


def _iso_seconds(text):
    try:
        return time.mktime(time.strptime(text[:19], "%Y-%m-%dT%H:%M:%S"))
    except (TypeError, ValueError):
        return None


def _duration_hours(out):
    """Wall hours of a finished task: task.json is written when it starts, COMPLETE.json when it ends.
    A task resumed after a kill counts from its last start, so this is a rough figure."""
    try:
        return max(0., ((out / "COMPLETE.json").stat().st_mtime - (out / "task.json").stat().st_mtime) / 3600.)
    except OSError:
        return None


def _train_progress(out):
    """(updates done, updates planned) of a running train task, from plan.json and latest.json."""
    try:
        total = len(read_json(out / "plan.json")["steps"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    try:
        done = int(read_json(out / "latest.json")["step"])
    except (OSError, ValueError, KeyError, TypeError):
        done = 0
    return done, total


def status(ctx, leases):
    """Everything `queue status` prints: totals, per-block counts, running tasks with progress, failures,
    what the waiting tasks wait on, and the time left measured from the tasks this cluster has finished."""
    tasks = build_queue(ctx)
    running = {(r.get("pair"), r.get("task")): r for r in leases.live()}
    failed_by_key = {(f.get("pair"), f.get("task")): f for f in leases.failed_list()}
    rows, totals = {}, {k: 0 for k in ("done", "running", "ready", "waiting", "failed")}
    measured = {}  # (block, kind) and kind -> finished wall hours
    waiting_on, states = {}, []
    for t in tasks:
        if ctx.done(t):
            state = "done"
            h = _duration_hours(ctx.dir(t))
            if h is not None:
                measured.setdefault((t["block"], t["kind"]), []).append(h)
                measured.setdefault(t["kind"], []).append(h)
        elif (t["pair"], t["id"]) in running:
            state = "running"
        elif leases.failed(t):
            state = "failed"
        else:
            missing = [d for d in t["deps"] if ctx.complete(t["pair"], d) is None]
            if not missing:
                state = "ready"
            else:
                state = "waiting"
                first = missing[0]
                gate = first if first.startswith("gates/") else None
                key = (f"gate {t['pair']}/{gate}" if gate else
                       f"earlier tasks {t['pair']}/{first.split('/')[0]}/... (e.g. {first})")
                waiting_on[key] = waiting_on.get(key, 0) + 1
        states.append((t, state))
        name = f"{t['pair']}/{t['block']}"
        row = rows.setdefault(name, {k: 0 for k in ("done", "running", "ready", "waiting", "failed")})
        row[state] += 1
        totals[state] += 1

    def estimate(t):
        for key in ((t["block"], t["kind"]), t["kind"]):
            values = measured.get(key)
            if values and len(values) >= 2:
                return sum(values) / len(values), True
        return float(t.get("hours") or 0.), False

    left, measured_share, run_rows = 0., [0, 0], []
    for t, state in states:
        if state == "done":
            continue
        h, real = estimate(t)
        if state == "running":
            prog = _train_progress(ctx.dir(t)) if t["kind"] == "train" else None
            if prog and prog[1]:
                h *= max(0., 1. - prog[0] / prog[1])
            r = running[(t["pair"], t["id"])]
            started = _iso_seconds(r.get("started"))
            run_rows.append({"task": f"{t['pair']}/{t['id']}", "kind": t["kind"], "worker": r.get("worker"),
                             "progress": prog, "for_s": None if started is None else max(0, time.time() - started)})
        if state == "failed":
            continue
        left += h
        measured_share[0 if real else 1] += 1
    workers = max(1, len({r.get("worker") for r in running.values()}))
    failed = []
    for (pair, task), f in failed_by_key.items():
        lines = [x for x in (f.get("error") or "").strip().splitlines() if x.strip()]
        failed.append({"task": f"{pair}/{task}", "count": f.get("count"), "time": f.get("time"),
                       "error": lines[-1].strip() if lines else "", "skipped": f.get("count", 0) >= leases.max_failures})
    for name, row in rows.items():
        row["total"] = sum(row[k] for k in ("done", "running", "ready", "waiting", "failed"))
    return {"blocks": rows, "totals": totals, "tasks": len(tasks), "running": run_rows, "failed": failed,
            "waiting_on": dict(sorted(waiting_on.items(), key=lambda kv: -kv[1])),
            "hours_left_gpu": left, "workers": workers, "hours_left_wall": left / workers,
            "estimate_from_cluster": measured_share[0], "estimate_from_plan": measured_share[1],
            "parallel_blocks": bool(ctx.cfg.get("queue", {}).get("parallel_blocks"))}


def _hm(seconds):
    if seconds is None:
        return "?"
    m = int(seconds // 60)
    return f"{m // 60}:{m % 60:02d}"


def format_status(result, now=None):
    """Text for `queue status` (and run.sh status). One screen: the totals and the next step come first."""
    T, n = result["totals"], result["tasks"]
    out = []
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(now or time.time()))
    pct = 100. * T["done"] / n if n else 0.
    out.append(f"== queue status {stamp} ==")
    out.append(f"done {T['done']}/{n} ({pct:.0f}%)   running {T['running']}   ready {T['ready']}   "
               f"waiting {T['waiting']}   failed {T['failed']}")
    if T["done"] == n and n:
        nxt = "all tasks done -> analysis runs at the end of run.sh (runs/analysis/verdicts.json)"
    elif T["failed"] and not (T["running"] or T["ready"]):
        nxt = "stopped on failures below; after a code fix run.sh retries them by itself"
    elif T["running"]:
        nxt = "running normally; nothing to do"
    elif T["ready"]:
        nxt = f"{T['ready']} task(s) can start but no worker is running -> bash scripts/h100/run.sh"
    else:
        nxt = "nothing can start: every left task waits on the gates/inputs listed below"
    out.append(f"next: {nxt}")
    if T["done"] < n:
        src = result["estimate_from_cluster"], result["estimate_from_plan"]
        basis = ("measured on this cluster" if not src[1] else
                 f"{src[0]} tasks from this cluster's times, {src[1]} from plan figures (no finished task of that kind yet)")
        out.append(f"time left: about {result['hours_left_wall']:.1f} h with {result['workers']} worker(s) "
                   f"({result['hours_left_gpu']:.1f} GPU-h; {basis}; tasks still to be listed after gates not included)")
    out.append("")
    out.append(f"{'block':18s} {'done/all':9s} {'run':>4s} {'ready':>5s} {'wait':>5s} {'fail':>4s}")
    for name, row in result["blocks"].items():
        cell = lambda k: f"{row[k]:d}" if row[k] else "."
        mark = "  ok" if row["done"] == row["total"] else ""
        out.append(f"{name:18s} {str(row['done']) + '/' + str(row['total']):9s} {cell('running'):>4s} {cell('ready'):>5s} "
                   f"{cell('waiting'):>5s} {cell('failed'):>4s}{mark}")
    if result["running"]:
        out.append("")
        out.append(f"running ({len(result['running'])})")
        for r in sorted(result["running"], key=lambda r: r["task"]):
            prog = r["progress"]
            p = f"update {prog[0]}/{prog[1]}" if prog else r["kind"]
            out.append(f"  {r['task']:34s} {p:16s} for {_hm(r['for_s']):>6s}  {r['worker']}")
    if result["failed"]:
        out.append("")
        out.append(f"failed ({len(result['failed'])}) - "
                   "skipped after 2 failures; run.sh retries them by itself when the code changes")
        for f in result["failed"]:
            out.append(f"  {f['task']:34s} x{f['count']}  {f['error'][:160]}")
    if result["waiting_on"]:
        out.append("")
        out.append("waiting on")
        for key, count in list(result["waiting_on"].items())[:8]:
            out.append(f"  {count:4d} task(s)  {key}")
    return "\n".join(out)


# ================================================================ executors
def execute(ctx, t):
    out = ctx.dir(t)
    out.mkdir(parents=True, exist_ok=True)
    if t["kind"] != "gate":
        need = ctx.cfg.get("queue", {}).get("min_free_gb", 30)
        free = shutil.disk_usage(out).free / 2 ** 30
        if free < need:
            raise RuntimeError(f"only {free:.1f} GB free under {out}; need {need} GB (design §8.2)")
    write_json(out / "task.json", t)
    result = {"measure": run_measure, "teacher_targets": run_teacher_targets, "train": run_train, "eval": run_eval,
              "latin_mass": run_latin, "gate": run_gate}[t["kind"]](ctx, t, out)
    write_json(out / "COMPLETE.json", {**result, "task": t["id"], "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    return result


def _items(ctx, p):
    seed = p.get("seed")
    split = ctx.splits(seed if seed is not None else ctx.cfg["seeds"]["grid"][0])
    return split[p["items"]]


def _load_adapter_from(ctx, pair, unit_id, runtime):
    from .train import load_adapter
    complete = ctx.complete(pair, unit_id)
    if complete is None:
        raise RuntimeError(f"{unit_id} is not complete")
    load_adapter(complete["adapter"], runtime, sha256=complete["adapter_sha256"])
    return complete


def run_measure(ctx, t, out):
    from .measure import measure
    p, cfg = t["params"], ctx.cfg
    runtime = ctx.runtime(t["pair"])
    role = p["role"]
    if role == "student":
        _load_adapter_from(ctx, t["pair"], p["adapter"], runtime)
    problems = ctx.dataset.problems(_items(ctx, p), p["lang"])
    summary, _ = measure(runtime, problems, role=role, condition=p["condition"], samples=cfg["stage0"]["samples"],
                         max_new=cfg["stage0"]["max_new_tokens"], label=cfg["stage0"]["label"],
                         identifier=ctx.identifier, out_dir=out, name="measure", batch_seqs=cfg["runtime"]["sample_batch"],
                         keep_ids=p.get("keep_ids", False), meta={"task": t["id"], "lang": p["lang"]})
    return {"summary": summary["summary"], "rows": str(out / "measure.rows.jsonl.gz"),
            "rows_sha256": sha256_file(out / "measure.rows.jsonl.gz")}


def run_teacher_targets(ctx, t, out):
    from .measure import measure
    from .records import response_row, write_rows
    from .targets import verified_target
    p, cfg = t["params"], ctx.cfg
    T = cfg["teacher_targets"]
    split = ctx.splits(p["seed"])
    ids = split[p["itemset"]]
    problems = ctx.dataset.problems(ids, p["lang"])
    runtime = ctx.runtime(t["pair"])
    if T["source"] == "gold_rationale":
        # Smoke/test mode only: fixture rationales stand in for teacher samples. Never used by default.
        rows = []
        for problem in problems:
            text = problem["rationale"]
            ids_ = runtime.tokenizer.encode(text, add_special_tokens=False) + [runtime.eos[0]]
            rows.append(response_row(text, problem, response_ids=ids_, truncated=False, identifier=ctx.identifier,
                                     extra={"sample": 0, "response_ids": ids_}))
        write_rows(out / "measure.rows.jsonl.gz", rows)
    else:
        cache = ctx.runs / t["pair"] / "targets" / p["condition"] / p["lang"] / "cache.rows.jsonl.gz"
        cached = {}
        if cache.exists():
            for r in read_rows(cache):
                cached.setdefault(r["id"], []).append(r)
        missing = [x for x in problems if x["id"] not in cached]
        rows = [r for x in problems if x["id"] in cached for r in cached[x["id"]]]
        if missing:
            _, new = measure(runtime, missing, role="teacher", condition=p["condition"], samples=T["samples"],
                             max_new=T["max_new"], label=T["label"], identifier=ctx.identifier, out_dir=out,
                             name="new", batch_seqs=cfg["runtime"]["sample_batch"], keep_ids=True)
            rows += new
            everything = [r for v in cached.values() for r in v] + new
            write_rows(cache, everything)
        write_rows(out / "measure.rows.jsonl.gz", rows)
    by_item = {}
    for r in rows:
        by_item.setdefault(r["id"], []).append(r)
    targets, statuses = {}, {}
    for problem in problems:
        samples = sorted(by_item.get(problem["id"], []), key=lambda r: r["sample"])
        samples = [{"response_ids": r["response_ids"], "truncated": bool(r["truncated"]), "text": r["text"]} for r in samples]
        target = verified_target(runtime.tokenizer, samples, problem["gold"], runtime.eos)
        targets[problem["id"]] = target
        if target:
            statuses[target["status"]] = statuses.get(target["status"], 0) + 1
    write_json(out / "targets.json", targets)
    from .records import summarize
    return {"verified": sum(v is not None for v in targets.values()), "items": len(problems), "statuses": statuses,
            "source": T["source"], "summary": summarize(rows), "targets_sha256": sha256_file(out / "targets.json")}


def _targets(ctx, pair, condition, itemset, seed, lang):
    path = ctx.runs / pair / "targets" / condition / f"{itemset}-s{seed}" / lang / "targets.json"
    return read_json(path)


def build_plan(ctx, t):
    """Turn a train task into a train.Unit plan (steps, targets, start adapter)."""
    p, cfg = t["params"], ctx.cfg
    tr = cfg["train"]
    seed = p["seed"]
    plan = {"kind": None, "seed": seed, "lr": tr["lr"], "clip": tr["clip"], "checkpoint_every": tr["checkpoint_every"],
            "adapter_seed": stable_seed(seed, "adapter"), "segment_first": tr["opd"]["segment_first"]}
    if p.get("start"):
        start = ctx.complete(t["pair"], p["start"])
        plan["start"] = {"task": p["start"], "adapter": start["adapter"], "sha256": start["adapter_sha256"]}
    name = p["plan"]
    if name in ("handoff", "washout_sft"):
        updates = p.get("updates") or tr["washout_sft"]["updates"]
        batch = tr["handoff"]["batch"] if name == "handoff" else tr["washout_sft"]["batch"]
        per_lang = {}
        for l in p["langs"]:
            targets = _targets(ctx, t["pair"], "T-def", "F", seed, l)
            per_lang[l] = shuffled([x for x, v in targets.items() if v], seed, name, l)
        sequence, counters = [], {l: 0 for l in p["langs"]}
        while len(sequence) < updates * batch:
            for l in p["langs"]:
                if per_lang[l]:
                    sequence.append((per_lang[l][counters[l] % len(per_lang[l])], l))
                    counters[l] += 1
        sequence = sequence[:updates * batch]
        cache = {l: _targets(ctx, t["pair"], "T-def", "F", seed, l) for l in p["langs"]}
        plan["targets"] = [{"item": x, "lang": l, "response_ids": cache[l][x]["response_ids"]} for x, l in sequence]
        plan["steps"] = [{"method": "sft", "targets": list(range(k * batch, (k + 1) * batch))} for k in range(updates)]
        plan["kind"] = "sft"
        plan["composition"] = {l: sum(1 for _, m in sequence if m == l) for l in p["langs"]}
        plan.pop("start", None)
        return plan
    if name == "sft_continue":
        sc = tr["sft_continue"]
        targets = _targets(ctx, t["pair"], p["condition"], "U", seed, p["lang"])
        usable = [x for x, v in targets.items() if v]
        sequence, epoch = [], 0
        while usable and len(sequence) < sc["updates"] * sc["batch"]:
            sequence += shuffled(usable, seed, "sft-continue", p["lang"], epoch)
            epoch += 1
        sequence = sequence[:sc["updates"] * sc["batch"]]
        plan["targets"] = [{"item": x, "lang": p["lang"], "response_ids": targets[x]["response_ids"]} for x in sequence]
        plan["steps"] = [{"method": "sft", "targets": list(range(k * sc["batch"], (k + 1) * sc["batch"]))}
                         for k in range(len(sequence) // sc["batch"])]
        plan["kind"] = "sft"
        plan["skipped_items"] = len(targets) - len(usable)
        return plan
    order = ctx.stage2_order(seed)
    key = {"opd": "opd", "opd_mask": "opd", "rl": "rl", "mixed": "opd"}[name]
    plan["rollout"] = {"samples": tr[key]["samples"], "max_new": tr[key]["max_new"], "tag": p.get("tag", "r0")}
    if name in ("opd", "opd_mask", "rl"):
        updates = tr["rl" if name == "rl" else "opd"]["updates"]
        plan["steps"] = [{"method": name, "item": order[k % len(order)], "lang": p["lang"], "cycle": k // len(order),
                          "condition": p.get("condition", "T-def")} for k in range(updates)]
        if name == "rl":
            for k, s in enumerate(plan["steps"]):
                s["candidates"] = rl_candidates(cfg, order, k)
        plan["kind"] = name
        if name == "opd_mask":
            plan["mask"] = dict(tr["opd_mask"])
        return plan
    if name == "mixed":
        groups, arm = p["groups"], p["arm"]
        methods = {"A": "opd", "B": "rl"} if arm == "matched" else {"A": "rl", "B": "opd"}
        steps = []
        for k in range(tr["allocation"]["updates"]):
            group = "A" if k % 2 == 0 else "B"
            j = k // 2
            langs = groups[group]
            steps.append({"method": methods[group], "group": group, "lang": langs[j % len(langs)],
                          "item": order[(j // len(langs)) % len(order)], "cycle": 0, "condition": "T-def"})
            if methods[group] == "rl":
                steps[-1]["candidates"] = rl_candidates(cfg, order, j // len(langs))
        plan["steps"], plan["kind"], plan["arm"] = steps, "mixed", arm
        return plan
    raise ValueError(name)


def rl_candidates(cfg, order, k):
    """Dynamic sampling (pre-registration change 2026-11-27): the RL update at position k tries order[k] first and,
    while all samples of a problem get the same reward (zero advantage, no gradient), the next problems
    order[k + j * stride], up to max_tries problems. The first candidate is the item of the original schedule, so
    the first rollout of an RL arm and an OPD arm with the same seed are still identical."""
    d = cfg["train"]["rl"].get("dynamic_sampling") or {}
    tries, stride, n = int(d.get("max_tries", 1)), int(d.get("stride", 1)), len(order)
    return [order[(k + j * stride) % n] for j in range(max(1, min(tries, n)))]


def run_train(ctx, t, out):
    from .train import Unit
    plan = build_plan(ctx, t)
    if not plan["steps"]:
        raise RuntimeError(f"{t['id']}: no steps (no verified targets?)")
    runtime = ctx.runtime(t["pair"])
    summary = Unit(runtime, ctx.dataset, plan, out, identifier=ctx.identifier).run()
    return summary


def run_eval(ctx, t, out):
    from .measure import evaluate
    p, cfg = t["params"], ctx.cfg
    runtime = ctx.runtime(t["pair"])
    role = "initial"
    meta = {"task": t["id"], "lang": p["lang"], "adapter_task": p.get("adapter")}
    if p.get("adapter"):
        complete = _load_adapter_from(ctx, t["pair"], p["adapter"], runtime)
        role = "student"
        meta["policy_hash"] = complete["final_policy_hash"]
    problems = ctx.dataset.eval_problems(p["lang"], sets=tuple(p["sets"]), limit=cfg["eval"].get("limit_per_set"))
    summary = evaluate(runtime, problems, role=role, eval_seed=p["eval_seed"], samples=cfg["eval"]["samples"],
                       max_new=cfg["eval"]["max_new_tokens"], identifier=ctx.identifier, out_dir=out, name="eval",
                       batch_seqs=cfg["runtime"]["sample_batch"], meta=meta)
    result = {"acc": summary["combined"]["acc"], "combined": summary["combined"], "per_set": summary["per_set"],
              "rows": str(out / "eval.rows.jsonl.gz"), "rows_sha256": sha256_file(out / "eval.rows.jsonl.gz")}
    result.update(_collapse(ctx, t, result["acc"]))
    return result


def _collapse(ctx, t, acc):
    """§5.5 collapse flag: more than 6 points below the start (the handoff checkpoint's evaluation in the same
    language, when that evaluation exists)."""
    from .measure import collapse_flag
    p = t["params"]
    adapter = p.get("adapter")
    plan_path = ctx.runs / t["pair"] / adapter / "plan.json" if adapter else None
    if not plan_path or not plan_path.exists():
        return {}
    start = read_json(plan_path).get("start", {}).get("task", "")
    if not start.startswith("handoff/"):
        return {}
    start_eval = ctx.complete(t["pair"], "start/" + start.split("/", 1)[1] + "/" + p["lang"])
    if start_eval is None:
        return {}
    return {"start_acc": start_eval["acc"],
            "collapse": collapse_flag(acc, start_eval["acc"], ctx.cfg["gates"]["collapse_pp"])}


def run_latin(ctx, t, out):
    p, cfg = t["params"], ctx.cfg
    runtime = ctx.runtime(t["pair"])
    measured = ctx.complete(t["pair"], p["measure"])
    rows = [r for r in read_rows(measured["rows"]) if r["sample"] == 0][:cfg["stage0"]["latin_items"]]
    masses = {"T-def": [], "T-en": []}
    with runtime.teacher_ready():
        _latin_loop(ctx, p, cfg, runtime, rows, masses)
    result = {c: (sum(v) / len(v) if v else None) for c, v in masses.items()}
    result["responses"] = len(masses["T-def"])
    result["source"] = measured["rows"]
    write_json(out / "latin.json", result)
    return result


def _latin_loop(ctx, p, cfg, runtime, rows, masses):
    for r in rows:
        if not r.get("response_ids"):
            continue
        problem = ctx.dataset.problem(r["id"], p["lang"])
        for c in masses:
            _, latin = runtime.teacher_logprobs(runtime.teacher_prompt(problem, c), r["response_ids"],
                                                latin_positions=cfg["stage0"]["latin_positions"])
            if latin:
                masses[c].append(sum(latin) / len(latin))


# ------------------------------------------------------------ gate executors
def _files_hash(paths):
    return {str(p): sha256_file(p) for p in paths}


def run_gate(ctx, t, out):
    name = t["params"]["name"]
    result = {"handoff_languages": gate_handoff_languages, "stage0_a": gate_stage0_a, "stage0_b": gate_stage0_b,
              "allocation": gate_allocation, "washout": gate_washout, "m2b": gate_m2b,
              "m3_pilot": gate_m3_pilot}[name](ctx, t)
    result["gates_source_sha256"] = sha256_file(GATES_SOURCE)
    return result


def gate_handoff_languages(ctx, t):
    p = t["params"]
    verified, files = {}, []
    for seed in p["seeds"]:
        verified[seed] = {}
        for l in p["langs"]:
            c = ctx.complete(t["pair"], f"targets/T-def/F-s{seed}/{l}")
            verified[seed][l] = c["verified"]
            files.append(ctx.runs / t["pair"] / f"targets/T-def/F-s{seed}/{l}" / "targets.json")
    out = G.handoff_languages(verified, ctx.cfg["gates"]["verified_min"])
    out["verified"] = verified
    out["input_files"] = _files_hash(files)
    return out


def _summary(ctx, pair, id):
    return ctx.complete(pair, id)["summary"]


def gate_stage0_a(ctx, t):
    cfg = ctx.cfg
    updates = t["params"]["updates"]
    seeds = cfg["seeds"]["grid"]
    handoff = ctx.complete("main", "gates/handoff-languages-" + "-".join(map(str, seeds)))
    langs = handoff["languages"]
    files = []

    def take(id):
        files.append(ctx.runs / "main" / id / "COMPLETE.json")
        return _summary(ctx, "main", id)
    q, r_t = {}, {}
    for c in CONDITIONS:
        q[c], r_t[c] = {}, {}
        for l in cfg["languages"]:
            source = f"s0a/teacher-{c}/{l}" if not (c == "T-priv" and l == "en") else f"s0a/teacher-T-def/{l}"
            s = take(source)
            q[c][l], r_t[c][l] = s["lenient_norm"], s["r_mean"]
    p_strict, p_len, r_s = {}, {}, {}
    for seed in seeds:
        p_strict[seed], p_len[seed], r_s[seed] = {}, {}, {}
        for l in langs:
            s = take(f"p/s{seed}-u{updates}/{l}")
            p_strict[seed][l], p_len[seed][l], r_s[seed][l] = s["strict"], s["lenient_norm"], s["r_mean"]
    latin = {}
    for l in langs:
        if l != "en":
            latin[l] = ctx.complete("main", f"latin/s{seeds[0]}-u{updates}/{l}")
            files.append(ctx.runs / "main" / f"latin/s{seeds[0]}-u{updates}/{l}" / "COMPLETE.json")
    scorer_rows = read_rows(ctx.complete("main", "s0/scorer-en")["rows"])
    files.append(Path(ctx.complete("main", "s0/scorer-en")["rows"]))
    scorer = G.scorer_gate(scorer_rows, cfg["gates"]["scorer_agreement"])
    s0b = {c: {l: take(f"s0b/{c}/{l}") for l in cfg["languages"]} for c in cfg["stage0"]["s0b_conditions"]}
    student0 = {l: take(f"s0a/student/{l}") for l in cfg["languages"]}
    out = G.stage0_a(cfg, handoff=handoff, q=q, r_teacher=r_t, p_strict=p_strict, p_lenient=p_len, r_student=r_s,
                     latin=latin, scorer=scorer, handoff_updates=updates)
    out["english_solving_advantage"] = {l: s0b["solve_en"][l]["lenient_norm"] - s0b["solve_native"][l]["lenient_norm"]
                                        for l in cfg["languages"]}
    out["s0b"] = {c: {l: {k: v[k] for k in ("strict", "lenient_norm", "language_match", "r_mean")} for l, v in d.items()}
                  for c, d in s0b.items()}
    out["initial_student"] = {l: {k: v[k] for k in ("strict", "lenient_norm", "lenient_ascii", "lenient_legacy",
                                                    "r_mean", "language_match")} for l, v in student0.items()}
    out["input_files"] = _files_hash(files)
    return out


def gate_stage0_b(ctx, t):
    cfg = ctx.cfg
    a = final_stage0(ctx)
    seed0 = cfg["seeds"]["grid"][0]
    updates = a["handoff_updates"]
    s0d, start, files = {"opd": {}, "rl": {}}, {}, []
    for l in a["selection"]["S0d"]:
        start[l] = ctx.complete("main", f"start/s{seed0}-u{updates}/{l}")["acc"]
        files.append(ctx.runs / "main" / f"start/s{seed0}-u{updates}/{l}" / "COMPLETE.json")
        for method in ("opd", "rl"):
            s0d[method][l] = []
            for tag in cfg["stage0"]["s0d_tags"]:
                id = f"s0d/{method}-{l}-{tag}/eval-{l}"
                s0d[method][l].append(ctx.complete("main", id)["acc"])
                files.append(ctx.runs / "main" / id / "COMPLETE.json")
    out = G.stage0_b(cfg, s0d=s0d, start=start)
    out["input_files"] = _files_hash(files)
    return out


def gate_allocation(ctx, t):
    from .analysis import collect_grid, m3_verdicts
    a = final_stage0(ctx)
    grid = collect_grid(ctx, a)
    b = ctx.gate("main", "b")
    m3 = m3_verdicts(ctx.cfg, grid, a, b)
    groups = a["selection"]["M5"]
    d = grid["D_mean"]
    d_mean = {k: (sum(d[l] for l in groups[k]) / len(groups[k]) if all(l in d for l in groups[k]) else None)
              for k in ("A", "B")}
    supported = any(m3[h]["status"] == "supported" for h in ("M3a", "M3a-p"))
    out = G.allocation_gate(ctx.cfg, m3a_supported=supported, d_mean=d_mean)
    out["d_mean_pp"] = d_mean
    if b is not None and not b.get("grid_seed_count"):
        out["go"] = False
        out["gate_b"] = b.get("decision")
    out["M3a"], out["M3a-p"] = m3["M3a"], m3["M3a-p"]
    return out


def gate_washout(ctx, t):
    a = final_stage0(ctx)
    cfg = ctx.cfg
    langs = [l for l in a["selection"]["M6"] if l != "en"]
    diffs = []
    for seed in cfg["seeds"]["washout"][:cfg["gates"]["washout_pilot_seeds"]]:
        # eval "a": the verdict's SFT-end difference uses eval "b", so selecting on the gate does not select on
        # the verdict's own noise.
        w = [ctx.complete("main", f"w/sft-W-s{seed}/eval-a-{l}")["acc"] for l in langs]
        n = [ctx.complete("main", f"w/sft-N-s{seed}/eval-a-{l}")["acc"] for l in langs]
        diffs.append(100 * (sum(w) / len(w) - sum(n) / len(n)))
    return G.washout_gate(cfg, pilot_diffs_pp=diffs)


def gate_m3_pilot(ctx, t):
    a = final_stage0(ctx)
    plan = m3_pilot_plan(ctx, a)
    d, files = {}, []
    for key in ("high_p", "low_p"):
        l, values = plan[key], []
        for s in plan["seeds"]:
            ids = [f"g/{m}-{l}-s{s}/eval-{l}" for m in ("opd", "rl")]
            values.append(100 * (ctx.complete("main", ids[0])["acc"] - ctx.complete("main", ids[1])["acc"]))
            files += [ctx.runs / "main" / i / "COMPLETE.json" for i in ids]
        d[key] = sum(values) / len(values)
    out = G.m3_pilot_gate(ctx.cfg, d_mean_pp=d)
    out.update(plan=plan, input_files=_files_hash(files))
    return out


def gate_m2b(ctx, t):
    from .analysis import verdicts
    v = verdicts(ctx)
    return {"M1b": v["M1b"], "go": v["M1b"]["status"] == "supported"}
