"""Generation + scoring of a model on a set of problems: evaluation on E_l (§5.1) and stage-0 measurements.

Evaluation seeds: stable_seed(eval_seed, "eval", lang, set, id, j) for sample j, the same for every checkpoint, so two
checkpoints are compared on the same items with the same sampler seeds.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import time

from .records import response_row, summarize, summarize_eval, write_rows
from .util import stable_seed, write_json


@contextmanager
def network(runtime, role):
    """role: "student" (current adapter), "initial" (adapters off) or "teacher"."""
    if role == "teacher":
        with runtime.teacher_ready() as teacher:
            yield teacher
    elif role == "initial":
        with runtime.initial_student() as student:
            yield student
    elif role == "student":
        yield runtime.student
    else:
        raise ValueError(role)


def prompt_for(runtime, problem, role, condition):
    """condition: a teacher condition (T-def/T-en/T-priv) for the teacher, an S0-b line name or None for the student."""
    if role == "teacher":
        return runtime.teacher_prompt(problem, condition or "T-def")
    if condition and condition != "T-def":
        from .prompts import s0b_messages
        return runtime.render(s0b_messages(problem["question"], condition, problem["lang"]))
    return runtime.student_prompt(problem)


def generate(runtime, problems, *, role, condition=None, samples, max_new, seed_fn, identifier, batch_seqs=32,
             keep_ids=False, greedy=False):
    prompts, seeds, owners = [], [], []
    for problem in problems:
        prompt = prompt_for(runtime, problem, role, condition)
        for j in range(samples):
            prompts.append(prompt)
            seeds.append(seed_fn(problem, j))
            owners.append((problem, j))
    started = time.monotonic()
    with network(runtime, role) as model:
        outputs = runtime.sample(model, prompts, seeds, max_new=max_new, batch_seqs=batch_seqs, greedy=greedy)
    rows = []
    for (problem, j), out in zip(owners, outputs):
        text = runtime.decode(out["response_ids"])
        extra = {"sample": j}
        if keep_ids:
            extra["response_ids"] = out["response_ids"]
        rows.append(response_row(text, problem, response_ids=out["response_ids"], truncated=out["truncated"],
                                 identifier=identifier, extra=extra))
    return rows, time.monotonic() - started


def evaluate(runtime, problems, *, role, eval_seed, samples, max_new, identifier, out_dir, name, batch_seqs=32,
             meta=None):
    rows, seconds = generate(runtime, problems, role=role, samples=samples, max_new=max_new, identifier=identifier,
                             seed_fn=lambda p, j: stable_seed(eval_seed, "eval", p["lang"], p.get("set"), p["id"], j),
                             batch_seqs=batch_seqs)
    out_dir = Path(out_dir)
    write_rows(out_dir / f"{name}.rows.jsonl.gz", rows)
    summary = {**(meta or {}), "eval_seed": eval_seed, "samples": samples, "max_new_tokens": max_new,
               "seconds": seconds, **summarize_eval(rows)}
    write_json(out_dir / f"{name}.json", summary)
    return summary


def measure(runtime, problems, *, role, condition, samples, max_new, label, identifier, out_dir, name,
            batch_seqs=32, keep_ids=False, meta=None):
    """Stage-0 style measurement: seeds depend on (label, role, condition, lang, id, j) only."""
    rows, seconds = generate(runtime, problems, role=role, condition=condition, samples=samples, max_new=max_new,
                             identifier=identifier, batch_seqs=batch_seqs, keep_ids=keep_ids,
                             seed_fn=lambda p, j: stable_seed(label, role, condition, p["lang"], p["id"], j))
    out_dir = Path(out_dir)
    write_rows(out_dir / f"{name}.rows.jsonl.gz", rows)
    summary = {**(meta or {}), "role": role, "condition": condition, "samples": samples, "max_new_tokens": max_new,
               "seconds": seconds, "summary": summarize(rows)}
    write_json(out_dir / f"{name}.json", summary)
    return summary, rows


def collapse_flag(acc_after, acc_start, threshold_pp=6.):
    """H15 rule: collapse = more than 6 points below the start."""
    if acc_after is None or acc_start is None:
        return None
    return int(100 * (acc_start - acc_after) > threshold_pp)
