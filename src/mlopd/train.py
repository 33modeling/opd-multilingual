"""SFT, OPD and RL updates with resumable units and the per-run records of design §5.5.

One unit = one training run: a start adapter (or the fresh adapter of the initial student), a fresh AdamW
(lr 1e-4, no weight decay), gradient clipping at 1, and a fixed list of steps.

  sft       one step = 8 target sequences; loss = summed token cross-entropy / tokens in the batch (English record).
  opd       one problem, 4 student samples (temperature 1, rollout cap 384 tokens); loss per sample = mean over
            response positions of the full-vocabulary forward KL(teacher || student); the update averages the 4.
            The teacher scores the student's response under its own prompt (teacher condition T-def / T-en / T-priv).
  opd_mask  design §3.3 M2b: positions in the first k tokens and in the answer line (and its EOS) use the student's
            cross-entropy on its own sampled token instead of the KL; the rest use the KL.
  rl        GRPO: the same 4 samples per problem, reward = strict with normalization (Appendix B), advantages
            (r - mean) / (std + 1e-4) with the unbiased std (English kdrl/stage_data rule), loss per sample =
            -A * mean token log-prob; the update averages the 4.
  mixed     a step list whose steps name their own method (block A: OPD on one language group, RL on the other).

Rollout seeds are stable_seed(seed, tag, item, lang, cycle, j), so two arms with the same seed, items, order and
rollout tag draw from the same seeds (the first update of two such arms sees identical samples).
"""
from __future__ import annotations

from pathlib import Path
import time

import torch

from . import lora
from .scoring import has_marker_line, strict_correct
from .targets import segments
from .util import read_json, read_jsonl, sha256_file, sha256_json, stable_seed, write_json, write_jsonl

SEGMENTS = ("first", "body", "answer", "eos")


def grpo_advantages(rewards):
    values = torch.as_tensor(rewards, dtype=torch.float32)
    return (values - values.mean()) / (values.std() + 1e-4)


def rollout_seeds(seed, tag, item, language, cycle, samples):
    return [stable_seed(seed, tag, item, language, cycle, j, "rollout") for j in range(samples)]


def save_adapter(path, runtime, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"adapter": lora.adapter_state(runtime.student), "policy_hash": runtime.policy_hash(),
                **(extra or {})}, tmp)
    tmp.replace(path)
    return sha256_file(path)


def load_adapter(path, runtime, *, sha256=None):
    if sha256 is not None and sha256_file(path) != sha256:
        raise ValueError(f"{path}: sha256 differs from the recorded value")
    state = torch.load(path, map_location="cpu", weights_only=False)
    lora.load_adapter_state(runtime.student, state["adapter"])
    if runtime.policy_hash() != state["policy_hash"]:
        raise ValueError(f"{path}: policy hash mismatch after loading")
    return state


def start_adapter(runtime, plan):
    if plan.get("start"):
        load_adapter(plan["start"]["adapter"], runtime, sha256=plan["start"].get("sha256"))
    else:
        runtime.reset_adapter(plan.get("adapter_seed"))


class Unit:
    def __init__(self, runtime, dataset, plan, out_dir, *, identifier=None):
        self.runtime, self.dataset, self.plan = runtime, dataset, plan
        self.out = Path(out_dir)
        self.identifier = identifier
        self.plan_hash = sha256_json(plan)
        self.eos = runtime.eos

    # ---------------------------------------------------------------- driver
    def run(self):
        complete = self.out / "COMPLETE.json"
        if complete.exists():
            return read_json(complete)
        self.out.mkdir(parents=True, exist_ok=True)
        write_json(self.out / "plan.json", self.plan)
        runtime = self.runtime
        start_adapter(runtime, self.plan)
        start_hash = runtime.policy_hash()
        optimizer = runtime.optimizer(self.plan["lr"])
        rows, step = [], 0
        latest = self.out / "latest.pt"
        meta = read_json(self.out / "latest.json") if (self.out / "latest.json").exists() else None
        if latest.exists() and (meta is None or sha256_file(latest) != meta["sha256"]):
            # Killed between writing latest.pt and latest.json: the pair does not match. Rollout seeds are fixed
            # per step, so starting the unit again gives the same run.
            for name in ("latest.pt", "latest.json", "updates.jsonl"):
                (self.out / name).unlink(missing_ok=True)
            meta = None
        if latest.exists():
            if meta["plan_hash"] != self.plan_hash:
                raise ValueError(f"{latest}: plan hash differs; refusing to resume")
            state = torch.load(latest, map_location="cpu", weights_only=False)
            lora.load_adapter_state(runtime.student, state["adapter"])
            optimizer.load_state_dict(state["optimizer"])
            step = state["step"]
            if runtime.policy_hash() != meta["policy_hash"]:
                raise ValueError("policy hash mismatch on resume")
            rows = read_jsonl(self.out / "updates.jsonl")[:step]
        steps = self.plan["steps"]
        every = int(self.plan.get("checkpoint_every", 16))
        while step < len(steps):
            row = self.step(optimizer, step, steps[step])
            row["step"] = step
            rows.append(row)
            step += 1
            if step % every == 0 or step == len(steps):
                write_jsonl(self.out / "updates.jsonl", rows)
                self._checkpoint(optimizer, step)
        sha = save_adapter(self.out / "adapter.pt", runtime, {"plan_hash": self.plan_hash})
        summary = {"kind": self.plan["kind"], "updates": len(steps), "plan_hash": self.plan_hash,
                   "start_policy_hash": start_hash, "final_policy_hash": runtime.policy_hash(),
                   "adapter": str(self.out / "adapter.pt"), "adapter_sha256": sha,
                   "records": summarize_updates(rows), "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        write_json(complete, summary)
        for name in ("latest.pt", "latest.json"):
            (self.out / name).unlink(missing_ok=True)
        return summary

    def _checkpoint(self, optimizer, step):
        path = self.out / "latest.pt"
        tmp = path.with_name("latest.pt.tmp")
        torch.save({"adapter": lora.adapter_state(self.runtime.student), "optimizer": optimizer.state_dict(),
                    "step": step}, tmp)
        tmp.replace(path)
        write_json(self.out / "latest.json", {"step": step, "sha256": sha256_file(path), "plan_hash": self.plan_hash,
                                              "policy_hash": self.runtime.policy_hash()})

    def step(self, optimizer, index, spec):
        method = spec.get("method", self.plan["kind"])
        began = time.monotonic()
        if method == "sft":
            row = self.sft_step(optimizer, spec)
        elif method in ("opd", "opd_mask"):
            row = self.opd_step(optimizer, spec, masked=method == "opd_mask")
        elif method == "rl":
            row = self.rl_step(optimizer, spec)
        else:
            raise ValueError(f"unknown method {method}")
        row["method"], row["seconds"] = method, time.monotonic() - began
        return row

    def _finish(self, optimizer):
        params = list(self.runtime.parameters().values())
        norm = torch.nn.utils.clip_grad_norm_(params, float(self.plan.get("clip", 1.)))
        if not torch.isfinite(norm):
            raise RuntimeError("nonfinite gradient")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.runtime.student.eval()
        return float(norm)

    # ---------------------------------------------------------------- SFT
    def sft_step(self, optimizer, spec):
        runtime = self.runtime
        targets = [self.plan["targets"][i] for i in spec["targets"]]
        tokens = sum(len(t["response_ids"]) for t in targets)
        optimizer.zero_grad(set_to_none=True)
        runtime.student.train()
        total = 0.
        for t in targets:
            prompt = runtime.student_prompt(self.dataset.problem(t["item"], t["lang"]))
            logp, _ = runtime.student_logprob_and_kl(prompt, t["response_ids"])
            loss = -logp.sum() / tokens
            loss.backward()
            total += float(loss.detach())
        norm = self._finish(optimizer)
        return {"loss": total, "tokens": tokens, "sequences": len(targets), "gradient_norm": norm,
                "items": [[t["item"], t["lang"]] for t in targets]}

    # ---------------------------------------------------------------- rollouts
    def rollouts(self, spec):
        runtime = self.runtime
        rollout = self.plan["rollout"]
        problem = self.dataset.problem(spec["item"], spec["lang"])
        prompt = runtime.student_prompt(problem)
        seeds = rollout_seeds(self.plan["seed"], rollout.get("tag", "r0"), spec["item"], spec["lang"],
                              spec.get("cycle", 0), rollout["samples"])
        samples = runtime.sample(runtime.student, [prompt] * len(seeds), seeds, max_new=rollout["max_new"])
        for s in samples:
            s["text"] = runtime.decode(s["response_ids"])
        return problem, prompt, samples

    @staticmethod
    def _common(samples):
        return {"response_tokens": [len(s["response_ids"]) for s in samples],
                "truncated": [int(s["truncated"]) for s in samples],
                "marker_line": [int(has_marker_line(s["text"])) for s in samples]}

    # ---------------------------------------------------------------- OPD
    def opd_step(self, optimizer, spec, *, masked=False):
        runtime = self.runtime
        problem, prompt, samples = self.rollouts(spec)
        condition = spec.get("condition", "T-def")
        teacher_prompt = runtime.teacher_prompt(problem, condition)
        first = int(self.plan.get("segment_first", 16))
        with runtime.teacher_ready():  # one device move for the 4 teacher passes
            teacher = [runtime.teacher_logprobs(teacher_prompt, s["response_ids"], latin_positions=first)
                       if s["response_ids"] else (None, []) for s in samples]
        optimizer.zero_grad(set_to_none=True)
        runtime.student.train()
        kl_seg = {k: 0. for k in SEGMENTS}
        tok_seg = {k: 0 for k in SEGMENTS}
        losses, kl_means, latin = [], [], []
        mask_cfg = self.plan.get("mask", {})
        for s, (logq, latin_mass) in zip(samples, teacher):
            if not s["response_ids"]:
                continue
            labels = segments(runtime.tokenizer, s["response_ids"], self.eos, first=first)
            logp, kl = runtime.student_logprob_and_kl(prompt, s["response_ids"], logq)
            if masked:
                k = int(mask_cfg.get("first", 16))
                ce_positions = torch.tensor([i < k or (mask_cfg.get("answer_line", True) and lab in ("answer", "eos"))
                                             for i, lab in enumerate(labels)], device=kl.device)
                per_position = torch.where(ce_positions, -logp, kl)
            else:
                per_position = kl
            loss = per_position.mean() / len(samples)
            loss.backward()
            detached = kl.detach().cpu().tolist()
            for value, lab in zip(detached, labels):
                kl_seg[lab] += value
                tok_seg[lab] += 1
            losses.append(float(per_position.detach().mean()))
            kl_means.append(sum(detached) / len(detached))
            latin.append(sum(latin_mass) / len(latin_mass) if latin_mass else None)
        norm = self._finish(optimizer) if losses else None
        rewards = [strict_correct(s["text"], problem["gold"]) for s in samples]
        return {"item": spec["item"], "lang": spec["lang"], "cycle": spec.get("cycle", 0), "condition": condition,
                "loss": sum(losses) / len(losses) if losses else None,
                "kl_mean": sum(kl_means) / len(kl_means) if kl_means else None,
                "kl_by_segment": kl_seg, "tokens_by_segment": tok_seg,
                "teacher_latin_mass_first": [x for x in latin if x is not None],
                "strict_rewards": rewards, "gradient_norm": norm, **self._common(samples)}

    # ---------------------------------------------------------------- RL
    def rl_step(self, optimizer, spec):
        runtime = self.runtime
        problem, prompt, samples = self.rollouts(spec)
        rewards = [strict_correct(s["text"], problem["gold"]) for s in samples]
        advantages = grpo_advantages(rewards)
        optimizer.zero_grad(set_to_none=True)
        runtime.student.train()
        losses = []
        zero = len(set(rewards)) == 1
        if not zero:
            for s, a in zip(samples, advantages.tolist()):
                if not s["response_ids"]:
                    continue
                logp, _ = runtime.student_logprob_and_kl(prompt, s["response_ids"])
                loss = -a * logp.mean() / len(samples)
                loss.backward()
                losses.append(float(loss.detach()) * len(samples))
            norm = self._finish(optimizer)
        else:
            # All advantages are exactly 0, so the policy-gradient loss has a zero gradient. As in the English runs
            # (which backpropagated the zero loss), AdamW still steps with zero gradients: its moments decay and
            # earlier momentum still moves the weights. Zero gradients are set directly instead of a forward pass.
            for parameter in runtime.parameters().values():
                parameter.grad = torch.zeros_like(parameter)
            norm = self._finish(optimizer)
        group = "all_correct" if all(rewards) else ("all_wrong" if not any(rewards) else "mixed")
        return {"item": spec["item"], "lang": spec["lang"], "cycle": spec.get("cycle", 0), "rewards": rewards,
                "reward_mean": sum(rewards) / len(rewards), "advantages": advantages.tolist(), "group": group,
                "zero_advantage": int(zero), "loss": sum(losses) / len(losses) if losses else 0.,
                "gradient_norm": norm, **self._common(samples)}


def summarize_updates(rows):
    """Run-level records of §5.5."""
    out = {"updates": len(rows)}
    tokens = [t for r in rows for t in r.get("response_tokens", [])]
    if tokens:
        truncated = [t for r in rows for t in r["truncated"]]
        marker = [t for r in rows for t in r["marker_line"]]
        out.update(mean_rollout_tokens=sum(tokens) / len(tokens), rollout_truncation_rate=sum(truncated) / len(truncated),
                   rollout_marker_line_rate=sum(marker) / len(marker), rollout_responses=len(tokens))
    rl = [r for r in rows if r["method"] == "rl"]
    if rl:
        groups = [r["group"] for r in rl]
        out["rl"] = {"updates": len(rl), "group_shares": {g: groups.count(g) / len(groups)
                                                          for g in ("all_correct", "all_wrong", "mixed")},
                     "reward_mean": sum(r["reward_mean"] for r in rl) / len(rl),
                     "zero_advantage_updates": sum(r["zero_advantage"] for r in rl),
                     "per_update_group": groups}
    opd = [r for r in rows if r["method"] in ("opd", "opd_mask")]
    if opd:
        kl = {k: sum(r["kl_by_segment"][k] for r in opd) for k in SEGMENTS}
        tok = {k: sum(r["tokens_by_segment"][k] for r in opd) for k in SEGMENTS}
        kl_total, tok_total = sum(kl.values()), sum(tok.values())
        latin = [x for r in opd for x in r["teacher_latin_mass_first"]]
        out["opd"] = {"updates": len(opd), "loss_mean": _mean([r["loss"] for r in opd]),
                      "kl_mean": _mean([r["kl_mean"] for r in opd]),
                      "kl_share": {k: (kl[k] / kl_total if kl_total else None) for k in SEGMENTS},
                      "token_share": {k: (tok[k] / tok_total if tok_total else None) for k in SEGMENTS},
                      "kl_to_token_share_ratio": {k: ((kl[k] / kl_total) / (tok[k] / tok_total)
                                                      if kl_total and tok[k] else None) for k in SEGMENTS},
                      "teacher_latin_mass_first_mean": _mean(latin),
                      "loss_by_update": [r["loss"] for r in opd]}
    sft = [r for r in rows if r["method"] == "sft"]
    if sft:
        out["sft"] = {"updates": len(sft), "loss_first": sft[0]["loss"], "loss_last": sft[-1]["loss"],
                      "tokens": sum(r["tokens"] for r in sft), "sequences": sum(r["sequences"] for r in sft)}
    return out


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None
