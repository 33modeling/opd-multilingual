"""Runtime (sampling, chunked KL), targets and the three trainers on tiny random models."""

import pytest
import torch

from conftest import make_dataset, smoke_overrides


@pytest.fixture(scope="module")
def setup(tiny, stub_identifier, tmp_path_factory):
    from mlopd.data import Dataset
    from mlopd.experiment import load_config
    from mlopd.runtime import Runtime
    base = tmp_path_factory.mktemp("rt")
    make_dataset(base / "p.json")
    cfg = load_config(overrides=smoke_overrides(tiny, base / "p.json"))
    runtime = Runtime(cfg["model_pairs"]["main"], cfg["runtime"], adapter_seed=11)
    return {"cfg": cfg, "runtime": runtime, "dataset": Dataset.load(base / "p.json"), "base": base}


def test_sampling_is_per_sequence_deterministic(setup):
    rt = setup["runtime"]
    problem = setup["dataset"].problem("train:0", "en")
    other = setup["dataset"].problem("train:1", "zh")
    p, q = rt.student_prompt(problem), rt.student_prompt(other)
    alone = rt.sample(rt.student, [p], [123], max_new=6)
    mixed = rt.sample(rt.student, [q, p, q], [5, 123, 6], max_new=6, batch_seqs=3)
    assert alone[0]["response_ids"] == mixed[1]["response_ids"]
    assert rt.sample(rt.student, [p], [124], max_new=6)[0]["response_ids"] != [] 


def test_chunked_kl_matches_full_computation(setup):
    rt = setup["runtime"]
    problem = setup["dataset"].problem("train:2", "ru")
    prompt = rt.student_prompt(problem)
    response = rt.sample(rt.student, [prompt], [7], max_new=9)[0]["response_ids"]
    logq, latin = rt.teacher_logprobs(rt.teacher_prompt(problem, "T-en"), response, latin_positions=3)
    assert logq.shape[0] == len(response) and len(latin) == min(3, len(response))
    logp, kl = rt.student_logprob_and_kl(prompt, response, logq)
    with torch.no_grad():
        tokens = torch.tensor([prompt + response])
        full = rt.student(input_ids=tokens).logits[0, len(prompt) - 1:-1].float().log_softmax(-1)
        reference = (logq.exp() * (logq - full)).sum(-1)
        chosen = full.gather(1, torch.tensor(response)[:, None]).squeeze(1)
    assert torch.allclose(kl.detach(), reference, atol=1e-5)
    assert torch.allclose(logp.detach(), chosen, atol=1e-5)
    kl.mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in rt.parameters().values())
    for p in rt.parameters().values():
        p.grad = None


def test_teacher_prompts_differ_by_condition(setup):
    rt = setup["runtime"]
    problem = setup["dataset"].problem("train:3", "zh")
    prompts = {c: rt.tokenizer.decode(rt.teacher_prompt(problem, c)) for c in ("T-def", "T-en", "T-priv")}
    assert prompts["T-def"] == rt.tokenizer.decode(rt.student_prompt(problem))
    assert "English" in prompts["T-en"] and problem["english_question"] in prompts["T-priv"]
    en = setup["dataset"].problem("train:3", "en")
    assert rt.teacher_prompt(en, "T-priv") == rt.teacher_prompt(en, "T-def")


def test_answer_line_rewrite_and_segments(setup):
    from mlopd.targets import rewrite_answer_line, segments, verified_target
    rt = setup["runtime"]
    tok = rt.tokenizer
    eos = rt.eos[0]
    ids = tok.encode("He has 7 apples.\n#### Final Answer: 7", add_special_tokens=False) + [eos]
    new, status = rewrite_answer_line(tok, ids, "7", rt.eos)
    assert status == "rewritten_marker" and tok.decode(new, skip_special_tokens=True) == "He has 7 apples.\n#### 7"
    ids = tok.encode("Всего 7 яблок", add_special_tokens=False) + [eos]
    new, status = rewrite_answer_line(tok, ids, "7", rt.eos)
    assert status == "appended_marker" and tok.decode(new, skip_special_tokens=True).endswith("\n\n#### 7")
    labels = segments(tok, tok.encode("ab\n#### 7", add_special_tokens=False) + [eos], rt.eos, first=2)
    assert labels[:2] == ["first", "first"] and labels[-1] == "eos" and "answer" in labels and "body" in labels
    samples = [{"response_ids": [1, 2], "truncated": True, "text": "7"},
               {"response_ids": tok.encode("so 8", add_special_tokens=False) + [eos], "truncated": False, "text": "so 8"},
               {"response_ids": tok.encode("so 7", add_special_tokens=False) + [eos], "truncated": False, "text": "so 7"}]
    target = verified_target(tok, samples, "7", rt.eos)
    assert target["sample_index"] == 2 and target["status"] == "appended_marker"


def _plan(cfg, kind, dataset, runtime=None, **extra):
    order = sorted(dataset.pool)[:3]
    plan = {"kind": kind, "seed": 3, "lr": 1e-3, "clip": 1.0, "checkpoint_every": 1, "adapter_seed": 17,
            "rollout": {"samples": 2, "max_new": 5, "tag": "r0"}, "segment_first": 2}
    if kind == "sft":
        ids = runtime.tokenizer.encode("#### 5", add_special_tokens=False) + [runtime.eos[0]]
        plan["targets"] = [{"item": x, "lang": "en", "response_ids": ids} for x in order]
        plan["steps"] = [{"method": "sft", "targets": [0, 1]}, {"method": "sft", "targets": [2, 0]}]
    else:
        plan["steps"] = [{"method": kind, "item": x, "lang": "zh", "cycle": 0, "condition": "T-en"} for x in order]
    if kind == "opd_mask":
        plan["mask"] = {"first": 2, "answer_line": True}
    plan.update(extra)
    return plan


@pytest.mark.parametrize("kind", ["sft", "opd", "rl", "opd_mask"])
def test_trainers_records_and_resume(setup, kind, tmp_path):
    from mlopd.train import Unit
    rt, ds = setup["runtime"], setup["dataset"]
    plan = _plan(setup["cfg"], kind, ds, rt)
    full = Unit(rt, ds, plan, tmp_path / "full").run()
    assert full["updates"] == len(plan["steps"]) and full["final_policy_hash"] != full["start_policy_hash"] or kind == "rl"
    records = full["records"]
    if kind in ("opd", "opd_mask"):
        shares = records["opd"]["kl_share"]
        assert abs(sum(v for v in shares.values() if v is not None) - 1) < 1e-6
        assert records["opd"]["teacher_latin_mass_first_mean"] is not None
    if kind == "rl":
        assert set(records["rl"]["group_shares"]) == {"all_correct", "all_wrong", "mixed"}
        assert records["rl"]["zero_advantage_updates"] >= 0 and "reward_mean" in records["rl"]
    if kind != "sft":
        assert 0 <= records["rollout_truncation_rate"] <= 1 and records["mean_rollout_tokens"] > 0
    # interrupted run: stop after the first update, then resume from latest.pt
    calls = {"n": 0}
    unit = Unit(rt, ds, plan, tmp_path / "resumed")
    original = unit.step

    def failing(optimizer, index, spec):
        if calls["n"] == 1:
            raise KeyboardInterrupt
        calls["n"] += 1
        return original(optimizer, index, spec)
    unit.step = failing
    with pytest.raises(KeyboardInterrupt):
        unit.run()
    assert (tmp_path / "resumed" / "latest.pt").exists()
    resumed = Unit(rt, ds, plan, tmp_path / "resumed").run()
    assert resumed["final_policy_hash"] == full["final_policy_hash"]


def test_rl_and_opd_share_rollouts(setup, tmp_path):
    from mlopd.train import Unit
    import json
    rt, ds = setup["runtime"], setup["dataset"]
    Unit(rt, ds, _plan(setup["cfg"], "rl", ds), tmp_path / "rl").run()
    Unit(rt, ds, _plan(setup["cfg"], "opd", ds), tmp_path / "opd").run()
    first_rl = json.loads((tmp_path / "rl" / "updates.jsonl").read_text().splitlines()[0])
    first_opd = json.loads((tmp_path / "opd" / "updates.jsonl").read_text().splitlines()[0])
    assert first_rl["tries"][0]["response_tokens"] == first_opd["response_tokens"]
    assert first_rl["tries"][0]["rewards"] == first_opd["strict_rewards"]


def test_rl_dynamic_sampling_moves_past_zero_advantage(setup, tmp_path, monkeypatch):
    import json
    from mlopd import train
    from mlopd.train import Unit
    rt, ds = setup["runtime"], setup["dataset"]
    order = sorted(ds.pool)[:3]
    plan = _plan(setup["cfg"], "rl", ds)
    plan["steps"] = [{"method": "rl", "item": order[0], "lang": "zh", "cycle": 0, "candidates": order}]
    # rewards: first problem all wrong (zero advantage), second mixed
    calls = []

    def fake(text, gold):
        calls.append(1)
        return int(len(calls) == 4)
    monkeypatch.setattr(train, "strict_correct", fake)
    out = Unit(rt, ds, plan, tmp_path / "dyn").run()
    row = json.loads((tmp_path / "dyn" / "updates.jsonl").read_text().splitlines()[0])
    assert [t["item"] for t in row["tries"]] == order[:2] and row["item"] == order[1]
    assert row["zero_advantage"] == 0 and out["records"]["rl"]["problems_tried"] == 2
    assert out["records"]["rl"]["first_try_zero_advantage"] == 1


def test_gumbel_sampling_matches_softmax_frequencies(setup):
    """Temperature-1 sampling: token frequencies follow the softmax of the logits."""
    rt = setup["runtime"]
    problem = setup["dataset"].problem("train:4", "en")
    prompt = rt.student_prompt(problem)
    with torch.no_grad():
        probs = rt.student(input_ids=torch.tensor([prompt])).logits[0, -1].float().softmax(-1)
    draws = rt.sample(rt.student, [prompt] * 3000, list(range(3000)), max_new=1, batch_seqs=500)
    counts = torch.bincount(torch.tensor([d["response_ids"][0] for d in draws]), minlength=probs.shape[0]).float()
    top = probs.argsort(descending=True)[:5]
    assert torch.allclose(counts[top] / 3000, probs[top], atol=0.02)
