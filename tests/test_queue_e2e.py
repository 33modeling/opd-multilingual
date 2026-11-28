"""The whole run grid on tiny models: stage 0, gates A/B, G, G-ref, G-x, I, K, W (pilot gate, then RL), A and B
(their conditional gates are forced in this test only), X (second pair), and the analysis verdicts."""
import json

import pytest

from conftest import make_dataset, smoke_overrides


@pytest.fixture(scope="module")
def finished(tiny, stub_identifier, tmp_path_factory):
    from mlopd.experiment import Context, load_config, run_queue
    root = tmp_path_factory.mktemp("queue")
    make_dataset(root / "data" / "p.json")
    cfg = load_config(overrides=smoke_overrides(tiny, root / "data" / "p.json"))
    ctx = Context(cfg, root=root)
    logs = []
    run_queue(ctx, log=logs.append)
    # The conditional blocks A (M5) and B (M2b) need M3a / M1b to be supported, which random models do not give.
    # Force both decisions here to exercise those blocks end to end.
    for name in ("m5", "m2b"):
        path = ctx.runs / "main" / "gates" / name / "COMPLETE.json"
        assert path.exists(), name
        decision = json.loads(path.read_text())
        decision["go"] = True
        path.write_text(json.dumps(decision))
    run_queue(ctx, log=logs.append)
    return ctx, logs


def test_every_block_ran(finished):
    ctx, logs = finished
    runs = ctx.runs
    expect = ["main/s0a/student/zh", "main/s0a/teacher-T-priv/ru", "main/s0b/solve_en/en", "main/s0/scorer-en",
              "main/handoff/s1-u2", "main/start/s2-u2/ru", "main/p/s1-u2/zh", "main/latin/s1-u2/zh",
              "main/gates/a-u2", "main/s0d/rl-en-r1", "main/gates/b", "main/g/opd-ru-s2/eval-ru",
              "main/g/rl-en-s1/eval-zh", "main/gref/sft-zh-s1/eval-zh", "main/gates/m6", "main/w/rl-W-s8/eval-ru",
              "main/w/sft-N-s7/eval-b-zh", "main/a/matched-s5/eval-en", "main/a/swapped-s6",
              "x/handoff/s1-u2", "x/g/rl-en-s2/eval-en"]
    for path in expect:
        assert (runs / path / "COMPLETE.json").exists(), path
    a = json.loads((runs / "main/gates/a-u2/COMPLETE.json").read_text())
    assert a["imprint"]["outcome"] == "go" and a["gap"]["outcome"] == "go" and "input_files" in a
    for l in a["selection"]["M1"]:
        assert (runs / f"main/i/opd-ten-{l}-s1/eval-{l}/COMPLETE.json").exists()
        assert (runs / f"main/i/sft-ten-{l}-s2/COMPLETE.json").exists()
    for l in a["selection"]["M4"]:
        assert (runs / f"main/k/opd-tpriv-{l}-s1/eval-{l}/COMPLETE.json").exists()
    for l in a["selection"]["B"]:
        assert (runs / f"main/b/opdmask-ten-{l}-s1/eval-{l}/COMPLETE.json").exists()


def test_records_present(finished):
    ctx, _ = finished
    opd = json.loads((ctx.runs / "main/g/opd-zh-s1/COMPLETE.json").read_text())["records"]
    assert set(opd["opd"]["kl_share"]) == {"first", "body", "answer", "eos"}
    rl = json.loads((ctx.runs / "main/g/rl-zh-s1/COMPLETE.json").read_text())["records"]
    assert "group_shares" in rl["rl"] and "zero_advantage_updates" in rl["rl"]
    ev = json.loads((ctx.runs / "main/g/rl-zh-s1/eval-zh/COMPLETE.json").read_text())
    assert set(ev["per_set"]) == {"mgsm", "msvamp"} and ev["collapse"] in (0, 1) and "start_acc" in ev
    for key in ("acc", "strict_acc", "r_acc", "match_acc", "lenient_norm_pass_at_k", "truncated", "language_consistent"):
        assert key in ev["combined"]
    mixed = json.loads((ctx.runs / "main/a/matched-s5/plan.json").read_text())
    methods = {(s["group"], s["method"]) for s in mixed["steps"]}
    assert methods == {("A", "opd"), ("B", "rl")}
    swapped = json.loads((ctx.runs / "main/a/swapped-s5/plan.json").read_text())
    assert [(s["item"], s["lang"]) for s in swapped["steps"]] == [(s["item"], s["lang"]) for s in mixed["steps"]]


def test_queue_is_resumable_and_idle(finished):
    from mlopd.experiment import dry_run, run_queue
    ctx, _ = finished
    assert run_queue(ctx, log=lambda m: None) == 0
    result = dry_run(ctx)
    assert all(line.startswith("done") for line in result["lines"])


def test_verdicts(finished):
    from mlopd.analysis import write_verdicts
    ctx, _ = finished
    path, verdicts = write_verdicts(ctx)
    for h in ("M1a", "M1b", "M3a", "M3b", "M3c", "M4", "M5", "M6a", "M6b", "M2b"):
        assert h in verdicts and "status" in verdicts[h], h
        assert verdicts[h]["status"] != "incomplete", (h, verdicts[h])
    # random models: every accuracy is 0, so the contrasts are degenerate; the plumbing is what is checked here
    assert verdicts["M3a"]["test"]["n"] == 3 and "mixed_model_slope" in verdicts["M3a"]
    assert verdicts["M1b"]["test"]["n"] == 2 * 2 and verdicts["M4"]["test"]["n"] == 2 * 2
    assert verdicts["M5"]["test"]["n"] == 2 and verdicts["M6a"]["test"]["n"] == 2
    assert verdicts["M2b"]["status"] == "not run (condition not met)"  # M1b is not supported
    assert set(verdicts["holm_family"]) <= {"M1b", "M2b", "M3a", "M4", "M5", "M6a", "M6b"}


def test_dry_run_before_any_run(tiny, stub_identifier, tmp_path):
    from mlopd.experiment import Context, dry_run, load_config
    make_dataset(tmp_path / "p.json")
    cfg = load_config(overrides=smoke_overrides(tiny, tmp_path / "p.json"))
    result = dry_run(Context(cfg, root=tmp_path))
    assert result["lines"] and any(name == "G" for name, _, _ in result["pending_blocks"])
    real = load_config()
    real_result = dry_run(Context(real, root=tmp_path))
    blocks = real_result["blocks"]
    assert blocks["S0-a"][0] == 10 + 29 and blocks["S0-b"][0] == 40


def test_parallel_blocks_run_despite_no_go_gates(tiny, stub_identifier, tmp_path_factory):
    """queue.parallel_blocks: with every hypothesis gate failing, blocks I, K, A, W and B still run, and their
    results come back labelled as records instead of verdicts."""
    from mlopd.analysis import verdicts
    from mlopd.experiment import Context, build_queue, load_config, run_queue
    root = tmp_path_factory.mktemp("parallel")
    make_dataset(root / "data" / "p.json")
    overrides = smoke_overrides(tiny, root / "data" / "p.json")
    overrides["gates"].update({"latin_mass_gap": 10.0, "teacher_ratio_gap": 10.0, "gap_manipulation": 10.0,
                               "washout_sft_gap_pp": 1000.0, "m3_pilot_spread_pp": 1000.0})
    overrides["queue"] = {"min_free_gb": 0, "parallel_blocks": True}
    cfg = load_config(overrides=overrides)
    ctx = Context(cfg, root=root)
    run_queue(ctx, log=lambda m: None)
    assert all(ctx.done(t) for t in build_queue(ctx))
    runs = ctx.runs / "main"
    a = json.loads((runs / "gates/a-u2/COMPLETE.json").read_text())
    assert a["imprint"]["outcome"] != "go" and a["gap"]["outcome"] != "go"
    assert not json.loads((runs / "gates/m6/COMPLETE.json").read_text())["go"]
    assert not json.loads((runs / "gates/m3-pilot/COMPLETE.json").read_text())["go"]
    l = a["selection"]["M1"][0]
    for path in (f"i/opd-ten-{l}-s1/eval-{l}", f"k/opd-tpriv-{a['selection']['M4'][0]}-s1",
                 "a/matched-s5", "w/rl-W-s8", f"b/opdmask-ten-{a['selection']['B'][0]}-s1",
                 "g/rl-en-s3", "g/opd-zh-s4"):
        assert (runs / path / "COMPLETE.json").exists(), path
    v = verdicts(ctx)
    for h in ("M1b", "M4", "M6a", "M5"):
        assert v[h]["status"] == "record (gate no-go)", (h, v[h].get("status"))
    assert "M3a-p" in v and v["M3a-p"]["alternative"] == "less"


def test_status_after_the_run(finished, tmp_path):
    from mlopd.experiment import format_status, status
    from mlopd.lease import Leases
    ctx, _ = finished
    result = status(ctx, Leases(ctx.runs))
    assert result["totals"]["done"] == result["tasks"] and result["tasks"] > 0
    text = format_status(result)
    assert text.splitlines()[1].startswith(f"done {result['tasks']}/{result['tasks']} (100%)")
    assert "next: all tasks done" in text and "main/G " in text
    # A running train task shows its update count; a failed one shows its last error line.
    t = next(t for t in __import__("mlopd.experiment", fromlist=["build_queue"]).build_queue(ctx) if t["kind"] == "train")
    leases = Leases(ctx.runs)
    out = ctx.dir(t)
    (out / "COMPLETE.json").rename(out / "COMPLETE.bak")
    try:
        (out / "latest.json").write_text(json.dumps({"step": 3}))
        with leases.acquire(t):
            r = status(ctx, leases)
            row = next(x for x in r["running"] if x["task"] == f"{t['pair']}/{t['id']}")
            assert row["progress"][0] == 3 and "update 3/" in format_status(r)
        leases.record_failure(t, RuntimeError("boom"))
        leases.record_failure(t, RuntimeError("boom"))
        r = status(ctx, leases)
        assert r["totals"]["failed"] == 1 and "RuntimeError: boom" in format_status(r)
    finally:
        leases.clear_failures()
        (out / "latest.json").unlink(missing_ok=True)
        (out / "COMPLETE.bak").rename(out / "COMPLETE.json")
