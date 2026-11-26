"""Leases for several workers on one runs/ directory: one owner per task, stale leases taken over, failures
counted and the task skipped after the limit, and two workers finishing the stage-0 queue together."""
import os
from pathlib import Path
import threading
import time

from conftest import make_dataset, smoke_overrides


def _task(id="s0a/student/zh"):
    return {"id": id, "pair": "main", "kind": "measure", "block": "S0-a", "deps": [], "hours": 0.1, "params": {}}


def test_one_owner(tmp_path):
    from mlopd.lease import Leases
    a, b = Leases(tmp_path, worker="a"), Leases(tmp_path, worker="b")
    t = _task()
    lease = a.acquire(t)
    assert lease is not None and b.acquire(t) is None
    assert b.held(t) and not a.held(t)
    assert [r["worker"] for r in b.live()] == ["a"]
    lease.release()
    assert not b.held(t) and b.acquire(t) is not None


def test_stale_lease_taken_over(tmp_path):
    from mlopd.lease import Leases, key_of
    a, b = Leases(tmp_path, worker="a", stale=5), Leases(tmp_path, worker="b", stale=5)
    t = _task()
    lease = a.acquire(t)
    lease._stop.set()  # the owner died: no more heartbeats
    old = time.time() - 60
    owner = tmp_path / ".leases" / key_of(t) / "owner.json"
    os.utime(owner, (old, old))
    assert not b.held(t) and b.live() == []
    assert b.acquire(t) is not None


def test_failure_limit(tmp_path):
    from mlopd.lease import Leases
    leases = Leases(tmp_path, worker="a", max_failures=2)
    t = _task()
    for expected in (1, 2):
        assert not leases.failed(t)
        assert leases.record_failure(t, RuntimeError("boom")) == expected
    assert leases.failed(t) and "boom" in leases.failed_list()[0]["error"]
    assert leases.clear_failures() == 1 and not leases.failed(t)


def test_failing_task_does_not_stop_worker(tiny, stub_identifier, tmp_path, monkeypatch):
    from mlopd import experiment
    from mlopd.lease import Leases
    make_dataset(tmp_path / "data" / "p.json")
    cfg = experiment.load_config(overrides=smoke_overrides(tiny, tmp_path / "data" / "p.json"))
    ctx = experiment.Context(cfg, root=tmp_path)
    real, bad = experiment.execute, {}

    def flaky(ctx, t):
        if not bad:
            bad["id"] = t["id"]
        if t["id"] == bad["id"]:
            raise RuntimeError("injected")
        return real(ctx, t)

    monkeypatch.setattr(experiment, "execute", flaky)
    leases = Leases(ctx.runs, worker="w", max_failures=2)
    experiment.run_queue(ctx, leases=leases, max_tasks=3, log=lambda m: None)
    assert leases.failure_count({"pair": "main", "id": bad["id"]}) == 2
    assert len(list(ctx.runs.glob("main/**/COMPLETE.json"))) == 3
    assert not any((ctx.runs / ".leases").iterdir())


def _worker(cfg, root, name, queue):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from conftest import StubIdentifier
    from mlopd import experiment
    from mlopd.lease import Leases
    from mlopd.records import register_identifier
    register_identifier("stub", StubIdentifier())
    ctx = experiment.Context(cfg, root=root)
    logs = []
    experiment.run_queue(ctx, leases=Leases(ctx.runs, worker=name), max_tasks=6, log=logs.append)
    queue.put((name, [m for m in logs if m.startswith(("done", "FAIL"))]))


def test_two_workers_share_stage0(tiny, tmp_path):
    """Two worker processes (as on two GPUs) on one runs/ directory: no task runs twice."""
    import multiprocessing as mp
    from mlopd import experiment
    from mlopd.lease import Leases
    make_dataset(tmp_path / "data" / "p.json")
    cfg = experiment.load_config(overrides=smoke_overrides(tiny, tmp_path / "data" / "p.json"))
    spawn = mp.get_context("spawn")
    queue = spawn.Queue()
    procs = [spawn.Process(target=_worker, args=(cfg, str(tmp_path), n, queue)) for n in ("w1", "w2")]
    for proc in procs:
        proc.start()
    out = dict(queue.get(timeout=900) for _ in procs)
    for proc in procs:
        proc.join(timeout=60)
    fails = Leases(tmp_path / "runs", worker="check").failed_list()
    assert not fails, fails[0]["error"]
    ids = [m.split()[1] for m in out["w1"] + out["w2"]]
    assert len(ids) == len(set(ids)) == 12 and out["w1"] and out["w2"]
