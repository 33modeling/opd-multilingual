"""Task leases for several workers (one per GPU, on one or more nodes) sharing one runs/ directory.

A lease is a directory runs/.leases/<pair>__<task id with / as ~>/ made with mkdir, which is atomic on local and
network file systems. Its owner.json names the worker (host, GPU, pid); a heartbeat thread rewrites it every
`beat` seconds. A lease whose owner.json is older than `stale` seconds belongs to a worker that died; another worker
removes it and may take the task (the task then resumes from its latest.pt where the executor supports it).

Failures: runs/.failures/<key>.json counts the exceptions of a task. After `max_failures` the task is skipped by
every worker until the file is deleted (`mlopd queue retry-failed`).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import threading
import time
import traceback


def key_of(t):
    return f"{t['pair']}__{t['id'].replace('/', '~')}"


def worker_name():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "cpu")
    return f"{socket.gethostname()}:gpu{gpu}:pid{os.getpid()}"


class Leases:
    def __init__(self, runs, *, worker=None, beat=60., stale=900., max_failures=2):
        self.root = Path(runs) / ".leases"
        self.failures = Path(runs) / ".failures"
        self.root.mkdir(parents=True, exist_ok=True)
        self.failures.mkdir(parents=True, exist_ok=True)
        self.worker = worker or worker_name()
        self.beat, self.stale, self.max_failures = beat, stale, max_failures

    # ------------------------------------------------------------ leases
    def _owner(self, path):
        try:
            return json.loads((path / "owner.json").read_text())
        except (OSError, ValueError):
            return None

    def _age(self, path):
        for name in ("owner.json", "."):
            try:
                return time.time() - (path / name).stat().st_mtime
            except OSError:
                continue
        return None

    def held(self, t):
        """True when a live worker (other than this one) holds the task."""
        path = self.root / key_of(t)
        if not path.exists():
            return False
        age = self._age(path)
        if age is not None and age > self.stale:
            return False
        owner = self._owner(path)
        return not (owner and owner.get("worker") == self.worker)

    def live(self):
        out = []
        for path in sorted(self.root.iterdir()) if self.root.exists() else []:
            if path.name.startswith("."):
                continue
            age = self._age(path)
            if age is not None and age <= self.stale:
                out.append({"key": path.name, "age_s": round(age), **(self._owner(path) or {})})
        return out

    def acquire(self, t):
        path = self.root / key_of(t)
        if path.exists():
            age = self._age(path)
            if age is None or age <= self.stale:
                return None
            # Dead owner. Only the worker holding the takeover lock may remove the lease, and it checks the age
            # again under that lock, so two workers cannot both remove and recreate it.
            guard = self.root / f".takeover-{key_of(t)}"
            try:
                guard.mkdir()
            except FileExistsError:
                try:
                    if time.time() - guard.stat().st_mtime > 120:
                        guard.rmdir()  # left by a worker that died during a takeover
                except OSError:
                    pass
                return None
            try:
                age = self._age(path)
                if path.exists() and (age is None or age <= self.stale):
                    return None
                shutil.rmtree(path, ignore_errors=True)
                if path.exists():
                    raise OSError(f"cannot remove stale lease {path}")
                path.mkdir()
            finally:
                guard.rmdir()
            return Lease(self, path, t)
        try:
            path.mkdir()
        except FileExistsError:
            return None
        return Lease(self, path, t)

    # ------------------------------------------------------------ failures
    def failure_path(self, t):
        return self.failures / f"{key_of(t)}.json"

    def failure_count(self, t):
        path = self.failure_path(t)
        try:
            return int(json.loads(path.read_text()).get("count", 0))
        except (OSError, ValueError):
            return 0

    def failed(self, t):
        return self.failure_count(t) >= self.max_failures

    def record_failure(self, t, error):
        path = self.failure_path(t)
        count = self.failure_count(t) + 1
        text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"task": t["id"], "pair": t["pair"], "count": count, "worker": self.worker,
                                   "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "error": text[-4000:]}, indent=1))
        os.replace(tmp, path)
        return count

    def clear_failures(self):
        removed = 0
        for path in self.failures.glob("*.json"):
            path.unlink()
            removed += 1
        return removed

    def failed_list(self):
        out = []
        for path in sorted(self.failures.glob("*.json")):
            try:
                out.append(json.loads(path.read_text()))
            except (OSError, ValueError):
                continue
        return out


class Lease:
    def __init__(self, leases, path, t):
        self.leases, self.path, self.task = leases, path, t
        self._stop = threading.Event()
        self._write()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _write(self):
        current = self.leases._owner(self.path)
        if current is not None and current.get("worker") != self.leases.worker:
            self._stop.set()  # taken over after a stall; stop beating on the other worker's lease
            return
        info = {"worker": self.leases.worker, "task": self.task["id"], "pair": self.task["pair"],
                "kind": self.task["kind"], "block": self.task["block"], "est_hours": self.task.get("hours"),
                "started": getattr(self, "started", None) or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "beat": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        self.started = info["started"]
        tmp = self.path / f"owner.{os.getpid()}.tmp"
        try:
            tmp.write_text(json.dumps(info))
            os.replace(tmp, self.path / "owner.json")
        except OSError:
            pass  # lease removed by another worker that judged it stale; the task re-checks COMPLETE.json

    def _loop(self):
        while not self._stop.wait(self.leases.beat):
            self._write()

    def release(self):
        self._stop.set()
        self._thread.join(timeout=5)
        owner = self.leases._owner(self.path)
        if owner is None or owner.get("worker") == self.leases.worker:  # never remove a lease taken over by another
            shutil.rmtree(self.path, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False
