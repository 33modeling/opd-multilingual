"""Shared helpers: deterministic seeds and draws, hashing, atomic JSON files."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import time

L10 = ("en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn")
NON_LATIN = ("ru", "zh", "ja", "th", "bn")
LATIN = ("en", "es", "fr", "de", "sw")
LANGUAGE_NAMES = {"en": "English", "es": "Spanish", "fr": "French", "de": "German", "ru": "Russian",
                  "zh": "Chinese", "ja": "Japanese", "th": "Thai", "sw": "Swahili", "bn": "Bengali"}


def stable_seed(*parts):
    """Same rule as the English code (continual-pair-opd route_features.stable_seed): the first 8 hex digits of
    sha256(":".join(parts))."""
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def draw(names, size, seed, *tags):
    """Seeded draw without replacement from the sorted names (English stage_data_campaign.draw with the label
    "mlopd" instead of "stage-data")."""
    left = sorted(names)
    if size > len(left):
        raise ValueError(f"{tags}: asked for {size}, only {len(left)}")
    return sorted(random.Random(stable_seed(seed, "mlopd", *tags)).sample(left, size))


def shuffled(names, seed, *tags):
    left = sorted(names)
    random.Random(stable_seed(seed, "mlopd-order", *tags)).shuffle(left)
    return left


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_json(value):
    return sha256_bytes(canonical_json(value).encode())


def write_json(path, value):
    """Atomic write (tmp file + rename) so an interrupted task never leaves a half file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False, sort_keys=False) + "\n")
    os.replace(tmp, path)
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return path


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None
