"""Per-response scoring rows and their summaries (design §6, §5.5)."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from . import langid
from .scoring import score
from .util import mean

_IDENTIFIERS = {}


def register_identifier(name, identifier):
    """Tests register a stub language identifier for the Latin-script languages."""
    _IDENTIFIERS[name] = identifier


def identifier_for(settings):
    backend = (settings or {}).get("backend", "fasttext")
    if backend in _IDENTIFIERS:
        return _IDENTIFIERS[backend]
    if backend == "fasttext":
        return langid.default_identifier((settings or {}).get("model_path"))
    raise ValueError(f"unknown language-ID backend {backend}")


class LazyIdentifier:
    """Builds the fastText model only when a Latin-script language is scored."""
    def __init__(self, settings):
        self.settings, self.inner = settings, None

    def predict(self, text):
        if self.inner is None:
            self.inner = identifier_for(self.settings)
        return self.inner.predict(text)


def response_row(text, problem, *, response_ids, truncated, identifier, extra=None):
    scores = score(text, problem["gold"], truncated=truncated)
    row = {"id": problem["id"], "lang": problem["lang"], "set": problem.get("set"), "text": text,
           "tokens": len(response_ids), "truncated": int(truncated), **scores}
    row.update(langid.language_metrics(text, problem["lang"], identifier=identifier,
                                       lenient_correct=scores["lenient_norm"]))
    if extra:
        row.update(extra)
    return row


def summarize(rows):
    """Response-level means (pass@1 with 4 samples = mean over responses), pass@4 per item, language metrics."""
    if not rows:
        return {"responses": 0}
    keys = ("strict", "lenient_norm", "lenient_ascii", "lenient_trim", "lenient_legacy", "strict_legacy",
            "language_match", "line_pass", "language_consistent", "truncated", "has_marker_line",
            "strict_dual_decided", "lenient_dual_decided")
    out = {"responses": len(rows), "items": len({r["id"] for r in rows})}
    for key in keys:
        values = [r[key] for r in rows if r.get(key) is not None]
        out[key] = sum(values) / len(values) if values else None
    out["r_mean"] = mean([r["r"] for r in rows])
    out["r_none"] = sum(r["r"] is None for r in rows)
    out["mean_tokens"] = mean([r["tokens"] for r in rows])
    out["marker_line_responses"] = sum(r["has_marker_line"] for r in rows)
    out["strict_dual_decided_count"] = sum(r["strict_dual_decided"] for r in rows)
    out["lenient_dual_decided_count"] = sum(r["lenient_dual_decided"] for r in rows)
    by_item = {}
    for r in rows:
        by_item.setdefault(r["id"], []).append(r["lenient_norm"])
    out["lenient_norm_pass_at_k"] = mean([float(any(v)) for v in by_item.values()])
    out["samples_per_item"] = round(len(rows) / len(by_item), 3)
    out["strict_over_lenient"] = (out["strict"] / out["lenient_norm"]) if out["lenient_norm"] else None
    return out


def summarize_eval(rows):
    """Per set and combined; the primary accuracy Acc is the simple mean of the per-set lenient_norm pass@1."""
    sets = sorted({r["set"] for r in rows})
    per_set = {s: summarize([r for r in rows if r["set"] == s]) for s in sets}
    combined = summarize(rows)
    combined["acc"] = mean([per_set[s]["lenient_norm"] for s in sets])
    combined["strict_acc"] = mean([per_set[s]["strict"] for s in sets])
    combined["r_acc"] = mean([per_set[s]["r_mean"] for s in sets])
    combined["match_acc"] = mean([per_set[s]["language_match"] for s in sets])
    return {"combined": combined, "per_set": per_set}


def item_accuracy(rows, key="lenient_norm"):
    """{(set, id): mean over samples} for the item bootstrap."""
    out = {}
    for r in rows:
        out.setdefault((r["set"], r["id"]), []).append(r[key] if r[key] is not None else 0.)
    return {k: sum(v) / len(v) for k, v in out.items()}


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
