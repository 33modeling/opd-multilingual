"""SFT targets from teacher samples and token segments of a response.

Target rule (design §5.2, English record stage_transition_campaign.teacher_target): the first teacher sample in
sampling order that ended with EOS, is lenient_norm-correct, and is strict-correct after its answer line is
rewritten to "#### N". N is the canonical ASCII form of the gold value (the sample's own final number has the same
value, because the sample is lenient-correct). The tokens before the answer line are kept verbatim. Items without
such a sample get no target (no gold fallback); the number of such items is recorded.
"""
from __future__ import annotations

import unicodedata

from .scoring import canonical_number, parse_gold, score, strict_correct


def _prefixes(tokenizer, ids):
    """decode(ids[:k]) for k = 0..len(ids)."""
    return [tokenizer.decode(ids[:k], skip_special_tokens=True) for k in range(len(ids) + 1)]


def _prefix_lengths(tokenizer, ids):
    return [len(p) for p in _prefixes(tokenizer, ids)]


def marker_line_start(text):
    """Character offset where the line holding the last "####" (NFKC-equivalent, e.g. full-width) starts."""
    norm = unicodedata.normalize("NFKC", text)
    position = norm.rfind("####")
    if position < 0:
        return None
    if len(norm) != len(text):
        # NFKC changed the length; locate the marker in the raw text by the same line index.
        line_index = norm[:position].count("\n")
        starts = [0] + [i + 1 for i, ch in enumerate(text) if ch == "\n"]
        return starts[min(line_index, len(starts) - 1)]
    return norm.rfind("\n", 0, position) + 1


def rewrite_answer_line(tokenizer, response_ids, gold, eos_ids):
    """Return (new_ids, status). Truncated responses (no final EOS) are returned unchanged."""
    eos_ids = set(eos_ids)
    if not response_ids or response_ids[-1] not in eos_ids:
        return list(response_ids), "unchanged_truncated"
    eos, content = response_ids[-1], list(response_ids[:-1])
    text = tokenizer.decode(content, skip_special_tokens=True)
    answer = "#### " + canonical_number(parse_gold(gold))
    start = marker_line_start(text)
    if start is None:
        tail = "\n\n" + answer
        return content + tokenizer.encode(tail, add_special_tokens=False) + [eos], "appended_marker"
    if text[start:].strip() == answer:
        return list(response_ids), "already_canonical"
    prefixes = _prefixes(tokenizer, content)
    keep = max(k for k, p in enumerate(prefixes) if len(p) <= start and text.startswith(p))
    tail = text[len(prefixes[keep]):start] + answer
    return content[:keep] + tokenizer.encode(tail, add_special_tokens=False) + [eos], "rewritten_marker"


def verified_target(tokenizer, samples, gold, eos_ids):
    """samples: [{response_ids, truncated, text}] in sampling order."""
    for index, sample in enumerate(samples):
        if sample["truncated"] or not score(sample["text"], gold)["lenient_norm"]:
            continue
        ids, status = rewrite_answer_line(tokenizer, sample["response_ids"], gold, eos_ids)
        if strict_correct(tokenizer.decode(ids, skip_special_tokens=True), gold):
            return {"response_ids": ids, "sample_index": index, "status": status}
    return None


def segments(tokenizer, response_ids, eos_ids, *, first=16):
    """Segment label per response token (§5.5): "first" (positions < first), "eos" (a final EOS token),
    "answer" (tokens of the last line holding "####"), "body" (the rest). Labels are disjoint in that priority."""
    eos_ids = set(eos_ids)
    n = len(response_ids)
    labels = ["body"] * n
    content = list(response_ids)
    if content and content[-1] in eos_ids:
        labels[-1] = "eos"
        content = content[:-1]
    text = tokenizer.decode(content, skip_special_tokens=True)
    start = marker_line_start(text)
    if start is not None:
        lengths = _prefix_lengths(tokenizer, content)
        for k in range(len(content)):
            if lengths[k + 1] > start:
                labels[k] = "answer"
    for k in range(min(first, n)):
        labels[k] = "first"
    return labels
