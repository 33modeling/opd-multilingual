"""Target-language ratio r and the language metrics of design §6.2-§6.3.

Text measured: the response up to the last "####" (the whole response when there is none). Digits, mathematical
symbols, LaTeX commands and markdown symbols are removed first.

  zh, ja, ru, th, bn (non-Latin scripts): share of target-script characters among the remaining letters and marks
      (Unicode categories L* and M*), from fixed Unicode block ranges (deterministic). Japanese counts kana and Han
      together; whether kana occur is recorded (`has_kana`) to tell Japanese from Chinese responses.
  en, es, fr, de, sw (Latin script): sentence-level language identification with fastText lid.176, weighted by
      sentence length in characters; r is the share of characters in sentences labelled with the target language.
      fastText is an optional dependency (pip install fasttext; model file from
      https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin given by MLOPD_LID_MODEL or the config).

A response without any letters after cleaning has r = None; it counts as not matching (r >= 0.5 fails) and is left
out of the mean of r. Language match = r >= 0.5. Line-level pass (Language Confusion Benchmark) = every non-empty
line (after cleaning) has its own r >= 0.5. Language-consistent accuracy = lenient_norm correct and r >= 0.5.
"""
from __future__ import annotations

import os
import re
import unicodedata

from .util import LATIN, NON_LATIN

MATCH_THRESHOLD = 0.5

SCRIPT_RANGES = {
    "han": ((0x2E80, 0x2FDF), (0x3005, 0x3007), (0x3021, 0x3029), (0x3038, 0x303B), (0x3400, 0x4DBF),
            (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2FA1F)),
    "kana": ((0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9F), (0x1B000, 0x1B16F)),
    "cyrillic": ((0x0400, 0x052F), (0x1C80, 0x1C8F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)),
    "thai": ((0x0E00, 0x0E7F),),
    "bengali": ((0x0980, 0x09FF),),
    "latin": ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF), (0x2C60, 0x2C7F),
              (0xA720, 0xA7FF), (0xFF21, 0xFF3A), (0xFF41, 0xFF5A)),
}
TARGET_SCRIPTS = {"zh": ("han",), "ja": ("han", "kana"), "ru": ("cyrillic",), "th": ("thai",), "bn": ("bengali",)}

LATEX_RE = re.compile(r"\\[A-Za-z]+")
# Sentence punctuation (. ! ? and the CJK forms) is kept for the sentence split of the Latin-script languages.
MARKDOWN_AND_MATH = set("#*_`~>|[](){}<>=+-−×÷*/^%$\\,;:\"'“”‘’«»…·•@&")
SENTENCE_RE = re.compile(r"[^.!?。！？\n]+[.!?。！？]?")


def script_of(ch):
    code = ord(ch)
    for name, ranges in SCRIPT_RANGES.items():
        if any(low <= code <= high for low, high in ranges):
            return name
    return "other"


def is_latin_letter(ch):
    return unicodedata.category(ch).startswith("L") and script_of(ch) == "latin"


def body_text(text):
    position = text.rfind("####")
    return text[:position] if position >= 0 else text


def clean(text):
    text = LATEX_RE.sub(" ", text)
    out = []
    for ch in text:
        if ch.isdigit() or unicodedata.category(ch) in ("Sm", "Sc") or ch in MARKDOWN_AND_MATH:
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def _letters(text):
    return [ch for ch in text if unicodedata.category(ch)[0] in ("L", "M")]


def script_ratio(text, language):
    letters = _letters(text)
    if not letters:
        return None
    targets = TARGET_SCRIPTS[language]
    return sum(script_of(ch) in targets for ch in letters) / len(letters)


class FastTextIdentifier:
    """fastText lid.176 wrapper; predict(text) -> ISO 639-1 code."""
    def __init__(self, model_path=None):
        path = model_path or os.environ.get("MLOPD_LID_MODEL")
        try:
            import fasttext
        except ImportError as error:
            raise ImportError("Latin-script languages (en, es, fr, de, sw) need fastText language ID: "
                              "pip install fasttext (or fasttext-wheel) and download lid.176.bin "
                              "(https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin); "
                              "set MLOPD_LID_MODEL or config langid.model_path") from error
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"fastText lid.176 model not found at {path!r}; set MLOPD_LID_MODEL")
        self.model = fasttext.load_model(path)
        self.path = path

    def predict(self, text):
        text = text.replace("\n", " ")
        try:
            # The low-level call: fasttext 0.9.x's Python predict() fails under numpy 2 (np.array(copy=False)).
            pairs = self.model.f.predict(text, 1, 0.0, "strict")
            labels = [label for _, label in pairs]
        except AttributeError:
            labels, _ = self.model.predict(text, k=1)
        return labels[0].replace("__label__", "") if labels else "unk"


_DEFAULT = {}


def default_identifier(model_path=None):
    key = model_path or os.environ.get("MLOPD_LID_MODEL")
    if key not in _DEFAULT:
        _DEFAULT[key] = FastTextIdentifier(model_path)
    return _DEFAULT[key]


def lid_ratio(text, language, identifier):
    total = hit = 0
    for sentence in SENTENCE_RE.findall(text):
        letters = _letters(sentence)
        if not letters:
            continue
        weight = len(sentence.strip())
        total += weight
        if identifier.predict(sentence.strip()) == language:
            hit += weight
    return hit / total if total else None


def ratio_of(cleaned, language, identifier=None):
    if language in NON_LATIN:
        return script_ratio(cleaned, language)
    if language in LATIN:
        return lid_ratio(cleaned, language, identifier or default_identifier())
    raise ValueError(f"unknown language {language}")


def language_metrics(text, language, *, identifier=None, lenient_correct=None):
    body = clean(body_text(text))
    r = ratio_of(body, language, identifier)
    lines = [line for line in body.splitlines() if _letters(line)]
    line_ratios = [ratio_of(line, language, identifier) for line in lines]
    out = {"r": r, "language_match": int(r is not None and r >= MATCH_THRESHOLD),
           "line_pass": int(bool(lines) and all(x is not None and x >= MATCH_THRESHOLD for x in line_ratios)),
           "latin_letter_share": (sum(map(is_latin_letter, _letters(body))) / len(_letters(body))
                                  if _letters(body) else None)}
    if language == "ja" or language == "zh":
        out["has_kana"] = int(any(script_of(ch) == "kana" for ch in body))
    if lenient_correct is not None:
        out["language_consistent"] = int(bool(lenient_correct) and out["language_match"])
    return out


def latin_token_mask(tokenizer):
    """Vocabulary ids whose decoded text holds a Latin-script letter and no letter of another script
    (for the teacher Latin-script probability mass of §5.4 / §5.5)."""
    import torch
    size = len(tokenizer)
    mask = torch.zeros(size, dtype=torch.bool)
    for i in range(size):
        piece = tokenizer.decode([i])
        letters = [ch for ch in piece if unicodedata.category(ch).startswith("L")]
        if letters and all(script_of(ch) == "latin" for ch in letters):
            mask[i] = True
    return mask
