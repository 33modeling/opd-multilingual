"""Answer scoring with the number normalization of design Appendix B (§6.1).

Rules (Appendix B):
  1. Unicode NFKC (full-width digits and signs, including the full-width "＃", become ASCII).
  2. Native digits -> ASCII: Bengali (০-৯), Thai (๐-๙), Devanagari (०-९), Arabic-Indic (٠-٩) and Extended
     Arabic-Indic (۰-۹). Han numerals and number words are left alone.
  3. Space-like separators between digits are removed (space, no-break and narrow no-break space, thin space,
     apostrophes ' and ’). Interpretation used here: a separator is removed only when it separates digit groups,
     i.e. when exactly three digits follow it (thousands grouping such as "1 000" or "1'000"), so that two
     separate numbers such as "3 4" are not merged.
  4. The number string is read two ways: (a) comma = thousands, period = decimal; (b) period = thousands,
     comma = decimal. A reading whose thousands groups are not three digits is dropped. The answer is correct when
     any remaining reading equals the gold within 1e-4. When the two readings differ and exactly one equals the
     gold, the response is counted in `dual_reading_decided` (the over-scoring upper bound of rule 7).
  5. Currency symbols (Unicode category Sc) and percent signs (%, ‰, ٪) are removed before extraction. Unit words
     need no removal because extraction only reads digit strings.
  6. strict reads the number right after the last "####" (only whitespace in between) after rules 1-5.
  7. lenient_ascii = lenient with rules 2 and 4 off (ASCII digits only, commas dropped, period = decimal).

Also returned: lenient_legacy and strict_legacy, the English record's rules on the raw text (for the scorer gate of
§5.4), and lenient_trim (§6.1: a truncated response loses its unfinished last line before lenient_norm).
"""
from __future__ import annotations

from fractions import Fraction
import re
import unicodedata

TOLERANCE = Fraction(1, 10000)
_NATIVE_ZEROS = (0x09E6, 0x0E50, 0x0966, 0x0660, 0x06F0)
DIGIT_MAP = {zero + k: ord("0") + k for zero in _NATIVE_ZEROS for k in range(10)}
_SEPARATORS = "    '’"
SEPARATOR_RE = re.compile(r"(?<=[0-9])[" + _SEPARATORS + r"](?=[0-9]{3}(?![0-9]))")
PERCENT = "%‰٪"
NUMBER_RE = re.compile(r"[-+]?[0-9](?:[0-9.,]*[0-9])?(?:/[0-9]+)?")
READ_A = (re.compile(r"[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?"), re.compile(r"[0-9]+(?:\.[0-9]+)?"))
READ_B = (re.compile(r"[0-9]{1,3}(?:\.[0-9]{3})+(?:,[0-9]+)?"), re.compile(r"[0-9]+(?:,[0-9]+)?"))
MARKER = "####"
LEGACY_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
LEGACY_STRICT = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)")


def _drop_symbols(text):
    return "".join(ch for ch in text if unicodedata.category(ch) != "Sc" and ch not in PERCENT)


def normalize(text, *, native_digits=True):
    """Rules 1, 2 (optional), 5 and 3, in that order (symbols go first so that "$1 000" groups correctly)."""
    text = unicodedata.normalize("NFKC", text)
    if native_digits:
        text = text.translate(DIGIT_MAP)
    text = _drop_symbols(text)
    return SEPARATOR_RE.sub("", text)


def parse_gold(gold):
    """Gold answers are canonical ASCII ("18", "8.0", "-3", "1,000" from some sources)."""
    if isinstance(gold, (int, float, Fraction)):
        return Fraction(str(gold))
    return Fraction(str(gold).strip().replace(",", ""))


def readings(number, *, dual=True):
    """Values of one number string under rule 4 (dual=True) or the single legacy reading (dual=False)."""
    sign = -1 if number.startswith("-") else 1
    body = number.lstrip("+-")
    if "/" in body:
        num, den = body.split("/", 1)
        if not num.isdigit() or not den.isdigit() or int(den) == 0:
            return []
        return [sign * Fraction(int(num), int(den))]
    if not dual:
        try:
            return [sign * Fraction(body.replace(",", ""))]
        except (ValueError, ZeroDivisionError):
            return []
    values = []
    if any(r.fullmatch(body) for r in READ_A):
        values.append(sign * Fraction(body.replace(",", "")))
    if any(r.fullmatch(body) for r in READ_B):
        value = sign * Fraction(body.replace(".", "").replace(",", "."))
        if value not in values:
            values.append(value)
    return values


def _match(values, gold):
    return [abs(v - gold) <= TOLERANCE for v in values]


def _judge(values, gold):
    """(correct, dual_reading_decided)."""
    hits = _match(values, gold)
    return any(hits), len(values) == 2 and sum(hits) == 1


def last_number(text, *, dual=True):
    """The last number string with at least one valid reading, and its readings."""
    for match in reversed(list(NUMBER_RE.finditer(text))):
        values = readings(match.group(), dual=dual)
        if values:
            return match.group(), values
    return None, []


def strict_number(text, *, dual=True):
    position = text.rfind(MARKER)
    if position < 0:
        return None, []
    match = re.match(r"\s*(" + NUMBER_RE.pattern + r")", text[position + len(MARKER):])
    if not match:
        return None, []
    return match.group(1), readings(match.group(1), dual=dual)


def legacy_lenient(text, gold):
    """English record rule (stage_data_campaign.lenient_answer + grade): last parseable number on the raw text."""
    target = str(parse_gold(gold))
    for raw in reversed(LEGACY_NUMBER.findall(text)):
        try:
            return int(str(Fraction(raw.replace(",", ""))) == target)
        except (ValueError, ZeroDivisionError):
            continue
    return 0


def legacy_strict(text, gold):
    """English record rule (route_features.extract_answer + grade)."""
    matches = LEGACY_STRICT.findall(text)
    if not matches:
        return 0
    try:
        return int(str(Fraction(matches[-1].replace(",", ""))) == str(parse_gold(gold)))
    except (ValueError, ZeroDivisionError):
        return 0


def trim_unfinished_line(text):
    cut = text.rstrip("\n").rfind("\n")
    return text[:cut] if cut >= 0 else ""


def has_marker_line(text):
    return any(line.lstrip().startswith(MARKER) for line in unicodedata.normalize("NFKC", text).splitlines())


def strict_correct(text, gold):
    """RL reward (design §5.2: strict reward with the normalization)."""
    _, values = strict_number(normalize(text))
    return int(_judge(values, parse_gold(gold))[0])


def score(text, gold, *, truncated=False):
    gold_value = parse_gold(gold)
    norm = normalize(text)
    s_raw, s_values = strict_number(norm)
    l_raw, l_values = last_number(norm)
    strict, strict_dual = _judge(s_values, gold_value)
    lenient, lenient_dual = _judge(l_values, gold_value)
    _, a_values = last_number(normalize(text, native_digits=False), dual=False)
    trimmed = trim_unfinished_line(text) if truncated else text
    _, t_values = last_number(normalize(trimmed))
    return {"strict": int(strict), "lenient_norm": int(lenient),
            "lenient_ascii": int(_judge(a_values, gold_value)[0]),
            "lenient_trim": int(_judge(t_values, gold_value)[0]),
            "lenient_legacy": legacy_lenient(text, gold), "strict_legacy": legacy_strict(text, gold),
            "strict_answer": s_raw, "lenient_answer": l_raw, "has_marker_line": int(has_marker_line(text)),
            "strict_dual_decided": int(strict_dual), "lenient_dual_decided": int(lenient_dual)}


def canonical_number(value):
    """ASCII form used in rewritten answer lines: integers without separators, otherwise a plain decimal."""
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    as_float = float(value)
    text = f"{as_float:.6f}".rstrip("0").rstrip(".")
    return text
