"""Appendix B, rule by rule."""
from mlopd.scoring import canonical_number, normalize, readings, score, strict_correct
from fractions import Fraction


def test_rule1_nfkc_fullwidth_digits_and_marker():
    s = score("答え ＃＃＃＃ １８", "18")
    assert s["strict"] == 1 and s["lenient_norm"] == 1
    assert s["strict_legacy"] == 0  # the English strict rule needs an ASCII "####"
    assert s["lenient_legacy"] == 1  # Python \\d in the English rule also matches full-width digits


def test_rule2_native_digits():
    assert score("উত্তর #### ১৮", "18")["strict"] == 1          # Bengali
    assert score("คำตอบ #### ๑๘", "18")["strict"] == 1          # Thai
    assert score("उत्तर #### १८", "18")["lenient_norm"] == 1    # Devanagari
    assert score("#### ١٨", "18")["strict"] == 1                # Arabic-Indic
    assert score("#### ۱۸", "18")["strict"] == 1                # Extended Arabic-Indic
    assert score("答案是十八", "18")["lenient_norm"] == 0         # Han numerals are not converted
    assert score("উত্তর ১৮", "18")["lenient_ascii"] == 0          # rule 2 off


def test_rule3_space_like_group_separators():
    for text in ("#### 1 000", "#### 1 000", "#### 1 000", "#### 1'000", "#### 1’000", "#### 1 000"):
        assert score(text, "1000")["strict"] == 1, text
    assert normalize("between 3 4 and 5") == "between 3 4 and 5"  # not a thousands group: not merged
    assert score("2 1 000 000", "1000000")["lenient_norm"] == 1


def test_rule4_dual_reading_and_over_scoring_count():
    assert readings("1,500") == [Fraction(1500), Fraction(3, 2)]
    assert readings("1.500") == [Fraction(3, 2), Fraction(1500)]
    assert readings("12,5") == [Fraction(25, 2)]
    assert readings("1,500.25") == [Fraction("1500.25")]
    assert readings("1.234.567") == [Fraction(1234567)]
    assert readings("1,2,3") == []          # neither reading has valid thousands groups
    s = score("Die Antwort ist 1.500", "1500")
    assert s["lenient_norm"] == 1 and s["lenient_dual_decided"] == 1
    s = score("#### 2,5", "2.5")
    assert s["strict"] == 1 and s["strict_dual_decided"] == 0  # only reading (b) is valid
    assert score("Le total est 12,5 euros", "12.5")["lenient_ascii"] == 0  # rule 4 off: 125
    assert score("#### 18.", "18")["strict"] == 1            # sentence period after the number


def test_rule5_currency_and_percent():
    assert score("#### $18", "18")["strict"] == 1
    assert score("#### 18%", "18")["strict"] == 1
    assert score("#### ৳১,০০০", "1000")["strict"] == 1
    assert score("#### €1.000", "1000")["strict"] == 1


def test_rule6_strict_reads_after_the_last_marker_only():
    assert strict_correct("#### 5\nmore\n#### 18", "18") == 1
    assert strict_correct("#### 18\nthen #### 5", "18") == 0
    assert strict_correct("#### Final Answer: 18", "18") == 0   # English record convention kept
    assert score("#### Final Answer: 18", "18")["lenient_norm"] == 1
    assert strict_correct("The answer is 18", "18") == 0


def test_lenient_variants_and_trim():
    s = score("He has 3 + 4 = 7.\n#### 7", "7")
    assert s["lenient_legacy"] == s["lenient_norm"] == s["lenient_ascii"] == s["strict"] == s["strict_legacy"] == 1
    assert s["has_marker_line"] == 1
    truncated = "So 7 apples.\nThen 12 + 1"
    assert score(truncated, "7", truncated=True)["lenient_trim"] == 1
    assert score(truncated, "7", truncated=True)["lenient_norm"] == 0
    assert score(truncated, "7", truncated=False)["lenient_trim"] == 0
    assert score("x = 8.0", "8.0")["lenient_norm"] == 1 and score("x = 8", "8.0")["lenient_norm"] == 1
    assert score("3/4 of it", "0.75")["lenient_norm"] == 1
    assert score("value 0.33335", "0.3333")["lenient_norm"] == 1  # within 1e-4


def test_canonical_number():
    assert canonical_number(Fraction(18)) == "18"
    assert canonical_number(Fraction("2.5")) == "2.5"
