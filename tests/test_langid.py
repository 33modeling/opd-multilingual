from mlopd.langid import clean, body_text, language_metrics, script_ratio
from conftest import StubIdentifier

ID = StubIdentifier()


def test_body_before_last_marker_and_cleaning():
    assert body_text("abc\n#### 5") == "abc\n"
    assert "\\frac" not in clean("\\frac{1}{2} = 3 **bold**")


def test_script_ratios_per_language():
    cases = {
        "zh": ("他有三个苹果，所以一共是七个。", 1.0),
        "ja": ("りんごは全部で七個あります。", 1.0),
        "ru": ("У него семь яблок.", 1.0),
        "th": ("เขามีแอปเปิ้ลเจ็ดผล", 1.0),
        "bn": ("তার কাছে সাতটি আপেল আছে।", 1.0),
    }
    for language, (text, expected) in cases.items():
        assert abs(script_ratio(clean(text), language) - expected) < 1e-9, language
    assert script_ratio(clean("He has seven 苹果"), "zh") == 2 / 12
    assert script_ratio(clean("12 + 3 = 15"), "ru") is None


def test_metrics_rows():
    m = language_metrics("He has seven apples.\n#### 7", "en", identifier=ID, lenient_correct=1)
    assert m["r"] == 1.0 and m["language_match"] == 1 and m["language_consistent"] == 1 and m["line_pass"] == 1
    m = language_metrics("У него 7 яблок.\nHe has 7 apples.\n#### 7", "ru", identifier=ID, lenient_correct=1)
    assert 0 < m["r"] < 1 and m["line_pass"] == 0
    m = language_metrics("He has seven apples. El tiene manzanas.", "es", identifier=ID)
    assert 0 < m["r"] < 1
    m = language_metrics("七個のりんご", "ja", identifier=ID)
    assert m["has_kana"] == 1
    m = language_metrics("#### 7", "ru", identifier=ID, lenient_correct=1)
    assert m["r"] is None and m["language_match"] == 0 and m["language_consistent"] == 0


def test_fasttext_missing_gives_clear_error(monkeypatch):
    import builtins
    import pytest
    from mlopd import langid
    real = builtins.__import__

    def fake(name, *args, **kwargs):
        if name == "fasttext":
            raise ImportError("no")
        return real(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match="lid.176"):
        langid.FastTextIdentifier("/nonexistent")
