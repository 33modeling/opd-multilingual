import json

from mlopd import data
from mlopd.util import stable_seed


def test_stable_seed_matches_english_rule():
    import hashlib
    assert stable_seed(87, "x", 3) == int(hashlib.sha256(b"87:x:3").hexdigest()[:8], 16)


def test_parse_parallel_groups_and_gold():
    def record(lang, question):
        return json.dumps({"prompt": f"Below ... in {lang}. Please answer in {lang}.\n\n### Instruction:\n{question}\n\n### Response:",
                           "chosen": "", "reject": ""})
    names = ["English", "Bengali", "German", "Spanish", "French", "Japanese", "Russian", "Swahili", "Thai", "Chinese"]
    lines = [record(n, f"q0 {n}") for n in names] + [record(n, f"q1 {n}") for n in names if n != "Bengali"]
    lines[0] = record("English", "Tom has 3 apples.")
    gsm = [{"question": "Tom has  3 apples.", "answer": "3 apples\n#### 1,003"}, {"question": "q1 English", "answer": "#### 4"}]
    items, report = data.parse_parallel(lines, gsm)
    assert items["train:0"]["complete"] and not items["train:1"]["complete"]
    assert items["train:0"]["gold"] == "1003" and items["train:1"]["questions"]["de"] == "q1 German"
    assert report["per_language"]["bn"] == 1


def test_english_pool_rule_and_fill():
    ids = [f"train:{i}" for i in range(3000)]
    pool = data.english_pool(ids)
    assert len(pool) == 1024 and pool == sorted(pool) and pool == data.english_pool(list(reversed(ids)))
    items = {x: {"complete": (int(x.split(":")[1]) % 7 != 0)} for x in ids}
    full, report = data.parallel_pool(items)
    assert len(full) == 1024 and report["filled"] == 1024 - report["complete_in_english_pool"]
    assert all(items[x]["complete"] for x in full)


def test_splits_deterministic_disjoint_shared_ids(tmp_path):
    from conftest import make_dataset
    make_dataset(tmp_path / "p.json", pool=60)
    ds = data.Dataset.load(tmp_path / "p.json")
    sizes = {"S0": 10, "F": 12, "U": 8}
    a = ds.splits(1, split_seed=9, sizes=sizes)
    assert a == ds.splits(1, split_seed=9, sizes=sizes)
    b = ds.splits(2, split_seed=9, sizes=sizes)
    assert a["S0"] == b["S0"] and a["F"] != b["F"]
    assert not (set(a["S0"]) & set(a["F"])) and not (set(a["F"]) & set(a["U"])) and not (set(a["S0"]) & set(a["U"]))
    en, zh = ds.problems(a["F"], "en"), ds.problems(a["F"], "zh")
    assert [p["id"] for p in en] == [p["id"] for p in zh] and en[0]["gold"] == zh[0]["gold"]


def test_parse_eval_formats():
    rows = data.parse_mgsm("Q one\t18\nQ two\t2,125\n", "en")
    assert rows[1]["gold"] == "2125" and rows[0]["id"] == "mgsm:0"
    text = "\n".join(json.dumps({"query": "en q", "m_query": f"q{k}", "response": "8.0"}) for k in range(300))
    rows = data.parse_msvamp(text, "th")
    assert len(rows) == 250 and rows[0]["gold"] == "8" and rows[0]["english_question"] == "en q"


def test_hf_endpoint_is_respected(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com/")
    assert data.source_url("mgsm", "mgsm_en.tsv").startswith("https://hf-mirror.com/datasets/juletxara/mgsm/resolve/f52417ca")


def test_degenerate_translation_is_flagged():
    from mlopd.data import degenerate
    english = "Tom has 3 apples and buys 2 more. How many apples does he have?"
    good = {"en": english, "bn": "টমের কাছে ৩টি আপেল আছে এবং সে আরও ২টি কেনে। তার কাছে কয়টি আপেল আছে?"}
    broken = dict(good, bn="টম " + "জন্য " * 200)
    assert degenerate(good) == []
    assert degenerate(broken) == ["bn"]


def test_processed_pool_has_no_degenerate_translation():
    import json
    from pathlib import Path
    from mlopd.data import degenerate
    processed = json.loads((Path(__file__).resolve().parents[1] / "data/processed/processed.json").read_text())
    assert len(processed["pool"]) == 1024
    assert not [x for x in processed["pool"] if degenerate(processed["items"][x]["questions"])]
