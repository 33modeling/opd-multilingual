"""Data (design §5.1): download, parallel pool P, per-seed splits, evaluation sets E_l, manifest.

Sources (revisions pinned; sha256 of every file checked after download and written to the manifest):
  MGSM8KInstruct  Mathoctopus/GSM8KInstruct_Parallel @ dd8076b8, MGSM8KInstruct_Parallel.json. JSON lines with
                  fields prompt / chosen / reject. The instruction prompt names the language ("... in Bengali.
                  Please answer in Bengali.") and holds the question between "### Instruction:\\n" and
                  "\\n\\n### Response:". Records come in groups that start with the English record; group k is
                  GSM8K train row k (checked against GSM8K train questions). Some groups lack languages (bn, ru, zh).
  GSM8K train     openai/grade-school-math @ 3101c7d5 (GitHub raw), train.jsonl; gold = number after "####".
                  (The Hugging Face copy openai/gsm8k is parquet only; GitHub raw avoids a pyarrow dependency.)
  MGSM            juletxara/mgsm @ f52417ca (the last revision with TSV files), mgsm_{lang}.tsv: question<TAB>answer,
                  250 rows; Telugu is not used.
  MSVAMP          Mathoctopus/MSVAMP @ 301e2b3b, test_{Language}.json: JSON lines with m_query (question in the
                  language), query (English; absent in the English file), response (gold, e.g. "8.0"). First 250.

Hugging Face files are fetched from $HF_ENDPOINT (default https://huggingface.co; this machine normally sets
https://hf-mirror.com).

Pool and splits:
  P0  the English STAGE-TRANSITION pool: 1024 GSM8K train ids "train:i" drawn exactly as the English record
      (random.Random(stable_seed(20261102, "stage-data-pool")).sample(sorted(ids), 1024)).
  P   the items of P0 with all ten L10 translations; when fewer than pool_size (1024), filled with complete items
      in the order of a seeded permutation of the other train ids (label "mlopd-pool-fill").
  S0  256 items drawn once from P with split_seed (shared by every seed so that stage-0 measurements never use an
      item that some seed trains on).
  F_s, U_s  112 and 128 items drawn per seed from P - S0 (F first, then U from the rest). Same ids in every language.
"""
from __future__ import annotations

from fractions import Fraction
import json
import os
from pathlib import Path
import random
import re
import urllib.request

from .scoring import canonical_number
from .util import L10, LANGUAGE_NAMES, draw, now, read_json, sha256_file, sha256_json, stable_seed, write_json

HF_DEFAULT = "https://huggingface.co"
SOURCES = {
    "mgsm8kinstruct": {"kind": "hf", "repo": "Mathoctopus/GSM8KInstruct_Parallel",
                       "revision": "dd8076b83bacd7c6ade2829317e8a500c6ffbec4",
                       "files": {"MGSM8KInstruct_Parallel.json":
                                 "57864b498f8ea0bc0f8903e8fd912c079ea1a824fad23b1077a0149b455f5022"},
                       "license": "apache-2.0"},
    "gsm8k_train": {"kind": "url", "repo": "openai/grade-school-math",
                    "revision": "3101c7d5072418e28b9008a6636bde82a006892c",
                    "url": "https://raw.githubusercontent.com/openai/grade-school-math/{revision}/grade_school_math/data/{file}",
                    "files": {"train.jsonl": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"},
                    "license": "MIT"},
    "mgsm": {"kind": "hf", "repo": "juletxara/mgsm", "revision": "f52417ca77bd71e9888ddc29f92587660725d2b4",
             "files": {"mgsm_bn.tsv": "6b00bc7cc635547e866989284afa924d789b5affa2eb8de623c385dd943ad977",
                       "mgsm_de.tsv": "4dfea30fede44b813e2e496f5f0534049e83c7cc8aed6e00600b30ec62053626",
                       "mgsm_en.tsv": "50021d0f28cc957edcb44e7806425b1c7fbd648ddcb9e0a8ec689d10e57d40fa",
                       "mgsm_es.tsv": "5bd27ebdf00140cec845c5298dc17715f5cd57d8edda6df1976db40bf3f0750a",
                       "mgsm_fr.tsv": "36207c1c03fd7cd3ea491441eb755408199f94e4a647851df531bbaedc99d606",
                       "mgsm_ja.tsv": "59a2b50debe77981fd784cb3b2bef1505e3abf2a37116dc9d7a366ab029b4637",
                       "mgsm_ru.tsv": "6bd30fd2e80c5bac23f566fb4ae0e6a55a19578401fbf9a01fb21beb3645fef8",
                       "mgsm_sw.tsv": "2bac828d77229e65d7c7197b1ad4a2cb5b1fe99b163f2cbdd66501de6a2115c4",
                       "mgsm_th.tsv": "f3932dc5ad8e9d0ea82b017adc1e1461dd647af861e7166d6741986602a0cfd6",
                       "mgsm_zh.tsv": "b2fa63151022370a0de1f4211c8c284eae74b0f5a3b003b1d5982c0d4a73f661"},
             "license": "cc-by-sa-4.0"},
    "msvamp": {"kind": "hf", "repo": "Mathoctopus/MSVAMP", "revision": "301e2b3b168be70058c89c21c2fbdc9262102add",
               "files": {"test_Bengali.json": "d1c0bb8e3a93127bdffab516ef9bb739ae5007f9071ae97c447c294b1fd65e82",
                         "test_Chinese.json": "31e136303b56ed1734d3e9fbcdcff4499f974a0857298816c41e1c5c0be6b4bd",
                         "test_English.json": "9431a24e993a8f83fd602393ffbb368e1d91fa36b2110ad8f1e0ec319792eed0",
                         "test_French.json": "3a756d228f3d9ee219b673f9dda091b6d64f7c4d0e71045d6a6dbf315539e0ea",
                         "test_German.json": "f5b854da48da762163fb3ccf60307bb9a97e0efac2fd44fdb5ba0969ee1464aa",
                         "test_Japanese.json": "04f89b370bb22a6a78de99fc36e30658186149858172abf8e68665890e81322b",
                         "test_Russian.json": "05d2d317c24f1180c0839d7813f6bebcaeeb57d919cd2d748a0b9964fd77a1c0",
                         "test_Spanish.json": "0aecc731e945d196c8eb38ddb5ad683821e5fd72c55273db1f2233d559495275",
                         "test_Swahili.json": "15f863a79b7e4adf820d76b61044f4e4489f7710d14128ad7b4e915e133a2c16",
                         "test_Thai.json": "60b81f4458f584f637a35e57e85f503173ef67434a83a449c6ee6670a730e9f8"},
               "license": "apache-2.0"},
}
NAME_TO_CODE = {name: code for code, name in LANGUAGE_NAMES.items()}
ENGLISH_POOL_SEED, ENGLISH_POOL_SIZE = 20261102, 1024
LANGUAGE_RE = re.compile(r"in (\w+)\. Please answer in")


# ---------------------------------------------------------------- download

def source_url(source, filename):
    spec = SOURCES[source]
    if spec["kind"] == "hf":
        endpoint = os.environ.get("HF_ENDPOINT", HF_DEFAULT).rstrip("/")
        return f"{endpoint}/datasets/{spec['repo']}/resolve/{spec['revision']}/{filename}"
    return spec["url"].format(revision=spec["revision"], file=filename)


def fetch(source, filename, cache):
    """Download one pinned file into cache/<source>/<revision>/ and check its sha256."""
    spec = SOURCES[source]
    path = Path(cache) / source / spec["revision"][:12] / filename
    expected = spec["files"][filename]
    if path.exists() and sha256_file(path) == expected:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = source_url(source, filename)
    tmp = path.with_name(path.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "mlopd-data/0.1"})
    with urllib.request.urlopen(request, timeout=600) as response, open(tmp, "wb") as handle:
        while block := response.read(1 << 20):
            handle.write(block)
    digest = sha256_file(tmp)
    if digest != expected:
        tmp.unlink()
        raise ValueError(f"{url}: sha256 {digest} differs from the pinned {expected}")
    os.replace(tmp, path)
    return path


def fetch_all(cache):
    return {source: {name: fetch(source, name, cache) for name in spec["files"]} for source, spec in SOURCES.items()}


# ---------------------------------------------------------------- parsing

def gsm8k_gold(answer):
    return canonical_number(Fraction(answer.rsplit("####", 1)[1].strip().replace(",", "")))


def instruction_question(prompt):
    return prompt.split("### Instruction:\n", 1)[1].rsplit("\n\n### Response:", 1)[0].strip()


def _squash(text):
    return " ".join(text.split())


# A translation more than DEGENERATE_RATIO times longer (in characters) than its English question is a broken
# machine translation: in MGSM8KInstruct ten such records (nine bn, one th) repeat one word hundreds of times and
# are 9-34 times the English length, while every other translation is at most 1.8 times. Items with one are not
# complete, so the pool fill replaces them (prompts stay short enough for max_context without truncation).
DEGENERATE_RATIO = 4.0


def degenerate(questions):
    """Languages whose translation is more than DEGENERATE_RATIO times the English question's length."""
    english = len(questions.get("en", ""))
    if not english:
        return []
    return [l for l in L10 if l != "en" and len(questions.get(l, "")) > DEGENERATE_RATIO * english]


def parse_parallel(lines, gsm8k_rows):
    """Group MGSM8KInstruct records into items (a new item starts at every English record) and attach GSM8K gold.
    Returns ({train id: item}, report)."""
    groups, current = [], None
    unknown = 0
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        match = LANGUAGE_RE.search(record["prompt"])
        name = match.group(1) if match else None
        code = NAME_TO_CODE.get(name)
        if code == "en":
            current = {"questions": {}, "duplicates": []}
            groups.append(current)
        if current is None or code is None:
            unknown += 1
            continue
        if code in current["questions"]:
            current["duplicates"].append(code)
        current["questions"][code] = instruction_question(record["prompt"])
    if len(groups) != len(gsm8k_rows):
        raise ValueError(f"{len(groups)} English groups but {len(gsm8k_rows)} GSM8K train rows")
    items, mismatch = {}, []
    for index, (group, row) in enumerate(zip(groups, gsm8k_rows)):
        if _squash(group["questions"]["en"]) != _squash(row["question"]):
            mismatch.append(index)
        items[f"train:{index}"] = {"id": f"train:{index}", "gsm8k_index": index, "gold": gsm8k_gold(row["answer"]),
                                   "english_question": row["question"].strip(), "questions": group["questions"],
                                   "complete": not group["duplicates"] and all(l in group["questions"] for l in L10)
                                               and not degenerate(group["questions"])}
    if len(mismatch) > 0.01 * len(groups):
        raise ValueError(f"{len(mismatch)} groups do not match GSM8K train questions; grouping rule broken")
    report = {"groups": len(groups), "records_outside_L10": unknown, "english_question_mismatch": mismatch,
              "complete_items": sum(i["complete"] for i in items.values()),
              "degenerate_translations": sorted([k, l] for k, i in items.items() for l in degenerate(i["questions"])),
              "per_language": {l: sum(l in i["questions"] for i in items.values()) for l in L10}}
    return items, report


def english_pool(train_ids, seed=ENGLISH_POOL_SEED, size=ENGLISH_POOL_SIZE):
    """English record stage_data_campaign.draw_pool, unchanged."""
    return sorted(random.Random(stable_seed(seed, "stage-data-pool")).sample(sorted(train_ids), size))


def parallel_pool(items, size=ENGLISH_POOL_SIZE, seed=ENGLISH_POOL_SEED):
    base = english_pool(items, seed, ENGLISH_POOL_SIZE)
    pool = [x for x in base if items[x]["complete"]]
    from_english = len(pool)
    if len(pool) < size:
        rest = sorted(set(items) - set(base))
        order = random.Random(stable_seed(seed, "mlopd-pool-fill")).sample(rest, len(rest))
        pool += [x for x in order if items[x]["complete"]][:size - len(pool)]
    return sorted(pool), {"english_pool": len(base), "complete_in_english_pool": from_english,
                          "filled": len(pool) - from_english, "size": len(pool)}


def parse_mgsm(text, language):
    rows = []
    for k, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        question, answer = line.rsplit("\t", 1)
        rows.append({"id": f"mgsm:{k}", "set": "mgsm", "lang": language, "question": question.strip(),
                     "gold": canonical_number(Fraction(answer.strip().replace(",", "")))})
    return rows


def parse_msvamp(text, language, count=250):
    rows = []
    for k, line in enumerate(l for l in text.splitlines() if l.strip()):
        if k >= count:
            break
        record = json.loads(line)
        rows.append({"id": f"msvamp:{k}", "set": "msvamp", "lang": language, "question": record["m_query"].strip(),
                     "english_question": record.get("query", record["m_query"]).strip(),
                     "gold": canonical_number(Fraction(record["response"].strip()))})
    return rows


def build(cache, out, *, pool_size=ENGLISH_POOL_SIZE, msvamp_count=250):
    """Download (or reuse) the pinned files and write out/processed.json plus out/manifest.json."""
    files = fetch_all(cache)
    gsm8k = [json.loads(l) for l in Path(files["gsm8k_train"]["train.jsonl"]).read_text().splitlines() if l.strip()]
    lines = Path(files["mgsm8kinstruct"]["MGSM8KInstruct_Parallel.json"]).read_text().splitlines()
    items, report = parse_parallel(lines, gsm8k)
    pool, pool_report = parallel_pool(items, pool_size)
    english_mgsm = parse_mgsm(Path(files["mgsm"]["mgsm_en.tsv"]).read_text(encoding="utf-8"), "en")
    evaluation = {}
    for language in L10:
        mgsm = parse_mgsm(Path(files["mgsm"][f"mgsm_{language}.tsv"]).read_text(encoding="utf-8"), language)
        for row, english in zip(mgsm, english_mgsm):
            row["english_question"] = english["question"]
            if row["gold"] != english["gold"]:
                raise ValueError(f"MGSM {language} {row['id']} gold differs from English")
        msvamp = parse_msvamp(Path(files["msvamp"][f"test_{LANGUAGE_NAMES[language]}.json"]).read_text(encoding="utf-8"),
                              language, msvamp_count)
        evaluation[language] = mgsm + msvamp
    processed = {"pool": pool, "items": {x: items[x] for x in pool}, "eval": evaluation}
    out = Path(out)
    write_json(out / "processed.json", processed)
    manifest = {"created": now(), "hf_endpoint": os.environ.get("HF_ENDPOINT", HF_DEFAULT),
                "sources": {s: {"repo": spec["repo"], "revision": spec["revision"], "license": spec["license"],
                                "files": {n: {"sha256": sha256_file(p), "bytes": Path(p).stat().st_size}
                                          for n, p in files[s].items()}}
                            for s, spec in SOURCES.items()},
                "parallel": report, "pool": pool_report,
                "eval_counts": {l: {"mgsm": sum(r["set"] == "mgsm" for r in v), "msvamp": sum(r["set"] == "msvamp" for r in v)}
                                for l, v in evaluation.items()},
                "processed_sha256": sha256_file(out / "processed.json"),
                "processed_content_sha256": sha256_json(processed)}
    write_json(out / "manifest.json", manifest)
    return processed, manifest


# ---------------------------------------------------------------- experiment-side access

class Dataset:
    """Read-only view of processed.json (real data or a test fixture with the same schema)."""

    def __init__(self, processed, *, path=None):
        self.raw = processed
        self.path = path
        self.pool = list(processed["pool"])
        self.items = processed["items"]
        self.eval = processed["eval"]
        self.sha256 = sha256_json(processed)

    @classmethod
    def load(cls, path):
        return cls(read_json(path), path=str(path))

    def languages(self):
        return [l for l in L10 if all(l in self.items[x]["questions"] for x in self.pool)]

    def problem(self, item_id, language):
        item = self.items[item_id]
        return {"id": item_id, "lang": language, "question": item["questions"][language], "gold": item["gold"],
                "english_question": item["english_question"], "rationale": item.get("rationales", {}).get(language)}

    def problems(self, ids, language):
        return [self.problem(x, language) for x in ids]

    def eval_problems(self, language, sets=("mgsm", "msvamp"), limit=None):
        rows = [dict(r) for r in self.eval[language] if r["set"] in sets]
        if limit:
            rows = [r for r in rows if int(r["id"].split(":")[1]) < limit]
        return rows

    def splits(self, seed, *, split_seed, sizes):
        """sizes: {"S0": 256, "F": 112, "U": 128}. S0 is shared (split_seed); F and U depend on seed."""
        s0 = draw(self.pool, sizes["S0"], split_seed, "S0")
        rest = sorted(set(self.pool) - set(s0))
        f = draw(rest, sizes["F"], seed, "F")
        u = draw(sorted(set(rest) - set(f)), sizes["U"], seed, "U")
        return {"S0": s0, "F": f, "U": u}
