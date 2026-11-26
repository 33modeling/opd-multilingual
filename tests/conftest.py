"""CPU fixtures: a byte-level tokenizer with a Qwen-like chat template, tiny random Qwen3 models, a three-language
fixture dataset (en, ru, zh) and a stub language identifier. Nothing is downloaded."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

torch.set_num_threads(4)

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}{% endif %}")


class StubIdentifier:
    """Stands in for fastText lid.176: 'en' when ASCII letters are the majority, else 'xx'."""
    def predict(self, text):
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return "unk"
        ascii_share = sum(c.isascii() for c in letters) / len(letters)
        words = text.lower().split()
        if ascii_share > 0.5:
            return "es" if any(w in ("el", "la", "los", "tiene", "manzanas") for w in words) else "en"
        return "xx"


QUESTIONS = {
    "en": "Tom has {a} apples and buys {b} more. How many apples does he have?",
    "ru": "У Тома {a} яблок, и он покупает ещё {b}. Сколько яблок у него?",
    "zh": "汤姆有{a}个苹果，又买了{b}个。他有多少个苹果？",
}
RATIONALES = {
    "en": "He has {a} + {b} = {c} apples.\n#### {c}",
    "ru": "У него {a} + {b} = {c} яблок.\n#### {c}",
    "zh": "他有{a}+{b}={c}个苹果。\n#### {c}",
}


def make_tokenizer(path):
    """Character-level tokenizer (every token decodes to one valid character, so random models emit letters of
    several scripts), with Qwen-style special tokens and chat template."""
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    chars = [chr(c) for c in range(32, 127)] + ["\n", "\u2028"]
    extra = "".join(QUESTIONS.values()) + "".join(RATIONALES.values()) + "яблокигэтомлбыхьсзжцчшщюфпрёУСТОМА汤姆有个苹果又买了他多少所以一共是七"
    for ch in extra:
        if ch not in chars:
            chars.append(ch)
    vocab = {"[UNK]": 0}
    for ch in chars:
        vocab.setdefault(ch, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tok.decoder = decoders.Fuse()
    tok.add_special_tokens([AddedToken(t, special=True) for t in ("<|endoftext|>", "<|im_start|>", "<|im_end|>")])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                   unk_token="[UNK]")
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(path)
    return fast


def make_model(path, tokenizer, *, layers, seed):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    config = Qwen3Config(vocab_size=len(tokenizer) + 5, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                         max_position_embeddings=1024, tie_word_embeddings=True,
                         eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    model = Qwen3ForCausalLM(config)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def make_dataset(path, languages=("en", "ru", "zh"), pool=40, evaluation=4):
    items = {}
    for k in range(pool):
        a, b = 2 + k, 3 + (k % 5)
        items[f"train:{k}"] = {"id": f"train:{k}", "gsm8k_index": k, "gold": str(a + b),
                               "english_question": QUESTIONS["en"].format(a=a, b=b),
                               "questions": {l: QUESTIONS[l].format(a=a, b=b) for l in languages},
                               "rationales": {l: RATIONALES[l].format(a=a, b=b, c=a + b) for l in languages},
                               "complete": True}
    ev = {}
    for l in languages:
        rows = []
        for s in ("mgsm", "msvamp"):
            for k in range(evaluation):
                a, b = 10 + k, 1 + k
                rows.append({"id": f"{s}:{k}", "set": s, "lang": l, "question": QUESTIONS[l].format(a=a, b=b),
                             "english_question": QUESTIONS["en"].format(a=a, b=b), "gold": str(a + b)})
        ev[l] = rows
    processed = {"pool": sorted(items), "items": items, "eval": ev}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(processed, ensure_ascii=False))
    return processed


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    base = tmp_path_factory.mktemp("tiny")
    tokenizer = make_tokenizer(base / "tok")
    student = make_model(base / "student", tokenizer, layers=2, seed=1)
    teacher = make_model(base / "teacher", tokenizer, layers=3, seed=2)
    x_student = make_model(base / "x_student", tokenizer, layers=1, seed=3)
    x_teacher = make_model(base / "x_teacher", tokenizer, layers=2, seed=4)
    return {"base": base, "tokenizer": tokenizer, "student": str(student), "teacher": str(teacher),
            "x_student": str(x_student), "x_teacher": str(x_teacher)}


@pytest.fixture(scope="session")
def stub_identifier():
    from mlopd.records import register_identifier
    identifier = StubIdentifier()
    register_identifier("stub", identifier)
    return identifier


def smoke_overrides(tiny, data_path, runs="runs"):
    return {
        "paths": {"data": str(data_path), "runs": runs},
        "languages": ["en", "ru", "zh"],
        "model_pairs": {"main": {"student": tiny["student"], "teacher": tiny["teacher"]},
                        "x": {"student": tiny["x_student"], "teacher": tiny["x_teacher"]}},
        "runtime": {"device": "cpu", "cpu_dtype": "float32", "vocab_chunk": 7, "sample_batch": 4,
                    "max_context": 1024},
        "splits": {"S0": 6, "F": 6, "U": 4, "pool_size": 40, "split_seed": 5},
        "seeds": {"grid": [1, 2], "grid_extra": [3, 4], "allocation": [5, 6], "washout": [7, 8], "x": [1, 2]},
        "train": {"checkpoint_every": 1, "handoff": {"updates": 2, "batch": 2, "extended_updates": 3},
                  "opd": {"updates": 2, "samples": 2, "max_new": 5, "segment_first": 2},
                  "rl": {"updates": 2, "samples": 2, "max_new": 5},
                  "sft_continue": {"updates": 2, "batch": 2}, "opd_mask": {"first": 2, "answer_line": True},
                  "allocation": {"updates": 4}, "washout_sft": {"updates": 2, "batch": 2}},
        "teacher_targets": {"samples": 2, "max_new": 5, "source": "gold_rationale"},
        "eval": {"samples": 2, "max_new_tokens": 5},
        "stage0": {"samples": 2, "max_new_tokens": 5, "latin_items": 2, "latin_positions": 2,
                   "s0d_tags": ["r0", "r1"]},
        "gates": {"p_min": 0.0, "q_min": 0.0, "verified_min": 1, "format_ratio": 0.0, "min_languages_for_m3": 3,
                  "range_min": -1.0, "rl_sd_max_pp": 1000.0, "rl_english_above_start": 0, "opd_sd_max_pp": 1000.0,
                  "latin_mass_gap": -1.0, "teacher_ratio_gap": -1.0, "gap_manipulation": -1.0,
                  "imprint_languages_needed": 1, "gap_languages_needed": 1,
                  "washout_sft_gap_pp": -1000.0, "washout_positive_pilots": 0, "washout_pilot_seeds": 2,
                  "scorer_agreement": 0.0, "m3_pilot_spread_pp": -1.0},
        "selection": {"m1_count": 2, "m4_count": 2, "m5_group": 1, "b_languages": 1},
        "langid": {"backend": "stub"},
        "analysis": {"bootstrap": 200, "permutations": 500},
        "queue": {"min_free_gb": 0},
    }
