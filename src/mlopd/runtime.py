"""Model runtime: load the student (with LoRA) and the frozen teacher, render prompts, sample with per-sequence
seeds, per-token log-probabilities and chunked full-vocabulary forward KL.

Memory plan for one 6 GB GPU: both models in bf16; with teacher_offload the teacher sits on the CPU and moves to the
GPU only for its no-grad passes. Full-vocabulary rows are never held for a whole sequence on the GPU: teacher
log-probabilities are computed `vocab_chunk` positions at a time and parked on the CPU, and the student side
recomputes its logits per chunk inside torch.utils.checkpoint, so only one [chunk, V] block lives at a time.

Sampling is temperature 1 without top-k/top-p (or greedy). Each sequence has its own torch.Generator seeded from a
stable_seed, and a token is drawn by the Gumbel-max rule on that generator's noise, so a sequence's sample does not
depend on which other sequences share its batch (needed for resumable runs and for shared rollout tags across arms).
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from . import lora
from .prompts import student_messages, teacher_messages


def _dtype(name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def tokenizer_fingerprint(tokenizer):
    backend = getattr(tokenizer, "backend_tokenizer", None)
    return backend.to_str() if backend is not None else repr(tokenizer.get_vocab())


class Runtime:
    def __init__(self, models, settings, *, adapter_seed, load_teacher=True):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.models, self.settings = models, settings
        device = settings.get("device", "auto")
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                                   ("cpu" if device == "auto" else device))
        dtype = _dtype(settings["dtype"] if self.device.type == "cuda" else settings.get("cpu_dtype", "float32"))
        local = settings.get("local_files_only", True)
        revisions = models.get("revisions") or {}

        def rev(name):  # pinned commit (configs/h100.json model_revisions); None = the cached default branch
            return revisions.get(name)

        self.tokenizer = AutoTokenizer.from_pretrained(models["student"], local_files_only=local, revision=rev(models["student"]))
        self.offload = bool(settings.get("teacher_offload", True)) and self.device.type == "cuda"
        self.student = AutoModelForCausalLM.from_pretrained(models["student"], local_files_only=local, dtype=dtype,
                                                            revision=rev(models["student"]))
        self.student.to(self.device).eval()
        lora_cfg = settings["lora"]
        lora.inject(self.student, targets=tuple(lora_cfg["targets"]), rank=lora_cfg["rank"],
                    alpha=lora_cfg["alpha"], seed=adapter_seed)
        self.adapter_seed = adapter_seed
        self.initial_adapter = lora.adapter_state(self.student)
        if settings.get("gradient_checkpointing", True) and self.device.type == "cuda":
            self.student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.teacher = None
        if load_teacher:
            other = AutoTokenizer.from_pretrained(models["teacher"], local_files_only=local, revision=rev(models["teacher"]))
            if tokenizer_fingerprint(other) != tokenizer_fingerprint(self.tokenizer):
                raise ValueError("teacher and student tokenizers differ; full-vocabulary OPD needs a shared tokenizer")
            self.teacher = AutoModelForCausalLM.from_pretrained(models["teacher"], local_files_only=local, dtype=dtype,
                                                                 revision=rev(models["teacher"]))
            self.teacher.requires_grad_(False).eval()
            self.teacher.to("cpu" if self.offload else self.device)
        eos = {self.tokenizer.eos_token_id}
        config_eos = getattr(self.student.generation_config, "eos_token_id", None)
        eos |= set(config_eos if isinstance(config_eos, list) else [config_eos])
        self.eos = sorted(e for e in eos if e is not None)
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eos[0]
        self.chunk = int(settings.get("vocab_chunk", 64))
        self.max_context = int(settings.get("max_context", 2048))
        self._latin = None
        self._teacher_depth = 0  # nested teacher_ready() calls move the teacher only once

    # ------------------------------------------------------------ prompts
    def render(self, messages):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def student_prompt(self, problem, extra_line=None):
        return self.render(student_messages(problem["question"], extra_line))

    def teacher_prompt(self, problem, condition):
        return self.render(teacher_messages(problem["question"], condition, problem.get("english_question")))

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def latin_mask(self):
        if self._latin is None:
            from .langid import latin_token_mask
            self._latin = latin_token_mask(self.tokenizer)
        return self._latin

    # ------------------------------------------------------------ devices and adapters
    @contextmanager
    def teacher_ready(self):
        if self.teacher is None:
            raise RuntimeError("runtime was built without a teacher")
        outer = self._teacher_depth == 0
        if self.offload and outer:
            self.teacher.to(self.device)
        self._teacher_depth += 1
        try:
            yield self.teacher
        finally:
            self._teacher_depth -= 1
            if self.offload and outer:
                self.teacher.to("cpu")
                torch.cuda.empty_cache()

    @contextmanager
    def initial_student(self):
        """The untrained student (adapters off)."""
        lora.set_enabled(self.student, False)
        try:
            yield self.student
        finally:
            lora.set_enabled(self.student, True)

    def reset_adapter(self, seed=None):
        """Fresh adapter (B = 0): with a seed, A is drawn from that seed; otherwise the construction-time A."""
        if seed is None:
            lora.load_adapter_state(self.student, self.initial_adapter)
        else:
            lora.reinitialise(self.student, seed)

    def parameters(self):
        return lora.adapter_parameters(self.student)

    def optimizer(self, lr):
        return torch.optim.AdamW(list(self.parameters().values()), lr=lr, weight_decay=0., betas=(.9, .999), eps=1e-8)

    def policy_hash(self):
        return lora.policy_hash(self.student)

    # ------------------------------------------------------------ sampling
    @torch.no_grad()
    def sample(self, network, prompts, seeds, *, max_new, greedy=False, batch_seqs=32):
        """One response per (prompt, seed). Returns [{response_ids, truncated}] in input order."""
        from transformers import DynamicCache

        if len(prompts) != len(seeds):
            raise ValueError("one seed per prompt")
        if prompts and max(map(len, prompts)) + max_new > self.max_context:
            raise ValueError("prompt + max_new exceeds max_context; no silent truncation")
        network.eval()
        eos = set(self.eos)
        out = [None] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        for start in range(0, len(order), batch_seqs):
            chunk = order[start:start + batch_seqs]
            rows = [prompts[i] for i in chunk]
            width = max(map(len, rows))
            ids = torch.tensor([[self.pad_id] * (width - len(p)) + list(p) for p in rows], device=self.device)
            mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in rows], device=self.device)
            position = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
            generators = []
            for i in chunk:
                g = torch.Generator(device=self.device)
                g.manual_seed(int(seeds[i]) % (2 ** 63))
                generators.append(g)
            cache = DynamicCache()
            logits = network(input_ids=ids, attention_mask=mask, position_ids=position, past_key_values=cache,
                             use_cache=True, logits_to_keep=1).logits[:, -1].float()
            alive = list(range(len(chunk)))
            responses = [[] for _ in chunk]
            truncated = [False] * len(chunk)
            following = position[:, -1] + 1
            for _ in range(max_new):
                if not torch.isfinite(logits).all():
                    raise RuntimeError("nonfinite logits during sampling")
                if greedy:
                    token = logits.argmax(-1)
                else:
                    noise = torch.stack([torch.rand(logits.shape[-1], generator=generators[r], device=self.device)
                                         for r in alive])
                    exponential = -torch.log(noise.clamp_min(1e-20))
                    gumbel = -torch.log(exponential.clamp_min(1e-20))
                    token = (logits.log_softmax(-1) + gumbel).argmax(-1)
                keep = []
                for k, (r, t) in enumerate(zip(alive, token.tolist())):
                    responses[r].append(t)
                    if t in eos:
                        continue
                    if len(responses[r]) >= max_new:
                        truncated[r] = True
                        continue
                    keep.append(k)
                if not keep:
                    break
                if len(keep) < len(alive):
                    index = torch.tensor(keep, device=self.device)
                    cache.batch_select_indices(index)
                    mask, token, following = mask[index], token[index], following[index]
                    alive = [alive[k] for k in keep]
                mask = torch.cat([mask, torch.ones(len(alive), 1, dtype=mask.dtype, device=self.device)], 1)
                logits = network(input_ids=token[:, None], attention_mask=mask, position_ids=following[:, None],
                                 past_key_values=cache, use_cache=True).logits[:, -1].float()
                following = following + 1
            del cache, logits
            for k, i in enumerate(chunk):
                out[i] = {"response_ids": responses[k], "truncated": truncated[k]}
        return out

    # ------------------------------------------------------------ scoring passes
    def hidden(self, network, prompt_ids, response_ids):
        """Final hidden states at the positions that predict each response token: [T, H]."""
        if not response_ids:
            raise ValueError("empty response")
        tokens = torch.tensor([list(prompt_ids) + list(response_ids)], device=self.device)
        base = getattr(network, network.base_model_prefix)
        states = base(input_ids=tokens, use_cache=False).last_hidden_state[0]
        start = len(prompt_ids) - 1
        return states[start:start + len(response_ids)]

    @staticmethod
    def _head(network):
        head = network.get_output_embeddings()
        return head.weight, head.bias

    @torch.no_grad()
    def teacher_logprobs(self, prompt_ids, response_ids, *, latin_positions=16):
        """Teacher log-probabilities over the full vocabulary for every response position, as float32 CPU chunks,
        plus the teacher probability mass on Latin-script tokens at the first `latin_positions` positions."""
        with self.teacher_ready() as teacher:
            states = self.hidden(teacher, prompt_ids, response_ids)
            weight, bias = self._head(teacher)
            chunks = []
            for s in range(0, states.shape[0], self.chunk):
                logits = F.linear(states[s:s + self.chunk], weight, bias).float()
                chunks.append(logits.log_softmax(-1).cpu())
            del states
        logq = torch.cat(chunks)
        mask = self.latin_mask()
        k = min(latin_positions, logq.shape[0])
        latin = logq[:k, :mask.shape[0]][:, mask].exp().sum(-1) if k else torch.zeros(0)
        return logq, latin.tolist()

    def student_logprob_and_kl(self, prompt_ids, response_ids, teacher_logq=None):
        """Student log-prob of each response token and, when teacher_logq is given, the per-position forward KL
        sum_v q(v) (log q(v) - log p(v)). Both carry gradients; computed per vocab_chunk positions with
        activation checkpointing."""
        states = self.hidden(self.student, prompt_ids, response_ids)
        weight, bias = self._head(self.student)
        tokens = torch.tensor(list(response_ids), device=self.device)
        logps, kls = [], []
        for s in range(0, states.shape[0], self.chunk):
            h = states[s:s + self.chunk]
            t = tokens[s:s + self.chunk]
            if teacher_logq is None:
                logps.append(checkpoint(_logp_chunk, h, weight, bias, t, use_reentrant=False))
            else:
                q = teacher_logq[s:s + self.chunk].to(self.device)
                lp, kl = checkpoint(_logp_kl_chunk, h, weight, bias, t, q, use_reentrant=False)
                logps.append(lp)
                kls.append(kl)
        return torch.cat(logps), (torch.cat(kls) if kls else None)

    @torch.no_grad()
    def sequence_logprob(self, network, prompt_ids, response_ids):
        states = self.hidden(network, prompt_ids, response_ids)
        weight, bias = self._head(network)
        tokens = torch.tensor(list(response_ids), device=self.device)
        return torch.cat([_logp_chunk(states[s:s + self.chunk], weight, bias, tokens[s:s + self.chunk])
                          for s in range(0, states.shape[0], self.chunk)]).cpu()


def _logp_chunk(h, weight, bias, tokens):
    logp = F.linear(h, weight, bias).float().log_softmax(-1)
    return logp.gather(1, tokens[:, None]).squeeze(1)


def _logp_kl_chunk(h, weight, bias, tokens, logq):
    logp = F.linear(h, weight, bias).float().log_softmax(-1)
    width = min(logp.shape[-1], logq.shape[-1])
    kl = (logq[:, :width].exp() * (logq[:, :width] - logp[:, :width])).sum(-1)
    return logp.gather(1, tokens[:, None]).squeeze(1), kl
