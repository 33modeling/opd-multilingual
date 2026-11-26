"""Minimal LoRA (no peft dependency): y = W x + (alpha / r) * B A x on chosen nn.Linear modules.

A is initialised like peft (uniform in +-1/sqrt(in_features), i.e. kaiming_uniform with a = sqrt(5)) from a seeded
generator, B = 0, so a fresh adapter leaves the model unchanged. Adapter parameters are kept in float32 and the
update is added in the base dtype.
"""
from __future__ import annotations

import hashlib
import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, generator):
        super().__init__()
        self.base = base
        self.rank, self.alpha = rank, alpha
        self.scale = alpha / rank
        bound = 1. / math.sqrt(base.in_features)
        a = (torch.rand(rank, base.in_features, generator=generator) * 2 - 1) * bound
        self.lora_A = nn.Parameter(a.to(base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        self.enabled = True

    def forward(self, x):
        out = self.base(x)
        if not self.enabled:
            return out
        delta = (x.to(self.lora_A.dtype) @ self.lora_A.t()) @ self.lora_B.t()
        return out + (self.scale * delta).to(out.dtype)


def inject(model, *, targets=("q_proj", "v_proj"), rank=4, alpha=8, seed=0):
    """Replace every nn.Linear whose attribute name is in targets; freeze everything else. Returns {name: param}."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    generator = torch.Generator().manual_seed(int(seed))
    names = sorted(name for name, module in model.named_modules()
                   if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in targets)
    if not names:
        raise ValueError(f"no Linear modules named {targets}")
    for name in names:
        parent_name, child = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child, LoRALinear(getattr(parent, child), rank, alpha, generator))
    return adapter_parameters(model)


def reinitialise(model, seed):
    """Fresh adapter in place: A from a generator seeded with `seed` (modules in sorted name order), B = 0."""
    generator = torch.Generator().manual_seed(int(seed))
    modules = sorted((n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear))
    with torch.no_grad():
        for _, module in modules:
            bound = 1. / math.sqrt(module.base.in_features)
            a = (torch.rand(module.rank, module.base.in_features, generator=generator) * 2 - 1) * bound
            module.lora_A.copy_(a.to(module.lora_A))
            module.lora_B.zero_()


def adapter_parameters(model):
    return {n: p for n, p in model.named_parameters() if "lora_" in n}


def set_enabled(model, enabled):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.enabled = enabled


def adapter_state(model):
    return {n: p.detach().cpu().clone() for n, p in adapter_parameters(model).items()}


def load_adapter_state(model, state):
    params = adapter_parameters(model)
    if set(params) != set(state):
        raise ValueError("adapter parameter names differ from the checkpoint")
    with torch.no_grad():
        for name, parameter in params.items():
            parameter.copy_(state[name].to(parameter))


def policy_hash(model):
    sha = hashlib.sha256()
    for name, parameter in sorted(adapter_parameters(model).items()):
        sha.update(name.encode())
        sha.update(parameter.detach().cpu().float().contiguous().numpy().tobytes())
    return sha.hexdigest()
