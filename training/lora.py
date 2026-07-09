"""Minimal LoRA for the GCT transformer blocks: y = W x + (B A x) * alpha/r,
with W frozen and B initialised to zero, so training starts exactly at the
released model.  merge() folds the update back into W, giving an ordinary
checkpoint the pipeline loads like the original."""
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):

    def __init__(self, base: nn.Linear, r=16, alpha=16):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        y = self.base(x)
        d = (x.to(self.lora_A.dtype) @ self.lora_A.t()) @ self.lora_B.t()
        return y + (d * self.scale).to(y.dtype)

    @torch.no_grad()
    def merged(self):
        w = self.base.weight.float() + self.scale * (self.lora_B @ self.lora_A)
        return w.to(self.base.weight.dtype)


TARGETS = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


def add_lora(module, r=16, alpha=16, targets=TARGETS):
    """Wrap matching nn.Linear children of every block in `module`."""
    wrapped = []
    for name, sub in list(module.named_modules()):
        if isinstance(sub, nn.Linear) and any(name.endswith(t) for t in targets):
            parent_name, _, leaf = name.rpartition(".")
            parent = module.get_submodule(parent_name)
            setattr(parent, leaf, LoRALinear(sub, r, alpha).to(sub.weight.device))
            wrapped.append(name)
    return wrapped


def lora_params(model):
    return [p for n, p in model.named_parameters() if "lora_" in n]


def lora_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n}


@torch.no_grad()
def merged_state_dict(model):
    """State dict with every LoRALinear folded back into a plain Linear."""
    sd = {}
    lora_mods = {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    for n, t in model.state_dict().items():
        if "lora_" in n:
            continue
        key = n.replace(".base.", ".")
        mod_name = n.rsplit(".base.", 1)[0] if ".base." in n else None
        if mod_name in lora_mods and n.endswith(".base.weight"):
            t = lora_mods[mod_name].merged()
        sd[key] = t.detach().cpu()
    return sd
