import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int, alpha: float, dropout: float, tag: str):
        super().__init__()
        self.base = base_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.tag = tag
        in_f = base_linear.in_features
        out_f = base_linear.out_features
        self.A = nn.Parameter(torch.zeros(in_f, r)) 
        self.B = nn.Parameter(torch.zeros(r, out_f))
        torch.nn.init.normal_(self.A, mean=0.0, std=0.02)
        nn.init.zeros_(self.B)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = F.linear(x, self.base.weight)
        lora_update = self.dropout(x) @ self.A @ self.B
        return out + self.scaling * lora_update

class LoRa:
    def __init__(self, r=0, alpha=1.0, dropout=0.0, apply_to=("key","query","value","output")):
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.apply_to = set(apply_to)

    def wrap(self, linear: nn.Linear, tag: str) -> nn.Module:
        if self.r > 0 and tag in self.apply_to:
            return LoRALinear(linear, self.r, self.alpha, self.dropout, tag)
        return linear

    def mark_only_lora_trainable(self, module: nn.Module):
        if self.r <= 0:
            return
        for p in module.parameters():
            p.requires_grad = False
        for m in module.modules():
            if isinstance(m, LoRALinear):
                m.A.requires_grad = True
                m.B.requires_grad = True
                m.base.weight.requires_grad = False