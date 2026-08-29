"""Tiny GPT that runs entirely on CPU.

Vocab is fixed at 16 tokens (0-9, +, -, =, space, BOS, EOS).
block_size (max context) is 1024 by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Vocab (16 tokens, fixed by spec) --------------------------------------
TOKENS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
          "+", "-", "=", " ", "<BOS>", "<EOS>"]
TOK_TO_ID = {tok: i for i, tok in enumerate(TOKENS)}
VOCAB_SIZE = len(TOKENS)  # 16


@dataclass
class TinyGPTConfig:
    vocab_size: int = VOCAB_SIZE   # 16
    block_size: int = 1024         # max context length
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64               # "d 无所谓" — keep it tiny
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TinyGPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device), diagonal=0).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: TinyGPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: TinyGPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: TinyGPTConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TinyGPTConfig()
        self.transformer = nn.Sequential(*[Block(self.cfg) for _ in range(self.cfg.n_layer)])
        self.ln_f = nn.LayerNorm(self.cfg.n_embd)
        self.token_embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.n_embd)
        self.position_embedding = nn.Embedding(self.cfg.block_size, self.cfg.n_embd)
        self.drop = nn.Dropout(self.cfg.dropout)
        self.head = nn.Linear(self.cfg.n_embd, self.cfg.vocab_size, bias=False)
        # Weight tie (embedding <-> head) — saves params, helps tiny models.
        self.head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"input length {T} exceeds block_size {self.cfg.block_size}"
        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)
        x = self.transformer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_parameters(self, non_embedding: bool = False) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params

    def configure_optimizers(self, weight_decay: float, learning_rate: float,
                             betas: tuple[float, float], device_type: str) -> dict:
        # LayerNorm weights, biases, and the head/embedding (tied) get no decay.
        decay = [p for n, p in self.named_parameters()
                 if p.requires_grad and "bias" not in n and "ln_" not in n and "head" not in n]
        nodecay = [p for n, p in self.named_parameters()
                   if p.requires_grad and ("bias" in n or "ln_" in n or "head" in n)]
        grouped = [{"params": decay, "weight_decay": weight_decay},
                   {"params": nodecay, "weight_decay": 0.0}]
        # CPU-only AdamW (no fused kernels, no torch.cuda).
        optim = torch.optim.AdamW(grouped, lr=learning_rate, betas=betas)
        return {"optimizer": optim, "param_groups": grouped}
