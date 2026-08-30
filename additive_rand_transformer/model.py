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
    attn_type: str = "causal"      # "causal" | "linear" | "dsa"
    attn_topk: int = 8             # for "dsa": keys kept per query
    n_experts: int = 0             # MoE: number of experts (0 = plain MLP)
    moe_topk: int = 2              # MoE: experts active per token
    moe_aux: float = 0.01          # MoE: load-balance aux loss coefficient
    lora_rank: int = 0             # LoRA: rank of adapters (0 = no LoRA)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TinyGPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_embd = cfg.n_embd
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


class MoE(nn.Module):
    """Sparse Mixture-of-Experts replacing the per-block MLP.

    A router (n_embd -> n_experts) picks the top-k experts per token; outputs
    are combined by softmax gate weights. `aux_loss` (switch-style load
    balance) is stored per forward and added in TinyGPT.forward.

    Note: at k=2 of E experts the *active* parameters per token are 2 MLPs,
    but the total parameters are E MLPs — the MoE comparison is "same total
    params, fewer active per token" or "more total params, same active".
    """

    def __init__(self, cfg: TinyGPTConfig) -> None:
        super().__init__()
        self.n_experts = cfg.n_experts
        self.topk = min(cfg.moe_topk, cfg.n_experts)
        self.aux_coef = cfg.moe_aux
        self.router = nn.Linear(cfg.n_embd, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([MLP(cfg) for _ in range(cfg.n_experts)])
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        logits = self.router(x)                      # (B,T,E)
        probs = torch.softmax(logits, dim=-1)
        gate, idx = probs.topk(self.topk, dim=-1)    # (B,T,k)
        # switch-style load-balance aux loss
        # fraction of tokens assigned to each expert:
        onehot = torch.zeros_like(probs).scatter_(-1, idx, 1.0)
        frac = onehot.mean(dim=(0, 1))               # (E,)
        mean_prob = probs.mean(dim=(0, 1))           # (E,)
        self.aux_loss = self.n_experts * (frac * mean_prob).sum()

        # gather expert outputs
        gate_n = gate / gate.sum(dim=-1, keepdim=True)
        flat = x.reshape(B * T, C)
        flat_g = gate_n.reshape(B * T, self.topk)
        flat_i = idx.reshape(B * T, self.topk)
        out = torch.zeros(B * T, C, device=x.device)
        for e in range(self.n_experts):
            mask = (flat_i == e).any(dim=-1)         # tokens that route to e
            if not mask.any():
                continue
            toks = flat[mask]
            e_out = self.experts[e](toks)
            w = flat_g[mask] * (flat_i[mask] == e).float()
            out[mask] += (w.sum(dim=-1, keepdim=True) * e_out)
        return out.reshape(B, T, C)


class LinearAttention(nn.Module):
    """Katharopoulos et al. linear attention (vectorized cumsum form).

    phi(x) = elu(x) + 1 applied to q/k; O(T * d_head^2) instead of O(T^2).
    Uses a running cumulative (phi(k)^T v) accumulator — the classic linear KV
    cache: per-step generation cost O(d^2) regardless of context, memory O(d^2).
    The vectorized cumsum builds an intermediate [.., T, d, d] (fine at our
    training widths ~44 tokens); a chunked form would be needed for very long
    sequences.
    """

    def __init__(self, cfg: TinyGPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_embd = cfg.n_embd
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.resid_drop = nn.Dropout(cfg.dropout)

    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = self._phi(q), self._phi(k)
        # kv_cum[t] = cumsum over t'<=t of phi(k[t']) * v[t'] : [B,H,T,d,d]
        kvt = torch.einsum("bhtd,bhte->bhtde", k, v)
        kv_cum = torch.cumsum(kvt, dim=2)
        z_cum = torch.cumsum(k, dim=2)
        num = torch.einsum("bhtd,bhtde->bhte", q, kv_cum)
        den = torch.einsum("bhtd,bhtd->bht", q, z_cum).unsqueeze(-1)
        y = num / (den + 1e-6)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.resid_drop(self.proj(y))


class DSAAttention(nn.Module):
    """Data-dependent Sparse Attention: keep only the top-k most relevant keys
    per query (selected by score), softmax over those, sparsify the rest.

    The sparsity mask is *data-dependent* (differs per query from the input),
    like DSA-style sparse attention. Cost is still O(T^2) for score computation
    on CPU here (torch lacks an efficient top-k scatter), but the attention
    matrix is sparsified to k/T density, which matters for KV-cache generation.
    """

    def __init__(self, cfg: TinyGPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_embd = cfg.n_embd
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.topk = cfg.attn_topk
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
        # data-dependent top-k selection per query
        k = min(self.topk, T)
        topk_vals, topk_idx = att.topk(k, dim=-1)
        att = torch.full_like(att, float("-inf"))
        att.scatter_(-1, topk_idx, topk_vals)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.resid_drop(self.proj(y))


class LoRALinear(nn.Module):
    """Low-Rank Adaptation of a frozen nn.Linear: y = Wx + alpha/r * x @ A @ B.

    Base weight is frozen (requires_grad=False); only A (in x r) and B (r x out)
    are trained. B is zero-initialized so LoRA starts as identity.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float = 1.0) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if base.bias is not None:
            self.base.bias.requires_grad_(False)
        in_f, out_f = base.weight.shape[1], base.weight.shape[0]
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.empty(in_f, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_f))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A @ self.B) * self.scaling


def apply_lora(model: nn.Module, rank: int, alpha: float = 1.0) -> int:
    """Wrap every nn.Linear in attention+MLP with a LoRALinear adapter.

    Returns number of adapters applied. Embedding/head/LayerNorm untouched
    (head is tied to embedding). Only A/B params remain trainable.
    Snapshot the modules first — mutating the tree while iterating a live
    `model.modules()` generator recurses into the freshly-wrapped submodules.
    """
    n = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank, alpha))
                n += 1
    return n


class Block(nn.Module):
    def __init__(self, cfg: TinyGPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        if cfg.attn_type == "linear":
            self.attn = LinearAttention(cfg)
        elif cfg.attn_type == "dsa":
            self.attn = DSAAttention(cfg)
        else:
            self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        if cfg.n_experts > 0:
            self.mlp = MoE(cfg)
        else:
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
            # MoE load-balance auxiliary loss
            if self.cfg.n_experts > 0:
                for blk in self.transformer:
                    if isinstance(blk.mlp, MoE):
                        loss = loss + self.cfg.moe_aux * blk.mlp.aux_loss
        return logits, loss

    def num_parameters(self, non_embedding: bool = False) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params

    def configure_optimizers(self, weight_decay: float, learning_rate: float,
                             betas: tuple[float, float], device_type: str) -> dict:
        # LayerNorm weights + biases get no decay; everything else (including
        # the tied token/head embedding, registered as "token_embedding") does.
        decay = [p for n, p in self.named_parameters()
                 if p.requires_grad and "bias" not in n and "ln_" not in n]
        nodecay = [p for n, p in self.named_parameters()
                   if p.requires_grad and ("bias" in n or "ln_" in n)]
        grouped = [{"params": decay, "weight_decay": weight_decay},
                   {"params": nodecay, "weight_decay": 0.0}]
        # CPU-only AdamW (no fused kernels, no torch.cuda).
        optim = torch.optim.AdamW(grouped, lr=learning_rate, betas=betas)
        return {"optimizer": optim, "param_groups": grouped}
